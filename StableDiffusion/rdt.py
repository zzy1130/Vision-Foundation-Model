import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy import ActionDDPMScheduler

class RDTBlock(nn.Module):
    """
    RDT Block. Implements a Transformer block with Multi-head Self-Attention
    and Adaptive Layer Normalization (AdaLN) for multimodal token sequence reasoning.
    """
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Adaptive Layer Normalization (AdaLN) modulation parameters regressed from time steps
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

    def forward(self, x, t_embed):
        """
        Args:
            x: (B, N_tokens, hidden_size) concatenated multimodal sequence
            t_embed: (B, hidden_size) timestep embedding
        """
        # Regress scale and shift factors
        mod = self.adaln(t_embed).unsqueeze(1)  # (B, 1, hidden_size * 6)
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = torch.chunk(mod, 6, dim=-1)
        
        # 1. Attention layer with AdaLN scale & shift
        h1 = (1.0 + gamma1) * self.norm1(x) + beta1
        attn_out, _ = self.self_attn(h1, h1, h1)
        x = x + alpha1 * attn_out
        
        # 2. Feed-Forward layer with AdaLN scale & shift
        h2 = (1.0 + gamma2) * self.norm2(x) + beta2
        ffn_out = self.ffn(h2)
        x = x + alpha2 * ffn_out
        
        return x


class RDT(nn.Module):
    """
    RDT (Robotics Diffusion Transformer) core backbone.
    Concatenates action tokens, proprioception state tokens, SigLIP image tokens, 
    and T5-XXL language tokens into a unified sequence and processes it.
    """
    def __init__(self, action_dim=128, horizon=64, hidden_size=128, depth=3, num_heads=4,
                 lang_token_dim=4096, img_token_dim=1152, state_token_dim=128,
                 max_lang_cond_len=32, img_cond_len=196):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Modality projection adapters
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.lang_proj = nn.Linear(lang_token_dim, hidden_size)
        self.img_proj = nn.Linear(img_token_dim, hidden_size)
        self.state_proj = nn.Linear(state_token_dim, hidden_size)
        
        # Positional embeddings for each sequence segment
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        self.lang_pos = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_pos = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))
        self.state_pos = nn.Parameter(torch.zeros(1, 1, hidden_size))  # single state step
        
        # Timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Stack of DiT blocks
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads=num_heads) for _ in range(depth)
        ])
        
        # Final output projection head
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Initialize positional embeddings
        nn.init.normal_(self.action_pos, std=0.02)
        nn.init.normal_(self.lang_pos, std=0.02)
        nn.init.normal_(self.img_pos, std=0.02)
        nn.init.normal_(self.state_pos, std=0.02)

    def forward(self, action_t, t, lang_cond, img_cond, state_traj):
        """
        Args:
            action_t: (B, T_horizon, action_dim) noisy actions
            t: (B, 1) diffusion timestep
            lang_cond: (B, L_lang, lang_token_dim) T5 language features
            img_cond: (B, L_img, img_token_dim) SigLIP image features (multi-view)
            state_traj: (B, 1, state_token_dim) proprioception trajectory
        """
        B = action_t.shape[0]
        
        # 1. Project all inputs to hidden_size
        act_tokens = self.action_proj(action_t) + self.action_pos
        lang_tokens = self.lang_proj(lang_cond) + self.lang_pos[:, :lang_cond.shape[1], :]
        img_tokens = self.img_proj(img_cond) + self.img_pos[:, :img_cond.shape[1], :]
        state_tokens = self.state_proj(state_traj) + self.state_pos
        
        # 2. Concatenate tokens into a single unified sequence
        # Sequence: [State, Language, Image, Action]
        seq = torch.cat([state_tokens, lang_tokens, img_tokens, act_tokens], dim=1)
        
        # 3. Project timestep
        t_embed = self.time_mlp(t.float())  # (B, hidden_size)
        
        # 4. Reason through RDT Transformer Blocks
        for block in self.blocks:
            seq = block(seq, t_embed)
            
        # 5. Extract action tokens segment from the end of the sequence
        # State: 1 token, Lang: L_lang tokens, Img: L_img tokens, Action: T_horizon tokens
        start_idx = 1 + lang_cond.shape[1] + img_cond.shape[1]
        out_act_tokens = seq[:, start_idx:, :]
        
        # 6. Project back to action dimensions (predict noise)
        noise_pred = self.final_layer(out_act_tokens)
        return noise_pred


class RDTRunner(nn.Module):
    """
    RDT (Robotics Diffusion Transformer) 1B-scale runner.
    Wraps the core DiT model and handles diffusion training loss and sampling logic.
    """
    def __init__(self, action_dim=128, pred_horizon=64, config=None,
                 lang_token_dim=4096, img_token_dim=1152, state_token_dim=128,
                 max_lang_cond_len=32, img_cond_len=196):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        
        # RDT Architecture Config
        depth = 3 if config is None else config.get('depth', 3)
        num_heads = 4 if config is None else config.get('num_heads', 4)
        hidden_size = 128 if config is None else config.get('hidden_size', 128)
        
        self.model = RDT(
            action_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            lang_token_dim=lang_token_dim,
            img_token_dim=img_token_dim,
            state_token_dim=state_token_dim,
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len
        )
        
        # Diffusion Scheduler
        self.scheduler = ActionDDPMScheduler(num_train_timesteps=100)

    def forward(self, actions, t, lang_cond, img_cond, state_traj):
        """
        Computes RDT training loss (MSE of noise prediction).
        Args:
            actions: (B, T_horizon, action_dim) target unified action trajectories
            t: (B, 1) diffusion timestep
        """
        # 1. Sample noise
        noise = torch.randn_like(actions)
        
        # 2. Add noise to action sequence (forward process)
        noisy_actions = self.scheduler.add_noise(actions, noise, t)
        
        # 3. Predict noise via RDT core DiT
        noise_pred = self.model(noisy_actions, t, lang_cond, img_cond, state_traj)
        
        # 4. Calculate MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def predict_action(self, lang_cond, img_cond, state_traj, num_inference_steps=20):
        """
        Performs reverse diffusion sampling to predict robotic actions.
        Returns:
            actions: (B, T_horizon, action_dim) predicted bimanual action trajectory
        """
        device = lang_cond.device
        B = lang_cond.shape[0]
        
        # Initialize action trajectory from Gaussian noise
        actions = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        self.scheduler = ActionDDPMScheduler(num_train_timesteps=num_inference_steps)
        for i in reversed(range(num_inference_steps)):
            timestep = torch.tensor([i], device=device)
            t_float = timestep.float().view(-1, 1).expand(B, -1)
            
            with torch.no_grad():
                noise_pred = self.model(actions, t_float, lang_cond, img_cond, state_traj)
                actions = self.scheduler.step(noise_pred, timestep, actions)
                
        return actions
