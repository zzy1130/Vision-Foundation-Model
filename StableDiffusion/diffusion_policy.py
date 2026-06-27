import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    """
    Visual-proprioceptive encoder for robot observations.
    Encodes visual camera frames (CNN) and joins them with proprioceptive state (joint angles, etc.).
    """
    def __init__(self, img_channels=3, state_dim=6, obs_dim=128):
        super().__init__()
        # Visual backbone (ResNet-like proxy)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projections
        self.img_proj = nn.Linear(64, 96)
        self.state_proj = nn.Linear(state_dim, 32)
        
        self.out_proj = nn.Sequential(
            nn.Linear(96 + 32, obs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obs_dim, obs_dim)
        )

    def forward(self, camera_img, robot_state):
        """
        Args:
            camera_img: (B, C, H, W) camera feed
            robot_state: (B, D_state) joint coordinates or pose
        Returns:
            obs_embed: (B, D_obs) combined observation embedding
        """
        # Visual feature extraction
        x = F.relu(self.bn1(self.conv1(camera_img)))
        x = F.relu(self.bn2(self.conv2(x)))
        img_feats = self.pool(x).flatten(1)  # (B, 64)
        
        # Project visual and state features
        img_proj = self.img_proj(img_feats)
        state_proj = self.state_proj(robot_state)
        
        # Concatenate and project to final observation space
        fused = torch.cat([img_proj, state_proj], dim=-1)
        obs_embed = self.out_proj(fused)
        return obs_embed


class TemporalUNet1D(nn.Module):
    """
    1D Temporal UNet used in Diffusion Policy (RSS 2023).
    Denoises action trajectories along the time/horizon dimension.
    """
    def __init__(self, action_dim=2, obs_dim=128):
        super().__init__()
        # Time step embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )
        
        # Downsampling blocks (1D Conv over action sequence horizon)
        self.down_conv1 = nn.Conv1d(action_dim, 64, kernel_size=5, padding=2)
        # Condition projection
        self.cond_proj = nn.Linear(obs_dim, 64)
        
        # Bottleneck
        self.mid_conv = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # Upsampling blocks
        self.up_conv = nn.Conv1d(64 + 64, 64, kernel_size=5, padding=2)
        
        self.out_conv = nn.Conv1d(64, action_dim, kernel_size=5, padding=2)

    def forward(self, action_t, t, obs_cond):
        """
        Args:
            action_t: (B, T_horizon, D_action) noisy action trajectory
            t: (B, 1) timestep values in [0, T]
            obs_cond: (B, D_obs) observation condition embedding
        """
        # Conv1D expects (B, D_action, T_horizon)
        x = action_t.transpose(1, 2)
        
        # 1. Embed diffusion step t
        t_embed = self.time_mlp(t).unsqueeze(-1)  # (B, 64, 1)
        
        # 2. Project observation condition
        cond_embed = self.cond_proj(obs_cond).unsqueeze(-1)  # (B, 64, 1)
        
        # Down sampling block
        x_down = self.down_conv1(x)
        # Inject condition and time embedding
        x_down = x_down + t_embed + cond_embed
        
        # Mid bottleneck
        x_mid = self.mid_conv(x_down)
        
        # Up sampling block (with skip connections)
        x_up = torch.cat([x_mid, x_down], dim=1)
        x_up = self.up_conv(x_up)
        
        # Output noise prediction
        out = self.out_conv(x_up)
        # Transpose back to (B, T_horizon, D_action)
        return out.transpose(1, 2)


class ActionDDPMScheduler:
    """
    1D Trajectory scheduler for diffusion policy.
    """
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        device = original_samples.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        device = sample.device
        idx = timestep.item()
        
        beta = self.betas[idx].to(device)
        alpha = self.alphas[idx].to(device)
        alpha_cumprod = self.alphas_cumprod[idx].to(device)
        
        if idx > 0:
            alpha_cumprod_prev = self.alphas_cumprod[idx - 1].to(device)
            variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        else:
            variance = torch.tensor(0.0, device=device)
            
        pred_original_sample = (sample - torch.sqrt(1.0 - alpha_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        
        mean = (torch.sqrt(alpha_cumprod_prev if idx > 0 else torch.tensor(1.0, device=device)) * beta * pred_original_sample + 
                torch.sqrt(alpha) * (1.0 - (alpha_cumprod_prev if idx > 0 else torch.tensor(1.0, device=device))) * sample) / (1.0 - alpha_cumprod)
                
        noise = torch.randn_like(sample) if idx > 0 else torch.tensor(0.0, device=device)
        prev_sample = mean + torch.sqrt(variance) * noise
        return prev_sample


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy (RSS 2023) model.
    Learns robot control trajectories by framing action generation as observation-conditioned diffusion.
    """
    def __init__(self, action_dim=2, state_dim=6, obs_dim=128):
        super().__init__()
        self.obs_encoder = ObservationEncoder(img_channels=3, state_dim=state_dim, obs_dim=obs_dim)
        self.unet = TemporalUNet1D(action_dim=action_dim, obs_dim=obs_dim)
        self.scheduler = ActionDDPMScheduler()

    def forward(self, camera_img, robot_state, actions, t):
        """
        Training forward pass. Calculates noise regression MSE loss.
        """
        # 1. Encode observations
        obs_cond = self.obs_encoder(camera_img, robot_state)
        
        # 2. Sample random noise for actions trajectory
        noise = torch.randn_like(actions)
        
        # 3. Add noise to target actions (forward diffusion)
        noisy_actions = self.scheduler.add_noise(actions, noise, t)
        
        # 4. Predict noise (1D UNet reverse diffusion)
        t_float = t.float().view(-1, 1)
        noise_pred = self.unet(noisy_actions, t_float, obs_cond)
        
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def predict_action(self, camera_img, robot_state, action_shape=(1, 16, 2), num_inference_steps=20):
        """
        Inference loop. Iteratively denoises random noise into a valid action trajectory sequence.
        """
        device = camera_img.device
        B = camera_img.shape[0]
        
        # 1. Encode observation
        obs_cond = self.obs_encoder(camera_img, robot_state)
        
        # 2. Sample random action noise trajectory
        actions = torch.randn(B, *action_shape[1:], device=device)
        
        # 3. Denoising iterations
        self.scheduler = ActionDDPMScheduler(num_train_timesteps=num_inference_steps)
        for i in reversed(range(num_inference_steps)):
            timestep = torch.tensor([i], device=device)
            t_float = timestep.float().view(-1, 1).expand(B, -1)
            
            with torch.no_grad():
                noise_pred = self.unet(actions, t_float, obs_cond)
                actions = self.scheduler.step(noise_pred, timestep, actions)
                
        return actions
