import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) block used in DiT (Peebles & Xie, ICCV 2023).
    Conditions token features on timestep and class embeddings via scale and shift parameters.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Projects conditioning vector y to 6 modulation parameters:
        # scale/shift for Self-Attention LN (gamma1, beta1)
        # scale/shift for FFN LN (gamma2, beta2)
        # scale gates for residual connections (alpha1, alpha2)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

    def forward(self, y):
        """
        Args:
            y: (B, D) condition vector
        Returns:
            gamma1, beta1, gamma2, beta2, alpha1, alpha2 scale/shift parameters
        """
        # Regress modulation parameters
        mod = self.modulation(y).unsqueeze(1)  # (B, 1, D * 6)
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = torch.chunk(mod, 6, dim=-1)
        return gamma1, beta1, gamma2, beta2, alpha1, alpha2


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block.
    Replaces standard ViT block normalization with Adaptive Layer Normalization (AdaLN).
    """
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.adaln = AdaLNBlock(hidden_size)

    def forward(self, x, y):
        """
        Args:
            x: (B, N, D) input tokens
            y: (B, D) conditioning vector
        """
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = self.adaln(y)
        
        # 1. Self-Attention Block with AdaLN modulation
        h1 = (1.0 + gamma1) * self.norm1(x) + beta1
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + alpha1 * attn_out
        
        # 2. Feed-Forward Block with AdaLN modulation
        h2 = (1.0 + gamma2) * self.norm2(x) + beta2
        ffn_out = self.ffn(h2)
        x = x + alpha2 * ffn_out
        
        return x


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) architecture.
    Replaces standard UNet denoising backbones in diffusion models with a ViT backbone.
    """
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=128, num_heads=4, depth=4, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # 1. Patch projection layer (Patchification)
        # Projects 2D latents into tokens: (B, C, H, W) -> (B, (H/p)*(W/p), D)
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (input_size // patch_size) ** 2
        
        # 3D spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # 2. Timestep and Class conditioning MLPs
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.class_embed = nn.Embedding(num_classes, hidden_size)
        
        # 3. Stack of DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads=num_heads) for _ in range(depth)
        ])
        
        # 4. Final Projection Head (Depatchification)
        # Regresses channels back to patch size
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        )
        
        # Initialization
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, z_t, t, class_labels):
        """
        Args:
            z_t: (B, C_latent, H_latent, W_latent) noisy latents
            t: (B, 1) timestep values
            class_labels: (B,) class conditioning labels
        Returns:
            noise_pred: (B, C_latent, H_latent, W_latent) predicted noise
        """
        B, C, H, W = z_t.shape
        
        # 1. Patchify input
        # (B, D, H/p, W/p) -> (B, N, D)
        x = self.patch_embed(z_t).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # 2. Embed conditioning variables
        t_emb = self.time_embed(t.float())  # (B, D)
        c_emb = self.class_embed(class_labels)  # (B, D)
        # Fused conditioning vector y
        y = t_emb + c_emb  # (B, D)
        
        # 3. Process through DiT blocks
        for block in self.blocks:
            x = block(x, y)
            
        # 4. Depatchify output
        x = self.final_layer(x)  # (B, N, p*p*C)
        
        # Reshape back to image grid: (B, H/p, W/p, p, p, C) -> (B, C, H, W)
        p = self.patch_size
        H_p, W_p = H // p, W // p
        x = x.reshape(B, H_p, W_p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        
        return x
