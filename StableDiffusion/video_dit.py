import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_diffusion import CrossAttention2D
from dit import AdaLNBlock

class VAE3D(nn.Module):
    """
    3D Spatiotemporal VAE Proxy.
    Compresses video tensors spatially and temporally into latent codes,
    and decodes them back to video space.
    """
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        # Encoder: compresses spatial by 8x, temporal frames by 2x
        # Input shape: (B, F, C, H, W) -> we transpose to (B, C, F, H, W) for Conv3D
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),   # F/2, H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),  # F/2, H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv3d(128, latent_channels * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1) # F/2, H/8, W/8
        )
        
        # Decoder: restores video dimensions
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), # x2 spatial
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),  # x4 spatial
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, in_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)), # x2 temporal, x8 spatial
            nn.Tanh()
        )

    def encode(self, x):
        # Input shape: (B, F, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        latent = mean + eps * std
        return latent.permute(0, 2, 1, 3, 4)  # (B, F_lat, C_lat, H_lat, W_lat)

    def decode(self, z):
        # Input shape: (B, F_lat, C_lat, H_lat, W_lat)
        z = z.permute(0, 2, 1, 3, 4)  # (B, C_lat, F_lat, H_lat, W_lat)
        out = self.decoder(z)
        return out.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)


class SpatiotemporalCrossAttention(nn.Module):
    """
    Spatiotemporal Cross-Attention layer for Text Injection in Video DiT.
    Injects textual conditioning prompt tokens into video patch tokens.
    """
    def __init__(self, hidden_size, cond_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(cond_dim, hidden_size)
        self.v_proj = nn.Linear(cond_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, text_cond):
        """
        Args:
            x: (B, N_tokens, hidden_size) spatiotemporal video tokens
            text_cond: (B, S, cond_dim) text conditioning embeddings
        """
        B, N, C = x.shape
        
        # Project Query, Key, Value
        q = self.q_proj(x)         # (B, N, C)
        k = self.k_proj(text_cond)  # (B, S, C)
        v = self.v_proj(text_cond)  # (B, S, C)
        
        # Reshape to multi-head format
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, S)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        context = context.transpose(1, 2).reshape(B, N, C)
        
        out = self.out_proj(context)
        return out + x


class SpatiotemporalDiTBlock(nn.Module):
    """
    Video DiT Block combining AdaLN timestep scaling, spatiotemporal self-attention, 
    and cross-attention for text prompt conditioning (text injection).
    """
    def __init__(self, hidden_size, cond_dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        # Text prompt injection layer (Cross-Attention)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = SpatiotemporalCrossAttention(hidden_size, cond_dim, num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.adaln = AdaLNBlock(hidden_size)

    def forward(self, x, y, text_cond):
        """
        Args:
            x: (B, N_tokens, hidden_size) spatiotemporal tokens
            y: (B, hidden_size) timestep/class conditioning
            text_cond: (B, S, cond_dim) text conditioning
        """
        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = self.adaln(y)
        
        # 1. Spatiotemporal Self-Attention (AdaLN modulated)
        h1 = (1.0 + gamma1) * self.norm1(x) + beta1
        attn_out, _ = self.self_attn(h1, h1, h1)
        x = x + alpha1 * attn_out
        
        # 2. Text Injection via Cross-Attention
        h_cross = self.norm_cross(x)
        x = self.cross_attn(h_cross, text_cond)
        
        # 3. FFN Block (AdaLN modulated)
        h2 = (1.0 + gamma2) * self.norm2(x) + beta2
        ffn_out = self.ffn(h2)
        x = x + alpha2 * ffn_out
        
        return x


class VideoDiT(nn.Module):
    """
    Video Diffusion Transformer (Video DiT) architecture.
    Applies spatiotemporal patching on 3D latents (from VAE3D) and processes
    them with spatiotemporal self-attention and text embedding cross-attention.
    """
    def __init__(self, latent_shape=(8, 4, 16, 16), patch_size=(2, 2, 2), hidden_size=128, cond_dim=128, num_heads=4, depth=3):
        super().__init__()
        # latent_shape: (F_lat, C_lat, H_lat, W_lat)
        self.latent_shape = latent_shape
        self.patch_size = patch_size  # (pt, ps, ps) -> temporal patch, spatial patch height/width
        self.hidden_size = hidden_size
        
        pt, ps, _ = patch_size
        F_lat, C_lat, H_lat, W_lat = latent_shape
        
        # 1. 3D Spatiotemporal VAE
        self.vae_3d = VAE3D(in_channels=3, latent_channels=C_lat)
        
        # 2. Spatiotemporal Patch projection: Conv3D with kernel_size = stride = patch_size
        self.patch_embed = nn.Conv3d(C_lat, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (F_lat // pt) * (H_lat // ps) * (W_lat // ps)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # 3. Timestep embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 4. Spatiotemporal DiT Blocks
        self.blocks = nn.ModuleList([
            SpatiotemporalDiTBlock(hidden_size, cond_dim, num_heads=num_heads) for _ in range(depth)
        ])
        
        # 5. Final Depatchification head
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Linear(hidden_size, pt * ps * ps * C_lat)
        )
        
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, z_t, t, text_cond):
        """
        Args:
            z_t: (B, F_lat, C_lat, H_lat, W_lat) noisy video latents
            t: (B, 1) timestep values
            text_cond: (B, S, cond_dim) text conditioning embeddings (CLIP)
        """
        B, F_l, C_l, H_l, W_l = z_t.shape
        pt, ps, _ = self.patch_size
        
        # Transpose z_t to Conv3D format: (B, C_l, F_l, H_l, W_l)
        z_t_conv = z_t.permute(0, 2, 1, 3, 4)
        
        # 1. Patchify both spatially and temporally: (B, hidden_size, F_l/pt, H_l/ps, W_l/ps)
        x = self.patch_embed(z_t_conv)
        
        # Flatten to sequence: (B, hidden_size, N_patches) -> (B, N_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # 2. Embed timestep
        y = self.time_embed(t.float())  # (B, hidden_size)
        
        # 3. Process blocks
        for block in self.blocks:
            x = block(x, y, text_cond)
            
        # 4. Depatchify
        x = self.final_layer(x)  # (B, N_patches, pt * ps * ps * C_l)
        
        # Reshape back to Conv3D grid: (B, F_l/pt, H_l/ps, W_l/ps, pt, ps, ps, C_l) -> (B, C_l, F_l, H_l, W_l)
        F_p, H_p, W_p = F_l // pt, H_l // ps, W_l // ps
        x = x.reshape(B, F_p, H_p, W_p, pt, ps, ps, C_l)
        
        # Permute and fold back: (B, C_l, F_p, pt, H_p, ps, W_p, ps) -> (B, C_l, F_l, H_l, W_l)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, C_l, F_l, H_l, W_l)
        
        # Restore to shape: (B, F_l, C_l, H_l, W_l)
        z_t_pred = x.permute(0, 2, 1, 3, 4)
        
        return z_t_pred
