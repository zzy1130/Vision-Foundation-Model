import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPA_ViT(nn.Module):
    """
    Vision Transformer with Dynamic Resolution Positional Interpolation (CLIPA).
    Supports dynamic start-low, end-high resolution training.
    """
    def __init__(self, default_img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=512, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.default_grid_size = default_img_size // patch_size
        num_patches = self.default_grid_size ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings defined for default resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Dynamically interpolate positional embeddings to fit the input image grid size.
        w, h: grid dimensions (e.g. input_width // patch_size)
        """
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == self.default_grid_size and h == self.default_grid_size:
            return self.pos_embed
        
        # Calculate grid coordinates
        pos_embed = self.pos_embed  # [1, N_default, D]
        dim = pos_embed.shape[-1]
        
        # Interpolate pos_embed [1, default_grid^2, D] to [1, h*w, D]
        # Reshape to [1, default_grid, default_grid, D] -> [1, D, default_grid, default_grid] for grid interpolation
        pos_embed_grid = pos_embed.reshape(1, self.default_grid_size, self.default_grid_size, dim).permute(0, 3, 1, 2)
        
        # 2D bilinear interpolation
        # align_corners=False matches standard scaling
        interpolated_pos_embed = F.interpolate(
            pos_embed_grid,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )  # [1, D, h, w]
        
        # Permute and flatten back to [1, h*w, D]
        interpolated_pos_embed = interpolated_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return interpolated_pos_embed

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Project patches
        x = self.patch_embed(x)  # [B, embed_dim, h_grid, w_grid]
        h_grid, w_grid = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 2. Get interpolated positional embeddings dynamically
        pos_embed = self.interpolate_pos_encoding(x, w_grid, h_grid)
        x = x + pos_embed
        
        # 3. Prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        x = self.blocks(x)
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        return self.proj(cls_rep)

class CLIPA_TextTransformer(nn.Module):
    """
    Standard Transformer for Text Encoder.
    """
    def __init__(self, vocab_size=49408, max_len=77, embed_dim=512, out_dim=512, depth=12, num_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, text, eot_indices):
        B, L = text.shape
        x = self.token_embed(text)
        x = x + self.pos_embed[:, :L, :]
        x = self.blocks(x)
        x = self.norm(x)
        eot_rep = x[torch.arange(B), eot_indices]
        return self.proj(eot_rep)

class CLIPA(nn.Module):
    """
    CLIPA model with Dynamic Resolution capability.
    """
    def __init__(self, vocab_size=49408, embed_dim=512):
        super().__init__()
        self.image_encoder = CLIPA_ViT(out_dim=embed_dim)
        self.text_encoder = CLIPA_TextTransformer(vocab_size=vocab_size, out_dim=embed_dim)
        
    def forward(self, images, text, eot_indices):
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(text, eot_indices)
        
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Testing CLIPA dynamic resolution functionality...")
    model = CLIPA()
    
    # 1. Forward with standard resolution: 224x224
    images_standard = torch.randn(1, 3, 224, 224)
    text = torch.randint(0, 49408, (1, 77))
    eot = torch.tensor([10])
    
    img_emb_std, _ = model(images_standard, text, eot)
    print(f"Image embeddings (224x224): {img_emb_std.shape}")
    
    # 2. Forward with low resolution: 112x112 (start of CLIPA training)
    images_low = torch.randn(1, 3, 112, 112)
    img_emb_low, _ = model(images_low, text, eot)
    print(f"Image embeddings (112x112): {img_emb_low.shape} - Success (Positional embeddings dynamically interpolated!)")
    
    # 3. Forward with high resolution: 448x448 (end of CLIPA training)
    images_high = torch.randn(1, 3, 448, 448)
    img_emb_high, _ = model(images_high, text, eot)
    print(f"Image embeddings (448x448): {img_emb_high.shape} - Success (Positional embeddings dynamically interpolated!)")
