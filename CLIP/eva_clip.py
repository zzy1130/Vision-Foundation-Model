import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU MLP layer.
    Replaces standard MLP to improve training stability and performance in large-scale ViTs (EVA-CLIP).
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # In SwiGLU, we project from in_features to hidden_features twice (for gate and value)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        # Swish(xW1) * xW2
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class LayerScaleBlock(nn.Module):
    """
    Transformer block with LayerScale.
    Allows scaling ViT training stably to over 1 Billion parameters.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # LayerScale parameters (initialized to small value like 1e-5)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # Attention with LayerScale
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.gamma_1 * attn_out
        
        # MLP with LayerScale
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x

class EVA_ViT(nn.Module):
    """
    EVA-CLIP Vision Encoder.
    Features: LayerScale, SwiGLU MLP, and LayerNorm stability.
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1024, out_dim=768, depth=24, num_heads=16, init_values=1e-5):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Blocks with LayerScale and SwiGLU
        self.blocks = nn.ModuleList([
            LayerScaleBlock(dim=embed_dim, num_heads=num_heads, init_values=init_values)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        return self.proj(cls_rep)

class EVA_CLIP(nn.Module):
    """
    EVA-CLIP Model wrapping EVA-ViT Vision encoder and Text encoder.
    """
    def __init__(self, vocab_size=49408, embed_dim=768):
        super().__init__()
        # Large scale vision encoder
        self.image_encoder = EVA_ViT(out_dim=embed_dim)
        
        # Standard large-scale Text Encoder (using embed_dim=768)
        self.text_token_embed = nn.Embedding(vocab_size, 768)
        self.pos_embed = nn.Parameter(torch.zeros(1, 77, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.text_blocks = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.text_norm = nn.LayerNorm(768)
        self.text_proj = nn.Linear(768, embed_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode_text(self, text, eot_indices):
        B, L = text.shape
        x = self.text_token_embed(text)
        x = x + self.pos_embed[:, :L, :]
        x = self.text_blocks(x)
        x = self.text_norm(x)
        eot_rep = x[torch.arange(B), eot_indices]
        return self.text_proj(eot_rep)

    def forward(self, images, text, eot_indices):
        image_feats = self.image_encoder(images)
        text_feats = self.encode_text(text, eot_indices)
        
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Testing EVA-CLIP components (LayerScale + SwiGLU)...")
    # Tiny depth for quick local verification
    model = EVA_CLIP()
    
    images = torch.randn(1, 3, 224, 224)
    text = torch.randint(0, 49408, (1, 77))
    eot = torch.tensor([10])
    
    img_emb, txt_emb = model(images, text, eot)
    print(f"EVA-CLIP Image embeddings shape: {img_emb.shape}")
    print(f"EVA-CLIP Text embeddings shape: {txt_emb.shape}")
