import torch
import torch.nn as nn
import torch.nn.functional as F

class FLIPViT(nn.Module):
    """
    Vision Transformer with Random Patch Masking (FLIP).
    Speeds up CLIP training by masking 50-75% of image patches.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=512, depth=12, num_heads=12, mask_ratio=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # In FLIP, positional embeddings are added *before* masking
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
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

    def random_masking(self, x, mask_ratio):
        """
        Perform random patch masking.
        x: [B, N, D]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Generate random noise for each patch position
        noise = torch.rand(B, N, device=x.device)  # [B, N]
        
        # Sort noise to get random indices
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, N]
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        
        # Gather the kept patches
        # ids_keep tensor shape: [B, len_keep], expand to [B, len_keep, D]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_masked = torch.gather(x, dim=1, index=ids_keep_expanded)
        
        return x_masked

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add position embedding to patch embeddings before masking
        x = x + self.pos_embed
        
        # Apply random masking (only during training)
        if self.training and self.mask_ratio > 0.0:
            x = self.random_masking(x, self.mask_ratio)  # [B, num_patches_kept, embed_dim]
            
        # Prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches_kept + 1, embed_dim]
        
        x = self.blocks(x)
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        return self.proj(cls_rep)

class FLIPTextTransformer(nn.Module):
    """
    Text Transformer for FLIP (same as CLIP text encoder).
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

class FLIP(nn.Module):
    """
    FLIP model combining Masked Image Encoder and Text Encoder.
    """
    def __init__(self, vocab_size=49408, embed_dim=512, mask_ratio=0.5):
        super().__init__()
        self.image_encoder = FLIPViT(out_dim=embed_dim, mask_ratio=mask_ratio)
        self.text_encoder = FLIPTextTransformer(vocab_size=vocab_size, out_dim=embed_dim)
        
    def forward(self, images, text, eot_indices):
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(text, eot_indices)
        
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Testing FLIP architecture...")
    # 50% mask ratio
    model = FLIP(mask_ratio=0.5)
    model.train() # Make sure masking is activated
    
    images = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 49408, (2, 77))
    eot = torch.tensor([10, 12])
    
    img_emb, txt_emb = model(images, text, eot)
    print(f"FLIP Image embeddings shape (during training): {img_emb.shape}")
    print(f"FLIP Text embeddings shape: {txt_emb.shape}")
