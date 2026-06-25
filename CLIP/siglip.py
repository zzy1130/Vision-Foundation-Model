import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttentionPooling(nn.Module):
    """
    Multihead Attention Pooling (MAP) layer.
    Used in SigLIP instead of standard ViT Class Token pooling.
    Uses a learnable query to pool features from sequence of patches.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Learnable latent vector acts as query
        self.latent = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.latent, std=0.02)

    def forward(self, x):
        """
        x: [B, N, dim] (Sequence of patch embeddings)
        """
        B = x.shape[0]
        # Query: [B, 1, dim]
        query = self.latent.expand(B, -1, -1)
        # Cross Attention: Query = query, Key/Value = x
        pooled_out, _ = self.attn(query, x, x)  # [B, 1, dim]
        return self.norm(pooled_out.squeeze(1))  # [B, dim]

class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language-Image Pre-training (SigLIP).
    Treats the pairwise matching task as independent binary classification.
    """
    def __init__(self, init_t=10.0, init_b=-10.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(init_t))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, image_embeds, text_embeds):
        sim = torch.matmul(image_embeds, text_embeds.t())
        batch_size = sim.shape[0]
        targets = 2 * torch.eye(batch_size, device=sim.device) - 1.0
        
        scale = torch.exp(self.t)
        logits = sim * scale + self.b
        
        # Pairwise binary cross entropy
        loss = -F.logsigmoid(targets * logits).sum() / batch_size
        return loss

class SigLIP_ViT(nn.Module):
    """
    SigLIP Vision Encoder with Multihead Attention Pooling (MAP).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=512, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
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
        
        # Multihead Attention Pooling instead of CLS Token
        self.attn_pool = MultiheadAttentionPooling(dim=embed_dim, num_heads=num_heads)
        self.proj = nn.Linear(embed_dim, out_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        x = self.blocks(x)
        x = self.norm(x)
        
        # Pool features using Attention Pooling
        pooled = self.attn_pool(x)  # [B, embed_dim]
        return self.proj(pooled)

class SigLIP_TextTransformer(nn.Module):
    """
    Text Transformer for SigLIP.
    """
    def __init__(self, vocab_size=32000, max_len=64, embed_dim=768, out_dim=512, depth=12, num_heads=12):
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

class SigLIP(nn.Module):
    """
    Complete SigLIP model.
    """
    def __init__(self, vocab_size=32000, embed_dim=512):
        super().__init__()
        self.image_encoder = SigLIP_ViT(out_dim=embed_dim)
        self.text_encoder = SigLIP_TextTransformer(vocab_size=vocab_size, out_dim=embed_dim)

    def forward(self, images, text, eot_indices):
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(text, eot_indices)
        
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Testing SigLIP architecture with MAP (Multihead Attention Pooling)...")
    model = SigLIP()
    loss_fn = SigLIPLoss()
    
    images = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 32000, (2, 64))
    eot = torch.tensor([10, 12])
    
    img_emb, txt_emb = model(images, text, eot)
    loss = loss_fn(img_emb, txt_emb)
    print(f"SigLIP Image embeddings shape: {img_emb.shape}")
    print(f"SigLIP Loss: {loss.item():.4f}")
