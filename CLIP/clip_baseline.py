import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss used in standard OpenAI CLIP.
    Computes symmetric cross-entropy over a similarity matrix.
    """
    def __init__(self, init_tau=2.659): # ln(1/0.07) ≈ 2.659
        super().__init__()
        # Learnable temperature parameter, clamped to log(100) to prevent overflow
        self.log_temperature = nn.Parameter(torch.tensor(init_tau))

    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: L2-normalized image embeddings [B, D]
            text_embeds: L2-normalized text embeddings [B, D]
        """
        # Calculate similarity matrix: shape [B, B]
        sim = torch.matmul(image_embeds, text_embeds.t())
        
        # Scale by learnable temperature: exp(log_temp) = 1 / temp
        temp = torch.exp(self.log_temperature.clamp(max=4.605)) # Max temperature scale is 100
        logits = sim * temp
        
        # Generate symmetric targets: [0, 1, 2, ..., B-1]
        targets = torch.arange(logits.shape[0], device=logits.device)
        
        # Cross entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        
        return (loss_i2t + loss_t2i) / 2

class CLIPViT(nn.Module):
    """
    Standard Vision Transformer (ViT) for CLIP Image Encoder.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=512, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Standard ViT Encoder Blocks
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
        
        # Projection layer to shared multimodal space
        self.proj = nn.Linear(embed_dim, out_dim)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed
        
        x = self.blocks(x)
        x = self.norm(x)
        
        cls_rep = x[:, 0]
        return self.proj(cls_rep)

class CLIPTextTransformer(nn.Module):
    """
    Standard Transformer for CLIP Text Encoder.
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
        x = self.token_embed(text)  # [B, L, embed_dim]
        x = x + self.pos_embed[:, :L, :]
        
        x = self.blocks(x)
        x = self.norm(x)
        
        # Pool text representation at the End-of-Text (EOT) token
        eot_rep = x[torch.arange(B), eot_indices]
        return self.proj(eot_rep)

class OpenAI_CLIP(nn.Module):
    """
    Baseline OpenAI CLIP model combining Image and Text Encoders.
    """
    def __init__(self, vocab_size=49408, embed_dim=512):
        super().__init__()
        self.image_encoder = CLIPViT(out_dim=embed_dim)
        self.text_encoder = CLIPTextTransformer(vocab_size=vocab_size, out_dim=embed_dim)
        
    def forward(self, images, text, eot_indices):
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(text, eot_indices)
        
        # L2 Normalize
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Testing OpenAI CLIP architecture...")
    model = OpenAI_CLIP()
    loss_fn = InfoNCELoss()
    
    images = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 49408, (2, 77))
    eot = torch.tensor([10, 12])
    
    img_emb, txt_emb = model(images, text, eot)
    loss = loss_fn(img_emb, txt_emb)
    print(f"Symmetric InfoNCE Loss: {loss.item():.4f}")
