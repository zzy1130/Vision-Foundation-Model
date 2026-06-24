import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss for Language-Image Pre-training (SigLIP).
    This loss treats the pairwise image-text matching task as independent binary classifications,
    eliminating the need for global softmax normalization and enabling O(N) communication scaling.
    """
    def __init__(self, init_t=10.0, init_b=-10.0):
        super().__init__()
        # t is log-scale temperature parameter, b is bias parameter. Both are learnable.
        self.t = nn.Parameter(torch.tensor(init_t))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: Tensor of shape [batch_size, embed_dim] (L2-normalized)
            text_embeds: Tensor of shape [batch_size, embed_dim] (L2-normalized)
        Returns:
            loss: scalar tensor representing the SigLIP loss
        """
        # Calculate similarity matrix: shape [batch_size, batch_size]
        # Since inputs are L2-normalized, matrix multiplication computes cosine similarities.
        sim = torch.matmul(image_embeds, text_embeds.t())
        
        batch_size = sim.shape[0]
        
        # Target matrix: 1 on diagonal (matching pairs), -1 elsewhere
        targets = 2 * torch.eye(batch_size, device=sim.device) - 1.0
        
        # Logits: t * sim + b
        # In SigLIP, both scale (t) and bias (b) are learnable
        scale = torch.exp(self.t)
        logits = sim * scale + self.b
        
        # Pairwise binary cross entropy loss
        # log(sigmoid(logits)) for matching pairs (target=1)
        # log(1 - sigmoid(logits)) = log(sigmoid(-logits)) for mismatched pairs (target=-1)
        loss = -F.logsigmoid(targets * logits).sum() / batch_size
        
        return loss

class SimpleViT(nn.Module):
    """
    A simple Vision Transformer (ViT) implementation as the Image Encoder.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=512, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection head to align with text embeddings
        self.proj = nn.Linear(embed_dim, num_classes)
        
        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: [batch_size, 3, img_size, img_size]
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Append class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)
        
        # Extract representation from cls token
        cls_rep = x[:, 0]  # [B, embed_dim]
        
        # Project to shared embedding space
        out = self.proj(cls_rep)  # [B, num_classes]
        return out

class SimpleTextTransformer(nn.Module):
    """
    A simple Transformer-based Text Encoder.
    """
    def __init__(self, vocab_size=10000, max_len=77, embed_dim=512, out_dim=512, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, eot_indices):
        """
        Args:
            text: Tensor of token IDs shape [batch_size, max_len]
            eot_indices: Tensor of shape [batch_size] containing index of the End of Text token for each sequence
        """
        B, L = text.shape
        x = self.token_embed(text)  # [B, L, embed_dim]
        x = x + self.pos_embed[:, :L, :]
        
        x = self.blocks(x)
        x = self.norm(x)
        
        # Take the representation at the End Of Text (EOT) token
        # (similar to class token pooling in ViT or EOS token pooling in GPT)
        eot_rep = x[torch.arange(B), eot_indices]  # [B, embed_dim]
        
        # Project to shared embedding space
        out = self.proj(eot_rep)  # [B, out_dim]
        return out

class SigLIP(nn.Module):
    """
    Complete SigLIP dual encoder model containing both Vision and Text encoders.
    """
    def __init__(self, vocab_size=10000, embed_dim=512):
        super().__init__()
        self.image_encoder = SimpleViT(num_classes=embed_dim)
        self.text_encoder = SimpleTextTransformer(vocab_size=vocab_size, out_dim=embed_dim)
        
    def forward(self, images, text, eot_indices):
        # 1. Encode image and text
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(text, eot_indices)
        
        # 2. L2 normalize embeddings
        image_embeds = F.normalize(image_feats, p=2, dim=-1)
        text_embeds = F.normalize(text_feats, p=2, dim=-1)
        
        return image_embeds, text_embeds

# Quick local test logic to verify dimensions and loss function behavior
if __name__ == "__main__":
    print("Testing SigLIP implementation components from scratch...")
    
    # Initialize components
    batch_size = 4
    vocab_size = 1000
    embed_dim = 128
    
    model = SigLIP(vocab_size=vocab_size, embed_dim=embed_dim)
    loss_fn = SigLIPLoss()
    
    # Fake inputs
    # Images: B x 3 x 224 x 224
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    # Texts: token indices [B, 77]
    dummy_texts = torch.randint(0, vocab_size, (batch_size, 77))
    # EOT indices: end of sentences (e.g. index 10 for all)
    dummy_eot = torch.tensor([10, 15, 8, 12])
    
    # Forward pass
    img_emb, txt_emb = model(dummy_images, dummy_texts, dummy_eot)
    
    print(f"Image embeddings shape: {img_emb.shape}") # Should be [4, 128]
    print(f"Text embeddings shape: {txt_emb.shape}")  # Should be [4, 128]
    
    # Calculate loss
    loss = loss_fn(img_emb, txt_emb)
    print(f"Calculated SigLIP Loss: {loss.item():.4f}")
    
    # Backward pass test
    loss.backward()
    print("Backward pass completed successfully. All components verified!")
