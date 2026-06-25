import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_v1 import SAM1PromptEncoder, SAM1MaskDecoder

class ConvStem(nn.Module):
    """
    A simple hierarchical CNN downsampler for TinyViT representation.
    Downsamples the input image by 16x.
    """
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256]):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )

    def forward(self, x):
        x = self.stage1(x)  # [B, 64, H/2, W/2]
        x = self.stage2(x)  # [B, 128, H/4, W/4]
        x = self.stage3(x)  # [B, 256, H/16, W/16]
        return x

class MobileImageEncoder(nn.Module):
    """
    TinyViT representation for MobileSAM.
    Highly efficient convolutional stem + hierarchical self-attention layers.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.stem = ConvStem(in_channels=in_channels, embed_dims=[64, 128, embed_dim])
        
        # Lightweight Mobile Attention Block (Self-Attention on downsampled patch tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        features = self.stem(x)  # [B, 256, H/16, W/16]
        B, D, H, W = features.shape
        flat_feats = features.flatten(2).transpose(1, 2)  # [B, N_tokens, D]
        flat_out = self.transformer(flat_feats)
        out_feats = flat_out.transpose(1, 2).reshape(B, D, H, W)
        return out_feats

class MobileSAM(nn.Module):
    """
    Fully-integrated MobileSAM.
    Replaces the heavy ViT image encoder with the lightweight TinyViT image encoder,
    while reusing SAM 1's prompt encoder and mask decoder.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_encoder = MobileImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.prompt_encoder = SAM1PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = SAM1MaskDecoder(embed_dim=embed_dim)

    def forward(self, images, points=None, labels=None, boxes=None, mask_priors=None):
        # 1. Efficient TinyViT feature extraction
        img_feats = self.image_encoder(images)  # [B, D, H_feat, W_feat]
        
        # 2. Reused Prompt encoder
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=points, labels=labels, boxes=boxes, mask_priors=mask_priors, feat_shape=img_feats.shape[2:]
        )
        
        # 3. Reused Mask decoder
        masks, iou_scores = self.mask_decoder(img_feats, sparse_prompts, dense_prompts)
        
        return masks, iou_scores

# Decoupled Knowledge Distillation Illustration
def train_decoupled_distillation_step(student_encoder, teacher_encoder, optimizer, images):
    """
    Mock function representing the Decoupled Knowledge Distillation (DKD) process of MobileSAM.
    DKD focuses ONLY on aligning the student image encoder output with the teacher image encoder output,
    keeping the prompt encoder and mask decoder completely frozen and untouched.
    """
    student_encoder.train()
    teacher_encoder.eval()
    
    # 1. Forward both encoders
    with torch.no_grad():
        teacher_feats = teacher_encoder(images)  # [B, D, H_f, W_f] (ViT-H target representation)
        
    student_feats = student_encoder(images)  # [B, D, H_f, W_f] (TinyViT student representation)
    
    # 2. Compute L2 distillation loss to align feature embeddings
    distill_loss = F.mse_loss(student_feats, teacher_feats)
    
    # 3. Update only the student TinyViT image encoder weights
    optimizer.zero_grad()
    distill_loss.backward()
    optimizer.step()
    
    return distill_loss.item()
