import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_v1 import SAM1PromptEncoder, SAM1ImageEncoder, TwoWayAttentionBlock

class HQImageEncoder(nn.Module):
    """
    HQ-SAM Image Encoder.
    Outputs both early-stage boundary features (low-level edges) and late-stage semantic features.
    """
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16, grid_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, grid_size, grid_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # We separate the transformer layers to extract intermediate activations
        self.layer1 = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation='gelu'
        )
        self.layer2_to_4 = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation='gelu'
            ) for _ in range(3)
        ])
        
    def forward(self, x):
        p = self.patch_embed(x)  # [B, D, H/16, W/16]
        if p.shape[2:] != self.pos_embed.shape[2:]:
            pos = F.interpolate(self.pos_embed, size=p.shape[2:], mode='bilinear', align_corners=False)
        else:
            pos = self.pos_embed
            
        p = p + pos
        B, D, H, W = p.shape
        x_flat = p.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 1. Early-layer forward pass to capture detailed boundaries
        early_flat = self.layer1(x_flat)
        early_feats = early_flat.transpose(1, 2).reshape(B, D, H, W)
        
        # 2. Complete remaining transformer layers for deep semantic context
        out_flat = early_flat
        for layer in self.layer2_to_4:
            out_flat = layer(out_flat)
            
        late_feats = out_flat.transpose(1, 2).reshape(B, D, H, W)
        return early_feats, late_feats

class HQMaskDecoder(nn.Module):
    """
    HQ-SAM Mask Decoder.
    Injects a learnable HQ Token, and fuses early-stage local boundaries and late-stage global features.
    """
    def __init__(self, embed_dim=256, num_multimask_outputs=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_outputs = num_multimask_outputs
        
        # Add 1 additional HQ-Token alongside original tokens
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)
        self.hq_token = nn.Embedding(1, embed_dim)  # HQ Output Token
        
        # Two-way transformer layers
        self.transformer_layers = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim=embed_dim, nhead=8) for _ in range(2)
        ])
        
        # Up-sampling layers for mask feature reconstruction
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Global-Local Feature Fusion Conv Blocks
        # Fuses early features (fine details) and deep upsampled features
        self.hq_feature_fuse = nn.Sequential(
            nn.Conv2d(embed_dim // 8 * 2, embed_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 8),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 8, kernel_size=3, padding=1)
        )
        
        # Original mask MLPs
        self.coeff_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim // 8)
            ) for _ in range(self.num_outputs)
        ])
        
        # Dedicated HQ Token coefficient MLP (projects to coefficients matching the fused boundary features)
        self.hq_coeff_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 8)
        )
        
        # IoU score MLP
        self.iou_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            # Predict scores for both original multimasks AND the HQ mask
            nn.Linear(embed_dim, self.num_outputs + 1)
        )

    def forward(self, early_feats, late_feats, sparse_prompts, dense_prompts):
        """
        Args:
            early_feats (Tensor): low-level edge features [B, D, H, W]
            late_feats (Tensor): high-level semantic features [B, D, H, W]
            sparse_prompts (Tensor): [B, N_tokens, D]
            dense_prompts (Tensor): [B, D, H, W]
        Returns:
            masks (Tensor): [B, num_outputs + 1, H_up, W_up] (includes the high-quality mask at index -1)
            iou_scores (Tensor): [B, num_outputs + 1]
        """
        B, D, H, W = late_feats.shape
        img_feats_fused = late_feats + dense_prompts
        img_flat = img_feats_fused.flatten(2).transpose(1, 2)  # [B, N_img, D]
        
        # Prepend both mask tokens AND the HQ token
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(B, 1, 1)
        hq_token = self.hq_token.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Concat tokens: [B, mask_tokens + hq_token + prompts, D]
        tokens = torch.cat([mask_tokens, hq_token, sparse_prompts], dim=1)
        
        # Run two-way cross attention layers
        for layer in self.transformer_layers:
            tokens, img_flat = layer(tokens, img_flat)
            
        # Extract representations
        # Index layout:
        # [0, 1, 2] -> original multimasks (Part, Whole, Subpart)
        # [3] -> HQ Token
        # [4] -> IoU score token
        mask_token_embeds = tokens[:, :self.num_outputs, :]  # [B, 3, D]
        hq_token_embed = tokens[:, self.num_outputs, :]      # [B, D]
        iou_token_embed = tokens[:, self.num_outputs + 1, :] # [B, D]
        
        # Standard query coefficients
        coeffs = []
        for i in range(self.num_outputs):
            coeffs.append(self.coeff_mlps[i](mask_token_embeds[:, i, :]))
        coeffs = torch.stack(coeffs, dim=1)  # [B, 3, D//8]
        
        # HQ query coefficients
        hq_coeff = self.hq_coeff_mlp(hq_token_embed).unsqueeze(1)  # [B, 1, D//8]
        
        # Merge all coefficients: [B, 4, D//8]
        all_coeffs = torch.cat([coeffs, hq_coeff], dim=1)
        
        # Predict IoU scores
        iou_scores = self.iou_mlp(iou_token_embed)  # [B, 4]
        
        # Upsample deep features
        img_recon = img_flat.transpose(1, 2).reshape(B, D, H, W)
        upsampled_feats = self.upsampler(img_recon)  # [B, D//8, H*4, W*4]
        
        # Upsample early boundary features for global-local fusion
        upsampled_early = self.upsampler(early_feats)  # [B, D//8, H*4, W*4]
        
        # Concatenate and fuse (Global-Local Fusion)
        fused_hq_feats = self.hq_feature_fuse(
            torch.cat([upsampled_feats, upsampled_early], dim=1)
        )  # [B, D//8, H*4, W*4]
        
        # Generate masks
        B, C_coeff, H_up, W_up = upsampled_feats.shape
        upsampled_feats_flat = upsampled_feats.reshape(B, C_coeff, H_up * W_up)
        fused_hq_feats_flat = fused_hq_feats.reshape(B, C_coeff, H_up * W_up)
        
        # Standard mask output
        std_masks = torch.bmm(coeffs, upsampled_feats_flat)  # [B, 3, N_up]
        
        # HQ mask output (computed using fused global-local boundary features)
        hq_mask = torch.bmm(hq_coeff, fused_hq_feats_flat)    # [B, 1, N_up]
        
        # Concatenate masks: [B, 4, N_up]
        all_masks_flat = torch.cat([std_masks, hq_mask], dim=1)
        all_masks = all_masks_flat.reshape(B, self.num_outputs + 1, H_up, W_up)
        
        return all_masks, iou_scores

class HQSAM(nn.Module):
    """
    High-Quality Segment Anything Model (HQ-SAM).
    Augments SAM with high-quality tokens and multi-scale boundary detail alignment.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_encoder = HQImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.prompt_encoder = SAM1PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = HQMaskDecoder(embed_dim=embed_dim)

    def forward(self, images, points=None, labels=None, boxes=None, mask_priors=None):
        # 1. Extract multi-stage features
        early_feats, late_feats = self.image_encoder(images)
        
        # 2. Encode prompts
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=points, labels=labels, boxes=boxes, mask_priors=mask_priors, feat_shape=late_feats.shape[2:]
        )
        
        # 3. HQ Decoded masks
        masks, iou_scores = self.mask_decoder(early_feats, late_feats, sparse_prompts, dense_prompts)
        
        return masks, iou_scores
