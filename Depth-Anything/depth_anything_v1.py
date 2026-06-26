"""
Depth Anything V1 (CVPR 2024) — Pure PyTorch From-Scratch Implementation
=========================================================================
Paper: "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"
Authors: Lihe Yang et al. (HKU + TikTok), CVPR 2024
GitHub: https://github.com/LiheYoung/Depth-Anything

Architecture Overview:
    1. DINOv2 ViT Encoder (ViT-S/B/L with 14x14 patch) — pre-trained on ImageNet-22k
    2. DPT (Dense Prediction Transformer) Decoder — multi-scale feature reassembly + fusion
    3. Depth Head — final 1x1 conv + ReLU to produce relative disparity

Training Pipeline:
    - Labeled Stage: 1.5M labeled images (MIX-6 benchmark) with affine-invariant loss
    - Unlabeled Stage: 62M+ unlabeled images with auxiliary student network (semi-supervised)
    - Auxiliary Student Loss: teacher generates soft pseudo-labels → student distills feature space

Loss Function:
    - L_ssi (Scale-Shift-Invariant Loss): handles diverse depth scale and shift across datasets
    - L_gm (Gradient Matching Loss): sharpens depth boundaries and fine structure

Key Innovation vs MiDaS:
    - Exploits massive unlabeled internet images via semi-supervised distillation
    - DINOv2 provides richer semantic features than BEiT/Swin used in MiDaS
    - ViT-S (24.8M) already outperforms MiDaS (345M) across 6 zero-shot benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────
#  1. DINOv2-style ViT Encoder (Lightweight Replica)
# ─────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping 14x14 patches and projects them to embed_dim.
    DINOv2 uses patch_size=14 (vs. standard ViT-16's 16px patches).
    """
    def __init__(self, in_channels=3, patch_size=14, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)           # [B, D, H/P, W/P]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]  where N = Hp * Wp
        return x, Hp, Wp


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention used inside each ViT block."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)    # each [B, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class ViTBlock(nn.Module):
    """One ViT Transformer block: LayerNorm → MHSA → residual → LayerNorm → FFN → residual."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2Encoder(nn.Module):
    """
    Simplified DINOv2 ViT encoder.
    Returns intermediate layer features for multi-scale DPT decoding.

    DINOv2 model scales:
        ViT-S: depth=12, embed=384, heads=6   →  24.8M params
        ViT-B: depth=12, embed=768, heads=12  →  97.5M params
        ViT-L: depth=24, embed=1024, heads=16 → 335.3M params
        ViT-G: depth=40, embed=1536, heads=24 →   1.3B params

    We implement ViT-S as the default, adjustable via constructor args.
    """
    def __init__(self, in_channels=3, patch_size=14, embed_dim=384,
                 depth=12, num_heads=6, img_size=518,
                 intermediate_layers=(2, 5, 8, 11)):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.embed_dim   = embed_dim
        self.patch_size  = patch_size
        self.intermediate_layers = intermediate_layers

        # Learnable [CLS] token and positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Returns list of intermediate spatial feature maps at selected encoder depths.
        These are fed into the DPT decoder for multi-scale depth prediction.
        """
        B = x.shape[0]
        tokens, Hp, Wp = self.patch_embed(x)   # [B, N, D]

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # [B, N+1, D]

        # Add positional embedding (interpolate if resolution differs)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            # Dynamic positional interpolation for arbitrary resolution
            cls_pos = self.pos_embed[:, :1, :]
            spatial_pos = self.pos_embed[:, 1:, :].transpose(1, 2)
            orig_len = int(spatial_pos.shape[-1] ** 0.5)
            spatial_pos = spatial_pos.reshape(1, self.embed_dim, orig_len, orig_len)
            spatial_pos = F.interpolate(spatial_pos, size=(Hp, Wp), mode='bicubic', align_corners=False)
            spatial_pos = spatial_pos.flatten(2).transpose(1, 2)
            pos = torch.cat([cls_pos, spatial_pos], dim=1)
        else:
            pos = self.pos_embed

        tokens = tokens + pos

        # Forward through blocks, collecting intermediates
        intermediates = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in self.intermediate_layers:
                # Drop [CLS] token, reshape to spatial feature map
                spatial = tokens[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
                intermediates.append(spatial)

        return intermediates, Hp, Wp


# ─────────────────────────────────────────────────
#  2. DPT (Dense Prediction Transformer) Decoder
# ─────────────────────────────────────────────────

class DPTReadout(nn.Module):
    """
    DPT Readout Operation: handles the [CLS] token by adding it to all patch tokens.
    Reassembles token sequence back into spatial feature map.
    """
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        # x: already a spatial feature map [B, D, Hp, Wp]
        B, D, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)   # [B, N, D]
        x_proj = self.proj(x_flat)               # [B, N, out_channels]
        return x_proj.transpose(1, 2).reshape(B, -1, H, W)


class DPTFusionBlock(nn.Module):
    """
    DPT Fusion Block: residual refinement at a given spatial scale.
    """
    def __init__(self, channels):
        super().__init__()
        self.resConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return self.resConv(x) + self.project(x)


class DPTDecoder(nn.Module):
    """
    DPT Decoder: processes 4 intermediate ViT feature maps (all same spatial size Hp×Wp),
    fuses them bottom-up with progressive 2x upsampling at each stage.
    
    In a standard ViT all intermediate layers share the same token grid (Hp×Wp),
    so we explicitly build the pyramid by upsampling the accumulated output and
    resizing skip connections to match at each fusion stage.
    
    Final output resolution: Hp * 2^4 × Wp * 2^4 = 16×Hp × 16×Wp
    (with img_size=518, patch=14 → 37×37 tokens → output 592×592 before final resize)
    """
    def __init__(self, embed_dim=384, feature_channels=256):
        super().__init__()
        self.layer_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, feature_channels, kernel_size=1) for _ in range(4)
        ])
        self.fusions = nn.ModuleList([
            DPTFusionBlock(feature_channels) for _ in range(4)
        ])

    def forward(self, intermediates):
        """
        intermediates: list of 4 spatial feature maps, each [B, D, Hp, Wp]
        Returns: fused feature map [B, C, 16*Hp, 16*Wp]
        """
        projected = [proj(feat) for proj, feat in zip(self.layer_projs, intermediates)]

        # Since all ViT intermediates share the same spatial size (Hp×Wp),
        # build the pyramid by upsampling progressively and resizing skips to match.
        out = self.fusions[3](projected[3])                                              # [B,C,Hp,Wp]
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)  # 2Hp×2Wp
        s2  = F.interpolate(projected[2], size=out.shape[2:], mode='bilinear', align_corners=True)
        out = self.fusions[2](out + s2)
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)  # 4Hp×4Wp
        s1  = F.interpolate(projected[1], size=out.shape[2:], mode='bilinear', align_corners=True)
        out = self.fusions[1](out + s1)
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)  # 8Hp×8Wp
        s0  = F.interpolate(projected[0], size=out.shape[2:], mode='bilinear', align_corners=True)
        out = self.fusions[0](out + s0)
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)  # 16Hp×16Wp
        return out


# ─────────────────────────────────────────────────
#  3. Depth Head
# ─────────────────────────────────────────────────

class DepthHead(nn.Module):
    """
    Final depth prediction head: conv layers → sigmoid → rescale to [0,1] relative disparity.
    Depth Anything outputs affine-invariant relative disparity (not metric depth in meters).
    """
    def __init__(self, feature_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_channels // 2, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))   # [B, 1, H', W'] in [0, 1]
        return x


# ─────────────────────────────────────────────────
#  4. Full Depth Anything V1 Model
# ─────────────────────────────────────────────────

class DepthAnythingV1(nn.Module):
    """
    Depth Anything V1 (CVPR 2024) — End-to-end monocular relative depth estimator.

    Architecture: DINOv2 ViT Encoder + DPT Multi-Scale Decoder + Depth Head

    Three model scales (matching official checkpoints):
        S: embed=384,  depth=12, heads=6   (24.8M params)
        B: embed=768,  depth=12, heads=12  (97.5M params)
        L: embed=1024, depth=24, heads=16  (335.3M params)

    Forward pass:
        Input:  x [B, 3, H, W] (normalized RGB, typically 518x518)
        Output: depth [B, 1, H, W] — affine-invariant relative disparity in [0,1]
                (larger value = closer/foreground; smaller = farther/background)
    """
    CONFIGS = {
        'S': dict(embed_dim=384,  depth=12, num_heads=6),
        'B': dict(embed_dim=768,  depth=12, num_heads=12),
        'L': dict(embed_dim=1024, depth=24, num_heads=16),
    }

    def __init__(self, scale='S', in_channels=3, patch_size=14,
                 feature_channels=256, img_size=518,
                 intermediate_layers=(2, 5, 8, 11)):
        super().__init__()
        cfg = self.CONFIGS[scale]
        embed_dim = cfg['embed_dim']

        self.encoder = DINOv2Encoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            img_size=img_size,
            intermediate_layers=intermediate_layers,
        )
        self.decoder = DPTDecoder(embed_dim=embed_dim, feature_channels=feature_channels)
        self.depth_head = DepthHead(feature_channels=feature_channels)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] — input RGB image tensor
        Returns:
            depth: [B, 1, H, W] — relative disparity map, upsampled to input resolution
        """
        H, W = x.shape[2], x.shape[3]

        # Encode: extract 4 intermediate multi-scale features
        intermediates, Hp, Wp = self.encoder(x)

        # Decode: DPT bottom-up fusion → dense feature map
        fused = self.decoder(intermediates)   # [B, C, 16*Hp, 16*Wp]

        # Predict relative depth
        depth = self.depth_head(fused)   # [B, 1, 16*Hp, 16*Wp]

        # Upsample to original input resolution
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
        return depth


# ─────────────────────────────────────────────────
#  5. Loss Functions
# ─────────────────────────────────────────────────

class ScaleShiftInvariantLoss(nn.Module):
    """
    Affine-Invariant (Scale-Shift-Invariant) Regression Loss.
    Handles depth data from heterogeneous labeled datasets with different scales and shifts.

    Formula (applied in disparity space):
        d_hat = (d - t(d)) / s(d)   where t = median, s = mean absolute deviation
        t(pred), s(pred) computed similarly
        L_ssi = mean( | d_hat_pred - d_hat_gt |^2 )

    This loss is invariant to global affine transformation: α·d + β
    meaning the model learns structural/relative depth ordering, not absolute scale.
    """
    def __init__(self):
        super().__init__()

    def _normalize(self, x):
        # x: [B, H*W] flattened depth
        # Normalize via median + mean absolute deviation (robust statistics)
        t = x.median(dim=1, keepdim=True).values
        s = (x - t).abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        return (x - t) / s

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred:   [B, 1, H, W] — predicted disparity map
            target: [B, 1, H, W] — ground truth disparity map
            mask:   [B, 1, H, W] bool — valid depth pixels (optional)
        Returns:
            scalar loss
        """
        B = pred.shape[0]
        pred_flat   = pred.view(B, -1)
        target_flat = target.view(B, -1)

        if mask is not None:
            mask_flat = mask.view(B, -1)
        else:
            mask_flat = torch.ones_like(pred_flat, dtype=torch.bool)

        total_loss = torch.tensor(0.0, device=pred.device)
        valid_count = 0
        for b in range(B):
            p = pred_flat[b][mask_flat[b]]
            t = target_flat[b][mask_flat[b]]
            if p.numel() < 2:
                continue
            p_norm = self._normalize(p.unsqueeze(0)).squeeze(0)
            t_norm = self._normalize(t.unsqueeze(0)).squeeze(0)
            total_loss = total_loss + ((p_norm - t_norm) ** 2).mean()
            valid_count += 1

        return total_loss / max(valid_count, 1)


class GradientMatchingLoss(nn.Module):
    """
    Gradient Matching Loss (L_gm):
    Penalizes inconsistencies in depth gradients (x and y directions).
    Key for sharpening boundaries and fine structural details in the predicted depth map.

    Formula:
        L_gm = mean( |∂x(pred) - ∂x(gt)| + |∂y(pred) - ∂y(gt)| )
    where ∂x and ∂y are finite difference gradients at multiple image scales.
    """
    def __init__(self, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, pred, target):
        """
        Args:
            pred, target: [B, 1, H, W]
        """
        loss = torch.tensor(0.0, device=pred.device)
        for scale in range(self.num_scales):
            if scale > 0:
                pred   = F.avg_pool2d(pred,   2, stride=2)
                target = F.avg_pool2d(target, 2, stride=2)
            # Finite difference gradients
            grad_pred_x   = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            grad_pred_y   = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
            grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
            loss = loss + (grad_pred_x - grad_target_x).abs().mean()
            loss = loss + (grad_pred_y - grad_target_y).abs().mean()
        return loss / self.num_scales


class DepthAnythingV1Loss(nn.Module):
    """
    Combined training loss for Depth Anything V1.
    L_total = L_ssi + λ * L_gm

    For unlabeled images: teacher generates pseudo-labels → student minimizes L_ssi
    against teacher's predictions (semi-supervised self-distillation).
    """
    def __init__(self, lambda_gm=0.5):
        super().__init__()
        self.ssi_loss = ScaleShiftInvariantLoss()
        self.gm_loss  = GradientMatchingLoss()
        self.lambda_gm = lambda_gm

    def forward(self, pred, target, mask=None):
        l_ssi = self.ssi_loss(pred, target, mask)
        l_gm  = self.gm_loss(pred, target)
        return l_ssi + self.lambda_gm * l_gm, l_ssi, l_gm


# ─────────────────────────────────────────────────
#  6. Auxiliary Student Network (Semi-Supervised)
# ─────────────────────────────────────────────────

class AuxiliaryStudent(nn.Module):
    """
    Auxiliary student network for semi-supervised training on unlabeled images.
    
    Training Protocol:
        1. Freeze teacher (main DepthAnythingV1 model) weights.
        2. Feed unlabeled images through teacher → get pseudo depth labels.
        3. Apply random augmentations to same images → feed through student.
        4. Minimize L_ssi between student predictions and teacher pseudo-labels.
        5. Propagate gradients to update student encoder, NOT teacher.
    
    This forces the student encoder to learn generalizable depth features
    from diverse, in-the-wild internet images.
    """
    def __init__(self, scale='S', feature_channels=256):
        super().__init__()
        # The student uses the same architecture as the teacher
        self.model = DepthAnythingV1(scale=scale, feature_channels=feature_channels)
        self.loss_fn = ScaleShiftInvariantLoss()

    def forward(self, x_aug):
        """Forward pass on augmented unlabeled image."""
        return self.model(x_aug)

    def distillation_step(self, x_aug, pseudo_labels, optimizer):
        """
        Single semi-supervised distillation step.
        Args:
            x_aug:         augmented unlabeled image [B, 3, H, W]
            pseudo_labels: teacher depth predictions  [B, 1, H, W]
            optimizer:     student optimizer
        Returns:
            distill_loss (scalar)
        """
        self.model.train()
        pred = self.model(x_aug)
        loss = self.loss_fn(pred, pseudo_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
