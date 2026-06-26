"""
Depth Anything V2 (NeurIPS 2024) — Pure PyTorch From-Scratch Implementation
=============================================================================
Paper: "Depth Anything V2"
Authors: Lihe Yang et al. (HKU + TikTok), NeurIPS 2024
GitHub: https://github.com/DepthAnything/Depth-Anything-V2

Key Improvements over V1:
    1. TRAINING DATA: Replaces noisy labeled real images with high-quality SYNTHETIC data
       (595K images from 8 synthetic datasets) for teacher training.
    2. TEACHER MODEL: Trains a large ViT-G teacher on pure synthetic data →
       dramatically reduces noise and structural errors in pseudo-labels.
    3. INTERMEDIATE LAYERS: V2 correctly uses intermediate encoder features for DPT
       (V1 accidentally used last 4 layers; V2 explicitly uses earlier layers).
    4. FINER DETAIL: Synthetic training provides pixel-perfect GT depth →
       much sharper predictions on fine-grained structures (e.g., thin objects).
    5. SCALE: 4 model sizes (S/B/L/G), with Giant (1.3B) also released.

Training Pipeline (3-Stage):
    Stage 1 — Teacher on Synthetic:
        ViT-G trained on 595K synthetic images (Hypersim, vKITTI, etc.)
        with pixel-perfect GT depth. Loss: L_ssi + L_gm.
    Stage 2 — Pseudo-Labeling:
        Teacher generates dense depth pseudo-labels for 62M real unlabeled images.
    Stage 3 — Student Distillation:
        Student models (S/B/L) trained on pseudo-labeled real data.
        Bridges synthetic-to-real domain gap.

Metric Depth Extension:
    After relative training, fine-tune with affine head on metric datasets:
        - NYUv2 (indoor): depth in [0, 10] meters
        - KITTI  (outdoor): depth in [0, 80] meters
    Uses camera field-of-view (FoV) conditioning for scale disambiguation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v1 import (
    DINOv2Encoder, DPTDecoder, DepthHead,
    ScaleShiftInvariantLoss, GradientMatchingLoss
)


# ─────────────────────────────────────────────────
#  1. Improved DPT Decoder (V2-specific fixes)
# ─────────────────────────────────────────────────

class DPTReassembleBlock(nn.Module):
    """
    DPT Reassembly Block: converts ViT tokens back to spatial feature maps.
    V2 fix: explicitly selects intermediate layers (not the last 4),
    then applies convolutional reassembly at the correct target strides.

    Stride mapping for ViT-S with patch=14, img=518 (→ 37x37 tokens):
        Layer 2  →  stride 4  (upsample 4x)
        Layer 5  →  stride 8  (upsample 2x)
        Layer 8  →  stride 16 (stride 1 — keep)
        Layer 11 →  stride 32 (downsample 2x)
    """
    def __init__(self, embed_dim, out_channels, scale_factor):
        """
        scale_factor: >1 = upsample, 1 = keep, <1 = downsample
        """
        super().__init__()
        self.scale_factor = scale_factor

        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        if scale_factor == 4.0:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4, padding=0)
            )
        elif scale_factor == 2.0:
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
            )
        elif scale_factor == 1.0:
            self.resample = nn.Identity()
        elif scale_factor == 0.5:
            self.resample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [B, embed_dim, Hp, Wp] — spatial feature map from ViT intermediate layer
        x = self.proj(x)
        x = self.resample(x)
        return x


class DPTDecoderV2(nn.Module):
    """
    V2-corrected DPT Decoder with explicit reassembly at 4 prescribed strides.
    Produces a 1/2-resolution feature map for the depth head.
    """
    def __init__(self, embed_dim=384, feature_channels=256):
        super().__init__()
        # Reassemble 4 intermediate features to common spatial resolution
        # Scale factors: [4.0, 2.0, 1.0, 0.5] → after reassembly all at 37*1 = 37px
        # Then bottom-up fusion doubles resolution at each step
        self.reassemble = nn.ModuleList([
            DPTReassembleBlock(embed_dim, feature_channels, scale_factor=4.0),   # Layer 2
            DPTReassembleBlock(embed_dim, feature_channels, scale_factor=2.0),   # Layer 5
            DPTReassembleBlock(embed_dim, feature_channels, scale_factor=1.0),   # Layer 8
            DPTReassembleBlock(embed_dim, feature_channels, scale_factor=0.5),   # Layer 11
        ])

        # Fusion convolutions (each receives fused + skip, outputs same channels)
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, feature_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

    def forward(self, intermediates):
        """
        intermediates: list of 4 spatial ViT feature maps [B, embed_dim, Hp, Wp]
        """
        # Reassemble all to their target strides
        reassembled = [block(feat) for block, feat in zip(self.reassemble, intermediates)]
        # Bottom-up fusion (deep to shallow)
        out = self.fusion_convs[3](reassembled[3])
        out = F.interpolate(out, size=reassembled[2].shape[2:], mode='bilinear', align_corners=True)
        out = self.fusion_convs[2](out + reassembled[2])
        out = F.interpolate(out, size=reassembled[1].shape[2:], mode='bilinear', align_corners=True)
        out = self.fusion_convs[1](out + reassembled[1])
        out = F.interpolate(out, size=reassembled[0].shape[2:], mode='bilinear', align_corners=True)
        out = self.fusion_convs[0](out + reassembled[0])
        # Final 2x upsample
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)
        return out


# ─────────────────────────────────────────────────
#  2. Metric Depth Head (V2 Extension)
# ─────────────────────────────────────────────────

class MetricDepthHead(nn.Module):
    """
    Metric depth prediction head for fine-tuned V2 models.
    Outputs absolute depth in meters using an affine mapping:
        d_metric = exp(α * d_relative + β) * max_depth

    Camera FoV Conditioning: The model uses the camera's horizontal FoV angle
    to disambiguate scale ambiguity (wide-angle cameras see closer objects as smaller,
    narrow-angle cameras see farther objects). FoV is encoded as a sinusoidal embedding
    and added to the depth features before the final projection.

    Dataset variants:
        Indoor  (NYUv2):  max_depth = 10.0m
        Outdoor (KITTI):  max_depth = 80.0m
    """
    def __init__(self, feature_channels=256, max_depth=10.0, fov_embed_dim=64):
        super().__init__()
        self.max_depth = max_depth

        # FoV conditioning embedding
        self.fov_embed = nn.Sequential(
            nn.Linear(1, fov_embed_dim),
            nn.GELU(),
            nn.Linear(fov_embed_dim, feature_channels),
        )

        self.conv1 = nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_channels // 2, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, fov_rad=None):
        """
        Args:
            x:       [B, C, H, W] — fused feature map
            fov_rad: [B] — camera horizontal field-of-view in radians (optional)
        Returns:
            depth:   [B, 1, H, W] — metric depth in meters
        """
        if fov_rad is not None:
            # Add FoV conditioning to spatial features
            fov_feat = self.fov_embed(fov_rad.unsqueeze(-1).float())  # [B, C]
            x = x + fov_feat.unsqueeze(-1).unsqueeze(-1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Sigmoid output × max_depth for metric prediction
        depth = torch.sigmoid(self.conv3(x)) * self.max_depth   # [B, 1, H, W]
        return depth


# ─────────────────────────────────────────────────
#  3. Full Depth Anything V2 Model
# ─────────────────────────────────────────────────

class DepthAnythingV2(nn.Module):
    """
    Depth Anything V2 (NeurIPS 2024) — Monocular Depth Estimation Foundation Model.

    Relative mode (default):
        Returns affine-invariant disparity in [0, 1]. Trained on synthetic + pseudo-labeled real.
    Metric mode (fine-tuned):
        Returns absolute depth in meters. Fine-tuned on NYUv2 or KITTI.

    Model Scales:
        'S': embed=384,  depth=12, heads=6   →  24.8M
        'B': embed=768,  depth=12, heads=12  →  97.5M
        'L': embed=1024, depth=24, heads=16  → 335.3M
        'G': embed=1536, depth=40, heads=24  →   1.3B  (conceptual — requires custom ViT)

    V2 Intermediate Layer Selection (relative to total depth):
        For depth=12: layers (2, 5, 8, 11)   — earlier than V1's last-4 bug
        For depth=24: layers (4, 11, 17, 23)
    """
    CONFIGS = {
        'S': dict(embed_dim=384,  depth=12, num_heads=6,  intermediate=(2, 5, 8, 11)),
        'B': dict(embed_dim=768,  depth=12, num_heads=12, intermediate=(2, 5, 8, 11)),
        'L': dict(embed_dim=1024, depth=24, num_heads=16, intermediate=(4, 11, 17, 23)),
        'G': dict(embed_dim=1536, depth=40, num_heads=24, intermediate=(9, 19, 29, 39)),
    }

    def __init__(self, scale='S', metric=False, max_depth=10.0,
                 feature_channels=256, img_size=518):
        super().__init__()
        cfg = self.CONFIGS[scale]
        embed_dim = cfg['embed_dim']
        self.metric = metric

        self.encoder = DINOv2Encoder(
            embed_dim=embed_dim,
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            img_size=img_size,
            intermediate_layers=cfg['intermediate'],
        )

        # V2 uses the improved decoder with explicit reassembly
        self.decoder = DPTDecoderV2(embed_dim=embed_dim, feature_channels=feature_channels)

        if metric:
            self.depth_head = MetricDepthHead(
                feature_channels=feature_channels, max_depth=max_depth
            )
        else:
            self.depth_head = DepthHead(feature_channels=feature_channels)

    def forward(self, x, fov_rad=None):
        """
        Args:
            x:       [B, 3, H, W] — input RGB image
            fov_rad: [B] float — camera FoV in radians (only used in metric mode)
        Returns:
            depth:   [B, 1, H, W] — relative disparity [0,1] or metric depth [meters]
        """
        H, W = x.shape[2], x.shape[3]

        intermediates, Hp, Wp = self.encoder(x)
        fused = self.decoder(intermediates)

        if self.metric and fov_rad is not None:
            depth = self.depth_head(fused, fov_rad)
        elif self.metric:
            depth = self.depth_head(fused)
        else:
            depth = self.depth_head(fused)

        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
        return depth


# ─────────────────────────────────────────────────
#  4. Three-Stage Training Pipeline
# ─────────────────────────────────────────────────

class TeacherStudentPipeline(nn.Module):
    """
    V2 Three-Stage Training Pipeline.

    Stage 1 (Teacher):
        Teacher = DepthAnythingV2(scale='G') trained on 595K synthetic images.
        Uses L_ssi + L_gm losses with pixel-perfect GT depth.
        Synthetic datasets: Hypersim, vKITTI, SceneFlow, etc.

    Stage 2 (Pseudo-Labeling):
        teacher.eval() → generate depth for 62M unlabeled real images.
        Pseudo-labels stored for stage 3.

    Stage 3 (Student):
        Students (S/B/L) trained on pseudo-labeled real images.
        Same L_ssi + L_gm losses.

    Key Insight: By replacing noisy labeled real images (V1) with
    high-quality synthetic GT depth (V2), teacher pseudo-labels are
    significantly more accurate → better student performance.
    """
    def __init__(self, teacher_scale='G', student_scale='S', feature_channels=256):
        super().__init__()
        self.teacher = DepthAnythingV2(scale=teacher_scale, feature_channels=feature_channels)
        self.student = DepthAnythingV2(scale=student_scale, feature_channels=feature_channels)

        self.ssi_loss = ScaleShiftInvariantLoss()
        self.gm_loss  = GradientMatchingLoss()
        self.lambda_gm = 0.5

    @torch.no_grad()
    def generate_pseudo_labels(self, real_images):
        """Stage 2: Teacher generates pseudo depth labels for unlabeled real images."""
        self.teacher.eval()
        return self.teacher(real_images)

    def student_step(self, real_images, optimizer):
        """Stage 3: One student training step using teacher pseudo-labels."""
        with torch.no_grad():
            pseudo_labels = self.teacher(real_images)

        self.student.train()
        pred = self.student(real_images)

        l_ssi = self.ssi_loss(pred, pseudo_labels)
        l_gm  = self.gm_loss(pred, pseudo_labels)
        loss  = l_ssi + self.lambda_gm * l_gm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), l_ssi.item(), l_gm.item()

    def teacher_step(self, synthetic_images, synthetic_depth_gt, optimizer):
        """Stage 1: Train teacher on synthetic images with pixel-perfect GT depth."""
        self.teacher.train()
        pred = self.teacher(synthetic_images)

        l_ssi = self.ssi_loss(pred, synthetic_depth_gt)
        l_gm  = self.gm_loss(pred, synthetic_depth_gt)
        loss  = l_ssi + self.lambda_gm * l_gm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


# ─────────────────────────────────────────────────
#  5. DA-2K Benchmark Evaluation Helper
# ─────────────────────────────────────────────────

class DA2KEvaluator:
    """
    Evaluation on DA-2K benchmark (introduced with Depth Anything V2).
    DA-2K: 2000 real-world images with relative ordering annotations
    (which of two pixels is closer/farther to camera).
    
    Metric: Accuracy of pairwise depth ordering (higher = better).
    Unlike absolute depth metrics (AbsRel, δ1), this tests structural
    correctness of relative depth estimation.
    """
    @staticmethod
    def pairwise_accuracy(pred_depth, annotations):
        """
        Args:
            pred_depth:  [N, H, W] — predicted disparity maps (larger = closer)
            annotations: list of (y1, x1, y2, x2, label) where label=1 if pixel1 is closer
        Returns:
            accuracy: float in [0, 1]
        """
        correct = 0
        total   = 0
        for i, (y1, x1, y2, x2, label) in enumerate(annotations):
            if i >= pred_depth.shape[0]:
                break
            d1 = pred_depth[i, y1, x1].item()
            d2 = pred_depth[i, y2, x2].item()
            # In disparity space: larger value = closer to camera
            pred_label = 1 if d1 > d2 else 0
            correct += int(pred_label == label)
            total   += 1
        return correct / max(total, 1)
