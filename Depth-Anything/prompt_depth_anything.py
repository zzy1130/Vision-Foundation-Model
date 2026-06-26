"""
Prompt Depth Anything (PromptDA, CVPR 2025) — Pure PyTorch From-Scratch Implementation
========================================================================================
Paper: "Prompt Depth Anything"
Project: https://promptda.github.io/
GitHub:  https://github.com/DepthAnything/PromptDA
ArXiv:   https://arxiv.org/abs/2412.14015

Core Motivation:
    Standard monocular depth models (including DA-V1/V2) produce RELATIVE depth
    (ordinal depth without absolute metric scale). This causes fundamental limitations:
        - Cannot determine if an object is 1m or 10m away
        - Scale ambiguity prevents direct 3D reconstruction from single camera
    
    Existing solutions:
        - Metric fine-tuning (NYUv2/KITTI): works only in-domain
        - Depth sensors (full LiDAR): expensive and heavy
    
    PromptDA insight: Use LOW-RESOLUTION, LOW-COST LiDAR (e.g., iPhone ARKit,
    mmWave radar, sparse structured light) as a "scale prompt" to ANCHOR the relative
    depth prediction to metric scale. This lifts 4K metric depth from sparse LiDAR points.

Architecture:
    Base:             Depth Anything V2 (frozen or partially frozen)
    Prompt Fusion:    Multi-scale LiDAR feature injected into DPT decoder layers
    Output:           4K-resolution metric depth map (absolute meters)

Multi-Scale Prompt Fusion:
    Low-res LiDAR (e.g., 32x24 pixels) is:
        1. Resized to match each DPT fusion layer's spatial resolution
        2. Processed by shallow ConvNets to extract depth-relevant features
        3. Additively fused into DPT decoder at multiple scales
    This "prompt" provides metric anchors at coarse scale while the RGB
    backbone recovers fine geometric detail at full 4K resolution.

Training Pipeline:
    - Real data: iPhone LiDAR + aligned RGB → ground truth depth pairs
    - Synthetic LiDAR simulation: random point sampling from GT depth surfaces
    - Pseudo-GT generation: DA-V2 teacher generates dense pseudo labels for scale alignment
    - Mixed training: real + synthetic for robust generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v1 import DINOv2Encoder, ScaleShiftInvariantLoss, GradientMatchingLoss
from depth_anything_v2 import DPTDecoderV2, MetricDepthHead


# ─────────────────────────────────────────────────
#  1. LiDAR Prompt Encoder
# ─────────────────────────────────────────────────

class LiDARPromptEncoder(nn.Module):
    """
    Encodes sparse, low-resolution LiDAR depth into multi-scale feature embeddings
    suitable for fusion into the DPT decoder at different spatial scales.

    Input:
        lidar_depth: [B, 1, H_l, W_l] — sparse LiDAR depth map (e.g., 32x24)
                     Zero values indicate missing LiDAR returns (invalid pixels).
    
    Validity Mask Processing:
        Missing LiDAR returns (zeros) are masked out before processing.
        The mask is injected as an extra channel to inform the network of data validity.
    
    Output:
        Multi-scale depth embeddings at 4 different spatial resolutions,
        matching the DPT decoder fusion stages.
    """
    def __init__(self, prompt_channels=64, feature_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        # Initial encoding of LiDAR depth + validity mask (2 input channels)
        self.stem = nn.Sequential(
            nn.Conv2d(2, prompt_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, prompt_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_channels, prompt_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, prompt_channels),
            nn.ReLU(inplace=True),
        )

        # Project prompt features to match DPT feature channel width at each scale
        self.scale_projections = nn.ModuleList([
            nn.Conv2d(prompt_channels, feature_channels, kernel_size=1)
            for _ in range(num_scales)
        ])

    def forward(self, lidar_depth, target_sizes):
        """
        Args:
            lidar_depth:  [B, 1, H_l, W_l] — raw LiDAR sparse depth
            target_sizes: list of (H, W) tuples — target spatial sizes for each DPT scale
        Returns:
            list of num_scales prompt embeddings, each [B, feature_channels, H_scale, W_scale]
        """
        # Create validity mask: 1 where LiDAR has valid reading, 0 where missing
        validity = (lidar_depth > 0).float()   # [B, 1, H_l, W_l]

        # Log-scale depth encoding (depth values span wide range)
        depth_encoded = torch.log1p(lidar_depth * validity)   # [B, 1, H_l, W_l]

        # Concatenate depth + validity mask
        prompt_input = torch.cat([depth_encoded, validity], dim=1)   # [B, 2, H_l, W_l]

        # Extract prompt features
        prompt_feat = self.stem(prompt_input)   # [B, prompt_C, H_l, W_l]

        # Project and resize to each DPT scale
        scale_embeddings = []
        for i, (H_target, W_target) in enumerate(target_sizes):
            resized = F.interpolate(
                prompt_feat, size=(H_target, W_target), mode='bilinear', align_corners=True
            )
            projected = self.scale_projections[i](resized)   # [B, feature_C, H_target, W_target]
            scale_embeddings.append(projected)

        return scale_embeddings


# ─────────────────────────────────────────────────
#  2. Prompt-Fused DPT Decoder
# ─────────────────────────────────────────────────

class PromptFusedDPTDecoder(nn.Module):
    """
    DPT Decoder with Multi-Scale LiDAR Prompt Fusion.
    
    At each fusion stage of the DPT bottom-up path, the LiDAR prompt embedding
    is ADDED to the current feature map (additive fusion). This injects metric
    scale information at multiple spatial resolutions, ensuring that both
    coarse-level scale alignment AND fine-grained detail preservation occur.

    Alternative fusion strategies considered (all inferior to additive fusion):
        - Concatenation + conv: increases parameters, risk of forgetting prompt
        - Attention-based: too expensive at 4K resolution
        - Late fusion: only coarse scale, misses fine structural cues
    """
    def __init__(self, embed_dim=384, feature_channels=256):
        super().__init__()
        self.base_decoder = DPTDecoderV2(embed_dim=embed_dim, feature_channels=feature_channels)
        # Learnable scale factors for each prompt injection
        # Init to small value so prompt starts as a small perturbation
        self.prompt_scales = nn.Parameter(torch.ones(4) * 0.1)

    def forward(self, intermediates, prompt_embeddings):
        """
        Args:
            intermediates:     list of 4 ViT feature maps [B, embed_dim, Hp, Wp]
            prompt_embeddings: list of 4 LiDAR prompt embeddings [B, feature_C, H_s, W_s]
        Returns:
            fused: [B, feature_C, H_out, W_out] — metric-scale-conditioned feature map
        """
        # Project ViT intermediates to feature_channels
        projected = [
            self.base_decoder.reassemble[i](intermediates[i])
            for i in range(4)
        ]

        # Inject LiDAR prompts: align sizes and add with learnable scale
        enriched = []
        for i, (proj_feat, prompt_feat) in enumerate(zip(projected, prompt_embeddings)):
            # Resize prompt to match projected feature map size
            if prompt_feat.shape[2:] != proj_feat.shape[2:]:
                prompt_feat = F.interpolate(
                    prompt_feat, size=proj_feat.shape[2:], mode='bilinear', align_corners=True
                )
            # Additive fusion with learnable scale (sigmoid for stability)
            alpha = torch.sigmoid(self.prompt_scales[i])
            enriched.append(proj_feat + alpha * prompt_feat)

        # Bottom-up DPT fusion with prompt-enriched features
        out = self.base_decoder.fusion_convs[3](enriched[3])
        out = F.interpolate(out, size=enriched[2].shape[2:], mode='bilinear', align_corners=True)
        out = self.base_decoder.fusion_convs[2](out + enriched[2])
        out = F.interpolate(out, size=enriched[1].shape[2:], mode='bilinear', align_corners=True)
        out = self.base_decoder.fusion_convs[1](out + enriched[1])
        out = F.interpolate(out, size=enriched[0].shape[2:], mode='bilinear', align_corners=True)
        out = self.base_decoder.fusion_convs[0](out + enriched[0])
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=True)
        return out


# ─────────────────────────────────────────────────
#  3. Full Prompt Depth Anything Model
# ─────────────────────────────────────────────────

class PromptDepthAnything(nn.Module):
    """
    Prompt Depth Anything (PromptDA) — 4K Metric Depth with Sparse LiDAR Prompt.

    Workflow:
        1. RGB Image → DINOv2 encoder → 4 intermediate feature maps
        2. Sparse LiDAR → LiDAR Prompt Encoder → 4 scale embeddings
        3. Feature maps + prompt embeddings → PromptFused DPT decoder
        4. Dense feature → Metric Depth Head → 4K depth map in meters

    Key Properties:
        - Output resolution: up to 4K (4096x2160), same as input RGB
        - Metric accuracy: works for both indoor [0-10m] and outdoor [0-80m]
        - LiDAR sparsity: robust to very sparse input (as few as 100 LiDAR points)
        - Cross-sensor: generalizes across iPhone ARKit, mmWave, ToF, structured light

    Metric Depth Head Variants:
        indoor:  max_depth=10.0m (trained with NYUv2)
        outdoor: max_depth=80.0m (trained with KITTI)
    """
    CONFIGS = {
        'S': dict(embed_dim=384,  depth=12, num_heads=6,  intermediate=(2, 5, 8, 11)),
        'B': dict(embed_dim=768,  depth=12, num_heads=12, intermediate=(2, 5, 8, 11)),
        'L': dict(embed_dim=1024, depth=24, num_heads=16, intermediate=(4, 11, 17, 23)),
    }

    def __init__(self, scale='S', max_depth=10.0, feature_channels=256,
                 prompt_channels=64, img_size=518):
        super().__init__()
        cfg = self.CONFIGS[scale]
        embed_dim = cfg['embed_dim']

        # RGB Encoder (DINOv2)
        self.encoder = DINOv2Encoder(
            embed_dim=embed_dim,
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            img_size=img_size,
            intermediate_layers=cfg['intermediate'],
        )

        # LiDAR Prompt Encoder
        self.prompt_encoder = LiDARPromptEncoder(
            prompt_channels=prompt_channels,
            feature_channels=feature_channels,
            num_scales=4
        )

        # Prompt-fused DPT Decoder
        self.decoder = PromptFusedDPTDecoder(embed_dim=embed_dim, feature_channels=feature_channels)

        # Metric depth head
        self.metric_head = MetricDepthHead(
            feature_channels=feature_channels, max_depth=max_depth
        )

    def forward(self, rgb, lidar_depth):
        """
        Args:
            rgb:         [B, 3, H_rgb, W_rgb] — high-resolution RGB image (up to 4K)
            lidar_depth: [B, 1, H_l, W_l]    — sparse low-res LiDAR depth (e.g., 32x24)
                         Zero values = invalid/missing LiDAR points
        Returns:
            metric_depth: [B, 1, H_rgb, W_rgb] — absolute depth in meters
        """
        H_rgb, W_rgb = rgb.shape[2], rgb.shape[3]

        # Step 1: Encode RGB image
        intermediates, Hp, Wp = self.encoder(rgb)

        # Compute target sizes for prompt injection at each DPT scale
        target_sizes = [
            (Hp * 4, Wp * 4),   # Scale 0: finest
            (Hp * 2, Wp * 2),   # Scale 1
            (Hp,     Wp    ),   # Scale 2
            (Hp // 2, Wp // 2), # Scale 3: coarsest
        ]
        # Filter out zero-size scales
        target_sizes = [(max(h, 1), max(w, 1)) for h, w in target_sizes]

        # Step 2: Encode LiDAR prompt at multiple scales
        prompt_embeddings = self.prompt_encoder(lidar_depth, target_sizes)

        # Step 3: Fused decode
        fused = self.decoder(intermediates, prompt_embeddings)

        # Step 4: Predict metric depth
        metric_depth = self.metric_head(fused)   # [B, 1, H', W']

        # Step 5: Upsample to input RGB resolution
        metric_depth = F.interpolate(
            metric_depth, size=(H_rgb, W_rgb), mode='bilinear', align_corners=True
        )
        return metric_depth


# ─────────────────────────────────────────────────
#  4. Metric Depth Loss Functions
# ─────────────────────────────────────────────────

class MetricDepthLoss(nn.Module):
    """
    Training loss for metric depth estimation.
    Combines scale-shift-invariant loss (for structural quality)
    with a direct metric L1 loss (for absolute scale accuracy).

    L_total = L_ssi + λ_l1 * L_l1 + λ_gm * L_gm

    L_l1 only computed at valid LiDAR/GT pixels (mask-aware).
    """
    def __init__(self, lambda_l1=1.0, lambda_gm=0.5):
        super().__init__()
        self.ssi_loss = ScaleShiftInvariantLoss()
        self.gm_loss  = GradientMatchingLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_gm = lambda_gm

    def forward(self, pred, target, valid_mask=None):
        """
        Args:
            pred:       [B, 1, H, W] — predicted metric depth (meters)
            target:     [B, 1, H, W] — GT metric depth (meters)
            valid_mask: [B, 1, H, W] bool — valid depth pixels
        Returns:
            total loss, ssi, l1, gm
        """
        l_ssi = self.ssi_loss(pred, target, valid_mask)
        l_gm  = self.gm_loss(pred, target)

        if valid_mask is not None:
            # Metric L1 loss only at valid GT pixels
            l1 = (pred - target).abs()
            l1 = (l1 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)
        else:
            l1 = (pred - target).abs().mean()

        total = l_ssi + self.lambda_l1 * l1 + self.lambda_gm * l_gm
        return total, l_ssi, l1, l_gm


# ─────────────────────────────────────────────────
#  5. Synthetic LiDAR Simulation
# ─────────────────────────────────────────────────

class SyntheticLiDARSimulator:
    """
    Simulates sparse LiDAR from dense GT depth for training data augmentation.
    
    This addresses the challenge of limited real (LiDAR, RGB, GT-depth) triplet data:
    We can generate arbitrary amounts of training data by:
        1. Starting with dense GT depth (from synthetic datasets or pseudo-labels)
        2. Randomly sampling N points to simulate LiDAR sparsity patterns
        3. Adding noise to simulate real LiDAR measurement error

    Sampling Strategies:
        'uniform':    randomly sample N pixels
        'beam':       simulate rotating LiDAR beams (horizontal bands)
        'iphone_ark': simulate iPhone ARKit depth pattern (sparse grid)
    """
    def __init__(self, num_points=500, noise_std=0.02, strategy='uniform'):
        self.num_points = num_points
        self.noise_std  = noise_std
        self.strategy   = strategy

    def simulate(self, dense_depth):
        """
        Args:
            dense_depth: [B, 1, H, W] — dense GT depth map
        Returns:
            sparse_lidar: [B, 1, H, W] — simulated sparse LiDAR (zeros = invalid)
        """
        B, _, H, W = dense_depth.shape
        sparse = torch.zeros_like(dense_depth)

        for b in range(B):
            depth_b = dense_depth[b, 0]   # [H, W]
            valid   = (depth_b > 0)
            valid_idx = valid.nonzero(as_tuple=False)   # [N_valid, 2]

            if valid_idx.shape[0] == 0:
                continue

            # Sample random valid pixels
            N = min(self.num_points, valid_idx.shape[0])
            perm = torch.randperm(valid_idx.shape[0])[:N]
            sampled = valid_idx[perm]   # [N, 2]

            ys, xs = sampled[:, 0], sampled[:, 1]
            # Add Gaussian noise to simulate LiDAR measurement error
            noise = torch.randn(N, device=dense_depth.device) * self.noise_std
            sparse_vals = depth_b[ys, xs] * (1.0 + noise)
            sparse_vals = sparse_vals.clamp(min=0.0)

            sparse[b, 0, ys, xs] = sparse_vals

        return sparse
