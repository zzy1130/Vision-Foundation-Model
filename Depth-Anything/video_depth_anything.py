"""
Video Depth Anything (CVPR 2025 Highlight) — Pure PyTorch From-Scratch Implementation
=======================================================================================
Paper: "Video Depth Anything"
Authors: Sili Chen, Hengkai Guo et al. (ByteDance), CVPR 2025 Highlight
GitHub: https://github.com/DepthAnything/Video-Depth-Anything
ArXiv:  https://arxiv.org/abs/2501.12375

Core Problem with Per-Frame Depth:
    Applying single-image depth estimation to every video frame independently
    produces temporally inconsistent results (flickering, jitter). The depth value
    of a static scene point changes randomly frame-to-frame due to:
        (a) Model randomness / ambiguity in relative scale
        (b) Lack of inter-frame communication

Key Innovations:
    1. SPATIAL-TEMPORAL HEAD: Adds a lightweight ConvLSTM (or temporal attention)
       on top of DPT features to propagate depth context across frames.
    2. KEY-FRAME STRATEGY: Processes "super-long" videos (minutes) in sliding
       window fashion with key-frame anchors, avoiding memory explosion.
    3. STREAMING MODE (2025 update): True online inference — processes each frame
       once with constant VRAM usage, regardless of video length.
    4. TEMPORAL CONSISTENCY LOSS: Penalizes depth inconsistency between frames
       for static regions (identified by optical flow warping residuals).

Architecture:
    Per-frame encoder: DINOv2 ViT (same as V2)
    Temporal module:   ConvLSTM / Temporal Window Attention
    Decoder:           DPT head (same as V2)
    Output:            temporally smooth relative disparity sequence

Training:
    Uses synthetic video datasets + pseudo-labeled real video sequences.
    Temporal consistency loss + per-frame SSI loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v1 import (
    DINOv2Encoder, DepthHead, ScaleShiftInvariantLoss, GradientMatchingLoss
)
from depth_anything_v2 import DPTDecoderV2


# ─────────────────────────────────────────────────
#  1. ConvLSTM Cell for Temporal Depth Propagation
# ─────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell: standard LSTM gating but with 2D convolutions
    instead of fully-connected layers. Preserves spatial structure while
    propagating temporal depth context across frames.

    State dimensions: same spatial resolution as input feature map.
    This is the core temporal propagation mechanism in Video Depth Anything.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # All 4 LSTM gates (i, f, g, o) fused into one convolution for efficiency
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        """
        Args:
            x:     [B, C, H, W] — current frame feature map
            state: tuple (h, c) each [B, hidden_C, H, W], or None for first frame
        Returns:
            (h_new, c_new): updated hidden and cell states
        """
        H_feat, W_feat = x.shape[2], x.shape[3]

        if state is None:
            h = torch.zeros(x.shape[0], self.hidden_channels, H_feat, W_feat, device=x.device)
            c = torch.zeros(x.shape[0], self.hidden_channels, H_feat, W_feat, device=x.device)
        else:
            h, c = state

        # Concatenate input and previous hidden state along channel dim
        combined = torch.cat([x, h], dim=1)   # [B, C+hidden, H, W]
        gates = self.conv(combined)            # [B, 4*hidden, H, W]

        # Split into 4 gates
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)    # input gate
        f = torch.sigmoid(f)    # forget gate
        g = torch.tanh(g)       # cell gate
        o = torch.sigmoid(o)    # output gate

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


# ─────────────────────────────────────────────────
#  2. Temporal Attention Window Module
# ─────────────────────────────────────────────────

class TemporalWindowAttention(nn.Module):
    """
    Temporal Window Attention for cross-frame feature alignment.
    Given a window of T frames' spatial features, applies self-attention
    ACROSS time steps at each spatial position.

    This allows the model to "look back" at previous frames and propagate
    depth context with a learned attention mechanism (rather than recurrent state).

    Used in the larger model variants where expressiveness is preferred over
    the memory efficiency of ConvLSTM.
    """
    def __init__(self, channels, num_heads=8, window_size=8):
        super().__init__()
        self.channels    = channels
        self.num_heads   = num_heads
        self.head_dim    = channels // num_heads
        self.window_size = window_size
        self.scale       = self.head_dim ** -0.5

        self.norm   = nn.GroupNorm(1, channels)
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.out    = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, frame_feats):
        """
        Args:
            frame_feats: list of T feature maps, each [B, C, H, W]
        Returns:
            enhanced: list of T enhanced feature maps, each [B, C, H, W]
        """
        T = len(frame_feats)
        B, C, H, W = frame_feats[0].shape

        # Stack and reshape: [B, T, C, H*W] → [B*H*W, C, T]
        stacked = torch.stack(frame_feats, dim=1)   # [B, T, C, H, W]
        stacked = stacked.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, T)

        # Temporal self-attention
        normed = self.norm(stacked)
        q = self.q_proj(normed)   # [B*H*W, C, T]
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        # Reshape to multi-head: [B*H*W, heads, head_dim, T]
        q = q.reshape(B * H * W, self.num_heads, self.head_dim, T)
        k = k.reshape(B * H * W, self.num_heads, self.head_dim, T)
        v = v.reshape(B * H * W, self.num_heads, self.head_dim, T)

        # Attention over time dimension T
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out  = torch.einsum('bhts,bhds->bhdt', attn, v)

        out = out.reshape(B * H * W, C, T)
        out = self.out(out) + stacked   # residual

        # Unpack back to list of T feature maps
        out = out.permute(2, 0, 1)   # [T, B*H*W, C]
        out = out.reshape(T, B, H * W, C).permute(0, 1, 3, 2).reshape(T, B, C, H, W)
        return [out[t] for t in range(T)]


# ─────────────────────────────────────────────────
#  3. Spatial-Temporal DPT Decoder
# ─────────────────────────────────────────────────

class SpatialTemporalDecoder(nn.Module):
    """
    Enhanced DPT decoder with temporal propagation.
    Structure:
        1. Standard DPT bottom-up fusion (per-frame)
        2. ConvLSTM temporal propagation on fused features
        3. Depth head (per-frame)

    The ConvLSTM hidden state is passed between frames, encoding
    the temporal context for smooth, consistent depth prediction.
    """
    def __init__(self, embed_dim=384, feature_channels=256):
        super().__init__()
        self.dpt_decoder = DPTDecoderV2(embed_dim=embed_dim, feature_channels=feature_channels)
        self.conv_lstm   = ConvLSTMCell(feature_channels, feature_channels)
        self.depth_head  = DepthHead(feature_channels=feature_channels)

    def forward(self, intermediates_sequence, lstm_state=None):
        """
        Args:
            intermediates_sequence: list of T intermediate feature lists,
                                    each element = [feat_l2, feat_l5, feat_l8, feat_l11]
            lstm_state: (h, c) from previous window, or None for first frame
        Returns:
            depths:     list of T [B, 1, H, W] relative depth maps
            lstm_state: updated (h, c) for next window
        """
        depths = []
        for intermediates in intermediates_sequence:
            # Per-frame DPT fusion
            fused = self.dpt_decoder(intermediates)    # [B, C, H', W']
            # Temporal ConvLSTM propagation
            h, c = self.conv_lstm(fused, lstm_state)
            lstm_state = (h, c)
            # Predict depth from temporally-enhanced features
            depth = self.depth_head(h)                 # [B, 1, H', W']
            depths.append(depth)
        return depths, lstm_state


# ─────────────────────────────────────────────────
#  4. Full Video Depth Anything Model
# ─────────────────────────────────────────────────

class VideoDepthAnything(nn.Module):
    """
    Video Depth Anything (CVPR 2025 Highlight).

    Processes arbitrarily long video streams with temporal consistency.
    Supports two inference modes:
        1. Window Mode: slide a window of W frames across the video,
                        pass LSTM state between windows.
        2. Streaming Mode (online): process each frame in real-time,
                                    maintaining constant VRAM via rolling LSTM state.

    Model Scales:
        'S': ViT-Small backbone (24.8M encoder params)
        'B': ViT-Base backbone  (97.5M encoder params)
        'L': ViT-Large backbone (335.3M encoder params)
    """
    CONFIGS = {
        'S': dict(embed_dim=384,  depth=12, num_heads=6,  intermediate=(2, 5, 8, 11)),
        'B': dict(embed_dim=768,  depth=12, num_heads=12, intermediate=(2, 5, 8, 11)),
        'L': dict(embed_dim=1024, depth=24, num_heads=16, intermediate=(4, 11, 17, 23)),
    }

    def __init__(self, scale='S', feature_channels=256, img_size=518):
        super().__init__()
        cfg = self.CONFIGS[scale]
        embed_dim = cfg['embed_dim']

        self.encoder = DINOv2Encoder(
            embed_dim=embed_dim,
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            img_size=img_size,
            intermediate_layers=cfg['intermediate'],
        )
        self.st_decoder = SpatialTemporalDecoder(
            embed_dim=embed_dim, feature_channels=feature_channels
        )
        self._lstm_state = None   # persistent state for streaming mode

    def reset_temporal_state(self):
        """Reset ConvLSTM state. Must be called between different video sequences."""
        self._lstm_state = None

    def forward_window(self, frames, reset=False):
        """
        Window-mode forward pass.
        Args:
            frames: [T, B, 3, H, W] — a window of T video frames
            reset:  bool — if True, reset LSTM state (start of new video)
        Returns:
            depths: [T, B, 1, H, W]  — temporally-consistent depth maps
        """
        if reset:
            self.reset_temporal_state()

        T = frames.shape[0]
        H, W = frames.shape[3], frames.shape[4]

        # Encode each frame independently (encoder is stateless)
        intermediates_seq = []
        for t in range(T):
            ints, Hp, Wp = self.encoder(frames[t])   # list of 4 feature maps
            intermediates_seq.append(ints)

        # Temporal decode with ConvLSTM state passing
        depths_lr, self._lstm_state = self.st_decoder(
            intermediates_seq, self._lstm_state
        )

        # Upsample each depth to original resolution
        depths_hr = [
            F.interpolate(d, size=(H, W), mode='bilinear', align_corners=True)
            for d in depths_lr
        ]
        return torch.stack(depths_hr, dim=0)   # [T, B, 1, H, W]

    def forward_streaming(self, frame):
        """
        Streaming-mode forward pass (one frame at a time, constant VRAM).
        Args:
            frame: [B, 3, H, W] — single video frame
        Returns:
            depth: [B, 1, H, W] — relative depth for this frame
        """
        H, W = frame.shape[2], frame.shape[3]

        # Encode current frame
        intermediates, Hp, Wp = self.encoder(frame)

        # One ConvLSTM step with persistent hidden state
        depths, self._lstm_state = self.st_decoder([intermediates], self._lstm_state)
        depth = depths[0]

        return F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, x, mode='single'):
        """
        Unified forward method.
        Args:
            x:    [B, 3, H, W] for single frame, or [T, B, 3, H, W] for window
            mode: 'single' | 'streaming' | 'window'
        """
        if mode == 'streaming' or mode == 'single':
            if x.dim() == 4:
                return self.forward_streaming(x)
            else:
                return self.forward_streaming(x[0])
        elif mode == 'window':
            return self.forward_window(x)


# ─────────────────────────────────────────────────
#  5. Temporal Consistency Loss
# ─────────────────────────────────────────────────

class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes depth inconsistency between adjacent video frames.
    
    Strategy:
        1. Use optical flow to warp frame t's depth map to frame t+1's viewpoint.
        2. Compare warped depth with frame t+1's predicted depth.
        3. Only penalize "static" regions (low flow magnitude → likely static scene point).
    
    Since we don't run a full optical flow network here, we use a simplified
    photometric proxy: regions with similar color between consecutive frames
    are assumed static, and their depth should be consistent.

    Formula:
        L_tc = Σ_t  mean( M_t * |depth_t_warped - depth_{t+1}|² )
    where M_t is a binary mask for (approximately) static pixels.
    """
    def __init__(self, flow_threshold=2.0):
        super().__init__()
        self.flow_threshold = flow_threshold

    def _photometric_static_mask(self, frame_t, frame_t1):
        """
        Approximate static region mask using photometric similarity.
        Pixels with small color change are assumed static.
        Returns: [B, 1, H, W] float mask in [0, 1]
        """
        diff = (frame_t - frame_t1).abs().mean(dim=1, keepdim=True)   # [B, 1, H, W]
        # Soft mask: smaller diff → more static → higher weight
        mask = torch.exp(-diff * 5.0)
        return mask

    def forward(self, depths, frames):
        """
        Args:
            depths: list of T depth maps [B, 1, H, W]
            frames: [T, B, 3, H, W] — original video frames (for static mask)
        Returns:
            scalar temporal consistency loss
        """
        T = len(depths)
        if T < 2:
            return torch.tensor(0.0, device=depths[0].device)

        total_loss = torch.tensor(0.0, device=depths[0].device)
        count = 0
        for t in range(T - 1):
            # Simple consistency: predicted depths of adjacent frames should be close
            # (after scale alignment, since relative depth can shift between frames)
            d_t   = depths[t]       # [B, 1, H, W]
            d_t1  = depths[t + 1]

            # Static region mask from photometric similarity
            mask = self._photometric_static_mask(frames[t], frames[t + 1])

            # Scale-align d_t to match d_t1's scale (using median ratio)
            with torch.no_grad():
                ratio = (d_t1.median() / d_t.median().clamp(min=1e-5)).clamp(0.5, 2.0)
            d_t_aligned = d_t * ratio

            consistency = (mask * (d_t_aligned - d_t1) ** 2).mean()
            total_loss = total_loss + consistency
            count += 1

        return total_loss / max(count, 1)


class VideoDepthLoss(nn.Module):
    """Combined training loss for Video Depth Anything."""
    def __init__(self, lambda_tc=0.1, lambda_gm=0.5):
        super().__init__()
        self.ssi_loss = ScaleShiftInvariantLoss()
        self.gm_loss  = GradientMatchingLoss()
        self.tc_loss  = TemporalConsistencyLoss()
        self.lambda_tc = lambda_tc
        self.lambda_gm = lambda_gm

    def forward(self, pred_depths, target_depths, frames):
        """
        Args:
            pred_depths:   list of T predicted depth maps [B, 1, H, W]
            target_depths: list of T GT / pseudo-label depth maps [B, 1, H, W]
            frames:        [T, B, 3, H, W] original frames (for temporal mask)
        """
        # Per-frame depth regression loss
        ssi_total = torch.tensor(0.0, device=pred_depths[0].device)
        gm_total  = torch.tensor(0.0, device=pred_depths[0].device)
        for pred, target in zip(pred_depths, target_depths):
            ssi_total = ssi_total + self.ssi_loss(pred, target)
            gm_total  = gm_total  + self.gm_loss(pred, target)
        ssi_total = ssi_total / len(pred_depths)
        gm_total  = gm_total  / len(pred_depths)

        # Temporal consistency loss
        tc = self.tc_loss(pred_depths, frames)

        total = ssi_total + self.lambda_gm * gm_total + self.lambda_tc * tc
        return total, ssi_total, gm_total, tc
