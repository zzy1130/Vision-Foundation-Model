import torch
import torch.nn as nn
import torch.nn.functional as F

def get_knn_indices(xyz, k):
    """
    Computes KNN indices based on pairwise L2 distance.
    Args:
        xyz: (B, N, 3) point coordinates
        k: number of nearest neighbors
    Returns:
        idx: (B, N, k) indices of nearest neighbors
    """
    B, N, _ = xyz.shape
    # Compute pairwise Euclidean distance: (B, N, N)
    inner = -2 * torch.bmm(xyz, xyz.transpose(2, 1))
    xx = torch.sum(xyz**2, dim=2, keepdim=True)
    pairwise_dist = xx + inner + xx.transpose(2, 1)
    
    # Grab the k smallest distances
    _, idx = torch.topk(pairwise_dist, k, dim=-1, largest=False)
    return idx


class PointTransformerAttention(nn.Module):
    """
    Vector Attention Layer of Point Transformer V1 (Zhao et al., ICCV 2021).
    Applies localized vector self-attention inside k-nearest neighbors with positional encoding.
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Projections for Query, Key, Value
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)
        
        # Position encoder: MLP mapping delta coordinates to feature space
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels)
        )
        
        # Mapping gamma mapping vector attention difference to attention weights
        self.gamma = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features
        Returns:
            out_features: (B, N, C) aggregated features
        """
        B, N, C = features.shape
        device = features.device
        
        # 1. Get KNN indices
        knn_idx = get_knn_indices(xyz, self.k)  # (B, N, k)
        
        # 2. Linear projection for Q, K, V
        q = self.linear_q(features)  # (B, N, C_out)
        k = self.linear_k(features)  # (B, N, C_out)
        v = self.linear_v(features)  # (B, N, C_out)
        
        # 3. Gather neighbor positions and features
        # Flatten batch indices for gathering
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, self.k)
        
        # Neighbor coordinates: (B, N, k, 3)
        xyz_neighbors = xyz[batch_indices, knn_idx, :]
        # Relative coordinates (delta position): (B, N, k, 3)
        pos_deltas = xyz.unsqueeze(2) - xyz_neighbors
        
        # Positional Encoding: (B, N, k, C_out)
        pe = self.pos_encoder(pos_deltas)
        
        # Gather neighbor keys and values
        k_neighbors = k[batch_indices, knn_idx, :]  # (B, N, k, C_out)
        v_neighbors = v[batch_indices, knn_idx, :]  # (B, N, k, C_out)
        
        # 4. Compute Vector Attention weights
        # attention relation = q_i - k_j + PE
        relation = q.unsqueeze(2) - k_neighbors + pe  # (B, N, k, C_out)
        attn_weights = self.gamma(relation)  # (B, N, k, C_out)
        
        # Softmax over neighbor dimension k
        attn_weights = F.softmax(attn_weights, dim=2)
        
        # 5. Aggregate values: sum_j (attn_weights_ij * (v_j + PE))
        v_shifted = v_neighbors + pe
        out_features = torch.sum(attn_weights * v_shifted, dim=2)  # (B, N, C_out)
        
        return out_features


class PointTransformerBlock(nn.Module):
    """
    Standard Point Transformer Block wrapping localized attention, residual connections, and MLPs.
    """
    def __init__(self, channels, k=16):
        super().__init__()
        self.linear_in = nn.Linear(channels, channels)
        self.attn = PointTransformerAttention(channels, channels, k=k)
        self.linear_out = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        residual = features
        x = self.relu(self.linear_in(features))
        x = self.attn(xyz, x)
        x = self.linear_out(x)
        return self.relu(x + residual)


class TransitionDown(nn.Module):
    """
    Downsampling block. Uses Farthest Point Sampling (FPS) or simple strided pooling 
    to reduce point count, and gathers local neighbor features to increase channels.
    """
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, xyz, features, target_num_points):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C_in)
            target_num_points: Target point count N_down
        Returns:
            down_xyz: (B, N_down, 3)
            down_features: (B, N_down, C_out)
        """
        B, N, C = features.shape
        device = xyz.device
        
        # Subsampling points (Farthest Point Sampling simulation/strided sampling)
        # Using simple strided selection as a robust PyTorch implementation
        stride = max(1, N // target_num_points)
        indices = torch.arange(0, target_num_points * stride, stride, device=device)[:target_num_points]
        
        down_xyz = xyz[:, indices, :]  # (B, N_down, 3)
        
        # Gather local features from high-res points to low-res points via KNN
        # Find neighbors of downsampled points in original point set
        # Query: down_xyz (B, N_down, 3), Keys: xyz (B, N, 3)
        inner = -2 * torch.bmm(down_xyz, xyz.transpose(2, 1))
        xx_down = torch.sum(down_xyz**2, dim=2, keepdim=True)
        xx_orig = torch.sum(xyz**2, dim=2, keepdim=True)
        dist = xx_down + inner + xx_orig.transpose(2, 1)
        _, knn_idx = torch.topk(dist, self.k, dim=-1, largest=False)  # (B, N_down, k)
        
        # Gather local features: (B, N_down, k, C)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, target_num_points, self.k)
        neighbor_features = features[batch_indices, knn_idx, :]
        
        # Max pool over neighbors and project
        local_max_feats = torch.max(neighbor_features, dim=2)[0]  # (B, N_down, C)
        
        # Conv1d expects (B, C_in, N_down)
        local_max_feats = local_max_feats.transpose(1, 2)
        down_features = self.mlp(local_max_feats).transpose(1, 2)  # (B, N_down, C_out)
        
        return down_xyz, down_features


class TransitionUp(nn.Module):
    """
    Upsampling block. Propagates features from low-resolution points to high-resolution points
    using linear interpolation, followed by feature concatenation and projection.
    """
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super().__init__()
        self.mlp_low = nn.Linear(in_channels_low, out_channels)
        self.mlp_high = nn.Linear(in_channels_high, out_channels)
        self.mlp_out = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz_high, xyz_low, feats_high, feats_low):
        """
        Interpolates feats_low (from xyz_low) to match xyz_high, then fuses with feats_high.
        """
        B, N_high, _ = xyz_high.shape
        B, N_low, C_low = feats_low.shape
        device = xyz_high.device
        
        # Three-nearest neighbors interpolation
        # Query: xyz_high (B, N_high, 3), Keys: xyz_low (B, N_low, 3)
        inner = -2 * torch.bmm(xyz_high, xyz_low.transpose(2, 1))
        xx_high = torch.sum(xyz_high**2, dim=2, keepdim=True)
        xx_low = torch.sum(xyz_low**2, dim=2, keepdim=True)
        dist = xx_high + inner + xx_low.transpose(2, 1)
        
        # Get 3 nearest neighbors
        dist_3, idx_3 = torch.topk(dist, 3, dim=-1, largest=False)  # (B, N_high, 3)
        
        # Inverse distance weights
        dist_3 = torch.clamp(dist_3, min=1e-10)
        weight = 1.0 / dist_3
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (B, N_high, 3)
        
        # Gather low-res features: (B, N_high, 3, C_low)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N_high, 3)
        neighbors_feats = feats_low[batch_indices, idx_3, :]
        
        # Weighted sum: (B, N_high, C_low)
        interpolated_feats = torch.sum(neighbors_feats * weight.unsqueeze(-1), dim=2)
        
        # Project and concatenate
        feats_low_proj = self.mlp_low(interpolated_feats)
        feats_high_proj = self.mlp_high(feats_high)
        
        fused = torch.cat([feats_low_proj, feats_high_proj], dim=-1)  # (B, N_high, out_channels * 2)
        
        # Conv1d expects (B, C_in, N)
        fused = fused.transpose(1, 2)
        out = self.mlp_out(fused).transpose(1, 2)
        
        return out


class PointTransformerV1(nn.Module):
    """
    Hierarchical Point Transformer V1 architecture for Point Cloud classification/segmentation.
    """
    def __init__(self, in_channels=6, num_classes=10, k=16):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        
        # Hierarchical layers
        self.enc1 = PointTransformerBlock(32, k=k)
        self.down1 = TransitionDown(32, 64, k=k)
        
        self.enc2 = PointTransformerBlock(64, k=k)
        self.down2 = TransitionDown(64, 128, k=k)
        
        self.enc3 = PointTransformerBlock(128, k=k)
        
        # Decoder layers (e.g., for semantic segmentation task)
        self.up1 = TransitionUp(128, 64, 64)
        self.dec1 = PointTransformerBlock(64, k=k)
        
        self.up2 = TransitionUp(64, 32, 32)
        self.dec2 = PointTransformerBlock(32, k=k)
        
        self.out_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) input features (e.g. RGB, normals, or simple constant features)
        """
        # Encoder 1
        x = self.in_proj(features)
        x_enc1 = self.enc1(xyz, x)
        
        # Encoder 2
        xyz_down1, x_down1 = self.down1(xyz, x_enc1, target_num_points=xyz.shape[1] // 2)
        x_enc2 = self.enc2(xyz_down1, x_down1)
        
        # Encoder 3
        xyz_down2, x_down2 = self.down2(xyz_down1, x_enc2, target_num_points=xyz_down1.shape[1] // 2)
        x_enc3 = self.enc3(xyz_down2, x_down2)
        
        # Decoder 1
        x_dec1 = self.up1(xyz_down1, xyz_down2, x_enc2, x_enc3)
        x_dec1 = self.dec1(xyz_down1, x_dec1)
        
        # Decoder 2
        x_dec2 = self.up2(xyz, xyz_down1, x_enc1, x_dec1)
        x_dec2 = self.dec2(xyz, x_dec2)
        
        logits = self.out_head(x_dec2)
        return logits
