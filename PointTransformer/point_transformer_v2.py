import torch
import torch.nn as nn
import torch.nn.functional as F
from point_transformer_v1 import get_knn_indices

class GroupedVectorAttention(nn.Module):
    """
    Grouped Vector Attention of Point Transformer V2 (Zhao et al., NeurIPS 2022).
    Divides feature channels into G groups, applies vector attention per group to save parameters, 
    and incorporates a Position Encoding Multiplier.
    """
    def __init__(self, channels, groups=4, k=16):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.k = k
        
        # Each group has d = channels // groups dimensions
        assert channels % groups == 0, "Channels must be divisible by groups"
        self.group_dim = channels // groups
        
        # Projections
        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        
        # Position Encoder (shared across groups or mapped to full channel size)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, channels)
        )
        
        # Position encoding multiplier generator
        self.pos_multiplier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        
        # Gamma relation function mapping difference to group weights
        # Instead of projecting to full channels, we map to groups.
        self.gamma = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )

    def forward(self, xyz, features):
        B, N, C = features.shape
        device = features.device
        
        # 1. KNN indices
        knn_idx = get_knn_indices(xyz, self.k)  # (B, N, k)
        
        # 2. Linear projection for Q, K, V
        q = self.linear_q(features)  # (B, N, C)
        k = self.linear_k(features)  # (B, N, C)
        v = self.linear_v(features)  # (B, N, C)
        
        # 3. Gather relative coordinates
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, self.k)
        xyz_neighbors = xyz[batch_indices, knn_idx, :]  # (B, N, k, 3)
        pos_deltas = xyz.unsqueeze(2) - xyz_neighbors  # (B, N, k, 3)
        
        # Raw Position Encoding
        pe = self.pos_encoder(pos_deltas)  # (B, N, k, C)
        
        # 4. Position Encoding Multiplier (PTv2 feature)
        # Learn multiplier from query features to scale positional encodings dynamically
        pe_multiplier = self.pos_multiplier(q).unsqueeze(2)  # (B, N, 1, C)
        pe_scaled = pe * pe_multiplier  # (B, N, k, C)
        
        # 5. Gather neighbor Key/Value
        k_neighbors = k[batch_indices, knn_idx, :]  # (B, N, k, C)
        v_neighbors = v[batch_indices, knn_idx, :]  # (B, N, k, C)
        
        # 6. Grouped Vector Attention calculation
        # Grouped difference: (B, N, k, C)
        relation = q.unsqueeze(2) - k_neighbors + pe_scaled
        
        # Reshape to separate groups: (B, N, k, groups, group_dim)
        relation_g = relation.view(B, N, self.k, self.groups, self.group_dim)
        
        # Compute weights via relation mapping gamma
        attn_weights = self.gamma(relation).view(B, N, self.k, self.groups, self.group_dim)
        attn_weights = F.softmax(attn_weights, dim=2)  # softmax over k neighbors
        
        # 7. Weighted sum per group
        v_neighbors_shifted = (v_neighbors + pe_scaled).view(B, N, self.k, self.groups, self.group_dim)
        out_g = torch.sum(attn_weights * v_neighbors_shifted, dim=2)  # (B, N, groups, group_dim)
        
        # Flatten groups back to channels
        out = out_g.view(B, N, C)
        return out


class PartitionBasedPooling(nn.Module):
    """
    Voxel Partition-Based Pooling of PTv2.
    Replaces Farthest Point Sampling (FPS) with grid/voxel pooling to speed up downsampling.
    """
    def __init__(self, in_channels, out_channels, grid_size=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz, features, max_down_points=128):
        """
        Subsamples point clouds into a voxel grid representation.
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features
            max_down_points: Target maximum downsampled points
        """
        B, N, C = features.shape
        device = xyz.device
        
        down_xyz_list = []
        down_features_list = []
        
        for b in range(B):
            xyz_b = xyz[b]  # (N, 3)
            feats_b = features[b]  # (N, C)
            
            # Map points to voxel coordinates (integer grids)
            min_xyz = xyz_b.min(dim=0)[0]
            grid_indices = torch.floor((xyz_b - min_xyz) / self.grid_size).long()
            
            # Compute unique 1D hash for each voxel index
            # Large primes to hash 3D indices into a unique ID
            hashes = grid_indices[:, 0] * 73856093 ^ grid_indices[:, 1] * 19349663 ^ grid_indices[:, 2] * 83492791
            
            unique_hashes, inverse_indices = torch.unique(hashes, return_inverse=True)
            num_voxels = unique_hashes.shape[0]
            
            # Downsample: compute centroid coordinate for each unique voxel index
            voxel_xyz = torch.zeros(num_voxels, 3, device=device)
            voxel_xyz.index_add_(0, inverse_indices, xyz_b)
            
            voxel_count = torch.zeros(num_voxels, 1, device=device)
            voxel_count.index_add_(0, inverse_indices, torch.ones(N, 1, device=device))
            voxel_xyz = voxel_xyz / voxel_count
            
            # Pool features: max pool features in each voxel grid
            # Create a large tensor filled with -inf, then scatter-max
            voxel_feats = torch.full((num_voxels, C), float('-inf'), device=device)
            for i in range(num_voxels):
                voxel_feats[i] = torch.max(feats_b[inverse_indices == i], dim=0)[0]
            
            # Truncate or pad to keep batch shape aligned
            if num_voxels > max_down_points:
                # Select top points based on variance or simply sample
                voxel_xyz = voxel_xyz[:max_down_points]
                voxel_feats = voxel_feats[:max_down_points]
            elif num_voxels < max_down_points:
                # Pad with duplicate points/zeros
                pad_size = max_down_points - num_voxels
                pad_xyz = voxel_xyz[-1:].expand(pad_size, -1) if num_voxels > 0 else torch.zeros(pad_size, 3, device=device)
                pad_feats = voxel_feats[-1:].expand(pad_size, -1) if num_voxels > 0 else torch.zeros(pad_size, C, device=device)
                voxel_xyz = torch.cat([voxel_xyz, pad_xyz], dim=0)
                voxel_feats = torch.cat([voxel_feats, pad_feats], dim=0)
                
            down_xyz_list.append(voxel_xyz)
            down_features_list.append(voxel_feats)
            
        down_xyz = torch.stack(down_xyz_list, dim=0)  # (B, max_down_points, 3)
        down_features = torch.stack(down_features_list, dim=0)  # (B, max_down_points, C)
        
        # Channel expansion MLP (Conv1d expects B, C, N)
        down_features = down_features.transpose(1, 2)
        down_features = self.mlp(down_features).transpose(1, 2)
        
        return down_xyz, down_features


class PointTransformerV2Block(nn.Module):
    """
    Standard Point Transformer V2 Block wrapping Grouped Vector Attention and residual MLP.
    """
    def __init__(self, channels, groups=4, k=16):
        super().__init__()
        self.linear_in = nn.Linear(channels, channels)
        self.attn = GroupedVectorAttention(channels, groups=groups, k=k)
        self.linear_out = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz, features):
        residual = features
        x = self.relu(self.linear_in(features))
        x = self.attn(xyz, x)
        x = self.linear_out(x)
        return self.relu(x + residual)


class PointTransformerV2(nn.Module):
    """
    Point Transformer V2 architecture featuring Grouped Vector Attention and Partition pooling.
    """
    def __init__(self, in_channels=6, num_classes=10, groups=4, k=16):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, 32)
        
        self.enc1 = PointTransformerV2Block(32, groups=groups, k=k)
        # Grid sizes (for partition based voxel pooling)
        self.down1 = PartitionBasedPooling(32, 64, grid_size=0.1)
        
        self.enc2 = PointTransformerV2Block(64, groups=groups, k=k)
        self.down2 = PartitionBasedPooling(64, 128, grid_size=0.2)
        
        self.enc3 = PointTransformerV2Block(128, groups=groups, k=k)
        
        # Simple global classification pooling head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, xyz, features):
        # Initial proj
        x = self.in_proj(features)
        
        # Stage 1
        x = self.enc1(xyz, x)
        
        # Downsample Stage 2
        xyz_d1, x_d1 = self.down1(xyz, x, max_down_points=xyz.shape[1] // 2)
        x_d1 = self.enc2(xyz_d1, x_d1)
        
        # Downsample Stage 3
        xyz_d2, x_d2 = self.down2(xyz_d1, x_d1, max_down_points=xyz_d1.shape[1] // 2)
        x_d2 = self.enc3(xyz_d2, x_d2)
        
        # Global Classifier
        # feats shape: (B, N_d2, C) -> pool to (B, C)
        global_feats = x_d2.transpose(1, 2)  # (B, C, N_d2)
        global_feats = self.global_pool(global_feats).squeeze(-1)  # (B, C)
        
        logits = self.classifier(global_feats)
        return logits
