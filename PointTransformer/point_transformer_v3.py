import torch
import torch.nn as nn
import torch.nn.functional as F

def coordinates_to_morton(xyz, grid_size=0.02):
    """
    Computes 3D Morton code (Z-order space-filling curve) in pure PyTorch.
    Args:
        xyz: (B, N, 3) point coordinates
        grid_size: voxel size for quantization
    Returns:
        morton_codes: (B, N) Morton codes of points
    """
    B, N, _ = xyz.shape
    device = xyz.device
    
    # Shift to positive space
    min_xyz = xyz.min(dim=1, keepdim=True)[0]
    quantized = torch.round((xyz - min_xyz) / grid_size).long()
    
    # Keep within safe bounds for 10-bit bitwise interleaving (1024 grid size)
    quantized = torch.clamp(quantized, min=0, max=1023)
    
    x = quantized[..., 0]
    y = quantized[..., 1]
    z = quantized[..., 2]
    
    # Differentiable bitwise interleaving simulation in PyTorch
    morton_codes = torch.zeros(B, N, dtype=torch.long, device=device)
    for i in range(10):
        morton_codes = morton_codes | (((x >> i) & 1) << (3 * i + 2))
        morton_codes = morton_codes | (((y >> i) & 1) << (3 * i + 1))
        morton_codes = morton_codes | (((z >> i) & 1) << (3 * i))
        
    return morton_codes


def serialize_and_sort_points(xyz, features, grid_size=0.02):
    """
    Serializes 3D point cloud into a 1D structured sequence using Morton codes.
    """
    B, N, C = features.shape
    device = xyz.device
    
    # 1. Get Morton codes
    codes = coordinates_to_morton(xyz, grid_size)
    
    # 2. Sort codes along point dimension
    sorted_codes, sort_indices = torch.sort(codes, dim=1)
    
    # 3. Gather sorted coordinates and features
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    sorted_xyz = xyz[batch_idx, sort_indices, :]
    sorted_features = features[batch_idx, sort_indices, :]
    
    return sorted_xyz, sorted_features, sort_indices


class SerializedLocalAttention(nn.Module):
    """
    PTv3 Local Attention block.
    Computes self-attention within local serialized point patches (windows) of size L,
    completely bypassing dynamic KNN query costs.
    """
    def __init__(self, channels, patch_size=32, num_heads=4):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        
        # Position Encoder (MLP mapping relative coordinates of patches to feature space)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, channels)
        )

    def forward(self, sorted_xyz, sorted_features):
        """
        Args:
            sorted_xyz: (B, N, 3) sorted coordinates
            sorted_features: (B, N, C) sorted features
        Returns:
            out_features: (B, N, C) attention output features
        """
        B, N, C = sorted_features.shape
        L = self.patch_size
        
        # We assume N is divisible by L for simplified batching.
        # If not, we pad the sequence.
        padding_needed = (L - (N % L)) % L
        if padding_needed > 0:
            pad_xyz = sorted_xyz[:, -1:].expand(-1, padding_needed, -1)
            pad_feats = torch.zeros(B, padding_needed, C, device=sorted_features.device)
            sorted_xyz = torch.cat([sorted_xyz, pad_xyz], dim=1)
            sorted_features = torch.cat([sorted_features, pad_feats], dim=1)
            
        N_padded = sorted_xyz.shape[1]
        num_patches = N_padded // L
        
        # Reshape to (B * num_patches, L, C)
        patch_feats = sorted_features.view(B * num_patches, L, C)
        patch_xyz = sorted_xyz.view(B * num_patches, L, 3)
        
        # 1. Generate local Positional Encodings inside each patch
        # Relative coordinates inside patch: (B*num_patches, L, L, 3)
        pos_deltas = patch_xyz.unsqueeze(2) - patch_xyz.unsqueeze(1)
        pe = self.pos_encoder(pos_deltas)  # (B*num_patches, L, L, C)
        
        # Max-pool relative coordinates to add a constant positional bias to values
        pe_bias = pe.max(dim=2)[0]  # (B*num_patches, L, C)
        
        # 2. Multi-Head Attention
        # Standard MultiheadAttention is run on the patches
        # For simplified code, we run MHA directly and add positional encoding bias
        attn_out, _ = self.mha(patch_feats, patch_feats, patch_feats)
        
        # Add spatial bias
        attn_out = attn_out + pe_bias
        
        # 3. Reshape back
        out_features = attn_out.view(B, num_patches * L, C)
        
        # Crop back to original sequence length N
        if padding_needed > 0:
            out_features = out_features[:, :N, :]
            
        return out_features


class PointTransformerV3Block(nn.Module):
    """
    Standard Point Transformer V3 Block.
    Integrates Morton Serialization attention, layernorm, and Feed-Forward network.
    """
    def __init__(self, channels, patch_size=32, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SerializedLocalAttention(channels, patch_size=patch_size, num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, sorted_xyz, sorted_features):
        # Attention Residual
        norm_feats = self.norm1(sorted_features)
        attn_feats = self.attn(sorted_xyz, norm_feats)
        x = sorted_features + attn_feats
        
        # FFN Residual
        x = x + self.ffn(self.norm2(x))
        return x


class PointTransformerV3(nn.Module):
    """
    Point Transformer V3 model showcasing serialized sequence processing.
    """
    def __init__(self, in_channels=6, num_classes=10, channels=64, patch_size=32, num_heads=4):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, channels)
        
        # Sequential processing blocks
        self.block1 = PointTransformerV3Block(channels, patch_size=patch_size, num_heads=num_heads)
        self.block2 = PointTransformerV3Block(channels, patch_size=patch_size, num_heads=num_heads)
        self.block3 = PointTransformerV3Block(channels, patch_size=patch_size, num_heads=num_heads)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C_in) point features
        """
        # 1. Project features
        x = self.in_proj(features)
        
        # 2. Serialize and sort point coordinates/features along Morton Z-order curve
        sorted_xyz, sorted_feats, sort_indices = serialize_and_sort_points(xyz, x, grid_size=0.02)
        
        # 3. Serialized Attention Blocks
        x_trans = self.block1(sorted_xyz, sorted_feats)
        x_trans = self.block2(sorted_xyz, x_trans)
        x_trans = self.block3(sorted_xyz, x_trans)
        
        # 4. Global Classifier Head
        x_trans_perm = x_trans.transpose(1, 2)  # (B, C, N)
        global_feats = self.global_pool(x_trans_perm).squeeze(-1)  # (B, C)
        
        logits = self.classifier(global_feats)
        return logits
