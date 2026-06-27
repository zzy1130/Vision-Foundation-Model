import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeatureExtractor(nn.Module):
    """
    A lightweight CNN-based feature extractor shared between the query crop and 
    rendered template crop, similar to the ResNet backbone used in FoundationPose.
    """
    def __init__(self, in_channels=4, out_channels=128):
        super().__init__()
        # Input channel is 4 (RGB + Depth)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.res_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: (B, C_in, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        residual = x
        x = self.res_block(x)
        x = self.relu(x + residual)
        return x  # Output shape: (B, out_channels, H/4, W/4)


class PoseRefineNet(nn.Module):
    """
    Pose Refinement Network of FoundationPose.
    It takes features of the query crop and the rendered crop,
    fuses them, and regresses the relative pose update (delta R, delta t).
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        # Concat along channel dimension: feature_dim * 2
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output heads for pose updates
        # Rotation is parameterized as a 6D representation (two columns of R matrix) for stability,
        # and Translation update is a 3D vector.
        self.fc_rot = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)  # 6D rotation representation
        )
        self.fc_trans = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # translation delta (dx, dy, dz)
        )

    def forward(self, query_feats, render_feats):
        # Concatenate features along the channel dimension
        fused = torch.cat([query_feats, render_feats], dim=1)
        fused = self.fuse_conv(fused)
        
        # Pool to a vector
        feat_vec = self.global_pool(fused).flatten(1)
        
        delta_rot_6d = self.fc_rot(feat_vec)
        delta_trans = self.fc_trans(feat_vec)
        
        return delta_rot_6d, delta_trans

    @staticmethod
    def orthog_6d_to_rotation_matrix(rot_6d):
        """
        Convert 6D rotation representation to standard 3x3 rotation matrix.
        """
        x_raw = rot_6d[:, 0:3]
        y_raw = rot_6d[:, 3:6]
        
        x = F.normalize(x_raw, dim=1)
        z = torch.cross(x, y_raw, dim=1)
        z = F.normalize(z, dim=1)
        y = torch.cross(z, x, dim=1)
        
        matrix = torch.stack([x, y, z], dim=2)  # (B, 3, 3)
        return matrix


class PoseScoreNet(nn.Module):
    """
    Pose Selection/Scoring Network of FoundationPose.
    It rates the quality/confidence of refined pose hypotheses.
    Uses self-attention across hypotheses to rank them globally.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Transformer encoder layer for self-attention across pose candidates
        self.transformer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=4, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # scalar score
        )

    def forward(self, query_feats, candidate_render_feats):
        """
        Args:
            query_feats: (B, C, H, W)
            candidate_render_feats: (B, N, C, H, W) where N is the number of pose hypotheses
        """
        B, N, C, H, W = candidate_render_feats.shape
        
        # Expand query_feats to match the number of candidates
        # query_expanded: (B * N, C, H, W)
        query_expanded = query_feats.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C, H, W)
        flat_render_feats = candidate_render_feats.reshape(B * N, C, H, W)
        
        # Pairwise comparison features
        pairwise = torch.cat([query_expanded, flat_render_feats], dim=1)
        pairwise = self.conv(pairwise)
        pairwise_vec = self.pool(pairwise).flatten(1)  # (B * N, C)
        
        # Reshape to (B, N, C) for Transformer sequence modeling
        seq_feats = pairwise_vec.reshape(B, N, -1)
        
        # Multi-head Self-Attention across candidates
        context_feats = self.transformer(seq_feats)  # (B, N, C)
        
        # Predict score
        scores = self.score_head(context_feats).squeeze(-1)  # (B, N)
        return scores


class FoundationPose(nn.Module):
    """
    Unified FoundationPose module wrapper.
    Implements:
      1. Shared feature extraction
      2. Iterative pose refinement
      3. Global candidate scoring and selection
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_extractor = SimpleFeatureExtractor(in_channels=4, out_channels=feature_dim)
        self.refine_net = PoseRefineNet(feature_dim=feature_dim)
        self.score_net = PoseScoreNet(feature_dim=feature_dim)

    def forward(self, query_img, candidate_renders, refine_iters=3):
        """
        Args:
            query_img: (B, 4, H, W) -> RGB-D query crop of the target object region
            candidate_renders: (B, N, 4, H, W) -> RGB-D renders from N candidate poses
            refine_iters: number of pose refinement iterations
        Returns:
            best_pose_index: (B,) index of the candidate pose with the highest confidence
            refined_deltas: tuple of (rot_matrices, trans_vectors) containing the relative pose updates
            scores: (B, N) score for each candidate pose
        """
        B, N, C, H, W = candidate_renders.shape
        
        # Extract features for query crop
        query_feats = self.feature_extractor(query_img)  # (B, C_feat, H_feat, W_feat)
        
        # Extract features for all candidate renders
        flat_renders = candidate_renders.reshape(B * N, C, H, W)
        flat_render_feats = self.feature_extractor(flat_renders)  # (B * N, C_feat, H_feat, W_feat)
        candidate_render_feats = flat_render_feats.reshape(B, N, -1, H // 4, W // 4)
        
        # 1. Pose Refinement
        # Simulating iterative refinement by updating candidate features (conceptually)
        # and regressing final pose deltas
        cur_render_feats = flat_render_feats.clone()
        for _ in range(refine_iters):
            # Predict delta updates using RefineNet
            delta_rot_6d, delta_trans = self.refine_net(
                query_feats.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, -1, H // 4, W // 4),
                cur_render_feats
            )
            # Update virtual rendering features (in a real pipeline, the object is re-rendered,
            # here we regress the updates for the final refined poses)
            # Feature updates are abstracted for demonstration
            cur_render_feats = cur_render_feats + 0.1 * torch.tanh(cur_render_feats)
        
        # Final pose updates
        refined_rot_matrices = self.refine_net.orthog_6d_to_rotation_matrix(delta_rot_6d).reshape(B, N, 3, 3)
        refined_trans = delta_trans.reshape(B, N, 3)
        
        # 2. Pose Selection (Scoring)
        scores = self.score_net(query_feats, candidate_render_feats)  # (B, N)
        
        # Select best pose index
        best_pose_idx = torch.argmax(scores, dim=1)  # (B,)
        
        return best_pose_idx, (refined_rot_matrices, refined_trans), scores

    def compute_loss(self, pred_rot_6d, pred_trans, gt_rot_matrix, gt_trans, pred_scores, gt_scores):
        """
        Loss formulation for training the Pose Refiner and Scorer.
        - Refine loss: standard MSE or Geodesic distance on rotations, L2 on translations.
        - Score loss: Binary Cross Entropy or ranking loss between predicted and ground-truth scores.
        """
        # Rotation 6D projection
        pred_rot_matrix = self.refine_net.orthog_6d_to_rotation_matrix(pred_rot_6d)
        
        # 1. Refine Loss: Rotation (Frobenius norm of matrix difference) + Translation (L2)
        rot_loss = F.mse_loss(pred_rot_matrix, gt_rot_matrix)
        trans_loss = F.mse_loss(pred_trans, gt_trans)
        refine_loss = rot_loss + 10.0 * trans_loss
        
        # 2. Score Loss: Contrastive or ranking loss (we use binary cross entropy here)
        score_loss = F.binary_cross_entropy_with_logits(pred_scores, gt_scores)
        
        total_loss = refine_loss + score_loss
        return {
            "total_loss": total_loss,
            "refine_loss": refine_loss,
            "score_loss": score_loss
        }
