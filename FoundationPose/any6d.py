import torch
import torch.nn as nn
import torch.nn.functional as F
from foundation_pose import FoundationPose

class InstantMeshProxy(nn.Module):
    """
    A proxy module representing the single-view Image-to-3D Reconstruction model (e.g., InstantMesh).
    It extracts normalized 3D geometry features and a point cloud representation from a single RGB image.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Image encoder (DINOv2 style)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        # Triplane / NeRF generator proxy
        self.triplane_conv = nn.Conv2d(latent_dim, latent_dim * 3, kernel_size=1)
        
        # 3D shape regressor (predicting query-level occupancy/SDF)
        self.sdf_decoder = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # signed distance
        )

    def forward(self, rgb_img, query_points=None):
        """
        Args:
            rgb_img: (B, 3, H, W) RGB anchor image
            query_points: (B, P, 3) 3D points to evaluate SDF
        Returns:
            triplane_feats: (B, 3 * latent_dim, H/4, W/4)
            sdf: (B, P, 1) predicted SDF values for the query points
        """
        B = rgb_img.shape[0]
        feats = self.image_encoder(rgb_img)
        triplanes = self.triplane_conv(feats)
        
        if query_points is None:
            # Generate a default mock grid of points if none provided
            query_points = torch.randn(B, 100, 3, device=rgb_img.device)
            
        P = query_points.shape[1]
        
        # Sample image features at projected point locations (simplified placeholder)
        # In a real model, this uses camera projection into the triplane.
        pooled_feat = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # (B, latent_dim)
        pooled_feat_expanded = pooled_feat.unsqueeze(1).expand(-1, P, -1)  # (B, P, latent_dim)
        
        sdf_input = torch.cat([pooled_feat_expanded, query_points], dim=-1)  # (B, P, latent_dim + 3)
        sdf = self.sdf_decoder(sdf_input)
        
        return triplanes, sdf


class CoarseScaleAligner(nn.Module):
    """
    Joint alignment module for 2D-3D alignment and metric scale estimation.
    Aligns the normalized 3D mesh with the depth map of the anchor image.
    """
    def __init__(self):
        super().__init__()
        # MLP to estimate scale factor and translation bias from mask/depth stats
        self.estimator = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # scale s (1), translation dx, dy, dz (3)
        )

    def forward(self, anchor_depth, anchor_mask, reconstructed_pts):
        """
        Args:
            anchor_depth: (B, 1, H, W) depth map of anchor image
            anchor_mask: (B, 1, H, W) mask of target object in anchor image
            reconstructed_pts: (B, P, 3) reconstructed points from InstantMesh
        Returns:
            scale: (B, 1) metric scale factor
            init_pose_trans: (B, 3) translation vector
        """
        B = anchor_depth.shape[0]
        
        # Extract features/statistics from depth map and mask
        # e.g., mean depth, mask area, bounding box width and height
        stats = []
        for i in range(B):
            depth_i = anchor_depth[i]
            mask_i = anchor_mask[i]
            pts_i = reconstructed_pts[i]
            
            mask_pts = depth_i[mask_i > 0.5]
            mean_depth = mask_pts.mean() if len(mask_pts) > 0 else torch.tensor(1.0, device=depth_i.device)
            std_depth = mask_pts.std() if len(mask_pts) > 1 else torch.tensor(0.0, device=depth_i.device)
            
            # Bounding box dimensions
            y_indices, x_indices = torch.where(mask_i[0] > 0.5)
            if len(y_indices) > 0:
                h_bbox = (y_indices.max() - y_indices.min()).float() / mask_i.shape[1]
                w_bbox = (x_indices.max() - x_indices.min()).float() / mask_i.shape[2]
            else:
                h_bbox = torch.tensor(0.0, device=depth_i.device)
                w_bbox = torch.tensor(0.0, device=depth_i.device)
                
            # Point cloud statistics
            mean_pts = pts_i.mean(dim=0)  # (3,)
            std_pts = pts_i.std(dim=0).mean()  # scalar
            
            stat_vec = torch.stack([
                mean_depth, std_depth, h_bbox, w_bbox,
                mean_pts[0], mean_pts[1], mean_pts[2], std_pts
            ])
            stats.append(stat_vec)
            
        stats = torch.stack(stats, dim=0)  # (B, 8)
        pred = self.estimator(stats)
        
        scale = torch.exp(pred[:, 0:1])  # ensure scale is positive
        trans = pred[:, 1:4]
        
        return scale, trans


class Any6D(nn.Module):
    """
    Any6D model: A Model-Free Framework for 6D Pose and Size Estimation of Unseen Objects.
    Leverages InstantMesh for 3D reconstruction and FoundationPose for tracking/refinement.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.reconstructor = InstantMeshProxy(latent_dim=feature_dim)
        self.aligner = CoarseScaleAligner()
        self.pose_estimator = FoundationPose(feature_dim=feature_dim)

    def forward(self, anchor_rgb, anchor_depth, anchor_mask, query_img, candidate_renders, refine_iters=3):
        """
        Args:
            anchor_rgb: (B, 3, H, W) RGB anchor image of the novel object
            anchor_depth: (B, 1, H, W) Depth map of anchor image
            anchor_mask: (B, 1, H, W) Mask of target object in anchor image
            query_img: (B, 4, H, W) RGB-D query crop
            candidate_renders: (B, N, 4, H, W) candidate pose templates rendered from the reconstructed model
            refine_iters: number of pose refinement iterations
        Returns:
            best_pose_index: (B,)
            refined_deltas: tuple of (rot_matrices, trans_vectors)
            scores: (B, N) pose confidence scores
            predicted_scale: (B, 1) metric scale factor of the novel object
        """
        B = anchor_rgb.shape[0]
        
        # Step 1: Reconstruct 3D shape from anchor image
        # We query a small set of mock points to get SDF
        mock_query_pts = torch.randn(B, 100, 3, device=anchor_rgb.device)
        triplanes, sdf = self.reconstructor(anchor_rgb, mock_query_pts)
        
        # Step 2: Joint alignment and metric scale estimation using anchor depth and mask
        # InstantMesh generates normalized points, we compute the scale factor to lift it to metric space.
        scale, init_trans = self.aligner(anchor_depth, anchor_mask, mock_query_pts)
        
        # Step 3: 6D Pose refinement and selection on query image using FoundationPose backbone
        best_pose_idx, refined_deltas, scores = self.pose_estimator(
            query_img, 
            candidate_renders, 
            refine_iters=refine_iters
        )
        
        return best_pose_idx, refined_deltas, scores, scale

    def compute_loss(self, pred_scale, gt_scale, pred_trans, gt_trans, pred_sdf, gt_sdf, pose_loss_dict):
        """
        Loss formulation for training Any6D:
        - Scale Loss: L1 or MSE between predicted and GT metric scale.
        - SDF Loss: L1 or MSE on signed distance values to supervise 3D shape reconstruction.
        - Pose loss: standard FoundationPose loss.
        """
        scale_loss = F.mse_loss(pred_scale, gt_scale)
        sdf_loss = F.l1_loss(pred_sdf, gt_sdf)
        
        total_loss = (
            scale_loss * 5.0 + 
            sdf_loss * 1.0 + 
            pose_loss_dict["total_loss"]
        )
        
        return {
            "total_loss": total_loss,
            "scale_loss": scale_loss,
            "sdf_loss": sdf_loss,
            "refine_loss": pose_loss_dict["refine_loss"],
            "score_loss": pose_loss_dict["score_loss"]
        }
