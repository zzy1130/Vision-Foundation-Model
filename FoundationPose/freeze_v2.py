import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2FeatureExtractorProxy(nn.Module):
    """
    Proxy representing the frozen DINOv2 foundation model used to extract dense patch descriptors.
    """
    def __init__(self, feature_dim=384):
        super().__init__()
        # Simple representation of a ViT backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=14, stride=14)  # ViT patch projection
        self.proj = nn.Conv2d(64, feature_dim, kernel_size=1)
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB crop of the target object
        Returns:
            descriptors: (B, feature_dim, H_patch, W_patch) dense patch descriptors
        """
        patches = self.conv1(x)
        descriptors = self.proj(patches)
        # Normalize descriptors along the feature channel dimension
        descriptors = F.normalize(descriptors, dim=1)
        return descriptors


class RANSACRegistration(nn.Module):
    """
    A PyTorch-based RANSAC and Kabsch SVD registration module.
    It takes 3D-3D correspondences and robustly solves for rotation R and translation t.
    """
    def __init__(self, num_iterations=100, inlier_threshold=0.02):
        super().__init__()
        self.num_iterations = num_iterations
        self.inlier_threshold = inlier_threshold

    def forward(self, src_pts, dst_pts):
        """
        Robustly registers source points to destination points.
        Args:
            src_pts: (B, P, 3) 3D template model points
            dst_pts: (B, P, 3) 3D query observation points (derived from depth map)
        Returns:
            R: (B, 3, 3) rotation matrix
            t: (B, 3) translation vector
            inliers_mask: (B, P) mask of inliers
        """
        B, P, _ = src_pts.shape
        device = src_pts.device
        
        best_R = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        best_t = torch.zeros(B, 3, device=device)
        best_inliers_count = torch.zeros(B, dtype=torch.long, device=device)
        best_inliers_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        
        # RANSAC loop
        # For efficiency in PyTorch, we sample 3 correspondences at each iteration
        for _ in range(self.num_iterations):
            # 1. Randomly sample 3 points for each batch item
            sample_idx = torch.randint(0, P, (B, 3), device=device)
            
            # Gather samples
            src_sample = torch.gather(src_pts, 1, sample_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, 3, 3)
            dst_sample = torch.gather(dst_pts, 1, sample_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, 3, 3)
            
            # 2. Compute Rigid Transform (Kabsch Algorithm)
            R_est, t_est = self.kabsch_svd(src_sample, dst_sample)
            
            # 3. Evaluate inliers
            # Transform all source points
            src_transformed = torch.bmm(src_pts, R_est.transpose(1, 2)) + t_est.unsqueeze(1)  # (B, P, 3)
            errors = torch.norm(src_transformed - dst_pts, dim=-1)  # (B, P)
            inliers = errors < self.inlier_threshold  # (B, P)
            inliers_count = inliers.sum(dim=1)  # (B,)
            
            # Update best fit for each batch item individually
            update_mask = inliers_count > best_inliers_count
            best_inliers_count[update_mask] = inliers_count[update_mask]
            best_inliers_mask[update_mask] = inliers[update_mask]
            
            # Update rotation and translation
            best_R = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), R_est, best_R)
            best_t = torch.where(update_mask.unsqueeze(-1), t_est, best_t)
            
        # 4. Refit using all inliers for the best hypothesis
        for b in range(B):
            inlier_idx = best_inliers_mask[b]
            if inlier_idx.sum() >= 3:
                src_inlier = src_pts[b, inlier_idx].unsqueeze(0)  # (1, K, 3)
                dst_inlier = dst_pts[b, inlier_idx].unsqueeze(0)  # (1, K, 3)
                R_refit, t_refit = self.kabsch_svd(src_inlier, dst_inlier)
                best_R[b] = R_refit[0]
                best_t[b] = t_refit[0]
                
        return best_R, best_t, best_inliers_mask

    def kabsch_svd(self, A, B_pts):
        """
        Kabsch SVD algorithm to calculate the optimal rotation and translation.
        Formula: A * R^T + t = B_pts
        Args:
            A: (B, K, 3) source point cloud
            B_pts: (B, K, 3) target point cloud
        """
        # Centroids
        centroid_A = A.mean(dim=1, keepdim=True)  # (B, 1, 3)
        centroid_B = B_pts.mean(dim=1, keepdim=True)  # (B, 1, 3)
        
        # Center the points
        A_centered = A - centroid_A
        B_centered = B_pts - centroid_B
        
        # Covariance matrix H
        H = torch.bmm(A_centered.transpose(1, 2), B_centered)  # (B, 3, 3)
        
        # SVD decomposition
        U, S, V = torch.linalg.svd(H)
        
        # Optimal rotation R = V * U^T
        R = torch.bmm(V, U.transpose(1, 2))
        
        # Handle reflection case to prevent left-handed coordinate systems
        det = torch.det(R)
        V_reflected = V.clone()
        V_reflected[:, :, 2] = V_reflected[:, :, 2] * det.unsqueeze(-1)
        R = torch.bmm(V_reflected, U.transpose(1, 2))
        
        # Optimal translation t = centroid_B - R * centroid_A
        t = centroid_B.squeeze(1) - torch.bmm(centroid_A, R.transpose(1, 2)).squeeze(1)
        
        return R, t


class FreeZeV2(nn.Module):
    """
    FreeZeV2 model: Training-Free Zero-Shot 6D Pose Estimation using frozen DINOv2 and geometric features.
    Winner of BOP Challenge 2024.
    """
    def __init__(self, feature_dim=384, ransac_iters=50, inlier_thresh=0.03):
        super().__init__()
        self.dino_extractor = DINOv2FeatureExtractorProxy(feature_dim=feature_dim)
        self.registration = RANSACRegistration(num_iterations=ransac_iters, inlier_threshold=inlier_thresh)

    def forward(self, query_rgb, query_depth, query_intrinsics, template_descriptors, template_pts_3d):
        """
        Args:
            query_rgb: (B, 3, H, W) RGB crop of the detected object
            query_depth: (B, 1, H, W) Depth crop of the detected object
            query_intrinsics: (B, 3, 3) Camera intrinsic matrix for backprojection
            template_descriptors: (B, P, feature_dim) pre-computed descriptors of sparse surface points on the 3D model
            template_pts_3d: (B, P, 3) 3D coordinate values of templates
        Returns:
            R: (B, 3, 3) estimated rotation matrix
            t: (B, 3) estimated translation vector
            confidence_score: (B,) matching confidence score based on feature similarity and inlier ratio
        """
        B, P, C = template_descriptors.shape
        device = query_rgb.device
        
        # Step 1: Extract DINOv2 features from query
        query_feats = self.dino_extractor(query_rgb)  # (B, C, H_patch, W_patch)
        H_patch, W_patch = query_feats.shape[-2:]
        
        # Step 2: Unproject query depth map pixels to 3D query points
        # Downsample depth map to match DINOv2 patch size
        query_depth_ds = F.interpolate(query_depth, size=(H_patch, W_patch), mode='nearest')
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(H_patch, device=device),
            torch.arange(W_patch, device=device),
            indexing='ij'
        )
        # Scale to match original resolution
        h_scale = query_depth.shape[-2] / H_patch
        w_scale = query_depth.shape[-1] / W_patch
        x_orig = (x.float() + 0.5) * w_scale
        y_orig = (y.float() + 0.5) * h_scale
        
        # Flatten and unproject to 3D space
        query_pts_3d = []
        query_flat_desc = []
        for b in range(B):
            depth_b = query_depth_ds[b, 0].flatten()  # (H_patch * W_patch,)
            x_b = x_orig.flatten()
            y_b = y_orig.flatten()
            
            # Intrinsics inversion
            K_inv = torch.inverse(query_intrinsics[b])
            
            # Homogeneous pixel coordinates
            pixels = torch.stack([x_b, y_b, torch.ones_like(x_b)], dim=0)  # (3, H_patch*W_patch)
            rays = torch.mm(K_inv, pixels)  # (3, H_patch*W_patch)
            
            # 3D points: rays * depth
            pts_3d = rays * depth_b.unsqueeze(0)  # (3, H_patch*W_patch)
            query_pts_3d.append(pts_3d.T)
            
            # Reshape query descriptors: (H_patch * W_patch, C)
            desc_b = query_feats[b].reshape(C, -1).T
            query_flat_desc.append(desc_b)
            
        query_pts_3d = torch.stack(query_pts_3d, dim=0)  # (B, H_patch * W_patch, 3)
        query_flat_desc = torch.stack(query_flat_desc, dim=0)  # (B, H_patch * W_patch, C)
        
        # Step 3: Establish 3D-3D Correspondences via Feature Matching
        # Compute cosine similarity between template descriptors and query descriptors
        # sim_matrix: (B, P, H_patch * W_patch)
        sim_matrix = torch.bmm(template_descriptors, query_flat_desc.transpose(1, 2))
        
        # Nearest neighbor matching
        matched_query_idx = torch.argmax(sim_matrix, dim=2)  # (B, P)
        
        # Gather matching 3D coordinates and descriptors from the query
        matched_query_pts = []
        matched_query_desc = []
        for b in range(B):
            idx = matched_query_idx[b]  # (P,)
            matched_query_pts.append(query_pts_3d[b, idx])
            matched_query_desc.append(query_flat_desc[b, idx])
            
        matched_query_pts = torch.stack(matched_query_pts, dim=0)  # (B, P, 3)
        matched_query_desc = torch.stack(matched_query_desc, dim=0)  # (B, P, C)
        
        # Step 4: Robust SVD Registration (RANSAC)
        # Register template_pts_3d with matched_query_pts
        R, t, inliers_mask = self.registration(template_pts_3d, matched_query_pts)
        
        # Step 5: Feature-Aware scoring
        # Score is proportional to the number of inliers and the cosine similarity of inliers
        inlier_ratio = inliers_mask.sum(dim=1).float() / P  # (B,)
        
        # Average similarity of inliers
        similarity_scores = []
        for b in range(B):
            mask_b = inliers_mask[b]
            if mask_b.sum() > 0:
                # Cosine similarity of inliers
                temp_desc = template_descriptors[b, mask_b]
                q_desc = matched_query_desc[b, mask_b]
                sim = F.cosine_similarity(temp_desc, q_desc, dim=-1).mean()
            else:
                sim = torch.tensor(0.0, device=device)
            similarity_scores.append(sim)
        similarity_scores = torch.stack(similarity_scores, dim=0)  # (B,)
        
        confidence_score = inlier_ratio * similarity_scores
        
        return R, t, confidence_score
