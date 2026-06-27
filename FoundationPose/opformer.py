import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTemplateEncoder(nn.Module):
    """
    Jointly encodes multiple object template views (rendered crops) 
    using self-attention to learn geometry-consistent visual features.
    """
    def __init__(self, feature_dim=128, num_templates=5):
        super().__init__()
        self.num_templates = num_templates
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        # Self-attention layer across templates
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        
    def forward(self, templates):
        """
        Args:
            templates: (B, num_templates, 3, H, W) template views
        Returns:
            fused_template_feats: (B, num_templates, feature_dim, H/2, W/2)
        """
        B, N, C, H, W = templates.shape
        flat_templates = templates.reshape(B * N, C, H, W)
        feats = self.conv(flat_templates)  # (B * N, feature_dim, H/2, W/2)
        
        _, C_f, H_f, W_f = feats.shape
        feats = feats.reshape(B, N, C_f, H_f, W_f)
        
        # Self-attention along the template count dimension
        # Reshape to token representation: (B, H_f * W_f, N, C_f) -> (B * H_f * W_f, N, C_f)
        feats_perm = feats.permute(0, 3, 4, 1, 2).reshape(B * H_f * W_f, N, C_f)
        attn_out, _ = self.self_attn(feats_perm, feats_perm, feats_perm)
        
        # Reshape back: (B, H_f, W_f, N, C_f) -> (B, N, C_f, H_f, W_f)
        attn_out = attn_out.reshape(B, H_f, W_f, N, C_f).permute(0, 3, 4, 1, 2)
        
        return attn_out


class NOCSPredictionHead(nn.Module):
    """
    NOCS (Normalized Object Coordinate Space) prediction head.
    Predicts the normalized 3D coordinates for each pixel in the target object region.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # (x, y, z) NOCS coordinates
        )

    def forward(self, x):
        """
        Args:
            x: (B, feature_dim, H_feat, W_feat) query visual features
        Returns:
            nocs_map: (B, 3, H_feat, W_feat) NOCS map (values range in [0, 1])
        """
        nocs_logits = self.conv(x)
        # NOCS coordinates are bounded between 0 and 1
        nocs_map = torch.sigmoid(nocs_logits)
        return nocs_map


class TransformerDecoderCorrespondences(nn.Module):
    """
    Transformer Decoder establishing robust 2D-3D correspondences.
    Uses query visual features to query template features and NOCS priors,
    predicting matching 3D points on the object model.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, 
            nhead=4, 
            dim_feedforward=256, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        
        # Head to regress 3D coordinates from features
        self.coord_regressor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # predicted 3D point location
        )

    def forward(self, query_feats, template_feats):
        """
        Args:
            query_feats: (B, feature_dim, H_q, W_q) query feature map
            template_feats: (B, N, feature_dim, H_t, W_t) joint template feature map
        Returns:
            pred_pts_3d: (B, H_q * W_q, 3) predicted 3D object-space points matching the 2D grid pixels
        """
        B, C, H_q, W_q = query_feats.shape
        _, N, _, H_t, W_t = template_feats.shape
        
        # Flatten query features to tokens: (B, H_q * W_q, C)
        query_tokens = query_feats.flatten(2).transpose(1, 2)
        
        # Flatten template features to tokens: (B, N * H_t * W_t, C)
        template_tokens = template_feats.reshape(B, N * H_t * W_t, C)
        
        # Transformer decoder: query_tokens query template_tokens
        decoded_tokens = self.decoder(query_tokens, template_tokens)  # (B, H_q * W_q, C)
        
        # Regress matching 3D points
        pred_pts_3d = self.coord_regressor(decoded_tokens)  # (B, H_q * W_q, 3)
        return pred_pts_3d


class OPFormer(nn.Module):
    """
    OPFormer (Object Pose Transformer) model:
    Unifying object detection, visual features, and NOCS geometric priors for 6DoF pose estimation.
    Uses transformer-decoder correspondences and Kabsch SVD solver to estimate R, t.
    """
    def __init__(self, feature_dim=128, num_templates=5):
        super().__init__()
        # Query encoder (DINOv2 style proxy)
        self.query_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.template_encoder = MultiTemplateEncoder(feature_dim=feature_dim, num_templates=num_templates)
        self.nocs_head = NOCSPredictionHead(feature_dim=feature_dim)
        self.correspondence_decoder = TransformerDecoderCorrespondences(feature_dim=feature_dim)

    def forward(self, query_rgb, query_depth, query_intrinsics, templates):
        """
        Args:
            query_rgb: (B, 3, H, W) RGB query image crop
            query_depth: (B, 1, H, W) Depth query image crop
            query_intrinsics: (B, 3, 3) Camera intrinsic matrix
            templates: (B, num_templates, 3, H, W) multi-view templates
        Returns:
            R: (B, 3, 3) estimated rotation matrix
            t: (B, 3) estimated translation vector
            nocs_map: (B, 3, H_feat, W_feat) predicted object coordinates
            pred_pts_3d: (B, H_feat * W_feat, 3) reconstructed 3D correspondence coordinates
        """
        B, C, H, W = query_rgb.shape
        
        # 1. Extract query features
        query_feats = self.query_encoder(query_rgb)  # (B, feature_dim, H_feat, W_feat)
        H_feat, W_feat = query_feats.shape[-2:]
        
        # 2. Extract multi-template features
        template_feats = self.template_encoder(templates)  # (B, num_templates, feature_dim, H_feat, W_feat)
        
        # 3. Predict NOCS map (pixel-aligned normalized 3D geometry representation)
        nocs_map = self.nocs_head(query_feats)  # (B, 3, H_feat, W_feat)
        
        # 4. Decode 2D-3D correspondences
        pred_pts_3d = self.correspondence_decoder(query_feats, template_feats)  # (B, H_feat * W_feat, 3)
        
        # 5. Get 2D camera coordinates for alignment
        # Downsample depth map to match feature map resolution
        depth_ds = F.interpolate(query_depth, size=(H_feat, W_feat), mode='nearest')
        
        # Build 3D camera-space points from depth and intrinsics (2D projection inverse)
        camera_pts_3d = []
        device = query_rgb.device
        
        # Grid coordinates
        y, x = torch.meshgrid(
            torch.arange(H_feat, device=device),
            torch.arange(W_feat, device=device),
            indexing='ij'
        )
        h_scale = H / H_feat
        w_scale = W / W_feat
        x_orig = (x.float() + 0.5) * w_scale
        y_orig = (y.float() + 0.5) * h_scale
        
        for b in range(B):
            depth_b = depth_ds[b, 0].flatten()  # (H_feat * W_feat,)
            K_inv = torch.inverse(query_intrinsics[b])
            
            pixels = torch.stack([x_orig.flatten(), y_orig.flatten(), torch.ones_like(x_orig.flatten())], dim=0)
            rays = torch.mm(K_inv, pixels)
            pts_3d = rays * depth_b.unsqueeze(0)  # (3, H_feat * W_feat)
            camera_pts_3d.append(pts_3d.T)
            
        camera_pts_3d = torch.stack(camera_pts_3d, dim=0)  # (B, H_feat * W_feat, 3)
        
        # 6. Pose determination using Kabsch SVD
        # Align pred_pts_3d (object model space) with camera_pts_3d (camera space)
        R, t = self.kabsch_svd(pred_pts_3d, camera_pts_3d)
        
        return R, t, nocs_map, pred_pts_3d

    def kabsch_svd(self, A, B_pts):
        """
        Kabsch SVD algorithm. Maps coordinates in A (object space) to B_pts (camera space).
        """
        centroid_A = A.mean(dim=1, keepdim=True)  # (B, 1, 3)
        centroid_B = B_pts.mean(dim=1, keepdim=True)  # (B, 1, 3)
        
        A_centered = A - centroid_A
        B_centered = B_pts - centroid_B
        
        H = torch.bmm(A_centered.transpose(1, 2), B_centered)  # (B, 3, 3)
        
        U, S, V = torch.linalg.svd(H)
        R = torch.bmm(V, U.transpose(1, 2))
        
        # Check reflection
        det = torch.det(R)
        V_reflected = V.clone()
        V_reflected[:, :, 2] = V_reflected[:, :, 2] * det.unsqueeze(-1)
        R = torch.bmm(V_reflected, U.transpose(1, 2))
        
        t = centroid_B.squeeze(1) - torch.bmm(centroid_A, R.transpose(1, 2)).squeeze(1)
        return R, t

    def compute_loss(self, pred_nocs, gt_nocs, pred_pts_3d, gt_pts_3d, pred_R, gt_R, pred_t, gt_t):
        """
        Multi-task loss formulation for training OPFormer:
        - NOCS Loss: Smooth L1 or L2 loss between predicted NOCS map and GT NOCS map.
        - Correspondence Loss: L2 loss between predicted 3D object-space points and GT correspondences.
        - Pose Loss: Rotation geodesic/matrix loss + translation L2.
        """
        nocs_loss = F.mse_loss(pred_nocs, gt_nocs)
        corr_loss = F.mse_loss(pred_pts_3d, gt_pts_3d)
        
        pose_r_loss = F.mse_loss(pred_R, gt_R)
        pose_t_loss = F.mse_loss(pred_t, gt_t)
        
        total_loss = nocs_loss * 2.0 + corr_loss * 1.0 + pose_r_loss * 5.0 + pose_t_loss * 10.0
        return {
            "total_loss": total_loss,
            "nocs_loss": nocs_loss,
            "corr_loss": corr_loss,
            "pose_r_loss": pose_r_loss,
            "pose_t_loss": pose_t_loss
        }
