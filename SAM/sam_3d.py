import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraLiftingModule(nn.Module):
    """
    SAM3D Point Cloud / SA3D NeRF Mask Lifting.
    Projects 2D mask pixels into 3D spatial coordinate points using camera intrinsics and extinsics.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mask_logits, depth_map, intrinsics, extrinsics):
        """
        Args:
            mask_logits (Tensor): 2D mask logits [B, 1, H, W]
            depth_map (Tensor): Depth estimation map [B, 1, H, W] (depth z for each pixel)
            intrinsics (Tensor): Camera intrinsic matrix K [B, 3, 3]
            extrinsics (Tensor): Camera extrinsic matrix [R | T] [B, 3, 4]
        Returns:
            point_cloud (Tensor): 3D points in world space [B, H*W, 3]
            mask_scores_3d (Tensor): 3D segmentation scores associated with points [B, H*W, 1]
        """
        B, _, H, W = mask_logits.shape
        device = mask_logits.device
        
        # 1. Generate pixel grid coordinates
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        x_grid = x_grid.float() + 0.5
        y_grid = y_grid.float() + 0.5
        
        # Reshape to homogeneous coordinates: [H*W, 3] (x, y, 1)
        pixels = torch.stack([x_grid.flatten(), y_grid.flatten(), torch.ones_like(x_grid.flatten())], dim=0) # [3, H*W]
        pixels = pixels.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, H*W]
        
        # 2. Backproject to camera space coordinates: P_cam = K^-1 * pixels * depth
        # Intrinsics K: [B, 3, 3]
        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        p_cam = torch.bmm(K_inv, pixels)    # [B, 3, H*W]
        
        depth_flat = depth_map.reshape(B, 1, H * W)  # [B, 1, H*W]
        p_cam = p_cam * depth_flat                   # [B, 3, H*W]
        
        # 3. Transform to world space: P_world = R^T * (P_cam - T)
        R = extrinsics[:, :, :3]  # [B, 3, 3]
        T = extrinsics[:, :, 3:4] # [B, 3, 1]
        
        R_inv = R.transpose(1, 2)  # Inverse rotation (since R is orthogonal)
        p_cam_shifted = p_cam - T   # [B, 3, H*W]
        p_world = torch.bmm(R_inv, p_cam_shifted)  # [B, 3, H*W]
        
        point_cloud = p_world.transpose(1, 2)  # [B, H*W, 3]
        mask_scores_3d = torch.sigmoid(mask_logits).reshape(B, H * W, 1)  # [B, H*W, 1]
        
        return point_cloud, mask_scores_3d

class Generative3DMeshHead(nn.Module):
    """
    Generative 3D Mesh reconstruction module (representing Meta's SAM 3D Objects).
    Takes 2D image features and generates a 3D mesh (Vertices and Faces) plus spatial layout.
    """
    def __init__(self, embed_dim=256, num_vertices=500, num_faces=1000):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        
        # Feature aggregator (pools image features into a global embedding)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3D bounding box layout predictor: [center(3), scale(3), orientation(3)]
        self.layout_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 9)
        )
        
        # 3D mesh vertices predictor: predicts coordinate offsets from a base sphere mesh
        self.vertices_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_vertices * 3)
        )
        
        # Static sphere base faces representation (faces are constant connectivity patterns)
        self.register_buffer(
            "base_faces",
            torch.randint(0, num_vertices, (num_faces, 3))
        )

    def forward(self, img_feats):
        """
        Args:
            img_feats (Tensor): [B, D, H_feat, W_feat] image features
        Returns:
            vertices (Tensor): 3D coordinates [B, num_vertices, 3]
            faces (Tensor): connection indices [num_faces, 3]
            layout (Tensor): 3D box center, scale, orientation [B, 9]
        """
        B = img_feats.shape[0]
        global_embed = self.pool(img_feats).flatten(1)  # [B, D]
        
        # 1. Predict 3D layout bounding boxes
        layout = self.layout_head(global_embed)  # [B, 9]
        
        # 2. Predict 3D mesh vertices offsets
        vertices_offset = self.vertices_head(global_embed)  # [B, num_vertices * 3]
        vertices = vertices_offset.reshape(B, self.num_vertices, 3).sigmoid() * 2.0 - 1.0  # normalize in [-1, 1] sphere
        
        # 3. Output faces connection topology
        faces = self.base_faces
        
        return vertices, faces, layout

class SAM3D(nn.Module):
    """
    SAM 3D: Unified 3D segmentation lifting and generative shape reconstruction.
    """
    def __init__(self, embed_dim=256, num_vertices=500, num_faces=1000):
        super().__init__()
        self.camera_lifter = CameraLiftingModule()
        self.generative_mesh_head = Generative3DMeshHead(
            embed_dim=embed_dim,
            num_vertices=num_vertices,
            num_faces=num_faces
        )

    def forward(self, img_feats, mask_logits=None, depth_map=None, intrinsics=None, extrinsics=None):
        """
        Args:
            img_feats: 2D feature maps [B, D, H_feat, W_feat]
            mask_logits, depth_map, intrinsics, extrinsics: optional params to perform 2D-to-3D projection
        """
        # 1. Generate 3D mesh shape reconstructions
        vertices, faces, layout = self.generative_mesh_head(img_feats)
        
        # 2. Lift 2D segmentation masks to 3D world space (if parameters provided)
        point_cloud = None
        mask_scores_3d = None
        
        if (mask_logits is not None and depth_map is not None 
                and intrinsics is not None and extrinsics is not None):
            point_cloud, mask_scores_3d = self.camera_lifter(
                mask_logits=mask_logits,
                depth_map=depth_map,
                intrinsics=intrinsics,
                extrinsics=extrinsics
            )
            
        return {
            "vertices": vertices,
            "faces": faces,
            "layout": layout,
            "point_cloud": point_cloud,
            "mask_scores_3d": mask_scores_3d
        }
