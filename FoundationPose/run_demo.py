import torch
from foundation_pose import FoundationPose
from any6d import Any6D
from freeze_v2 import FreeZeV2
from opformer import OPFormer

def run_foundation_pose_demo(device):
    print("\n--- Running FoundationPose Demo ---")
    model = FoundationPose(feature_dim=128).to(device)
    model.eval()
    
    # Inputs:
    # B=2 (batch size), N=5 (candidates), C=4 (RGB-D)
    query_img = torch.randn(2, 4, 112, 112, device=device)
    candidate_renders = torch.randn(2, 5, 4, 112, 112, device=device)
    
    with torch.no_grad():
        best_idx, (rot, trans), scores = model(query_img, candidate_renders, refine_iters=3)
        
    print(f"Query Image Shape: {query_img.shape}")
    print(f"Candidate Renders Shape: {candidate_renders.shape}")
    print(f"Predicted Best Candidate Index: {best_idx.tolist()}")
    print(f"Refined Rotation Matrices Shape: {rot.shape}")  # (B, N, 3, 3)
    print(f"Refined Translations Shape: {trans.shape}")  # (B, N, 3)
    print(f"Hypotheses Confidence Scores Shape: {scores.shape}")  # (B, N)
    print("FoundationPose forward pass executed successfully!")


def run_any6d_demo(device):
    print("\n--- Running Any6D (CVPR 2025) Demo ---")
    model = Any6D(feature_dim=128).to(device)
    model.eval()
    
    # Inputs:
    anchor_rgb = torch.randn(2, 3, 112, 112, device=device)
    anchor_depth = torch.randn(2, 1, 112, 112, device=device)
    anchor_mask = (torch.randn(2, 1, 112, 112, device=device) > 0).float()
    
    query_img = torch.randn(2, 4, 112, 112, device=device)
    candidate_renders = torch.randn(2, 5, 4, 112, 112, device=device)
    
    with torch.no_grad():
        best_idx, (rot, trans), scores, scale = model(
            anchor_rgb, anchor_depth, anchor_mask, 
            query_img, candidate_renders, refine_iters=2
        )
        
    print(f"Anchor RGB Image Shape: {anchor_rgb.shape}")
    print(f"Anchor Depth Image Shape: {anchor_depth.shape}")
    print(f"Anchor Mask Shape: {anchor_mask.shape}")
    print(f"Predicted Scale Factor: {scale.squeeze(-1).tolist()}")
    print(f"Refined Rotation Shape: {rot.shape}")
    print(f"Refined Translation Shape: {trans.shape}")
    print(f"Scores Shape: {scores.shape}")
    print("Any6D forward pass executed successfully!")


def run_freeze_v2_demo(device):
    print("\n--- Running FreeZeV2 (BOP 2024 Winner) Demo ---")
    # P=20 sparse model points, feature_dim=384 (DINOv2 standard)
    model = FreeZeV2(feature_dim=384, ransac_iters=30, inlier_thresh=0.05).to(device)
    model.eval()
    
    # Inputs:
    query_rgb = torch.randn(2, 3, 112, 112, device=device)
    query_depth = torch.randn(2, 1, 112, 112, device=device)
    
    # Camera intrinsics matrix
    query_intrinsics = torch.tensor([
        [[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]],
        [[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]]
    ], device=device)
    
    template_descriptors = torch.randn(2, 20, 384, device=device)
    # Ensure they are normalized
    template_descriptors = torch.nn.functional.normalize(template_descriptors, dim=-1)
    
    template_pts_3d = torch.randn(2, 20, 3, device=device)
    
    with torch.no_grad():
        R, t, conf_score = model(
            query_rgb, query_depth, query_intrinsics, 
            template_descriptors, template_pts_3d
        )
        
    print(f"Query RGB Shape: {query_rgb.shape}")
    print(f"Query Depth Shape: {query_depth.shape}")
    print(f"Template 3D Points Shape: {template_pts_3d.shape}")
    print(f"Estimated Rotations Shape: {R.shape}")  # (B, 3, 3)
    print(f"Estimated Translations Shape: {t.shape}")  # (B, 3)
    print(f"Confidence Scores: {conf_score.tolist()}")
    print("FreeZeV2 forward pass executed successfully!")


def run_opformer_demo(device):
    print("\n--- Running OPFormer (CVPR 2026 / WACV 2026) Demo ---")
    model = OPFormer(feature_dim=128, num_templates=5).to(device)
    model.eval()
    
    # Inputs:
    query_rgb = torch.randn(2, 3, 112, 112, device=device)
    query_depth = torch.randn(2, 1, 112, 112, device=device)
    query_intrinsics = torch.tensor([
        [[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]],
        [[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]]
    ], device=device)
    
    # 5 template views
    templates = torch.randn(2, 5, 3, 112, 112, device=device)
    
    with torch.no_grad():
        R, t, nocs, pred_pts_3d = model(query_rgb, query_depth, query_intrinsics, templates)
        
    print(f"Query RGB Shape: {query_rgb.shape}")
    print(f"Templates Shape: {templates.shape}")
    print(f"Predicted NOCS Map Shape: {nocs.shape}")  # (B, 3, H_feat, W_feat)
    print(f"Predicted 3D Correspondences Shape: {pred_pts_3d.shape}")  # (B, H_feat*W_feat, 3)
    print(f"Estimated Rotations Shape: {R.shape}")  # (B, 3, 3)
    print(f"Estimated Translations Shape: {t.shape}")  # (B, 3)
    print("OPFormer forward pass executed successfully!")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_foundation_pose_demo(device)
    run_any6d_demo(device)
    run_freeze_v2_demo(device)
    run_opformer_demo(device)
    
    print("\nAll demos completed successfully!")
