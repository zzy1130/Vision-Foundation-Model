import torch
from point_transformer_v1 import PointTransformerV1
from point_transformer_v2 import PointTransformerV2
from point_transformer_v3 import PointTransformerV3
from point_mae import PointMAE

def run_pt_v1_demo(device):
    print("\n--- Running Point Transformer V1 Demo ---")
    # Inputs: B=2, N=128, C_in=6 (XYZ + RGB/Normals), num_classes=10
    xyz = torch.randn(2, 128, 3, device=device)
    features = torch.randn(2, 128, 6, device=device)
    
    model = PointTransformerV1(in_channels=6, num_classes=10, k=16).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(xyz, features)
        
    print(f"Input Point Coordinates Shape: {xyz.shape}")
    print(f"Input Point Features Shape: {features.shape}")
    print(f"Output Segmentation/Classification Logits Shape: {logits.shape}")  # (B, N, num_classes)
    print("Point Transformer V1 executed successfully!")


def run_pt_v2_demo(device):
    print("\n--- Running Point Transformer V2 Demo ---")
    # Inputs: B=2, N=128, C_in=6, num_classes=10, groups=4
    xyz = torch.randn(2, 128, 3, device=device)
    features = torch.randn(2, 128, 6, device=device)
    
    model = PointTransformerV2(in_channels=6, num_classes=10, groups=4, k=16).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(xyz, features)
        
    print(f"Input Point Coordinates Shape: {xyz.shape}")
    print(f"Input Point Features Shape: {features.shape}")
    print(f"Output Voxel-Classified Logits Shape: {logits.shape}")  # (B, num_classes)
    print("Point Transformer V2 executed successfully!")


def run_pt_v3_demo(device):
    print("\n--- Running Point Transformer V3 Demo ---")
    # Inputs: B=2, N=128, C_in=6, num_classes=10
    xyz = torch.randn(2, 128, 3, device=device)
    features = torch.randn(2, 128, 6, device=device)
    
    model = PointTransformerV3(in_channels=6, num_classes=10, channels=64, patch_size=32, num_heads=4).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(xyz, features)
        
    print(f"Input Point Coordinates Shape: {xyz.shape}")
    print(f"Input Point Features Shape: {features.shape}")
    print(f"Output Morton-Serialized Logits Shape: {logits.shape}")  # (B, num_classes)
    print("Point Transformer V3 executed successfully!")


def run_point_mae_demo(device):
    print("\n--- Running Point-MAE (ECCV 2022) Demo ---")
    # Inputs: B=2, N=256 points
    xyz = torch.randn(2, 256, 3, device=device)
    
    model = PointMAE(embed_dim=128, depth_enc=3, depth_dec=1, mask_ratio=0.6, k=16).to(device)
    model.eval()
    
    with torch.no_grad():
        # Target 64 patches. Mask ratio is 0.6, so 60% of patches are masked.
        reconstructed_patches, gt_masked_patches, masked_indices = model(xyz, target_num_patches=64)
        
        # Calculate Chamfer reconstruction loss
        loss = model.compute_loss(reconstructed_patches, gt_masked_patches)
        
    print(f"Input Point Cloud Coordinates: {xyz.shape}")
    print(f"Reconstructed Masked Patches Shape: {reconstructed_patches.shape}")  # (B, M_masked, k, 3)
    print(f"Ground Truth Masked Patches Shape: {gt_masked_patches.shape}")  # (B, M_masked, k, 3)
    print(f"Masked Patch Indices Shape: {masked_indices.shape}")
    print(f"Calculated Chamfer Loss: {loss.item():.4f}")
    print("Point-MAE executed successfully!")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_pt_v1_demo(device)
    run_pt_v2_demo(device)
    run_pt_v3_demo(device)
    run_point_mae_demo(device)
    
    print("\nAll Point cloud model demos completed successfully!")
