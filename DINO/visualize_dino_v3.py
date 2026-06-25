import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dino_v3 import DINOv3

def create_synthetic_image():
    """
    Creates a synthetic 224x224 image containing a circle and a square
    on a dark background.
    """
    # Create a 224x224 RGB image
    img = Image.new("RGB", (224, 224), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)
    
    # Draw a bright circle (representing an object)
    # bounding box: [x0, y0, x1, y1]
    draw.ellipse([40, 40, 100, 100], fill=(220, 100, 100))
    
    # Draw a bright square (representing another object)
    draw.rectangle([120, 110, 180, 170], fill=(100, 180, 220))
    
    return img

def main():
    print("==================================================================")
    print(" DINOv3 Emerging Spatial Representation & Gram Anchoring Demo")
    print("==================================================================")
    
    # 1. Initialize the DINOv3 model
    # We use small dimensions for quick execution
    embed_dim = 192
    patch_size = 14
    model = DINOv3(embed_dim=embed_dim, out_dim=1024, patch_out_dim=256)
    model.eval()
    
    # 2. Create the synthetic image and convert to tensor
    img = create_synthetic_image()
    # Normalize image to [0, 1] range tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
    
    # 3. Pass through the backbone to get patch features
    # DINOv3 forward returns:
    # (student_cls, student_patches_proj, teacher_cls, teacher_patches_proj,
    #  koleo_features, student_raw_patches, teacher_raw_patches)
    # For a single image, we can just run the student backbone directly
    with torch.no_grad():
        cls_rep, patch_reps = model.student_backbone(img_tensor)
        
    # patch_reps shape: [1, num_patches, embed_dim]
    B, num_patches, D = patch_reps.shape
    h_patches = int(num_patches ** 0.5)  # 224 // 14 = 16
    w_patches = h_patches
    print(f"Input image shape: {img_tensor.shape}")
    print(f"Number of ViT patches: {num_patches} ({h_patches}x{w_patches})")
    print(f"Patch feature embedding dimension: {D}")
    
    # 4. Compute Gram Matrix (Patch similarities)
    # Standard DINOv3 Gram Anchoring normalizes features to Sit on the unit hypersphere
    # so that matrix multiply computes Cosine Similarities
    patch_reps_norm = F.normalize(patch_reps, p=2, dim=-1)  # [1, num_patches, D]
    
    # G = X X^T -> shape: [num_patches, num_patches]
    G = torch.bmm(patch_reps_norm, patch_reps_norm.transpose(1, 2)).squeeze(0)
    
    # 5. Visualize similarity maps from different query points
    # Query 1: Inside the circle (located around patch (5, 5))
    query_1_y, query_1_x = 5, 5
    query_1_idx = query_1_y * w_patches + query_1_x
    sim_map_1 = G[query_1_idx].reshape(h_patches, w_patches).cpu().numpy()
    
    # Query 2: Inside the square (located around patch (10, 11))
    query_2_y, query_2_x = 10, 11
    query_2_idx = query_2_y * w_patches + query_2_x
    sim_map_2 = G[query_2_idx].reshape(h_patches, w_patches).cpu().numpy()
    
    # Query 3: On the background (located around patch (2, 13))
    query_3_y, query_3_x = 2, 13
    query_3_idx = query_3_y * w_patches + query_3_x
    sim_map_3 = G[query_3_idx].reshape(h_patches, w_patches).cpu().numpy()
    
    # 6. Plotting the results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot Input Image
    axes[0].imshow(np.array(img))
    # Mark the query locations
    axes[0].scatter([query_1_x * 14 + 7], [query_1_y * 14 + 7], color='yellow', marker='x', s=100, label='Query 1 (Circle)')
    axes[0].scatter([query_2_x * 14 + 7], [query_2_y * 14 + 7], color='cyan', marker='x', s=100, label='Query 2 (Square)')
    axes[0].scatter([query_3_x * 14 + 7], [query_3_y * 14 + 7], color='white', marker='x', s=100, label='Query 3 (BG)')
    axes[0].set_title("Input Image & Queries")
    axes[0].axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    
    # Plot Similarity Map 1 (Circle Query)
    axes[1].imshow(sim_map_1, cmap="hot", interpolation="nearest")
    axes[1].set_title("Similarity Map (Circle Query)")
    axes[1].axis("off")
    
    # Plot Similarity Map 2 (Square Query)
    axes[2].imshow(sim_map_2, cmap="hot", interpolation="nearest")
    axes[2].set_title("Similarity Map (Square Query)")
    axes[2].axis("off")
    
    # Plot Similarity Map 3 (BG Query)
    axes[3].imshow(sim_map_3, cmap="hot", interpolation="nearest")
    axes[3].set_title("Similarity Map (BG Query)")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    # Create directory for output demo images
    os.makedirs("DINO/demo_images", exist_ok=True)
    out_path = "DINO/demo_images/dino_v3_similarity_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nDemo visualization saved successfully to: {out_path}")
    print("==================================================================")
    print("EXPLANATION:")
    print("1. DINOv3 processes the input image into 16x16=256 spatial patch tokens.")
    print("2. The Gram Matrix (G = X X^T) represents the similarity between all patch pairs.")
    print("3. During training, DINOv3 uses 'Gram Anchoring' (L2 distance to a stable Gram teacher)")
    print("   to regularize this patch similarity layout, preventing dense representation collapse.")
    print("4. Even with random initialization, the similarity maps demonstrate how queries")
    print("   correlate with spatial neighborhoods. In a pre-trained model, the maps would")
    print("   segment the exact object boundaries (circle/square) automatically without labels!")
    print("==================================================================")

if __name__ == "__main__":
    main()
