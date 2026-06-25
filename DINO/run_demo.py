import torch
from dino_v1 import DINOv1, DINOLoss
from dino_v2 import DINOv2, DINOv2Loss
from dino_v3 import DINOv3, DINOv3Loss
from dino_detr import DinoDETR
from grounding_dino import GroundingDINO

def test_dino_v1():
    print("\n-------------------------------------------")
    print("▶ Running Demo for DINO v1 (Meta Self-Distillation)")
    print("-------------------------------------------")
    model = DINOv1(embed_dim=384, out_dim=2048)
    loss_fn = DINOLoss(out_dim=2048)
    
    # 2 global crops (224x224) + 4 local crops (96x96)
    B = 2
    crops = [
        torch.randn(B, 3, 224, 224), # Global 1
        torch.randn(B, 3, 224, 224), # Global 2
        torch.randn(B, 3, 96, 96),   # Local 1
        torch.randn(B, 3, 96, 96),   # Local 2
        torch.randn(B, 3, 96, 96),   # Local 3
        torch.randn(B, 3, 96, 96)    # Local 4
    ]
    
    student_projs, teacher_projs = model(crops)
    loss = loss_fn(student_projs, teacher_projs)
    
    print(f"Student Outputs shape: {student_projs.shape}")  # [B * 6, out_dim] -> [12, 2048]
    print(f"Teacher Outputs shape: {teacher_projs.shape}")  # [B * 2, out_dim] -> [4, 2048]
    print(f"DINOv1 Cross-Entropy Loss: {loss.item():.4f}")
    
    model.update_teacher()
    print("Teacher weights updated successfully.")

def test_dino_v2():
    print("\n-------------------------------------------")
    print("▶ Running Demo for DINOv2 (with SwiGLU, LayerScale, iBOT & KoLeo)")
    print("-------------------------------------------")
    model = DINOv2(embed_dim=384, out_dim=2048, patch_out_dim=512)
    loss_fn = DINOv2Loss(out_dim=2048, patch_out_dim=512)
    
    B = 2
    crops = [
        torch.randn(B, 3, 224, 224), # Global 1
        torch.randn(B, 3, 224, 224), # Global 2
        torch.randn(B, 3, 96, 96),   # Local 1
        torch.randn(B, 3, 96, 96)    # Local 2
    ]
    
    # Generate random masks for global crops (ViT Patch size = 14)
    num_patches = (224 // 14) ** 2  # 256
    masks = [
        torch.rand(B, num_patches) < 0.5, # mask for Global 1
        torch.rand(B, num_patches) < 0.5  # mask for Global 2
    ]
    concat_masks = torch.cat(masks, dim=0) # [2*B, num_patches]
    
    student_cls, student_patches, teacher_cls, teacher_patches, koleo_feats = model(crops, masks)
    
    total_loss, g_loss, p_loss, k_loss = loss_fn(
        student_cls=student_cls,
        student_patches=student_patches,
        teacher_cls=teacher_cls,
        teacher_patches=teacher_patches,
        mask=concat_masks,
        koleo_features=koleo_feats
    )
    
    print(f"Student CLS shape: {student_cls.shape}")      # [B * 4, out_dim] -> [8, 2048]
    print(f"Student Patches shape: {student_patches.shape}")  # [2 * B, num_patches, patch_out_dim] -> [4, 256, 512]
    print(f"DINOv2 Compound Loss: {total_loss.item():.4f}")
    print(f"  ├─ Global Loss: {g_loss.item():.4f}")
    print(f"  ├─ iBOT Loss: {p_loss.item():.4f}")
    print(f"  └─ KoLeo Regularization: {k_loss.item():.4f}")

def test_dino_v3():
    print("\n-------------------------------------------")
    print("▶ Running Demo for DINOv3 (with Gram Anchoring)")
    print("-------------------------------------------")
    model = DINOv3(embed_dim=384, out_dim=2048, patch_out_dim=512)
    loss_fn = DINOv3Loss(out_dim=2048, patch_out_dim=512)
    
    B = 2
    crops = [
        torch.randn(B, 3, 224, 224), # Global 1
        torch.randn(B, 3, 224, 224), # Global 2
        torch.randn(B, 3, 96, 96),   # Local 1
        torch.randn(B, 3, 96, 96)    # Local 2
    ]
    
    num_patches = (224 // 14) ** 2
    masks = [
        torch.rand(B, num_patches) < 0.5,
        torch.rand(B, num_patches) < 0.5
    ]
    concat_masks = torch.cat(masks, dim=0)
    
    (student_cls, student_patches_proj, teacher_cls, teacher_patches_proj, koleo_feats,
     student_raw_patches, teacher_raw_patches) = model(crops, masks)
     
    total_loss, g_loss, p_loss, k_loss, gram_loss = loss_fn(
        student_cls=student_cls,
        student_patches_proj=student_patches_proj,
        teacher_cls=teacher_cls,
        teacher_patches_proj=teacher_patches_proj,
        mask=concat_masks,
        koleo_features=koleo_feats,
        student_raw_patches=student_raw_patches,
        teacher_raw_patches=teacher_raw_patches
    )
    
    print(f"Student Raw Patches shape (for Gram): {student_raw_patches.shape}")  # [2 * B, num_patches, embed_dim] -> [4, 256, 384]
    print(f"DINOv3 Compound Loss: {total_loss.item():.4f}")
    print(f"  ├─ DINO/iBOT/KoLeo base: {(g_loss + p_loss + k_loss*0.1).item():.4f}")
    print(f"  └─ Gram Anchoring Loss: {gram_loss.item():.4f}")

def test_dino_detr():
    print("\n-------------------------------------------")
    print("▶ Running Demo for DINO-DETR Object Detector")
    print("-------------------------------------------")
    model = DinoDETR(num_classes=80, num_queries=20, decoder_layers=6)
    model.train()  # CDN active
    
    B = 2
    images = torch.randn(B, 3, 256, 256)
    
    # Dummy Ground truths
    gt_boxes = torch.tensor([
        [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.2, 0.2]],
        [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.4, 0.4]]
    ])
    gt_labels = torch.tensor([
        [1, 5],
        [12, 0]
    ])
    
    output = model(images, gt_boxes, gt_labels)
    
    print(f"Predicted Class Logits shape (last layer): {output['pred_logits'].shape}") # [B, num_queries, num_classes] -> [2, 20, 80]
    print(f"Predicted Box Coords shape (last layer): {output['pred_boxes'].shape}")   # [B, num_queries, 4] -> [2, 20, 4]
    print(f"Denoising Outputs count: {len(output['dn_outputs']['pred_boxes'])}") # 6 layers of denoising boxes

def test_grounding_dino():
    print("\n-------------------------------------------")
    print("▶ Running Demo for Grounding DINO Open-Set Detector")
    print("-------------------------------------------")
    model = GroundingDINO(vocab_size=30522, num_queries=25, decoder_layers=6)
    
    B = 2
    images = torch.randn(B, 3, 256, 256)
    input_ids = torch.randint(0, 30522, (2, 15))  # Text length = 15 tokens
    
    output = model(images, input_ids)
    
    print(f"Predicted Token Logits shape: {output['pred_logits'].shape}") # [B, num_queries, text_seq_len] -> [2, 25, 15]
    print(f"Predicted Box Coords shape: {output['pred_boxes'].shape}")     # [B, num_queries, 4] -> [2, 25, 4]

if __name__ == "__main__":
    test_dino_v1()
    test_dino_v2()
    test_dino_v3()
    test_dino_detr()
    test_grounding_dino()
    print("\nAll DINO models verified successfully!")
