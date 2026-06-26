import torch
from sam_v1 import SAM1, SegmentAnythingLoss
from sam_v2 import SAM2
from hq_sam import HQSAM
from mobile_sam import MobileSAM, train_decoupled_distillation_step
from grounded_sam import GroundedSAM

def test_sam_v1():
    print("\n--- Testing SAM 1 (Segment Anything Model) ---")
    model = SAM1(in_channels=3, embed_dim=256)
    model.eval()
    
    # Simulate single image [1, 3, 256, 256]
    images = torch.randn(1, 3, 256, 256)
    
    # Prompt: 2 coordinate points with labels (1=fg, 0=bg)
    points = torch.tensor([[[0.3, 0.4], [0.7, 0.8]]])  # [1, 2, 2]
    labels = torch.tensor([[1, 0]])                     # [1, 2]
    
    # Prompt: 1 Bounding Box [x1, y1, x2, y2]
    boxes = torch.tensor([[0.25, 0.35, 0.75, 0.85]])    # [1, 4]
    
    # Forward Pass with Point prompts
    masks_pts, iou_pts = model(images, points=points, labels=labels)
    print("Point prompt masks shape:", masks_pts.shape)  # Expect [1, 3, 64, 64] (upsampled features)
    print("Point prompt IoU scores shape:", iou_pts.shape) # Expect [1, 3]
    
    # Forward Pass with Box prompt
    masks_box, iou_box = model(images, boxes=boxes)
    print("Box prompt masks shape:", masks_box.shape)
    print("Box prompt IoU scores shape:", iou_box.shape)
    
    # Verify focal + dice loss calculation
    loss_fn = SegmentAnythingLoss()
    target_masks = torch.randint(0, 2, (1, 3, 64, 64)).float()
    total_loss, focal, dice = loss_fn(masks_pts, target_masks)
    print(f"SAM 1 Loss verification: Total={total_loss.item():.4f} (Focal={focal.item():.4f}, Dice={dice.item():.4f})")

def test_sam_v2():
    print("\n--- Testing SAM 2 (Segment Anything 2 - Video Tracking) ---")
    model = SAM2(in_channels=3, embed_dim=256)
    model.reset_video_memory()
    model.eval()
    
    # Simulate a stream of 3 video frames
    seq_len = 3
    print(f"Processing a video stream of {seq_len} frames sequentially...")
    
    # Frame 0: Interactive Frame (user clicks foreground point)
    print("  [Frame 0] User interacts (clicks foreground point):")
    img0 = torch.randn(1, 3, 256, 256)
    pts0 = torch.tensor([[[0.5, 0.5]]])  # [1, 1, 2]
    lbls0 = torch.tensor([[1]])          # [1, 1]
    
    masks0, iou0 = model(img0, points=pts0, labels=lbls0, is_video_frame=True)
    print(f"    Predicted mask shape: {masks0.shape}, Top IoU score: {iou0.max().item():.4f}")
    
    # Frame 1: Tracking Frame (no new prompt, model relies on memory of Frame 0)
    print("  [Frame 1] Tracking mode (relying on Memory Bank):")
    img1 = torch.randn(1, 3, 256, 256)
    
    masks1, iou1 = model(img1, is_video_frame=True)
    print(f"    Predicted mask shape: {masks1.shape}, Top IoU score: {iou1.max().item():.4f}")
    
    # Frame 2: Interactive Correction Frame (user adds background point to refine)
    print("  [Frame 2] User refines tracking (adds background point):")
    img2 = torch.randn(1, 3, 256, 256)
    pts2 = torch.tensor([[[0.2, 0.2]]])
    lbls2 = torch.tensor([[0]])
    
    masks2, iou2 = model(img2, points=pts2, labels=lbls2, is_video_frame=True)
    print(f"    Predicted mask shape: {masks2.shape}, Top IoU score: {iou2.max().item():.4f}")
    
    # Check that memory bank is populated
    print("Current Memory Bank queue length:", len(model.memory_bank.queue))

def test_hq_sam():
    print("\n--- Testing HQ-SAM (High-Quality SAM) ---")
    model = HQSAM(in_channels=3, embed_dim=256)
    model.eval()
    
    images = torch.randn(1, 3, 256, 256)
    points = torch.tensor([[[0.4, 0.6]]])
    labels = torch.tensor([[1]])
    
    # Forward Pass
    masks, iou_scores = model(images, points=points, labels=labels)
    # HQ-SAM outputs 3 original multimasks + 1 high-quality mask = 4 masks total
    print("HQ-SAM masks shape:", masks.shape)  # Expect [1, 4, 64, 64]
    print("HQ-SAM IoU scores shape:", iou_scores.shape)  # Expect [1, 4]
    print("  First 3 masks: standard SAM outputs (whole/part/subpart)")
    print("  Index 3 mask: High-Quality fused boundary mask")

def test_mobile_sam():
    print("\n--- Testing MobileSAM & Decoupled Distillation ---")
    mobile_model = MobileSAM(in_channels=3, embed_dim=256)
    mobile_model.eval()
    
    images = torch.randn(1, 3, 256, 256)
    boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]])
    
    # Forward Pass
    masks, iou = mobile_model(images, boxes=boxes)
    print("MobileSAM masks shape:", masks.shape)  # Expect [1, 3, 64, 64]
    print("MobileSAM IoU scores shape:", iou.shape)
    
    # Distillation step illustration
    print("Simulating a Decoupled Distillation step (TinyViT student aligns with ViT-H teacher)...")
    # Initialize heavy teacher image encoder and lightweight student image encoder
    from sam_v1 import SAM1ImageEncoder
    teacher_encoder = SAM1ImageEncoder(in_channels=3, embed_dim=256)  # simulating ViT-H
    student_encoder = mobile_model.image_encoder
    
    optimizer = torch.optim.Adam(student_encoder.parameters(), lr=1e-4)
    loss = train_decoupled_distillation_step(student_encoder, teacher_encoder, optimizer, images)
    print(f"  Distillation Feature MSE Loss: {loss:.6f}")

def test_grounded_sam():
    print("\n--- Testing Grounded-SAM (Open-Vocabulary Segmentation Pipeline) ---")
    model = GroundedSAM(vocab_size=30522, num_queries=15, embed_dim=256)
    model.eval()
    
    # Simulate single image and text inputs
    images = torch.randn(1, 3, 256, 256)
    # Token-level input (length 10) representing a query like "segment the puppy and the ball"
    input_ids = torch.randint(0, 30522, (1, 10))
    
    # Pipeline execution (Confidence threshold set to a low value to ensure we trigger segments in mock run)
    masks_batch, boxes_batch = model(images, input_ids, confidence_threshold=0.1)
    
    for idx, (masks, boxes) in enumerate(zip(masks_batch, boxes_batch)):
        if masks is not None:
            print(f"Image {idx}: Detected {boxes.shape[0]} matching items.")
            print(f"  Box prompts xyxy coordinates:\n{boxes}")
            print(f"  SAM high-precision masks shape: {masks.shape}")  # [N_detected, 1, 64, 64]
        else:
            print(f"Image {idx}: No items detected matching the text prompt.")

def test_sam_v3():
    print("\n--- Testing SAM 3 (Segment Anything with Concepts) ---")
    from sam_v3 import SAM3, SAM3Loss
    model = SAM3(in_channels=3, embed_dim=256)
    model.eval()
    
    images = torch.randn(1, 3, 256, 256)
    # Natural language concept prompt: "yellow school bus" -> encoded tokens
    text_input_ids = torch.randint(0, 30522, (1, 5))
    # Visual exemplar crop prompt: [1, 3, 64, 64]
    exemplar_crops = torch.randn(1, 3, 64, 64)
    
    # Forward Pass with Concept prompts
    masks, scores, presence_logits = model(
        images, text_input_ids=text_input_ids, exemplar_crops=exemplar_crops
    )
    print("SAM 3 Concept masks shape:", masks.shape)            # Expect [1, 3, 64, 64]
    print("SAM 3 Presence logits:", presence_logits)            # Expect [1, 1]
    
    # Verify SAM 3 Loss (segmentation + presence classification)
    loss_fn = SAM3Loss()
    target_masks = torch.randint(0, 2, (1, 3, 64, 64)).float()
    target_presence = torch.tensor([[1]])
    
    total_loss, seg_loss, presence_loss = loss_fn(
        masks, target_masks, presence_logits, target_presence
    )
    print(f"SAM 3 Loss verification: Total={total_loss.item():.4f} (Seg={seg_loss.item():.4f}, Presence={presence_loss.item():.4f})")

def test_sam_3d():
    print("\n--- Testing SAM 3D (Lifting & Mesh Generation) ---")
    from sam_3d import SAM3D
    model = SAM3D(embed_dim=256, num_vertices=100, num_faces=200)
    model.eval()
    
    # 2D features map
    img_feats = torch.randn(1, 256, 16, 16)
    
    # Camera matrix parameters for 3D Lifting
    mask_logits = torch.randn(1, 1, 32, 32)
    depth_map = torch.randn(1, 1, 32, 32).abs() + 0.1  # depth must be positive
    intrinsics = torch.eye(3).unsqueeze(0)             # identity matrix [1, 3, 3]
    extrinsics = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).unsqueeze(0) # [1, 3, 4]
    
    # Forward Pass
    outputs = model(
        img_feats,
        mask_logits=mask_logits,
        depth_map=depth_map,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    
    print("Generative 3D Mesh vertices shape:", outputs["vertices"].shape)  # [1, 100, 3]
    print("Generative 3D Mesh faces shape:", outputs["faces"].shape)        # [200, 3]
    print("Predicted 3D bounding box layout shape:", outputs["layout"].shape) # [1, 9]
    print("Lifted 3D Point Cloud coordinates shape:", outputs["point_cloud"].shape) # [1, 1024, 3]
    print("Lifted 3D Mask segmentation scores shape:", outputs["mask_scores_3d"].shape) # [1, 1024, 1]

def main():
    test_sam_v1()
    test_sam_v2()
    test_hq_sam()
    test_mobile_sam()
    test_grounded_sam()
    test_sam_v3()
    test_sam_3d()

if __name__ == "__main__":
    main()
