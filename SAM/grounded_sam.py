import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Append the DINO directory to system path to import GroundingDINO
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "DINO"))
from grounding_dino import GroundingDINO
from sam_v1 import SAM1

class GroundedSAM(nn.Module):
    """
    Grounded Segment Anything Model (Grounded-SAM) Pipeline.
    Combines Grounding DINO (Open-Set Detection) and SAM (High-Precision Segmentation)
    to segment objects from free-form text prompts.
    """
    def __init__(self, vocab_size=30522, num_queries=25, embed_dim=256):
        super().__init__()
        # Grounding DINO detector
        self.detector = GroundingDINO(vocab_size=vocab_size, num_queries=num_queries, embed_dim=embed_dim)
        # SAM 1 Segmenter
        self.segmenter = SAM1(in_channels=3, embed_dim=embed_dim)

    def _box_cxcywh_to_xyxy(self, boxes):
        """
        Converts Grounding DINO coordinates [cx, cy, w, h] to SAM corner coordinates [x1, y1, x2, y2].
        Args:
            boxes (Tensor): [B, Q, 4] normalized coordinates
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, images, input_ids, confidence_threshold=0.25):
        """
        Args:
            images (Tensor): [B, 3, H, W]
            input_ids (Tensor): [B, L] text token IDs
            confidence_threshold (float): threshold to filter detected bounding boxes
        Returns:
            segmented_masks (list of Tensors): list of length B, where each element contains
                                              the predicted high-precision masks for the detected targets
            detected_boxes (list of Tensors): list of length B containing the filtered corner bounding boxes [N_detected, 4]
        """
        B = images.shape[0]
        
        # 1. Run Grounding DINO to predict text-aligned bounding boxes
        detections = self.detector(images, input_ids)
        pred_logits = detections["pred_logits"]  # [B, Q, L] similarity matrix
        pred_boxes = detections["pred_boxes"]    # [B, Q, 4] normalized [cx, cy, w, h]
        
        # Convert detector boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_boxes_xyxy = self._box_cxcywh_to_xyxy(pred_boxes)  # [B, Q, 4]
        
        segmented_masks_batch = []
        detected_boxes_batch = []
        
        for b in range(B):
            # Compute confidence score as max similarity across text tokens for each query
            scores = pred_logits[b].max(dim=-1)[0]  # [Q]
            
            # Filter boxes using threshold
            keep_indices = scores > confidence_threshold
            filtered_boxes = pred_boxes_xyxy[b, keep_indices]  # [N_detected, 4]
            
            if filtered_boxes.shape[0] > 0:
                # 2. Run SAM using the filtered boxes as prompt inputs
                # Extract image features for the batch image
                img_single = images[b : b+1]  # [1, 3, H, W]
                img_feats = self.segmenter.image_encoder(img_single)  # [1, D, H_feat, W_feat]
                
                # Expand image features to match the number of detected boxes
                N_detected = filtered_boxes.shape[0]
                img_feats_expanded = img_feats.repeat(N_detected, 1, 1, 1) # [N_detected, D, H_feat, W_feat]
                img_single_expanded = img_single.repeat(N_detected, 1, 1, 1)
                
                # Pass boxes as prompt encoder constraints
                sparse_prompts, dense_prompts = self.segmenter.prompt_encoder(
                    boxes=filtered_boxes, feat_shape=img_feats_expanded.shape[2:]  # [N_detected, 4]
                )
                
                # Decode masks
                masks, iou_scores = self.segmenter.mask_decoder(
                    img_feats_expanded, sparse_prompts, dense_prompts
                )  # [N_detected, 3, H_up, W_up]
                
                # For each detected box, select the highest-scoring mask
                best_mask_idx = torch.argmax(iou_scores, dim=1)  # [N_detected]
                best_masks = []
                for idx in range(N_detected):
                    best_masks.append(masks[idx, best_mask_idx[idx] : best_mask_idx[idx] + 1, :, :])
                best_masks = torch.stack(best_masks, dim=0)  # [N_detected, 1, H_up, W_up]
                
                segmented_masks_batch.append(best_masks)
                detected_boxes_batch.append(filtered_boxes)
            else:
                # No objects detected above threshold
                segmented_masks_batch.append(None)
                detected_boxes_batch.append(None)
                
        return segmented_masks_batch, detected_boxes_batch
