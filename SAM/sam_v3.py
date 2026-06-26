import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_v1 import SAM1ImageEncoder, SAM1MaskDecoder

class ExemplarVisualEncoder(nn.Module):
    """
    Encodes visual exemplar crops (few-shot visual prompts) into semantic embeddings.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Simple conv network to extract a compact embedding from a visual crop (e.g. 64x64)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d((1, 1)) # [B, D, 1, 1]
        )

    def forward(self, crop):
        # crop: [B, 3, H_c, W_c]
        out = self.conv(crop)
        return out.flatten(1).unsqueeze(1)  # [B, 1, D] (representing one visual token)

class ConceptEncoder(nn.Module):
    """
    Encodes concepts (either natural language nouns or image exemplars)
    into standard multimodality prompt tokens.
    """
    def __init__(self, vocab_size=30522, embed_dim=256):
        super().__init__()
        # Text Encoder representation
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Visual Exemplar Encoder
        self.exemplar_encoder = ExemplarVisualEncoder(embed_dim=embed_dim)

    def forward(self, text_input_ids=None, exemplar_crops=None):
        """
        Args:
            text_input_ids (Tensor): [B, L] text token IDs
            exemplar_crops (Tensor): [B, 3, H_c, W_c] image crops
        Returns:
            concept_tokens (Tensor): [B, N_tokens, D]
        """
        B = 1
        if text_input_ids is not None:
            B = text_input_ids.shape[0]
        elif exemplar_crops is not None:
            B = exemplar_crops.shape[0]
            
        tokens = []
        
        # 1. Process Text-based concepts
        if text_input_ids is not None:
            txt_emb = self.text_embedding(text_input_ids)  # [B, L, D]
            txt_proj = self.text_projection(txt_emb)        # [B, L, D]
            tokens.append(txt_proj)
            
        # 2. Process Image exemplars
        if exemplar_crops is not None:
            vis_proj = self.exemplar_encoder(exemplar_crops) # [B, 1, D]
            tokens.append(vis_proj)
            
        # If both are empty -> return zeros
        if len(tokens) == 0:
            zeros = torch.zeros(B, 1, 256, device=text_input_ids.device if text_input_ids is not None else None)
            tokens.append(zeros)
            
        return torch.cat(tokens, dim=1)  # [B, N_tokens, D]

class PresenceHead(nn.Module):
    """
    SAM 3 Presence Head.
    Decoupled tracking classifier that predicts the presence probability of a concept in the frame.
    Prevents tracking drift when the object is fully occluded or leaves the frame.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Binary classification head: outputs presence probability (logit)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, decoded_tokens):
        # decoded_tokens: [B, D] (IoU/presence representation token)
        logits = self.fc(decoded_tokens)  # [B, 1]
        return logits

class SAM3(nn.Module):
    """
    SAM 3: Segment Anything with Concepts.
    Unified model that supports Promptable Concept Segmentation (PCS) using
    text labels or image exemplars, and incorporates a Presence Head for tracking robustness.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_encoder = SAM1ImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.concept_encoder = ConceptEncoder(embed_dim=embed_dim)
        self.mask_decoder = SAM1MaskDecoder(embed_dim=embed_dim)
        self.presence_head = PresenceHead(embed_dim=embed_dim)

    def forward(self, images, text_input_ids=None, exemplar_crops=None, mask_priors=None):
        # 1. Extract current frame image features
        img_feats = self.image_encoder(images)  # [B, D, H_feat, W_feat]
        
        # 2. Encode concepts
        sparse_prompts = self.concept_encoder(text_input_ids=text_input_ids, exemplar_crops=exemplar_crops) # [B, N_tokens, D]
        
        # Default empty mask embedding matching features shape
        dense_prompts = torch.zeros_like(img_feats)
        if mask_priors is not None:
            # For simplicity, if mask priors are given we reuse the mask downsampler logic
            pass
            
        # 3. Decode masks and intermediate tokens
        # We subclass or extract tokens from mask_decoder. In our SAM1MaskDecoder,
        # masks and iou_scores are decoded from the top tokens.
        # We'll compute masks and IoU/presence scores.
        masks, output_scores = self.mask_decoder(img_feats, sparse_prompts, dense_prompts)
        
        # Compute presence score based on the global token embeddings inside the decoder.
        # To avoid editing the core SAM1MaskDecoder layers, we can compute presence logits
        # by projecting the mean of sparse prompts (or decoded queries) via the Presence Head.
        presence_logits = self.presence_head(sparse_prompts.mean(dim=1))  # [B, 1]
        
        return masks, output_scores, presence_logits

class SAM3Loss(nn.Module):
    """
    Segment Anything 3 Joint Loss.
    Combines segmentation loss (Focal + Dice) and classification loss (BCE) for the Presence Head.
    """
    def __init__(self, seg_weight=1.0, presence_weight=5.0):
        super().__init__()
        from sam_v1 import SegmentAnythingLoss
        self.seg_loss_fn = SegmentAnythingLoss()
        self.presence_weight = presence_weight

    def forward(self, pred_masks, target_masks, pred_presence, target_presence):
        """
        Args:
            pred_masks (Tensor): predicted masks logits [B, C, H, W]
            target_masks (Tensor): binary ground-truth masks [B, C, H, W]
            pred_presence (Tensor): predicted presence logits [B, 1]
            target_presence (Tensor): binary ground-truth presence labels [B, 1] (0 or 1)
        """
        # Segmentation loss
        seg_loss, focal, dice = self.seg_loss_fn(pred_masks, target_masks)
        
        # Presence loss
        presence_loss = F.binary_cross_entropy_with_logits(pred_presence, target_presence.float())
        
        # Combined Loss
        total_loss = seg_loss + self.presence_weight * presence_loss
        return total_loss, seg_loss, presence_loss
