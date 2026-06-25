import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_v1 import SAM1ImageEncoder, SAM1PromptEncoder, SAM1MaskDecoder

class MemoryAttentionBlock(nn.Module):
    """
    SAM 2 Memory Attention Layer.
    Current frame features query the historical spatial features stored in the memory bank.
    """
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        # Cross Attention: current frame queries history
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Self Attention on updated current frame features
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, curr_feats, memory_feats):
        """
        Args:
            curr_feats (Tensor): [B, N_curr, D] current frame image features
            memory_feats (Tensor): [B, N_mem, D] concatenated features from Memory Bank
        """
        # 1. Cross Attention: Current queries Memory
        cross_out, _ = self.cross_attn(curr_feats, memory_feats, memory_feats)
        curr_feats = self.norm1(curr_feats + cross_out)
        
        # 2. Self Attention
        self_out, _ = self.self_attn(curr_feats, curr_feats, curr_feats)
        curr_feats = self.norm2(curr_feats + self_out)
        
        # 3. MLP FFN
        curr_feats = self.norm3(curr_feats + self.ffn(curr_feats))
        
        return curr_feats

class MemoryBank:
    """
    Memory Bank storage for Segment Anything 2.
    Stores spatial features, masks, and prompt pointers of past video frames.
    """
    def __init__(self, max_memory_size=16):
        self.max_memory_size = max_memory_size
        self.reset()

    def reset(self):
        # Stores tuples: (spatial_features [B, D, H, W], mask_predictions [B, 1, H, W], is_interactive_frame [bool])
        self.queue = []

    def add_frame(self, spatial_features, mask_predictions, is_interactive=False):
        if len(self.queue) >= self.max_memory_size:
            self.queue.pop(0)  # Evict oldest frame
        self.queue.append((spatial_features.detach(), mask_predictions.detach(), is_interactive))

    def get_concatenated_memory(self):
        """
        Returns:
            memory_features (Tensor): [B, N_mem, D] concatenated features
        """
        if len(self.queue) == 0:
            return None
            
        B = self.queue[0][0].shape[0]
        D = self.queue[0][0].shape[1]
        
        all_features = []
        for feat, mask, is_int in self.queue:
            # Flatten spatial dims to tokens: [B, D, H, W] -> [B, H*W, D]
            B_f, D_f, H_f, W_f = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)
            
            # Incorporate mask information by mapping mask [B, 1, H, W] -> [B, H*W, D]
            # Downsample/flatten mask to match feature size
            mask_down = F.interpolate(mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
            mask_flat = mask_down.flatten(2).transpose(1, 2)  # [B, N, 1]
            
            # Simple addition of mask embedding: multiply mask by a scale or just add
            # For learning purposes, we add the mask values as a simple cue to features
            feat_fused = feat_flat + mask_flat * 0.1
            all_features.append(feat_fused)
            
        # Concatenate along token length dimension -> [B, N_mem = N_frames * N_img, D]
        return torch.cat(all_features, dim=1)

class SAM2(nn.Module):
    """
    Segment Anything 2 (SAM 2) streaming video and image segmentation model.
    """
    def __init__(self, in_channels=3, embed_dim=256, max_memory_size=16):
        super().__init__()
        self.image_encoder = SAM1ImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.prompt_encoder = SAM1PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = SAM1MaskDecoder(embed_dim=embed_dim)
        
        # Memory attention block to fuse temporal details
        self.memory_attention = MemoryAttentionBlock(embed_dim=embed_dim, nhead=8)
        self.memory_bank = MemoryBank(max_memory_size=max_memory_size)

    def reset_video_memory(self):
        self.memory_bank.reset()

    def forward(self, images, points=None, labels=None, boxes=None, mask_priors=None, is_video_frame=False):
        """
        Runs SAM 2 forward pass. If tracking in video, queries the memory bank of past frames.
        Args:
            images (Tensor): current frame [B, 3, H, W]
            points, labels, boxes, mask_priors: user prompt constraints
            is_video_frame (bool): whether this is a frame in a video stream
        """
        B = images.shape[0]
        
        # 1. Extract current frame image features
        curr_feats = self.image_encoder(images)  # [B, D, H_feat, W_feat]
        B_f, D_f, H_f, W_f = curr_feats.shape
        
        # 2. Query historical frame features if tracking video
        memory_feats = self.memory_bank.get_concatenated_memory()
        
        if is_video_frame and memory_feats is not None:
            # Flatten current features to query memory
            curr_flat = curr_feats.flatten(2).transpose(1, 2)  # [B, N_curr, D]
            # Perform temporal memory attention
            fused_flat = self.memory_attention(curr_flat, memory_feats)  # [B, N_curr, D]
            # Reshape back to spatial dimensions
            curr_feats_fused = fused_flat.transpose(1, 2).reshape(B_f, D_f, H_f, W_f)
        else:
            curr_feats_fused = curr_feats
            
        # 3. Encode user prompts
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=points, labels=labels, boxes=boxes, mask_priors=mask_priors, feat_shape=curr_feats_fused.shape[2:]
        )
        
        # 4. Decode masks and IoU scores
        masks, iou_scores = self.mask_decoder(curr_feats_fused, sparse_prompts, dense_prompts)
        
        # 5. Push current predictions to Memory Bank if in video mode
        if is_video_frame:
            # Determine if this was an interactive frame (i.e. if user prompts were supplied)
            is_interactive = (points is not None or boxes is not None)
            
            # Select the highest scoring mask prediction to save to memory
            best_mask_idx = torch.argmax(iou_scores, dim=1)  # [B]
            best_masks = []
            for b in range(B):
                best_masks.append(masks[b, best_mask_idx[b] : best_mask_idx[b] + 1, :, :])
            best_masks = torch.stack(best_masks, dim=0)  # [B, 1, H_up, W_up]
            
            # Save frame details to memory bank
            self.memory_bank.add_frame(curr_feats, best_masks, is_interactive=is_interactive)
            
        return masks, iou_scores
