import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEmbeddingRandom(nn.Module):
    """
    Random spatial position embedding using sine/cosine for coordinate prompt encoding.
    """
    def __init__(self, num_pos_feats=128, scale=10.0):
        super().__init__()
        self.register_buffer(
            "positional_folder",
            torch.randn(2, num_pos_feats) * scale
        )

    def _pe_encode(self, coords):
        # coords: [B, N, 2] in [0, 1] range
        # Project coordinates
        coords = coords * 2.0 - 1.0  # normalize to [-1, 1]
        # Project via positional random matrix: [B, N, 2] @ [2, D] -> [B, N, D]
        projected = torch.matmul(coords, self.positional_folder)
        # Sin and Cos features
        sin_feats = torch.sin(projected * math.pi)
        cos_feats = torch.cos(projected * math.pi)
        # Cat along feature dim -> [B, N, 2*D]
        return torch.cat([sin_feats, cos_feats], dim=-1)

    def forward(self, coords):
        return self._pe_encode(coords)

class SAM1ImageEncoder(nn.Module):
    """
    A lightweight Vision Transformer (ViT) representation acting as the SAM Image Encoder.
    It takes an image and outputs spatial feature maps.
    """
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16, grid_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Simple spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, grid_size, grid_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Lightweight ViT layers (1x1 convs & self-attention simulated via Transformer Encoder Layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        p = self.patch_embed(x)  # [B, D, H/16, W/16]
        # Add position embeddings (interpolated if input resolution differs)
        if p.shape[2:] != self.pos_embed.shape[2:]:
            pos = F.interpolate(self.pos_embed, size=p.shape[2:], mode='bilinear', align_corners=False)
        else:
            pos = self.pos_embed
            
        p = p + pos
        B, D, H, W = p.shape
        p_flat = p.flatten(2).transpose(1, 2)  # [B, N, D] where N = H*W
        p_fused = self.transformer(p_flat)
        feat_map = p_fused.transpose(1, 2).reshape(B, D, H, W)
        return feat_map

class SAM1PromptEncoder(nn.Module):
    """
    Encodes geometric prompts (Points, Bounding Boxes, Mask priors) into token embeddings.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_generator = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)
        
        # Point prompt types: 0 (background point), 1 (foreground point), -1 (placeholder/padding)
        self.point_type_embeds = nn.ModuleDict({
            "bg": nn.Embedding(1, embed_dim),
            "fg": nn.Embedding(1, embed_dim),
            "padding": nn.Embedding(1, embed_dim)
        })
        
        # Box prompts types: top-left & bottom-right coordinates
        self.box_tl_embed = nn.Embedding(1, embed_dim)
        self.box_br_embed = nn.Embedding(1, embed_dim)
        
        # Mask prompt downsampler (2x conv layers)
        self.mask_downsampler = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=4, stride=4),
            nn.GroupNorm(1, embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=4, stride=4),
            nn.GroupNorm(1, embed_dim),
            nn.GELU()
        )
        self.no_prompt_embed = nn.Embedding(1, embed_dim)

    def forward(self, points=None, labels=None, boxes=None, mask_priors=None, feat_shape=None):
        """
        Args:
            points (Tensor): [B, N_pts, 2] values in [0, 1] range representing coordinate points
            labels (Tensor): [B, N_pts] labels (0=bg, 1=fg)
            boxes (Tensor): [B, 4] normalized box coords [x1, y1, x2, y2]
            mask_priors (Tensor): [B, 1, H, W] mask priors
        Returns:
            sparse_embeddings (Tensor): prompt tokens of shape [B, N_tokens, D]
            dense_embeddings (Tensor): prompt feature map of shape [B, D, H_feat, W_feat]
        """
        B = 1
        if points is not None:
            B = points.shape[0]
        elif boxes is not None:
            B = boxes.shape[0]
        elif mask_priors is not None:
            B = mask_priors.shape[0]
            
        sparse_embeddings = []
        
        # 1. Encode Point prompts
        if points is not None and labels is not None:
            pt_embeds = self.pe_generator(points)  # [B, N_pts, D]
            for i in range(points.shape[1]):
                lbl = labels[:, i]  # [B]
                # Match embedding based on label
                point_emb = torch.zeros_like(pt_embeds[:, i, :])
                bg_mask = (lbl == 0)
                fg_mask = (lbl == 1)
                
                if bg_mask.any():
                    point_emb[bg_mask] = self.point_type_embeds["bg"].weight
                if fg_mask.any():
                    point_emb[fg_mask] = self.point_type_embeds["fg"].weight
                    
                pt_embeds[:, i, :] = pt_embeds[:, i, :] + point_emb
            sparse_embeddings.append(pt_embeds)
            
        # 2. Encode Box prompts (top-left & bottom-right corners)
        if boxes is not None:
            # boxes: [B, 4] -> split to [B, 2] tl and [B, 2] br
            tl_coords = boxes[:, :2]
            br_coords = boxes[:, 2:]
            
            tl_pe = self.pe_generator(tl_coords.unsqueeze(1))[:, 0, :] + self.box_tl_embed.weight # [B, D]
            br_pe = self.pe_generator(br_coords.unsqueeze(1))[:, 0, :] + self.box_br_embed.weight # [B, D]
            
            box_embeds = torch.stack([tl_pe, br_pe], dim=1)  # [B, 2, D]
            sparse_embeddings.append(box_embeds)
            
        # 3. Handle default case if no sparse prompts
        if len(sparse_embeddings) == 0:
            default_emb = self.no_prompt_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, 1, D]
            sparse_embeddings.append(default_emb)
            
        sparse_embeddings = torch.cat(sparse_embeddings, dim=1)  # [B, N_tokens, D]
        
        # 4. Encode Mask priors
        if mask_priors is not None:
            dense_embeddings = self.mask_downsampler(mask_priors)  # [B, D, H_feat, W_feat]
        else:
            # No mask -> return zeros matching feature shape
            h_f, w_f = feat_shape if feat_shape is not None else (64, 64)
            dense_embeddings = torch.zeros(B, self.embed_dim, h_f, w_f, device=sparse_embeddings.device)
            
        return sparse_embeddings, dense_embeddings

class TwoWayAttentionBlock(nn.Module):
    """
    SAM's custom two-way Transformer layer to align tokens and image embeddings.
    """
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        # 1. Token Self-Attention
        self.token_self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm_token1 = nn.LayerNorm(embed_dim)
        
        # 2. Token-to-Image Cross Attention (Token queries Image)
        self.token_to_img_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm_token2 = nn.LayerNorm(embed_dim)
        
        # 3. MLP for Tokens
        self.ffn_token = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
        self.norm_token3 = nn.LayerNorm(embed_dim)
        
        # 4. Image-to-Token Cross Attention (Image queries Token)
        self.img_to_token_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm_img1 = nn.LayerNorm(embed_dim)
        
        # 5. MLP for Image features
        self.ffn_img = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
        self.norm_img2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens, img_feats):
        """
        Args:
            tokens (Tensor): [B, N_tokens, D]
            img_feats (Tensor): [B, N_img, D] (flattened image feature map)
        """
        # 1. Token Self-Attention
        token_self, _ = self.token_self_attn(tokens, tokens, tokens)
        tokens = self.norm_token1(tokens + token_self)
        
        # 2. Token queries Image features
        token_cross, _ = self.token_to_img_attn(tokens, img_feats, img_feats)
        tokens = self.norm_token2(tokens + token_cross)
        
        # 3. Image features query Tokens
        img_cross, _ = self.img_to_token_attn(img_feats, tokens, tokens)
        img_feats = self.norm_img1(img_feats + img_cross)
        
        # 4. FFN/MLP updates
        tokens = self.norm_token3(tokens + self.ffn_token(tokens))
        img_feats = self.norm_img2(img_feats + self.ffn_img(img_feats))
        
        return tokens, img_feats

class SAM1MaskDecoder(nn.Module):
    """
    Decodes prompt tokens and image embeddings into multi-granularity masks and IoU scores.
    """
    def __init__(self, embed_dim=256, num_multimask_outputs=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_outputs = num_multimask_outputs
        
        # Query tokens for mask generation (3 outputs corresponding to whole/part/subpart + 1 for IoU score)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, embed_dim)
        
        # Two-way transformer layers (depth = 2)
        self.transformer_layers = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim=embed_dim, nhead=8) for _ in range(2)
        ])
        
        # Mask projection heads (transposed convolutions to upsample features from H/16 to H/4)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # MLPs to project token representations to coefficients matching the upsampled features
        self.coeff_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim // 8)
            ) for _ in range(self.num_outputs)
        ])
        
        # MLP for predicting IoU scores
        self.iou_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.num_outputs)
        )

    def forward(self, img_feats, sparse_prompts, dense_prompts):
        """
        Args:
            img_feats (Tensor): [B, D, H_feat, W_feat]
            sparse_prompts (Tensor): [B, N_tokens, D]
            dense_prompts (Tensor): [B, D, H_feat, W_feat]
        Returns:
            masks (Tensor): [B, num_outputs, H_feat*4, W_feat*4]
            iou_scores (Tensor): [B, num_outputs]
        """
        B, D, H, W = img_feats.shape
        # Add dense mask prompt to image feature maps
        img_feats_fused = img_feats + dense_prompts
        img_flat = img_feats_fused.flatten(2).transpose(1, 2)  # [B, N_img, D]
        
        # Prepend mask tokens to prompt tokens
        # mask_tokens: [num_mask_tokens, D] -> [B, num_mask_tokens, D]
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(B, 1, 1)
        tokens = torch.cat([mask_tokens, sparse_prompts], dim=1)  # [B, num_mask_tokens + N_tokens, D]
        
        # Run two-way cross attention layers
        for layer in self.transformer_layers:
            tokens, img_flat = layer(tokens, img_flat)
            
        # Extract mask outputs tokens (first 3 tokens) and IoU token (last token)
        mask_token_embeds = tokens[:, :self.num_outputs, :]  # [B, 3, D]
        iou_token_embed = tokens[:, self.num_outputs, :]       # [B, D]
        
        # Predict mask coefficients
        coeffs = []
        for i in range(self.num_outputs):
            coeffs.append(self.coeff_mlps[i](mask_token_embeds[:, i, :])) # [B, D//8]
        coeffs = torch.stack(coeffs, dim=1)  # [B, 3, D//8]
        
        # Predict IoU Scores
        iou_scores = self.iou_mlp(iou_token_embed)  # [B, 3]
        
        # Upsample fused image features
        img_recon = img_flat.transpose(1, 2).reshape(B, D, H, W)
        upsampled_feats = self.upsampler(img_recon)  # [B, D//8, H*4, W*4]
        
        # Dot product between coefficients and features to generate masks
        # coeffs: [B, 3, D//8], upsampled_feats: [B, D//8, H*4, W*4]
        B, C_coeff, H_up, W_up = upsampled_feats.shape
        upsampled_feats_flat = upsampled_feats.reshape(B, C_coeff, H_up * W_up)  # [B, D//8, N_up]
        
        # Compute masks: [B, 3, D//8] @ [B, D//8, N_up] -> [B, 3, N_up]
        masks = torch.bmm(coeffs, upsampled_feats_flat)
        masks = masks.reshape(B, self.num_outputs, H_up, W_up)  # [B, 3, H_up, W_up]
        
        return masks, iou_scores

class SegmentAnythingLoss(nn.Module):
    """
    Loss function for training Segment Anything Models.
    Combines Focal Loss and Dice Loss to handle imbalanced masks and fine boundaries.
    """
    def __init__(self, alpha=0.5, gamma=2.0, dice_weight=1.0, focal_weight=20.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_masks, target_masks):
        """
        Args:
            pred_masks (Tensor): [B, C, H, W] raw logits
            target_masks (Tensor): [B, C, H, W] ground-truth binary masks (0 or 1)
        """
        # Sigmoid activation to turn logits into probabilities
        probs = torch.sigmoid(pred_masks)
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        
        # 1. Focal Loss (pixel-wise binary cross entropy with focusing parameter)
        bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction='none')
        # Focal multiplier
        p_t = probs * target_masks + (1.0 - probs) * (1.0 - target_masks)
        focal_loss = ((1.0 - p_t) ** self.gamma) * bce
        focal_loss = focal_loss.mean()
        
        # 2. Dice Loss
        intersection = (probs * target_masks).sum(dim=(-1, -2))
        union = (probs ** 2).sum(dim=(-1, -2)) + (target_masks ** 2).sum(dim=(-1, -2))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0) # added epsilon for stability
        dice_loss = dice_loss.mean()
        
        # Combined Loss
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss, focal_loss, dice_loss

class SAM1(nn.Module):
    """
    Fully-integrated Segment Anything Model (SAM 1) interface.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_encoder = SAM1ImageEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.prompt_encoder = SAM1PromptEncoder(embed_dim=embed_dim)
        self.mask_decoder = SAM1MaskDecoder(embed_dim=embed_dim)

    def forward(self, images, points=None, labels=None, boxes=None, mask_priors=None):
        # 1. Extract image features
        img_feats = self.image_encoder(images)  # [B, D, H_feat, W_feat]
        
        # 2. Encode user prompts
        sparse_prompts, dense_prompts = self.prompt_encoder(
            points=points, labels=labels, boxes=boxes, mask_priors=mask_priors, feat_shape=img_feats.shape[2:]
        )
        
        # 3. Decode masks and IoU predictions
        masks, iou_scores = self.mask_decoder(img_feats, sparse_prompts, dense_prompts)
        
        return masks, iou_scores
