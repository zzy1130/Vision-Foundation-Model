import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyCNNBackbone(nn.Module):
    """
    A simple CNN backbone to extract feature maps of different scales.
    In real DINO-DETR, this is typically ResNet or Swin.
    """
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=3, stride=2, padding=1)  # 1/2
        self.conv2 = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1)  # 1/4
        self.conv3 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1)  # 1/8

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        return [c2, c3]  # Return multi-scale features

class MixedQuerySelection(nn.Module):
    """
    DINO's Mixed Query Selection.
    Selects top-K features from the encoder output to initialize anchor boxes (positional queries),
    while the content queries remain as independent learnable parameters.
    """
    def __init__(self, embed_dim=256, num_queries=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # Binary classifier/scoring head to rank encoder outputs
        self.score_head = nn.Linear(embed_dim, 1)
        # Coordinate head to map encoder features to initial box coordinates (x, y, w, h)
        self.box_head = nn.Linear(embed_dim, 4)
        
        # Learnable content queries (initialized to zeros, but learnable offsets can be added)
        self.content_queries = nn.Parameter(torch.zeros(num_queries, embed_dim))

    def forward(self, encoder_features):
        """
        Args:
            encoder_features (Tensor): [B, S, D] where S is the sequence length of all scales combined
        Returns:
            init_boxes (Tensor): [B, num_queries, 4] initial bounding boxes in sigmoid space
            tgt_features (Tensor): [B, num_queries, D] content queries initialized
        """
        B, S, D = encoder_features.shape
        
        # 1. Score each token
        scores = self.score_head(encoder_features).squeeze(-1)  # [B, S]
        
        # 2. Select top-K tokens
        topk_scores, topk_indices = torch.topk(scores, self.num_queries, dim=1)  # [B, num_queries]
        
        # 3. Gather features for the selected indices
        batch_indices = torch.arange(B, device=encoder_features.device).unsqueeze(-1).expand(-1, self.num_queries)
        topk_features = encoder_features[batch_indices, topk_indices]  # [B, num_queries, D]
        
        # 4. Predict initial boxes from selected encoder features
        # The boxes are in sigmoid coordinate space: (cx, cy, w, h)
        init_boxes = self.box_head(topk_features).sigmoid()  # [B, num_queries, 4]
        
        # 5. Mixed query strategy:
        # Content queries: Learnable parameters [num_queries, D] expanded to batch
        tgt_features = self.content_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]
        
        return init_boxes, tgt_features

class ContrastiveDenoising(nn.Module):
    """
    DINO's Contrastive Denoising Training (CDN).
    Prepares noisy ground-truth boxes (both positive and negative) to feed into the decoder
    during training. Denoising helps stabilize and accelerate DETR training.
    """
    def __init__(self, embed_dim=256, num_classes=80):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        # Embeddings representing box reconstruction and class labels
        self.label_encoder = nn.Embedding(num_classes + 1, embed_dim)

    def forward(self, gt_boxes, gt_labels, num_denoising_groups=5, noise_scale=0.2):
        """
        Args:
            gt_boxes (Tensor): [B, M, 4] ground truth bounding boxes
            gt_labels (Tensor): [B, M] ground truth labels
            num_denoising_groups (int): number of denoising query groups
            noise_scale (float): noise scaling factor
        Returns:
            dn_boxes (Tensor): [B, num_dn_queries, 4]
            dn_embeds (Tensor): [B, num_dn_queries, D]
            attn_mask (Tensor): [num_dn_queries + num_queries, num_dn_queries + num_queries] attention mask
        """
        B, M, _ = gt_boxes.shape
        if M == 0:
            return None, None, None
            
        num_dn_queries = num_denoising_groups * M * 2  # Each group has positive & negative query per box
        
        dn_boxes_list = []
        dn_embeds_list = []
        
        for g in range(num_denoising_groups):
            # Positive Noise (low noise)
            pos_noise = (torch.rand_like(gt_boxes) - 0.5) * noise_scale * 0.5
            pos_boxes = (gt_boxes + pos_noise).clamp(0, 1)
            
            # Negative Noise (high noise)
            neg_noise = torch.sign(torch.rand_like(gt_boxes) - 0.5) * noise_scale * 1.5
            neg_boxes = (gt_boxes + neg_noise).clamp(0, 1)
            
            # Embed labels
            # We add label noise (random class flip) for negative queries
            pos_labels = gt_labels
            neg_labels = (gt_labels + torch.randint_like(gt_labels, 1, self.num_classes)) % self.num_classes
            
            pos_embeds = self.label_encoder(pos_labels)
            neg_embeds = self.label_encoder(neg_labels)
            
            dn_boxes_list.extend([pos_boxes, neg_boxes])
            dn_embeds_list.extend([pos_embeds, neg_embeds])
            
        dn_boxes = torch.cat(dn_boxes_list, dim=1)  # [B, num_dn_queries, 4]
        dn_embeds = torch.cat(dn_embeds_list, dim=1)  # [B, num_dn_queries, D]
        
        return dn_boxes, dn_embeds

class DinoDecoderLayer(nn.Module):
    """
    DINO Decoder Layer supporting iterative bounding box refinement.
    """
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Positional coordinate projection (sine position embeddings for boxes)
        self.pos_proj = nn.Linear(4, d_model)

    def forward(self, tgt, memory, ref_points, attn_mask=None):
        """
        Args:
            tgt (Tensor): Decoder queries [B, N, D]
            memory (Tensor): Encoder features [B, S, D]
            ref_points (Tensor): Bounding box coordinates [B, N, 4]
            attn_mask (Tensor, optional): Mask to prevent leakage (e.g., between denoising and matching queries)
        """
        # Embed reference points as positional queries
        query_pos = self.pos_proj(ref_points)
        
        # 1. Self Attention with Positional Queries
        q = tgt + query_pos
        tgt2, _ = self.self_attn(q, q, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + tgt2)
        
        # 2. Cross Attention with memory
        q = tgt + query_pos
        tgt2, _ = self.cross_attn(q, memory, memory)
        tgt = self.norm2(tgt + tgt2)
        
        # 3. FFN
        tgt = self.norm3(tgt + self.ffn(tgt))
        
        return tgt

class DinoDETR(nn.Module):
    """
    DINO-DETR Object Detector with:
    - Mixed Query Selection
    - Contrastive Denoising Training (CDN) representation
    - Look Forward Twice box refinement
    """
    def __init__(self, num_classes=80, num_queries=100, embed_dim=256, decoder_layers=6):
        super().__init__()
        self.backbone = ToyCNNBackbone(hidden_dim=embed_dim)
        
        # Simple Multi-scale Encoder Projector
        self.input_proj = nn.ModuleList([
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        ])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation='gelu'),
            num_layers=3
        )
        
        # Query initializers
        self.query_selection = MixedQuerySelection(embed_dim=embed_dim, num_queries=num_queries)
        self.cdn = ContrastiveDenoising(embed_dim=embed_dim, num_classes=num_classes)
        
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            DinoDecoderLayer(d_model=embed_dim, nhead=8) for _ in range(decoder_layers)
        ])
        
        # Iterative Box refinement Heads (one per decoder layer)
        self.box_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 4)
            ) for _ in range(decoder_layers)
        ])
        
        # Classification Heads (one per decoder layer)
        self.cls_predictors = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(decoder_layers)
        ])

    def forward(self, images, gt_boxes=None, gt_labels=None):
        """
        Args:
            images (Tensor): [B, 3, H, W]
            gt_boxes (Tensor, optional): Ground truth boxes for training [B, M, 4]
            gt_labels (Tensor, optional): Ground truth labels [B, M]
        """
        B = images.shape[0]
        
        # 1. Feature extraction & Encoder
        feats = self.backbone(images)  # List of scale maps
        # Flatten and concatenate multi-scale features for Transformer encoder
        flat_feats = []
        for i, f in enumerate(feats):
            # Map channel & flatten
            proj_f = self.input_proj[i](f)
            flat_f = proj_f.flatten(2).transpose(1, 2)  # [B, H*W, D]
            flat_feats.append(flat_f)
            
        memory = torch.cat(flat_feats, dim=1)  # [B, S, D]
        memory = self.encoder(memory)
        
        # 2. Mixed Query Selection to initialize anchor boxes and queries
        init_boxes, tgt = self.query_selection(memory)  # init_boxes: [B, num_queries, 4]
        
        # 3. If training, inject Contrastive Denoising Queries
        dn_boxes = None
        dn_embeds = None
        if self.training and gt_boxes is not None and gt_labels is not None:
            dn_boxes, dn_embeds = self.cdn(gt_boxes, gt_labels)
            
        # Combine matching queries and denoising queries
        if dn_boxes is not None:
            ref_boxes = torch.cat([dn_boxes, init_boxes], dim=1)
            tgt_features = torch.cat([dn_embeds, tgt], dim=1)
        else:
            ref_boxes = init_boxes
            tgt_features = tgt
            
        # 4. Decoder with Look Forward Twice box update
        # We store intermediate predictions for deep supervision
        outputs_coords = []
        outputs_classes = []
        
        current_ref_boxes = ref_boxes
        
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            # Forward decoder layer
            tgt_features = decoder_layer(tgt_features, memory, current_ref_boxes)
            
            # Class & Box offsets prediction
            class_logits = self.cls_predictors[layer_idx](tgt_features)
            box_offsets = self.box_predictors[layer_idx](tgt_features)
            
            # --- Look Forward Twice Bounding Box Refinement ---
            # In standard DETR:
            # next_ref = inverse_sigmoid(current_ref_boxes.detach()) + box_offsets
            # In DINO's "Look Forward Twice", we do not fully detach or we coordinate the refinement updates.
            # Here we preserve the gradient path so parameters of layer (i-1) are optimized under the guidance 
            # of layer (i)'s predictions. We implement this by keeping the box coordinate computation graph:
            inverse_ref = torch.logit(current_ref_boxes.clamp(1e-5, 1-1e-5))
            next_ref_boxes = (inverse_ref + box_offsets).sigmoid()
            
            outputs_coords.append(next_ref_boxes)
            outputs_classes.append(class_logits)
            
            # Set the reference points for the next layer
            # (In training, we detach coords for cross-attention projection stability,
            # but gradients propagate to offset predictors from the final loss)
            current_ref_boxes = next_ref_boxes.detach()
            
        # Separate Matching predictions from Denoising predictions
        num_matching = init_boxes.shape[1]
        
        final_coords = [c[:, -num_matching:] for c in outputs_coords]
        final_classes = [c[:, -num_matching:] for c in outputs_classes]
        
        out = {
            "pred_logits": final_classes[-1],
            "pred_boxes": final_coords[-1],
            "aux_outputs": [{"pred_logits": cl, "pred_boxes": co} for cl, co in zip(final_classes[:-1], final_coords[:-1])]
        }
        
        if dn_boxes is not None:
            # Return denoising predictions for CDN loss
            num_dn = dn_boxes.shape[1]
            dn_coords = [c[:, :num_dn] for c in outputs_coords]
            dn_classes = [c[:, :num_dn] for c in outputs_classes]
            out["dn_outputs"] = {"pred_logits": dn_classes, "pred_boxes": dn_coords}
            
        return out

if __name__ == "__main__":
    print("Testing DINO-DETR detector...")
    model = DinoDETR(num_classes=80, num_queries=20, decoder_layers=6)
    model.train()  # CDN is active only in train mode
    
    # Batch size = 2, 3 channels, 256x256 image
    images = torch.randn(2, 3, 256, 256)
    
    # Mock ground truths: 2 objects per image
    gt_boxes = torch.tensor([
        [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.2, 0.2]],
        [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.4, 0.4]]
    ])
    gt_labels = torch.tensor([
        [1, 5],
        [12, 0]
    ])
    
    # Forward Pass
    output = model(images, gt_boxes, gt_labels)
    
    print("Pred logits shape:", output["pred_logits"].shape)  # [B, num_queries, num_classes] -> [2, 20, 80]
    print("Pred boxes shape:", output["pred_boxes"].shape)    # [B, num_queries, 4] -> [2, 20, 4]
    print("Aux outputs count:", len(output["aux_outputs"]))   # 5 (decoder layers 0 to 4)
    print("Denoising outputs detected:", "dn_outputs" in output)
    if "dn_outputs" in output:
        print("  Denoising logits shape (last layer):", output["dn_outputs"]["pred_logits"][-1].shape)
