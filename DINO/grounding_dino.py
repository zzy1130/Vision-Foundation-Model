import torch
import torch.nn as nn
import torch.nn.functional as F
from dino_detr import ToyCNNBackbone

class ToyTextEncoder(nn.Module):
    """
    A simple Text Encoder (representing BERT) to extract token-level text embeddings.
    """
    def __init__(self, vocab_size=30522, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Standard Transformer to model token relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input_ids):
        # input_ids: [B, L] where L is sequence length
        x = self.embedding(input_ids)  # [B, L, D]
        x = self.transformer(x)
        return x

class BiAttentionBlock(nn.Module):
    """
    Bi-directional Multi-Modality Attention layer.
    Allows image tokens to attend to text tokens, and text tokens to attend to image tokens.
    """
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        # Image Self-Attention
        self.img_self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        # Text Self-Attention
        self.txt_self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        
        # Cross Attention: Image-to-Text (Q=Image, K/V=Text)
        self.img2txt_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        # Cross Attention: Text-to-Image (Q=Text, K/V=Image)
        self.txt2img_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        
        # LayerNorms
        self.norm_img1 = nn.LayerNorm(embed_dim)
        self.norm_img2 = nn.LayerNorm(embed_dim)
        self.norm_txt1 = nn.LayerNorm(embed_dim)
        self.norm_txt2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward Networks
        self.ffn_img = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
        self.ffn_txt = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
        self.norm_img3 = nn.LayerNorm(embed_dim)
        self.norm_txt3 = nn.LayerNorm(embed_dim)

    def forward(self, img_feats, txt_feats):
        """
        Args:
            img_feats (Tensor): [B, N, D] (flattened image feature map)
            txt_feats (Tensor): [B, L, D] (text token embeddings)
        """
        # 1. Modality-specific Self-Attention
        img_self, _ = self.img_self_attn(img_feats, img_feats, img_feats)
        img = self.norm_img1(img_feats + img_self)
        
        txt_self, _ = self.txt_self_attn(txt_feats, txt_feats, txt_feats)
        txt = self.norm_txt1(txt_feats + txt_self)
        
        # 2. Cross-Modality Interaction
        # Image-to-Text: Image features query the Text features
        img_cross, _ = self.img2txt_attn(img, txt, txt)
        img = self.norm_img2(img + img_cross)
        
        # Text-to-Image: Text features query the Image features
        txt_cross, _ = self.txt2img_attn(txt, img, img)
        txt = self.norm_txt2(txt + txt_cross)
        
        # 3. FFN
        img = self.norm_img3(img + self.ffn_img(img))
        txt = self.norm_txt3(txt + self.ffn_txt(txt))
        
        return img, txt

class FeatureEnhancer(nn.Module):
    """
    Feature Enhancer stacking BiAttentionBlocks to align image and text spaces.
    """
    def __init__(self, embed_dim=256, depth=2, nhead=8):
        super().__init__()
        self.layers = nn.ModuleList([
            BiAttentionBlock(embed_dim=embed_dim, nhead=nhead) for _ in range(depth)
        ])

    def forward(self, img_feats, txt_feats):
        for layer in self.layers:
            img_feats, txt_feats = layer(img_feats, txt_feats)
        return img_feats, txt_feats

class LanguageGuidedQuerySelection(nn.Module):
    """
    Language-guided query selection.
    Uses text features to select highly relevant image patches to initialize object queries.
    """
    def __init__(self, embed_dim=256, num_queries=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        
        # Project selected features to boxes (cx, cy, w, h)
        self.box_head = nn.Linear(embed_dim, 4)
        # Learnable content queries to mix in
        self.content_queries = nn.Parameter(torch.zeros(num_queries, embed_dim))

    def forward(self, img_feats, txt_feats):
        """
        Args:
            img_feats (Tensor): [B, N, D] fused image features
            txt_feats (Tensor): [B, L, D] fused text features
        Returns:
            init_boxes (Tensor): [B, num_queries, 4]
            tgt_features (Tensor): [B, num_queries, D]
        """
        B, N, D = img_feats.shape
        L = txt_feats.shape[1]
        
        # 1. Compute similarity between image patches and text tokens
        # Norm features to compute cosine-like similarity
        img_norm = F.normalize(img_feats, p=2, dim=-1)
        txt_norm = F.normalize(txt_feats, p=2, dim=-1)
        
        # Dot product: [B, N, L]
        sim = torch.bmm(img_norm, txt_norm.transpose(1, 2))
        
        # Find maximum similarity value across all text tokens for each image patch
        max_sim, _ = sim.max(dim=-1)  # [B, N]
        
        # 2. Select top-K image patches with highest text relevance
        topk_scores, topk_indices = torch.topk(max_sim, self.num_queries, dim=1)  # [B, num_queries]
        
        # Gather corresponding image features
        batch_indices = torch.arange(B, device=img_feats.device).unsqueeze(-1).expand(-1, self.num_queries)
        selected_img_feats = img_feats[batch_indices, topk_indices]  # [B, num_queries, D]
        
        # 3. Predict box coordinates (x, y, w, h) in sigmoid space
        init_boxes = self.box_head(selected_img_feats).sigmoid()  # [B, num_queries, 4]
        
        # 4. Content queries initialized with learnable parameters mixed with top-k text-guided features
        tgt_features = self.content_queries.unsqueeze(0).expand(B, -1, -1) + selected_img_feats * 0.1
        
        return init_boxes, tgt_features

class GroundingDinoDecoderLayer(nn.Module):
    """
    Cross-Modality Decoder Layer.
    Fuses object queries with both image features and text features.
    """
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        # Cross Attention with Image memory
        self.img_cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        # Cross Attention with Text memory
        self.txt_cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim)
        )
        self.pos_proj = nn.Linear(4, embed_dim)

    def forward(self, tgt, img_memory, txt_memory, ref_points):
        """
        Args:
            tgt (Tensor): queries [B, Q, D]
            img_memory (Tensor): fused image maps [B, N, D]
            txt_memory (Tensor): fused text representations [B, L, D]
            ref_points (Tensor): current anchor box coordinates [B, Q, 4]
        """
        query_pos = self.pos_proj(ref_points)
        
        # 1. Query Self-Attention
        q = tgt + query_pos
        tgt2, _ = self.self_attn(q, q, tgt)
        tgt = self.norm1(tgt + tgt2)
        
        # 2. Query-to-Text Attention (guides queries with prompt semantics)
        q = tgt + query_pos
        tgt2, _ = self.txt_cross_attn(q, txt_memory, txt_memory)
        tgt = self.norm2(tgt + tgt2)
        
        # 3. Query-to-Image Attention (localizes features matching boxes)
        q = tgt + query_pos
        tgt2, _ = self.img_cross_attn(q, img_memory, img_memory)
        tgt = self.norm3(tgt + tgt2)
        
        # 4. FFN
        tgt = self.norm4(tgt + self.ffn(tgt))
        
        return tgt

class GroundingDINO(nn.Module):
    """
    Grounding DINO architecture for Open-Set Object Detection.
    Fuses image features and text embeddings, performs language-guided query selection,
    and returns predictions matching text tokens.
    """
    def __init__(self, vocab_size=30522, num_queries=100, embed_dim=256, decoder_layers=6):
        super().__init__()
        self.image_backbone = ToyCNNBackbone(hidden_dim=embed_dim)
        self.text_encoder = ToyTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        
        # 1x1 Conv to match channel size
        self.image_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Feature Enhancer
        self.enhancer = FeatureEnhancer(embed_dim=embed_dim, depth=2, nhead=8)
        
        # Language Guided Query Selection
        self.query_selection = LanguageGuidedQuerySelection(embed_dim=embed_dim, num_queries=num_queries)
        
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            GroundingDinoDecoderLayer(embed_dim=embed_dim, nhead=8) for _ in range(decoder_layers)
        ])
        
        # Iterative Box heads
        self.box_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 4)
            ) for _ in range(decoder_layers)
        ])
        
        # Contrastive Projection Heads (projects query to multimodal space)
        # Text similarity classification is computed as dot product between query projection and text token embedding
        self.query_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(decoder_layers)
        ])
        self.text_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, images, input_ids):
        """
        Args:
            images (Tensor): [B, 3, H, W]
            input_ids (Tensor): [B, L] text token IDs
        """
        B = images.shape[0]
        
        # 1. Multi-scale Image features (using single scaled map for simplicity)
        img_maps = self.image_backbone(images)
        c3 = img_maps[-1]  # Use highest resolution map [B, D, H_feat, W_feat]
        c3_proj = self.image_proj(c3)
        img_feats = c3_proj.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 2. Text features
        txt_feats = self.text_encoder(input_ids)  # [B, L, D]
        
        # 3. Align Modalities using Feature Enhancer
        img_feats, txt_feats = self.enhancer(img_feats, txt_feats)
        
        # 4. Language Guided Query Selection
        ref_boxes, tgt_queries = self.query_selection(img_feats, txt_feats)
        
        # 5. Decoder layers (Iterative update with Look Forward Twice coordinates graph)
        outputs_coords = []
        outputs_similarity = []
        
        current_ref_boxes = ref_boxes
        
        # Project text features for similarity calculation
        projected_txt = self.text_proj(txt_feats)  # [B, L, D]
        projected_txt_norm = F.normalize(projected_txt, p=2, dim=-1)
        
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            # Decode queries
            tgt_queries = decoder_layer(tgt_queries, img_feats, txt_feats, current_ref_boxes)
            
            # Predict box offset
            box_offsets = self.box_predictors[layer_idx](tgt_queries)
            
            # Look Forward Twice coordinate update
            inverse_ref = torch.logit(current_ref_boxes.clamp(1e-5, 1-1e-5))
            next_ref_boxes = (inverse_ref + box_offsets).sigmoid()
            
            # Compute Contrastive Logits (similarity matrix between queries and text tokens)
            # [B, Q, D] -> projected to multimodal space
            projected_queries = self.query_proj[layer_idx](tgt_queries)
            projected_queries_norm = F.normalize(projected_queries, p=2, dim=-1)
            
            # Dot product similarity: [B, Q, D] * [B, D, L] -> [B, Q, L]
            similarity = torch.bmm(projected_queries_norm, projected_txt_norm.transpose(1, 2))
            
            outputs_coords.append(next_ref_boxes)
            outputs_similarity.append(similarity)
            
            # Update reference points (detached for local layer computations)
            current_ref_boxes = next_ref_boxes.detach()
            
        return {
            "pred_logits": outputs_similarity[-1],  # [B, num_queries, text_seq_len]
            "pred_boxes": outputs_coords[-1],       # [B, num_queries, 4]
            "aux_outputs": [{"pred_logits": sim, "pred_boxes": co} for sim, co in zip(outputs_similarity[:-1], outputs_coords[:-1])]
        }

if __name__ == "__main__":
    print("Testing Grounding DINO model...")
    # Initialize model
    model = GroundingDINO(vocab_size=30522, num_queries=25, decoder_layers=6)
    
    # Simulate batch of 2 images and text sequences of length 15 tokens
    images = torch.randn(2, 3, 256, 256)
    input_ids = torch.randint(0, 30522, (2, 15))
    
    # Forward Pass
    output = model(images, input_ids)
    
    print("Logits shape (queries to text tokens similarity):", output["pred_logits"].shape)  # [B, num_queries, text_seq_len] -> [2, 25, 15]
    print("Pred boxes shape:", output["pred_boxes"].shape)  # [B, num_queries, 4] -> [2, 25, 4]
    print("Aux outputs count:", len(output["aux_outputs"]))
