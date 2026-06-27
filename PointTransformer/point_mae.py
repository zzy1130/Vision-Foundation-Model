import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetPatchEmbed(nn.Module):
    """
    Mini-PointNet to project irregular 3D point patches (M, K, 3) 
    into dense embedding tokens (M, C).
    """
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, patches):
        """
        Args:
            patches: (B, M, K, C_in) point coordinates in local patches
        Returns:
            tokens: (B, M, C_embed) patch tokens
        """
        B, M, K, C = patches.shape
        flat_patches = patches.reshape(B * M, K, C)
        
        # PointNet MLP on each point inside patches
        x = flat_patches.transpose(1, 2)  # (B * M, C_in, K)
        x = self.mlp(x)  # (B * M, C_embed, K)
        
        # Max pool over points in the patch to get shape representation
        tokens = torch.max(x, dim=-1)[0]  # (B * M, C_embed)
        tokens = tokens.reshape(B, M, -1)
        return tokens


class ChamferLoss(nn.Module):
    """
    Chamfer Distance Loss to calculate reconstruction errors between two 3D point sets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, N, 3) predicted point coordinates
            gt: (B, N, 3) ground truth point coordinates
        """
        B, N, _ = pred.shape
        # Compute pairwise distance: (B, N, N)
        inner = -2 * torch.bmm(pred, gt.transpose(2, 1))
        xx = torch.sum(pred**2, dim=2, keepdim=True)
        yy = torch.sum(gt**2, dim=2, keepdim=True)
        dist = xx + inner + yy.transpose(2, 1)  # (B, N, N)
        
        # For each predicted point, find distance to nearest GT point
        dist_pred_to_gt = torch.min(dist, dim=2)[0]  # (B, N)
        
        # For each GT point, find distance to nearest predicted point
        dist_gt_to_pred = torch.min(dist, dim=1)[0]  # (B, N)
        
        # Chamfer distance is the sum of both directions
        loss = torch.mean(dist_pred_to_gt) + torch.mean(dist_gt_to_pred)
        return loss


class PointMAE(nn.Module):
    """
    Point-MAE model (ECCV 2022) for self-supervised point cloud representation learning.
    """
    def __init__(self, embed_dim=128, depth_enc=4, depth_dec=2, mask_ratio=0.6, k=16):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.k = k  # patch local neighbor count
        self.embed_dim = embed_dim
        
        # Patch embedding and positioning
        self.patch_embed = PointNetPatchEmbed(in_channels=3, embed_dim=embed_dim)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )
        
        # Encoder Transformer Blocks (ViT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth_enc)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Decoder Transformer Blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth_dec)
        
        # Reconstruction head: regresses (k x 3) coordinates for each patch
        self.reconstruct_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, k * 3)
        )
        
        self.chamfer_loss = ChamferLoss()
        
        # Initialize mask token weights
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, xyz, target_num_patches=64):
        """
        Args:
            xyz: (B, N, 3) point cloud
            target_num_patches: Number of patches M
        Returns:
            reconstructed_patches: (B, M_masked, k, 3) points reconstructed for masked regions
            gt_masked_patches: (B, M_masked, k, 3) original points for masked regions
            mask_indices: (B, M_masked) index of masked patches
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        # 1. Group point cloud into local patches (using simple FPS stride approximation + KNN)
        stride = max(1, N // target_num_patches)
        center_indices = torch.arange(0, target_num_patches * stride, stride, device=device)[:target_num_patches]
        centers = xyz[:, center_indices, :]  # (B, M, 3)
        
        # Pairwise distance to extract K nearest neighbors for each center
        inner = -2 * torch.bmm(centers, xyz.transpose(2, 1))
        xx_centers = torch.sum(centers**2, dim=2, keepdim=True)
        xx_orig = torch.sum(xyz**2, dim=2, keepdim=True)
        dist = xx_centers + inner + xx_orig.transpose(2, 1)
        _, knn_idx = torch.topk(dist, self.k, dim=-1, largest=False)  # (B, M, k)
        
        # Gather local patches: (B, M, k, 3)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, target_num_patches, self.k)
        patches = xyz[batch_indices, knn_idx, :]
        # Normalize patches coordinates locally (relative to center)
        patches_normalized = patches - centers.unsqueeze(2)
        
        # 2. Embed patches to tokens
        tokens = self.patch_embed(patches_normalized)  # (B, M, C)
        pos_encodings = self.pos_encoder(centers)  # (B, M, C)
        
        # Add positional encodings to tokens
        tokens = tokens + pos_encodings
        
        # 3. Random Masking
        # Mask a ratio of patches
        num_masked = int(target_num_patches * self.mask_ratio)
        num_visible = target_num_patches - num_masked
        
        visible_tokens_list = []
        visible_pos_list = []
        masked_pos_list = []
        masked_patch_idx_list = []
        gt_masked_patches_list = []
        
        for b in range(B):
            # Shuffle indices
            rand_idx = torch.randperm(target_num_patches, device=device)
            visible_idx = rand_idx[:num_visible]
            masked_idx = rand_idx[num_visible:]
            
            # Slice visible tokens & position encodings
            visible_tokens_list.append(tokens[b, visible_idx])
            visible_pos_list.append(pos_encodings[b, visible_idx])
            
            # Slice masked position encodings & ground truth patches
            masked_pos_list.append(pos_encodings[b, masked_idx])
            masked_patch_idx_list.append(masked_idx)
            gt_masked_patches_list.append(patches_normalized[b, masked_idx])
            
        visible_tokens = torch.stack(visible_tokens_list, dim=0)  # (B, M_visible, C)
        visible_pos = torch.stack(visible_pos_list, dim=0)  # (B, M_visible, C)
        masked_pos = torch.stack(masked_pos_list, dim=0)  # (B, M_masked, C)
        masked_patch_indices = torch.stack(masked_patch_idx_list, dim=0)  # (B, M_masked)
        gt_masked_patches = torch.stack(gt_masked_patches_list, dim=0)  # (B, M_masked, k, 3)
        
        # 4. Encoder: processes only visible tokens
        encoded_tokens = self.encoder(visible_tokens)  # (B, M_visible, C)
        
        # 5. Decoder: processes both visible features and learnable mask tokens
        # Prepare mask tokens: (B, M_masked, C)
        mask_tokens = self.mask_token.expand(B, num_masked, -1) + masked_pos
        
        # Decoded output
        # Multi-layer Transformer decoder: mask_tokens query encoded_tokens
        decoded_tokens = self.decoder(mask_tokens, encoded_tokens)  # (B, M_masked, C)
        
        # 6. Reconstruction head
        # Regress patch coordinates: (B, M_masked, k * 3)
        pred_flat = self.reconstruct_head(decoded_tokens)
        reconstructed_patches = pred_flat.reshape(B, num_masked, self.k, 3)
        
        return reconstructed_patches, gt_masked_patches, masked_patch_indices

    def compute_loss(self, pred_patches, gt_patches):
        """
        Calculates reconstruction loss using Chamfer Distance.
        Args:
            pred_patches: (B, M_masked, k, 3)
            gt_patches: (B, M_masked, k, 3)
        """
        B, M_masked, k, _ = pred_patches.shape
        # Flatten patches into single point sets per batch element for Chamfer distance computation
        pred_flat = pred_patches.reshape(B, M_masked * k, 3)
        gt_flat = gt_patches.reshape(B, M_masked * k, 3)
        
        loss = self.chamfer_loss(pred_flat, gt_flat)
        return loss
