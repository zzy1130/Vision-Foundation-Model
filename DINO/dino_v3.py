import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from dino_v2 import DINOv2ViT, DINOv2Head, DINOv2Loss, KoLeoLoss

class GramAnchoringLoss(nn.Module):
    """
    Gram Anchoring Loss introduced in DINOv3 (Meta AI, 2025).
    Constrains the student's patch similarity structure (Gram matrix) to match 
    a stable teacher's Gram matrix, preventing dense feature degradation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_patches, teacher_patches):
        """
        Args:
            student_patches (Tensor): Student patch embeddings [B, P, D]
            teacher_patches (Tensor): Gram teacher patch embeddings [B, P, D]
        """
        # 1. L2 Normalize patch features along the channel dimension
        # This makes the dot products represent cosine similarity
        s_norm = F.normalize(student_patches, p=2, dim=-1)
        t_norm = F.normalize(teacher_patches, p=2, dim=-1)
        
        # 2. Compute pairwise patch similarity matrix (Gram matrix) for each batch element
        # Shape: [B, P, P]
        G_s = torch.bmm(s_norm, s_norm.transpose(1, 2))
        G_t = torch.bmm(t_norm, t_norm.transpose(1, 2))
        
        # 3. Compute Mean Squared Error (Frobenius norm squared, scaled by size)
        loss = F.mse_loss(G_s, G_t)
        return loss

class DINOv3Loss(nn.Module):
    """
    DINOv3 Compound Loss incorporating:
    1. Global DINO Loss
    2. iBOT Masked Patch Loss
    3. KoLeo Regularization
    4. Gram Anchoring Loss (DINOv3 specific)
    """
    def __init__(self, out_dim, patch_out_dim=8192, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9, gram_weight=2.0):
        super().__init__()
        self.dino_v2_loss = DINOv2Loss(
            out_dim=out_dim, 
            patch_out_dim=patch_out_dim, 
            student_temp=student_temp, 
            teacher_temp=teacher_temp, 
            center_momentum=center_momentum
        )
        self.gram_anchoring = GramAnchoringLoss()
        self.gram_weight = gram_weight

    def forward(self, student_cls, student_patches_proj, teacher_cls, teacher_patches_proj, mask, koleo_features,
                student_raw_patches, teacher_raw_patches):
        """
        Args:
            student_cls (Tensor): Student CLS predictions
            student_patches_proj (Tensor): Student projected patch features [2*B, N, patch_out_dim]
            teacher_cls (Tensor): Teacher CLS predictions
            teacher_patches_proj (Tensor): Teacher projected patch features [2*B, N, patch_out_dim]
            mask (Tensor): Mask boolean tensor [2*B, N]
            koleo_features (Tensor): Student CLS for KoLeo
            student_raw_patches (Tensor): Student raw patch representations [2*B, N, D] (before head, unmasked)
            teacher_raw_patches (Tensor): Teacher raw patch representations [2*B, N, D] (before head)
        """
        # 1. Compute DINOv2 loss components (Global + Patch + KoLeo)
        total_dino_v2, g_loss, p_loss, k_loss = self.dino_v2_loss(
            student_cls=student_cls,
            student_patches=student_patches_proj,
            teacher_cls=teacher_cls,
            teacher_patches=teacher_patches_proj,
            mask=mask,
            koleo_features=koleo_features
        )
        
        # 2. Compute Gram Anchoring loss between student and teacher raw patch maps
        # Normally applied to unmasked patches or global crops to regularize structure
        gram_loss = self.gram_anchoring(student_raw_patches, teacher_raw_patches)
        
        # Combine
        total_loss = total_dino_v2 + self.gram_weight * gram_loss
        
        return total_loss, g_loss, p_loss, k_loss, gram_loss

class DINOv3(nn.Module):
    """
    DINOv3 model architecture featuring Gram Anchoring for stable large-scale SSL.
    """
    def __init__(self, embed_dim=384, out_dim=65536, patch_out_dim=8192, teacher_momentum=0.996):
        super().__init__()
        # 1. Student Network
        self.student_backbone = DINOv2ViT(embed_dim=embed_dim)
        self.student_cls_head = DINOv2Head(in_dim=embed_dim, out_dim=out_dim)
        self.student_patch_head = DINOv2Head(in_dim=embed_dim, out_dim=patch_out_dim)
        
        # 2. Teacher Network (EMA)
        self.teacher_backbone = DINOv2ViT(embed_dim=embed_dim)
        self.teacher_cls_head = DINOv2Head(in_dim=embed_dim, out_dim=out_dim)
        self.teacher_patch_head = DINOv2Head(in_dim=embed_dim, out_dim=patch_out_dim)
        
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_cls_head.load_state_dict(self.student_cls_head.state_dict())
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict())
        
        # Freeze teacher
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_cls_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False
            
        # 3. Gram Teacher: Snapshot of student at a stable checkpoint
        # DINOv3 keeps a stable copy of student (e.g. updated every 10k steps) to anchor patch relations
        self.gram_teacher_backbone = DINOv2ViT(embed_dim=embed_dim)
        self.gram_teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        for p in self.gram_teacher_backbone.parameters():
            p.requires_grad = False
            
        self.teacher_momentum = teacher_momentum

    def forward(self, crops, masks=None):
        """
        Args:
            crops (list of Tensors): Full crop list. crops[0:2] are global crops, crops[2:] are local crops.
            masks (list of Tensors): Mask boolean tensors for global crops.
        """
        B = crops[0].shape[0]
        
        # Process student on crops
        student_cls_list = []
        student_patches_list = []
        student_raw_patches_list = []  # For Gram Anchoring
        koleo_features = None
        
        for i in range(2):
            mask = masks[i] if masks is not None else None
            # Forward student on global crop
            cls_rep, patch_rep = self.student_backbone(crops[i], mask=mask)
            
            if i == 0:
                koleo_features = cls_rep
                
            student_cls_list.append(self.student_cls_head(cls_rep))
            student_patches_list.append(self.student_patch_head(patch_rep))
            
            # For Gram anchoring, we also get student features WITHOUT mask to compare spatial layouts
            _, raw_patch_rep = self.student_backbone(crops[i], mask=None)
            student_raw_patches_list.append(raw_patch_rep)
            
        for local_crop in crops[2:]:
            cls_rep, _ = self.student_backbone(local_crop, mask=None)
            student_cls_list.append(self.student_cls_head(cls_rep))
            
        student_cls = torch.cat(student_cls_list, dim=0)
        student_patches_proj = torch.cat(student_patches_list, dim=0)
        student_raw_patches = torch.cat(student_raw_patches_list, dim=0)
        
        # Forward regular teacher on unmasked global crops
        teacher_cls_list = []
        teacher_patches_list = []
        with torch.no_grad():
            for i in range(2):
                cls_rep, patch_rep = self.teacher_backbone(crops[i], mask=None)
                teacher_cls_list.append(self.teacher_cls_head(cls_rep))
                teacher_patches_list.append(self.teacher_patch_head(patch_rep))
            teacher_cls = torch.cat(teacher_cls_list, dim=0)
            teacher_patches_proj = torch.cat(teacher_patches_list, dim=0)
            
        # Forward Gram Teacher (stable checkpoint) on unmasked global crops
        teacher_raw_patches_list = []
        with torch.no_grad():
            for i in range(2):
                _, patch_rep = self.gram_teacher_backbone(crops[i], mask=None)
                teacher_raw_patches_list.append(patch_rep)
            teacher_raw_patches = torch.cat(teacher_raw_patches_list, dim=0)
            
        return (student_cls, student_patches_proj, teacher_cls, teacher_patches_proj, koleo_features,
                student_raw_patches, teacher_raw_patches)

    @torch.no_grad()
    def update_teacher(self):
        # Update backbone
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)
        # Update heads
        for param_q, param_k in zip(self.student_cls_head.parameters(), self.teacher_cls_head.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)
        for param_q, param_k in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)

    @torch.no_grad()
    def update_gram_teacher(self):
        """
        Snapshots the student parameters to the Gram Teacher.
        Called periodically (e.g., every K steps) rather than every step.
        """
        for param_q, param_k in zip(self.student_backbone.parameters(), self.gram_teacher_backbone.parameters()):
            param_k.data.copy_(param_q.data)

if __name__ == "__main__":
    print("Testing DINOv3 architecture...")
    model = DINOv3(embed_dim=384, out_dim=2048, patch_out_dim=512)
    loss_fn = DINOv3Loss(out_dim=2048, patch_out_dim=512)
    
    B = 2
    # 2 global crops, 2 local crops
    g_crop1 = torch.randn(B, 3, 224, 224)
    g_crop2 = torch.randn(B, 3, 224, 224)
    l_crop1 = torch.randn(B, 3, 96, 96)
    l_crop2 = torch.randn(B, 3, 96, 96)
    crops = [g_crop1, g_crop2, l_crop1, l_crop2]
    
    num_patches = (224 // 14) ** 2
    mask1 = torch.rand(B, num_patches) < 0.5
    mask2 = torch.rand(B, num_patches) < 0.5
    masks = [mask1, mask2]
    concat_masks = torch.cat(masks, dim=0)
    
    # Forward Pass
    (student_cls, student_patches_proj, teacher_cls, teacher_patches_proj, koleo_feats,
     student_raw_patches, teacher_raw_patches) = model(crops, masks)
     
    # Loss
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
    
    print(f"Student raw patch shape for Gram loss: {student_raw_patches.shape}") # [4, 256, 384]
    print(f"Total DINOv3 Loss: {total_loss.item():.4f}")
    print(f"  └─ Global DINO Loss: {g_loss.item():.4f}")
    print(f"  └─ iBOT Patch Loss: {p_loss.item():.4f}")
    print(f"  └─ KoLeo Regularizer: {k_loss.item():.4f}")
    print(f"  └─ Gram Anchoring Loss: {gram_loss.item():.4f}")
    
    model.update_teacher()
    model.update_gram_teacher()
    print("Successfully updated both DINOv3 EMA teacher and Gram teacher.")
