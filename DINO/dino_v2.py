import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) MLP block.
    Often used in modern LLMs and vision foundation models like DINOv2.
    """
    def __init__(self, in_features, hidden_features, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        # We split the hidden features into gate and value paths
        # Hidden dimension is typically 8/3 of in_features for SwiGLU (to match parameter counts)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # Swish(xW1) * xW2
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    """
    Transformer Block with LayerScale and optional SwiGLU MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        
        # SwiGLU MLP: hidden dim is scaled appropriately
        hidden_dim = int(dim * mlp_ratio * 2 / 3)
        self.mlp = SwiGLU(in_features=dim, hidden_features=hidden_dim)
        
        # LayerScale parameters
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # Attention with LayerScale
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ls1 * attn_out
        
        # MLP with LayerScale
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x

class DINOv2ViT(nn.Module):
    """
    ViT encoder for DINOv2 with LayerScale, SwiGLU, and Masked Image Modeling (iBOT) support.
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Learnable Mask Token for iBOT masked image modeling
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _interpolate_pos_embed(self, x, pos_embed):
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        
        dim = pos_embed.shape[-1]
        old_h_w = int(patch_pos.shape[1] ** 0.5)
        new_h_w = int((x.shape[1] - 1) ** 0.5)
        
        patch_pos = patch_pos.reshape(1, old_h_w, old_h_w, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_h_w, new_h_w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input images [B, C, H, W]
            mask (Tensor, optional): Boolean tensor of shape [B, num_patches].
                                     True represents masked patches.
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Apply iBOT masked image modeling if mask is provided
        if mask is not None:
            # Broadcast mask token
            w_mask = mask.unsqueeze(-1).type_as(x)
            x = x * (1 - w_mask) + self.mask_token * w_mask
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self._interpolate_pos_embed(x, self.pos_embed)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Separate CLS token and Patch tokens
        cls_rep = x[:, 0]
        patch_reps = x[:, 1:]
        
        return cls_rep, patch_reps

class DINOv2Head(nn.Module):
    """
    DINOv2 Head returning both global and patch outputs.
    """
    def __init__(self, in_dim, out_dim, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        # x can be [B, in_dim] or [B, N, in_dim]
        is_sequence = len(x.shape) == 3
        if is_sequence:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
            
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        
        if is_sequence:
            x = x.reshape(B, N, -1)
        return x

class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss to encourage uniform distribution on unit sphere.
    Prevents representation collapse.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): Feature embeddings [B, D]
        """
        # 1. Normalize features to sit on the unit hypersphere
        x = F.normalize(x, dim=-1, p=2)
        
        # 2. Pairwise Euclidean distances
        dist_matrix = torch.cdist(x, x, p=2)
        
        # 3. Mask the diagonal (distance to itself is 0) by adding a huge number
        dist_matrix = dist_matrix + torch.eye(x.shape[0], device=x.device) * 1e5
        
        # 4. Find the distance to the nearest neighbor for each sample
        nearest_neighbor_dist, _ = torch.min(dist_matrix, dim=1)
        
        # 5. Entropic loss: -mean(log(nearest_neighbor_distance))
        loss = -torch.log(nearest_neighbor_dist + 1e-8).mean()
        return loss

class DINOv2Loss(nn.Module):
    """
    DINOv2 Compound Loss combining:
    1. Global DINO Loss (CLS tokens)
    2. iBOT Masked Patch Loss (Patch tokens)
    3. KoLeo Regularization
    """
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9, patch_out_dim=8192):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        # Running centers for global and patch representation
        self.register_buffer("global_center", torch.zeros(1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, patch_out_dim))
        
        # KoLeo loss module
        self.koleo_loss = KoLeoLoss()

    def forward(self, student_cls, student_patches, teacher_cls, teacher_patches, mask, koleo_features):
        """
        Args:
            student_cls (Tensor): Student CLS predictions [2*B + local_crops_B, out_dim]
            student_patches (Tensor): Student patch predictions [B * 2, num_patches, patch_out_dim] (only for global crops)
            teacher_cls (Tensor): Teacher CLS predictions [2*B, out_dim] (only for global crops)
            teacher_patches (Tensor): Teacher patch predictions [B * 2, num_patches, patch_out_dim]
            mask (Tensor): Boolean mask indicating masked patches in student global crops [B * 2, num_patches]
            koleo_features (Tensor): CLS representations to be regularized [B, out_dim]
        """
        # 1. Global DINO Loss on CLS tokens (symmetric cross-entropy across global crops + local crops)
        # Assuming 2 global crops for teacher
        B = teacher_cls.shape[0] // 2
        student_cls_chunks = student_cls.chunk(student_cls.shape[0] // B)
        teacher_cls_chunks = teacher_cls.chunk(2)
        
        global_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_cls_chunks):
            q_centered = F.softmax((q - self.global_center) / self.teacher_temp, dim=-1).detach()
            for ip, p in enumerate(student_cls_chunks):
                if ip == iq:
                    continue
                p_log_softmax = F.log_softmax(p / self.student_temp, dim=-1)
                loss = torch.sum(-q_centered * p_log_softmax, dim=-1).mean()
                global_loss += loss
                n_loss_terms += 1
        global_loss /= n_loss_terms
        
        # Update global centering buffer
        self.update_global_center(teacher_cls)
        
        # 2. iBOT Patch-level DINO Loss (Masked Image Modeling)
        # We only compute loss on the patches that are masked in the student network
        # student_patches shape: [2*B, N, patch_out_dim]
        # teacher_patches shape: [2*B, N, patch_out_dim]
        # mask shape: [2*B, N] (True where masked)
        
        patch_loss = torch.tensor(0.0, device=student_cls.device)
        if mask.any():
            # Apply mask to select student masked patches and corresponding teacher patches
            # We apply centering and sharpening for patch level as well
            
            # Student predictions at masked locations
            s_masked = student_patches[mask] / self.student_temp
            s_log_softmax = F.log_softmax(s_masked, dim=-1)
            
            # Teacher predictions at masked locations
            t_masked = (teacher_patches[mask] - self.patch_center) / self.teacher_temp
            t_prob = F.softmax(t_masked, dim=-1).detach()
            
            # Cross-entropy
            patch_loss = torch.sum(-t_prob * s_log_softmax, dim=-1).mean()
            
            # Update patch centering buffer
            self.update_patch_center(teacher_patches[mask])
            
        # 3. KoLeo Regularization (Typically applied to student CLS embeddings)
        koleo_loss_val = self.koleo_loss(koleo_features)
        
        # Weighted sum of losses
        total_loss = global_loss + 1.0 * patch_loss + 0.1 * koleo_loss_val
        return total_loss, global_loss, patch_loss, koleo_loss_val

    @torch.no_grad()
    def update_global_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.global_center = self.global_center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_patch_center(self, teacher_patches_masked):
        batch_center = torch.mean(teacher_patches_masked, dim=0, keepdim=True)
        self.patch_center = self.patch_center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOv2(nn.Module):
    """
    DINOv2 model architecture featuring:
    - ViT with LayerScale and SwiGLU
    - Multi-crop + iBOT Masked Patch prediction
    - EMA Teacher updates
    """
    def __init__(self, embed_dim=384, out_dim=65536, patch_out_dim=8192, teacher_momentum=0.996):
        super().__init__()
        # 1. Student Networks
        self.student_backbone = DINOv2ViT(embed_dim=embed_dim)
        self.student_cls_head = DINOv2Head(in_dim=embed_dim, out_dim=out_dim)
        self.student_patch_head = DINOv2Head(in_dim=embed_dim, out_dim=patch_out_dim)
        
        # 2. Teacher Networks (initialized as copy)
        self.teacher_backbone = DINOv2ViT(embed_dim=embed_dim)
        self.teacher_cls_head = DINOv2Head(in_dim=embed_dim, out_dim=out_dim)
        self.teacher_patch_head = DINOv2Head(in_dim=embed_dim, out_dim=patch_out_dim)
        
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_cls_head.load_state_dict(self.student_cls_head.state_dict())
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict())
        
        # Freeze teacher weights
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_cls_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False
            
        self.teacher_momentum = teacher_momentum

    def forward(self, crops, masks=None):
        """
        Args:
            crops (list of Tensors): Full crop list. crops[0:2] are global crops, crops[2:] are local crops.
            masks (list of Tensors): Mask boolean tensors for the global crops. masks[0] and masks[1]
                                     specify patch mask coordinates [B, num_patches].
        """
        # Run student on global crops (potentially with masks) and local crops (without masks)
        B = crops[0].shape[0]
        
        # Process global crops
        student_cls_list = []
        student_patches_list = []
        koleo_features = None
        
        for i in range(2):
            mask = masks[i] if masks is not None else None
            cls_rep, patch_rep = self.student_backbone(crops[i], mask=mask)
            
            if i == 0:
                koleo_features = cls_rep
                
            student_cls_list.append(self.student_cls_head(cls_rep))
            student_patches_list.append(self.student_patch_head(patch_rep))
            
        # Process local crops (no masking applied)
        for local_crop in crops[2:]:
            cls_rep, _ = self.student_backbone(local_crop, mask=None)
            student_cls_list.append(self.student_cls_head(cls_rep))
            
        # Concatenate student outputs
        # student_cls contains global + local crop representations
        student_cls = torch.cat(student_cls_list, dim=0)
        # student_patches contains patch representations for the two global crops
        student_patches = torch.cat(student_patches_list, dim=0) # Shape [2 * B, N, patch_out_dim]
        
        # Run teacher on unmasked global crops
        teacher_cls_list = []
        teacher_patches_list = []
        with torch.no_grad():
            for i in range(2):
                cls_rep, patch_rep = self.teacher_backbone(crops[i], mask=None)
                teacher_cls_list.append(self.teacher_cls_head(cls_rep))
                teacher_patches_list.append(self.teacher_patch_head(patch_rep))
                
            teacher_cls = torch.cat(teacher_cls_list, dim=0)
            teacher_patches = torch.cat(teacher_patches_list, dim=0)
            
        return student_cls, student_patches, teacher_cls, teacher_patches, koleo_features

    @torch.no_grad()
    def update_teacher(self):
        # Update backbone
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)
        # Update CLS head
        for param_q, param_k in zip(self.student_cls_head.parameters(), self.teacher_cls_head.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)
        # Update patch head
        for param_q, param_k in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)

if __name__ == "__main__":
    print("Testing DINOv2 architecture...")
    model = DINOv2(embed_dim=384, out_dim=2048, patch_out_dim=512)
    loss_fn = DINOv2Loss(out_dim=2048, patch_out_dim=512)
    
    # Simulate multi-crop inputs (B=2)
    B = 2
    # 2 global crops (224x224), 2 local crops (96x96)
    g_crop1 = torch.randn(B, 3, 224, 224)
    g_crop2 = torch.randn(B, 3, 224, 224)
    l_crop1 = torch.randn(B, 3, 96, 96)
    l_crop2 = torch.randn(B, 3, 96, 96)
    
    crops = [g_crop1, g_crop2, l_crop1, l_crop2]
    
    # 14x14 patch size on 224x224 images results in 16x16 = 256 patches
    num_patches = (224 // 14) ** 2 # 256
    
    # Generate random boolean masks for student global crops
    mask1 = torch.rand(B, num_patches) < 0.5
    mask2 = torch.rand(B, num_patches) < 0.5
    masks = [mask1, mask2]
    concat_masks = torch.cat(masks, dim=0) # [2*B, num_patches]
    
    # Forward Pass
    student_cls, student_patches, teacher_cls, teacher_patches, koleo_feats = model(crops, masks)
    
    # Compute Loss
    total_loss, g_loss, p_loss, k_loss = loss_fn(
        student_cls=student_cls,
        student_patches=student_patches,
        teacher_cls=teacher_cls,
        teacher_patches=teacher_patches,
        mask=concat_masks,
        koleo_features=koleo_feats
    )
    
    print(f"Student CLS output shape: {student_cls.shape}")       # [4*B, out_dim] -> [8, 2048]
    print(f"Student patch output shape: {student_patches.shape}")   # [2*B, num_patches, patch_out_dim] -> [4, 256, 512]
    print(f"Total DINOv2 Loss: {total_loss.item():.4f}")
    print(f"  └─ Global DINO Loss: {g_loss.item():.4f}")
    print(f"  └─ iBOT Patch Loss: {p_loss.item():.4f}")
    print(f"  └─ KoLeo Regularizer: {k_loss.item():.4f}")
    
    model.update_teacher()
    print("Successfully updated DINOv2 teacher via EMA.")
