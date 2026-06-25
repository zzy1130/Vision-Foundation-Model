import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ViT(nn.Module):
    """
    Standard Vision Transformer (ViT) implementation for DINO.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=6, num_classes=0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Init weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        # x shape: [B, C, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Interp pos embed if resolution changes (simplistic for standard sizes)
        if x.shape[1] != self.pos_embed.shape[1]:
            # Interpolate position embeddings
            pos_embed = self._interpolate_pos_embed(x, self.pos_embed)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
            
        x = self.blocks(x)
        x = self.norm(x)
        
        # DINO uses the CLS token representation
        cls_rep = x[:, 0]
        return cls_rep

    def _interpolate_pos_embed(self, x, pos_embed):
        # Simplistic interpolation helper
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        
        dim = pos_embed.shape[-1]
        old_h_w = int(patch_pos.shape[1] ** 0.5)
        new_h_w = int((x.shape[1] - 1) ** 0.5)
        
        patch_pos = patch_pos.reshape(1, old_h_w, old_h_w, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_h_w, new_h_w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat((cls_pos, patch_pos), dim=1)

class DINOHead(nn.Module):
    """
    DINO Projection Head.
    Consists of a 3-layer MLP followed by an L2 normalization and a weight-normalized linear layer.
    """
    def __init__(self, in_dim, out_dim, bottleneck_dim=256, use_bn=False):
        super().__init__()
        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048) if use_bn else nn.Identity(),
            nn.GELU(),
            nn.Linear(2048, bottleneck_dim),
        )
        self.apply(self._init_weights)
        
        # Last layer: weight normalized linear layer projecting to high dim output
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # L2 normalize bottleneck features
        x = F.normalize(x, dim=-1, p=2)
        # Project to output classification dim
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    """
    DINO Loss implementing Cross-Entropy between Student and Teacher outputs.
    Includes centering and sharpening for teacher outputs to prevent collapse.
    """
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        # Register buffer for centering vector
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Computes the cross-entropy loss between student and teacher projections.
        Args:
            student_output (Tensor): Shape [B * num_crops, out_dim]
            teacher_output (Tensor): Shape [B * num_global_crops, out_dim]
        """
        # 1. Sharpening student output
        student_out = student_output / self.student_temp
        
        # 2. Centering & sharpening teacher output (gradient detached)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()
        
        # Multi-crop chunking: DINOv1 uses 2 global crops + M local crops
        # Let's say batch size is B.
        # student_output has shape [B * (num_global + num_local), out_dim]
        # teacher_output has shape [B * num_global, out_dim]
        
        # In this educational version, we assume standard DINO settings:
        # 2 global crops (indices 0 and 1) and M local crops (indices 2 to N)
        # We compute cross-entropy for all student-teacher pairs except when the crop is the same
        
        n_student_crops = student_output.shape[0] // teacher_output.shape[0] * 2
        # For simplicity, we chunk student and teacher outputs by crops
        B = teacher_output.shape[0] // 2  # Assuming 2 global crops
        
        student_chunks = student_output.chunk(student_output.shape[0] // B)
        teacher_chunks = teacher_output.chunk(2)
        
        total_loss = 0
        n_loss_terms = 0
        
        # Pairwise comparison: Student predicts teacher representations of other global crops
        for iq, q in enumerate(teacher_chunks):
            # q shape [B, out_dim]
            q_centered = F.softmax((q - self.center) / self.teacher_temp, dim=-1).detach()
            
            for ip, p in enumerate(student_chunks):
                if ip == iq:
                    # Skip matching a crop with itself
                    continue
                # p shape [B, out_dim]
                p_log_softmax = F.log_softmax(p / self.student_temp, dim=-1)
                
                loss = torch.sum(-q_centered * p_log_softmax, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        
        # Update centering vector
        self.update_center(teacher_output)
        
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center vector with the mean of the current teacher outputs (via EMA).
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        # EMA center update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINOv1(nn.Module):
    """
    DINO v1 wrapper matching the student-teacher self-distillation paradigm.
    """
    def __init__(self, embed_dim=384, out_dim=65536, teacher_momentum=0.996):
        super().__init__()
        # 1. Student Network
        self.student_backbone = ViT(embed_dim=embed_dim)
        self.student_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
        
        # 2. Teacher Network (initialized as a copy of the student)
        self.teacher_backbone = ViT(embed_dim=embed_dim)
        self.teacher_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Teacher gradients are frozen (updated via EMA)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        self.teacher_momentum = teacher_momentum

    def forward_student(self, crops):
        """
        Runs the student backbone + head on all crops (global + local).
        Args:
            crops: List of Tensors representing different crops of images.
        """
        projs = []
        for crop in crops:
            embeds = self.student_backbone(crop)
            proj = self.student_head(embeds)
            projs.append(proj)
        return torch.cat(projs, dim=0)

    def forward_teacher(self, global_crops):
        """
        Runs the teacher backbone + head on global crops only.
        """
        projs = []
        for crop in global_crops:
            embeds = self.teacher_backbone(crop)
            proj = self.teacher_head(embeds)
            projs.append(proj)
        return torch.cat(projs, dim=0)

    def forward(self, crops):
        """
        Args:
            crops (list of Tensors): Full crop list. crops[0:2] are global crops (e.g. 224x224),
                                    crops[2:] are local crops (e.g. 96x96).
        """
        student_projs = self.forward_student(crops)
        
        # Teacher only gets global crops
        global_crops = crops[:2]
        with torch.no_grad():
            teacher_projs = self.forward_teacher(global_crops)
            
        return student_projs, teacher_projs

    @torch.no_grad()
    def update_teacher(self):
        """
        Update teacher weights as Exponential Moving Average (EMA) of student weights.
        """
        # Update backbone
        for param_q, param_k in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)
        # Update projection head
        for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data = param_k.data * self.teacher_momentum + param_q.data * (1.0 - self.teacher_momentum)

if __name__ == "__main__":
    print("Testing DINOv1 architecture...")
    # Instantiate DINOv1 model and DINO loss
    model = DINOv1(embed_dim=384, out_dim=2048)  # Out dim set to 2048 for quick testing
    loss_fn = DINOLoss(out_dim=2048)
    
    # 2 global crops (224x224), 4 local crops (96x96)
    B = 2
    global_crop1 = torch.randn(B, 3, 224, 224)
    global_crop2 = torch.randn(B, 3, 224, 224)
    local_crop1 = torch.randn(B, 3, 96, 96)
    local_crop2 = torch.randn(B, 3, 96, 96)
    local_crop3 = torch.randn(B, 3, 96, 96)
    local_crop4 = torch.randn(B, 3, 96, 96)
    
    crops = [global_crop1, global_crop2, local_crop1, local_crop2, local_crop3, local_crop4]
    
    student_projs, teacher_projs = model(crops)
    loss = loss_fn(student_projs, teacher_projs)
    
    print(f"Student projections shape: {student_projs.shape}") # (B * 6, out_dim) = (12, 2048)
    print(f"Teacher projections shape: {teacher_projs.shape}") # (B * 2, out_dim) = (4, 2048)
    print(f"DINOv1 Cross-Entropy Loss: {loss.item():.4f}")
    
    # Update teacher weights
    model.update_teacher()
    print("Successfully updated DINOv1 teacher parameters via EMA.")
