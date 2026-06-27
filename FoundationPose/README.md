# FoundationPose & SOTA Follow-up Models (6D Object Pose Estimation)

本目录包含了基于 PyTorch 从零实现的 <strong>FoundationPose</strong> 及其重要后续发展模型的核心架构与前向传播逻辑。这些模型代表了 6D 物体姿态估计（6D Object Pose Estimation & Tracking）领域从“特定物体微调”走向“零样本泛化（Zero-Shot Generalization）”与“免训练（Training-Free）”的技术演进路线。

---

## 目录
1. [模型家族概览](#模型家族概览)
2. [核心模型实现与解析](#核心模型实现与解析)
    - [FoundationPose (CVPR 2024 Highlight)](#1-foundationpose-cvpr-2024-highlight)
    - [Any6D (CVPR 2025)](#2-any6d-cvpr-2025)
    - [FreeZeV2 (BOP Challenge 2024 Winner)](#3-freezev2-bop-challenge-2024-winner)
    - [OPFormer (CVPR 2026)](#4-opformer-cvpr-2026)
3. [快速开始与 Demo 运行](#快速开始与-demo-运行)
4. [公式与图示说明](#公式与图示说明)

---

## 模型家族概览

<p align="center">
  <img src="images/foundationpose_intro.jpg" width="85%" alt="FoundationPose Intro Teaser" />
</p>

在机器人抓取、增强现实（AR）等物理交互任务中，获取物体相对于相机的 3D 旋转 R 和 3D 平移 t（合称 6D 姿态）是感知系统的核心。
*   <strong>FoundationPose</strong> 奠定了统一的姿态估计与跟踪框架。它通过在海量合成数据上训练“渲染-对比（Render-and-Compare）”网络，实现了对未知物体的即插即用（零样本泛化）。
*   <strong>Any6D</strong> 解决了“无 CAD 模型”情况下的尺度不确定性。它通过单张 RGB-D 锚图重建 3D 形状并对其进行联合对齐，完成了绝对尺度（Metric Scale）的预测。
*   <strong>FreeZeV2</strong> 探索了彻底的“免训练（Training-Free）”方案。直接冻结大规模视觉基座模型（如 DINOv2），在 3D-3D 点匹配上通过 RANSAC 和 Kabsch 算法解出最优姿态，获得了 BOP 竞赛冠军。
*   <strong>OPFormer</strong> 则将目标检测与姿态估计进行了端到端（End-to-End）的 Transformer 统一，引入了 NOCS（归一化物体坐标空间）几何先验，在效率和鲁棒性之间取得了极佳的平衡。

---

## 核心模型实现与解析

### 1. FoundationPose (CVPR 2024 Highlight)
*   <strong>代码实现</strong>: [foundation_pose.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/FoundationPose/foundation_pose.py)
*   <strong>核心机制</strong>:
    1.  <strong>特征提取器（Shared Feature Extractor）</strong>: 提取查询区域 RGB-D 裁剪图与渲染姿态模板的共享特征。
    2.  <strong>姿态细化网络（Pose Refinement Network）</strong>: 回归相对姿态更新量 (ΔR, Δt)。为确保旋转回归的数值稳定，采用连续的 6D 旋转表示法，然后通过施密特正交化还原为 3x3 旋转矩阵。正交化公式如下：
        <p align="center"><img src="images/eq1_rot_6d.png" width="65%" alt="6D Rotation Orthogonalization" /></p>
    3.  <strong>姿态选择网络（Pose Selection/Scoring Network）</strong>: 接收多个细化后的姿态假设，使用 Transformer 自注意力机制（Self-Attention）在各个姿态候选（Tokens）之间进行全局上下文对比，最终回归出置信度得分，挑选最高得分者。
    4.  <strong>多任务损失</strong>:
        <p align="center"><img src="images/eq2_refine_loss.png" width="35%" alt="Refinement Loss" /></p>

#### 代码调用示例
```python
import torch
from foundation_pose import FoundationPose

# 初始化模型
model = FoundationPose(feature_dim=128)

# 模拟输入: 1个 Batch, 5个候选姿态渲染图, 分辨率 112x112, 4通道(RGB-D)
query_img = torch.randn(1, 4, 112, 112)
candidate_renders = torch.randn(1, 5, 4, 112, 112)

# 前向传播
best_idx, (refined_rot, refined_trans), scores = model(query_img, candidate_renders, refine_iters=3)
print("Best pose candidate index:", best_idx.item())
print("Refined Rotations:", refined_rot.shape)  # torch.Size([1, 5, 3, 3])
```

---

### 2. Any6D (CVPR 2025)
*   <strong>代码实现</strong>: [any6d.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/FoundationPose/any6d.py)
*   <strong>核心机制</strong>:
    1.  <strong>单视角 3D 重建（InstantMesh Proxy）</strong>: 输入单张 RGB 锚图（Anchor Image），网络重建出物体的 normalized 3D 网格模型。
    2.  <strong>联合尺度与粗对齐（Coarse Scale Aligner）</strong>: 利用锚图深度和掩码（Mask）与生成的 normalized 3D 形状进行点云尺度对齐，预测出物体的物理尺度因子 s 和初始相机坐标系平移。尺度损失采用 L1 形式：
        <p align="center"><img src="images/eq3_scale_loss.png" width="32%" alt="Metric Scale Loss" /></p>
    3.  <strong>级联姿态精细化</strong>: 将缩放后的 3D 模型渲染出的模板，与查询图像（Query Image）送入 FoundationPose 模型进行 6D 姿态优化。

<p align="center">
  <img src="images/any6d_teaser.png" width="85%" alt="Any6D Pipeline Overview" />
</p>
<p align="center">
  <img src="images/any6d_robot.gif" width="55%" alt="Any6D Robotics Manipulation Demo" />
</p>

#### 代码调用示例
```python
from any6d import Any6D
import torch

model = Any6D(feature_dim=128)

# 模拟输入: 锚图 RGB-D + 目标掩码
anchor_rgb = torch.randn(1, 3, 112, 112)
anchor_depth = torch.randn(1, 1, 112, 112)
anchor_mask = (torch.randn(1, 1, 112, 112) > 0).float()

query_img = torch.randn(1, 4, 112, 112)
candidate_renders = torch.randn(1, 5, 4, 112, 112)

# 前向传播预测 6D 姿态与绝对尺度 s
best_idx, (rot, trans), scores, scale = model(
    anchor_rgb, anchor_depth, anchor_mask, 
    query_img, candidate_renders, refine_iters=2
)
print("Reconstructed scale:", scale.item())
```

---

### 3. FreeZeV2 (BOP Challenge 2024 Winner)
*   <strong>代码实现</strong>: [freeze_v2.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/FoundationPose/freeze_v2.py)
*   <strong>核心机制</strong>:
    1.  <strong>冻结的视觉特征提取（DINOv2 Proxy）</strong>: 直接使用预训练并冻结的 DINOv2 作为视觉编码器，提取极其鲁棒的局部特征描述子（Descriptors）。
    2.  <strong>3D 空间反投影</strong>: 利用查询图的像素坐标、相机内参矩阵 K<sub>inv</sub> 和深度图将其投影到 3D 相机空间中。
    3.  <strong>双向描述子匹配</strong>: 计算 3D 模板描述子与查询点描述子的余弦相似度矩阵，通过最近邻匹配建立 3D-3D 的点云对应关系。
    4.  <strong>RANSAC & Kabsch 闭式求解器</strong>: 对匹配点对在 RANSAC 框架下进行高频采样，通过 Kabsch SVD 算子迭代求解出刚体变换（旋转 R 与平移 t），并抵抗野值干扰。Kabsch 旋转计算公式：
        <p align="center"><img src="images/eq4_kabsch.png" width="22%" alt="Kabsch SVD Solver" /></p>
    5.  <strong>特征感知的置信度得分</strong>: 根据 RANSAC 内点率（Inlier Ratio）与匹配点之间的特征余弦相似度计算最终的置信度。

#### 代码调用示例
```python
from freeze_v2 import FreeZeV2
import torch

# 初始化免训练姿态估计器
model = FreeZeV2(feature_dim=384, ransac_iters=30)

query_rgb = torch.randn(1, 3, 112, 112)
query_depth = torch.randn(1, 1, 112, 112)
query_intrinsics = torch.tensor([[[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]]])

# 20个稀疏 3D 模型点及其预计算特征
template_descriptors = torch.randn(1, 20, 384)
template_pts_3d = torch.randn(1, 20, 3)

R, t, conf = model(query_rgb, query_depth, query_intrinsics, template_descriptors, template_pts_3d)
print("Estimated Rotation Matrix:\n", R[0])
print("Confidence Score:", conf.item())
```

---

### 4. OPFormer (CVPR 2026)
*   <strong>代码实现</strong>: [opformer.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/FoundationPose/opformer.py)
*   <strong>核心机制</strong>:
    1.  <strong>联合多模板视图编码器（Multi-Template Encoder）</strong>: 将 5 个不同视角下的模板图整合，使用自注意力机制提取几何一致（Geometry-Consistent）的跨视图特征。
    2.  <strong>NOCS（归一化物体坐标空间）估计器</strong>: 预测每个查询像素所对应的物体局部 3D 坐标图，强行施加 3D 几何结构先验。
    3.  <strong>Transformer 解码器（Correspondences Decoder）</strong>: 查询图像特征作为 Query，多模板交互特征作为 Key/Value。通过交叉注意力解码出高精度的 2D 像素与 3D 物体空间点云的稠密对应关系。
    4.  <strong>多任务学习目标</strong>:
        <p align="center"><img src="images/eq5_opformer_loss.png" width="55%" alt="OPFormer Loss" /></p>

#### 代码调用示例
```python
from opformer import OPFormer
import torch

model = OPFormer(feature_dim=128, num_templates=5)

query_rgb = torch.randn(1, 3, 112, 112)
query_depth = torch.randn(1, 1, 112, 112)
query_intrinsics = torch.tensor([[[100.0, 0.0, 56.0], [0.0, 100.0, 56.0], [0.0, 0.0, 1.0]]])
templates = torch.randn(1, 5, 3, 112, 112)

# 前向传播预测 6D 姿态与 NOCS 坐标图
R, t, nocs_map, pred_pts_3d = model(query_rgb, query_depth, query_intrinsics, templates)
print("NOCS Map Shape:", nocs_map.shape)  # torch.Size([1, 3, 56, 56])
print("Pose R Shape:", R.shape)          # torch.Size([1, 3, 3])
```

---

## 快速开始与 Demo 运行

项目提供了一个完整的单元测试与前向推理验证脚本 [run_demo.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/FoundationPose/run_demo.py)。它将使用代表性的随机张量执行以上所有姿态估计流水线，并在终端输出输入与预测输出张量的详细尺寸信息。

运行 demo 命令：
```bash
python FoundationPose/run_demo.py
```

---

## 公式与图示说明

*   <strong>静态公式</strong>: 所有的 Block Math 公式已经全部在本地使用 `matplotlib` 渲染为白底的高 DPI 图片并存放在 `images/` 中，以保证在暗色模式和不同的移动端浏览器上能瞬间且高保真地渲染显示。
*   <strong>论文图示</strong>: 已通过自动化脚本从官方开源库中获取：
    -   `foundationpose_intro.jpg` 说明了基础模型通过渲染和选择来估计姿态的总体框架。
    -   `any6d_teaser.png` 和 `any6d_robot.gif` 演示了 Any6D 联合估计未知物体尺度并实施机械臂抓取任务的流程。
