# Stable Diffusion & Diffusion Policy (Generative Models, World Models & Robot Control)

本目录包含了基于 PyTorch 从零实现的 <strong>Stable Diffusion (Latent Diffusion Models)</strong>、机器人决策中的 <strong>Diffusion Policy</strong>、<strong>Flow Matching Policy</strong>，以及现代生成式基座的核心架构 <strong>DiT (Diffusion Transformer)</strong> 与 <strong>Video DiT (时空视频扩散 Transformer)</strong> 的完整设计与前向/推理逻辑。这些模型共同构成了现代生成式 AI 从“高维多模态媒体合成”走向“机器人动作轨迹决策”与“物理世界模型模拟（World Models）”的理论基石。

---

## 目录
1. [从扩散模型到世界模型（World Models）](#从扩散模型到世界模型world-models)
2. [核心技术演进与原理解析](#核心技术演进与原理解析)
    - [DDPM 与 DDIM：随机加噪与确定性加速采样](#1-ddpm-与-ddim随机加噪与确定性加速采样)
    - [无分类器引导（Classifier-Free Guidance, CFG）](#2-无分类器引导classifier-free-guidance-cfg)
    - [LDM 与 VAE：在隐空间中去噪的必要性](#3-ldm-与-vae在隐空间中去噪的必要性)
    - [DiT (Diffusion Transformer)：Backbone 的 ViT 变革](#4-dit-diffusion-transformerbackbone-的-vit-变革)
    - [Video DiT：3D VAE + 时空 Patch + 文本注入（Sora 模拟器核心）](#5-video-dit3d-vae--时空-patch--文本注入sora-模拟器核心)
3. [机器人决策应用：Diffusion Policy & Flow Matching](#6-机器人决策应用diffusion-policy--flow-matching)
4. [快速开始与 Demo 运行](#快速开始与-demo-运行)
5. [公式与图示说明](#公式与图示说明)

---

## 从扩散模型到世界模型（World Models）

世界模型（World Models）的核心目标是让智能体（如机器人或自动驾驶系统）在脑海中对外部物理世界的时空演化进行模拟。通过输入当前观测与智能体采取的动作，预测物理世界的未来状态变迁（即状态转移函数 $s_{t+1} = \mathcal{T}(s_t, a_t)$）。
*   <strong>视频生成即物理模拟</strong>: 现代世界模型（如 OpenAI Sora, Google Genie, Decartes 等）将物理世界的演变直接建模为条件视频生成。
*   <strong>Video DiT 的核心作用</strong>: 借助 3D VAE 将视频数据进行时空压缩，再利用时空双向注意力（Spatiotemporal Attention）学习不同帧、不同空间斑块（Spatiotemporal Patches）之间的物理规律（如重力、碰撞、流体力学），从而具备了模拟物理世界演变的能力。

---

## 核心技术演进与原理解析

### 1. DDPM 与 DDIM：随机加噪与确定性加速采样
*   <strong>DDPM (Denoising Diffusion Probabilistic Model)</strong>:
    -   <strong>前向过程（加噪）</strong>: 将高斯噪声逐步注入数据中，加噪公式为：
        <p align="center"><img src="images/eq1_ddpm_add_noise.png" width="40%" alt="DDPM Forward Process" /></p>
    -   <strong>反向过程（去噪）</strong>: 这是一个随机的马尔可夫链过程，去噪时需要沿预设的加噪步数一步步回退（通常需要 1000 步），采样速度极慢。
*   <strong>DDIM (Denoising Diffusion Implicit Model)</strong>:
    -   <strong>确定性采样</strong>: DDIM 重新设计了反向采样公式，使得反向过程不再依赖随机性，可以通过确定性的隐式路径进行加速采样：
        <p align="center"><img src="images/eq5_ddim_step.png" width="70%" alt="DDIM Sampling Step" /></p>
    -   <strong>加速采样率</strong>: 由于采样路径是确定性的，我们可以跳过大部分时间步（例如仅抽取其中的 20 步或 50 步），从而将反向去噪过程加速数十倍。

---

### 2. 无分类器引导（Classifier-Free Guidance, CFG）
在有条件扩散生成（如文本生成图像、观测生成机器人动作）中，模型需要权衡<strong>样本多样性（Diversity）</strong>与<strong>条件一致性（Fidelity）</strong>。
*   <strong>原理</strong>: 训练时，以一定概率随机将条件（如文本 Embedding $c$）置为空值 $\emptyset$（即进行无条件训练）。
*   <strong>推理外推</strong>: 在反向去噪时，将有条件预测的噪声与无条件预测的噪声做差值放大，以强化条件语义：
    <p align="center"><img src="images/eq2_cfg.png" width="55%" alt="Classifier-Free Guidance" /></p>
*   <strong>在机器人中的应用</strong>: CFG 权重 $w$ 越大，机器人动作生成的确定性越强，动作轨迹越倾向于严格对齐输入的摄像头观测；CFG 较小时则多样性更丰富。

---

### 3. LDM 与 VAE：在隐空间中去噪的必要性
直接在像素空间（Pixel Space，如 512x512 图像）中进行扩散去噪，会导致大量的计算资源浪费在编码高频视觉冗余（如墙壁纹理、背景噪点）上。
*   <strong>VAE 隐空间压缩</strong>: <strong>Latent Diffusion Models (LDM)</strong> 引入变分自编码器（VAE）。VAE Encoder 将高维像素图像压缩到低维特征隐空间（Latent Space，如 64x64，降低 8x 空间分辨率），去除大部分感知冗余。
*   <strong>在隐空间去噪</strong>: 扩散 UNet 或 DiT 仅在 64x64 的隐特征 z 上进行加噪和去噪，最后通过 VAE Decoder 将生成隐变量还原为高维图像，使训练与推理计算开销降低了数百倍。

---

### 4. DiT (Diffusion Transformer)：Backbone 的 ViT 变革
*   <strong>代码实现</strong>: [dit.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/dit.py)
*   <strong>核心机制</strong>:
    1.  <strong>Patchification（分块嵌入）</strong>: 将 2D Latent（如 32x32x4）切分为 $p \times p$ 的图像块（Patches），平坦化并映射为一维 Token 序列。
    2.  <strong>AdaLN (Adaptive Layer Normalization)</strong>: 传统的 Transformer 使用层归一化（LayerNorm），而 DiT 使用 AdaLN 将时间步和条件信息（如类别标签）注入每一层 Transformer Block 中。AdaLN 计算公式为：
        <p align="center"><img src="images/eq6_adaln.png" width="50%" alt="AdaLN Block" /></p>
        其中 $\gamma(y)$ 和 $\beta(y)$ 是由 timestep 和类别特征通过 MLP 回归得到的通道级缩放和偏移参数。
    3.  <strong>参数规模扩展（Scaling Up）</strong>: 彻底摆脱了 UNet 复杂的卷积 Skip Connection 限制，DiT 的参数规模和生成质量表现出极其完美的 Scaling Law。

#### DiT 代码调用示例
```python
import torch
from dit import DiffusionTransformer

# 初始化 2D DiT 模型 (输入大小 32x32，patch 尺寸 2，隐藏维度 128)
dit_model = DiffusionTransformer(input_size=32, patch_size=2, in_channels=4, hidden_size=128)

z_t = torch.randn(1, 4, 32, 32)
t = torch.tensor([[250.0]])
class_labels = torch.tensor([5])  # 模拟类别条件

noise_pred = dit_model(z_t, t, class_labels)
print("DiT Predicted noise shape:", noise_pred.shape)  # torch.Size([1, 4, 32, 32])
```

---

### 5. Video DiT：3D VAE + 时空 Patch + 文本注入（Sora 模拟器核心）
*   <strong>代码实现</strong>: [video_dit.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/video_dit.py)
*   <strong>核心机制</strong>:
    1.  <strong>3D 变分自编码器 (VAE3D)</strong>: 将视频帧序列（B, F, C, H, W）输入 3D 卷积层，在空间维度压缩 8x 的同时，在<strong>时间（帧）维度进行压缩（如 2x）</strong>，从而获得时空隐表征（B, F_lat, C_lat, H_lat, W_lat）。
    2.  <strong>时空分块（Spatiotemporal Patchification）</strong>: 提取 $p_t \times p_s \times p_s$（如 2 帧 $\times$ 2 像素 $\times$ 2 像素）的时空超像素块（3D Patches），映射为 1D 时空 Tokens。
    3.  <strong>时空联合自注意力（Spatiotemporal Self-Attention）</strong>: 所有的时空 Tokens 在 Transformer 中进行全局注意力交互，使得模型可以同时学习空间结构与跨帧的时间演变物理规律（World Model 的物理模拟基础）。
    4.  <strong>交叉注意力文本注入 (Text Injection)</strong>: 在每一层 Spatiotemporal DiT Block 中，插入一个专用的 Cross-Attention 层，让时空视频 Tokens 扮演 Query，去检索由 CLIP/T5 提取的文本 prompt 特征（Key/Value），将复杂的文本指令注入到视频生成的每一个像素时空轨道中。

<p align="center">
  <img src="images/pointmae_net.jpg" width="80%" style="display:none" alt="Placeholder" />
</p>

#### Video DiT 代码调用示例
```python
import torch
from video_dit import VideoDiT

# 初始化 Video DiT (输入隐视频帧数为 8, 4通道, 空间大小 16x16)
video_dit = VideoDiT(latent_shape=(8, 4, 16, 16), patch_size=(2, 2, 2))

# 模拟 3D VAE 压缩编码与解码
raw_video = torch.randn(1, 16, 3, 128, 128)  # 16 帧原始 RGB 视频
latents = video_dit.vae_3d.encode(raw_video)
print("3D VAE latents size:", latents.shape)    # torch.Size([1, 8, 4, 16, 16])

# 模拟 Video DiT 去噪前向传播 (文本注入 embedding 长度为 10, 维度为 128)
t = torch.tensor([[120.0]])
text_cond = torch.randn(1, 10, 128)
noise_pred = video_dit(latents, t, text_cond)
print("Video DiT noise prediction shape:", noise_pred.shape)  # torch.Size([1, 8, 4, 16, 16])
```

---

## 机器人决策应用：Diffusion Policy & Flow Matching

在机器人闭环控制中，扩散策略将高维环境观测（视觉 + Proprioception）作为 Conditioning 变量，利用加噪-去噪来拟合复杂的动作轨迹分布。
*   <strong>仿射动作生成（Diffusion Policy）</strong>: 使用 1D 时间卷积 UNet 逐步将动作轨迹噪声去噪还原为流畅的操控动作。详细代码实现：[diffusion_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/diffusion_policy.py)。
*   <strong>低延迟流匹配（Flow Matching Policy）</strong>: 使用直线向量场映射（Straight Conditional Flow Matching）取代弯曲的扩散轨迹。在推理时通过 Euler 积分快速求解 ODE 轨迹，大幅降低机器人闭环交互的推理延迟。详细代码实现：[flow_matching_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/flow_matching_policy.py)。

<p align="center">
  <img src="images/diffusion_policy_teaser.png" width="85%" alt="Diffusion Policy Robotics Teaser" />
</p>
<p align="center">
  <img src="images/diffusion_policy_multimodal.png" width="70%" alt="Multimodal Action Trajectory Learning" />
</p>

---

## 快速开始与 Demo 运行

我们提供了一个测试验证脚本 [run_demo.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/run_demo.py)。它包含了 **Stable Diffusion、Diffusion Policy、Flow Matching Policy、DiT、Video DiT 3D VAE 编码/解码**的一体化前向推理单元测试。

运行测试命令：
```bash
python StableDiffusion/run_demo.py
```

---

## 公式与图示说明

*   <strong>静态公式</strong>: 所有的 Block Math 公式已经全部在本地使用 `matplotlib` 渲染为白底的高 DPI 图片并存放在 `images/` 中（例如 `eq5_ddim_step.png`, `eq6_adaln.png` 等），以保证在暗色模式和不同的移动端浏览器上能瞬间且高保真地渲染显示。
*   <strong>论文图示</strong>:
    -   `diffusion_policy_teaser.png` 和 `diffusion_policy_multimodal.png` 见前文图示，它们说明了生成式扩散模型在多模态运动规划及机械臂闭环抓取任务中的泛化能力与直观优势。
