# Stable Diffusion & Diffusion Policy (Generative Models for Vision & Robot Control)

本目录包含了基于 PyTorch 从零实现的 <strong>Stable Diffusion (Latent Diffusion Models)</strong>、机器人决策中的 <strong>Diffusion Policy</strong> 以及现代改进方案 <strong>Flow Matching Policy</strong> 的核心架构与前向传播/推理逻辑。这些模型代表了生成式 AI 从“高维图像合成”走向“机器人高维多模态动作轨迹生成（Imitation Learning）”的技术跨越。

---

## 目录
1. [模型家族与技术演进](#模型家族与技术演进)
2. [核心模型实现与解析](#核心模型实现与解析)
    - [Stable Diffusion (Latent Diffusion)](#1-stable-diffusion-latent-diffusion)
    - [Diffusion Policy (RSS 2023)](#2-diffusion-policy-rss-2023)
    - [Flow Matching Policy (ICLR 2023)](#3-flow-matching-policy-iclr-2023)
3. [快速开始与 Demo 运行](#快速开始与-demo-运行)
4. [公式与图示说明](#公式与图示说明)

---

## 模型家族与技术演进

生成式扩散模型（Diffusion Models）在计算机视觉领域掀起了革命（如 Stable Diffusion）。其核心理念是通过神经网络预测噪声，逐步将高斯分布噪声去噪重构为高质量数据。
*   <strong>Stable Diffusion (LDM)</strong>: 为降低计算开销，先利用 VAE 将图像编码到低维潜在空间（Latent Space），在隐空间内运行扩散去噪过程。通过交叉注意力机制（Cross-Attention）注入 CLIP 编码的文本向量，以实现受控文本生成。
*   <strong>Diffusion Policy (机器人动作扩散策略)</strong>: 在机器人模仿学习（Imitation Learning）中，人类演示往往是<strong>多模态（Multimodal）</strong>的（例如：绕过障碍物去抓取目标，既可以从左边绕，也可以从右边绕）。传统确定性策略（如 MLP 行为克隆）在这种平均化多峰分布时会彻底失效，输出撞墙的折中动作。
    -   <strong>解决方案</strong>: 将机器人的连续动作序列（Action Trajectory）建模为一个条件去噪扩散过程（Conditional Denoising Diffusion Process），以当前的相机图像和机械臂状态（Proprioception）作为 Conditioning 变量，利用 1D 卷积 UNet 或 Transformer 逐步将高斯动作噪声去噪为符合人类操作分布的顺滑轨迹。
*   <strong>Flow Matching Policy</strong>: 传统的扩散模型训练时前向加噪轨迹是弯曲且随机的，导致去噪推理时需要 50--100 个步骤（即便使用加速采样器也需要 15--20 步），在机器人高频闭环控制中延迟严重。<strong>流匹配（Flow Matching）</strong>构建了从噪声到动作轨迹之间的直线概率路径（Straight Flow），利用确定性的常微分方程（ODE）以极少的积分步数（如 5--10 步）完成推理，非常适合机器人的高频低延迟控制需求。

---

## 核心模型实现与解析

### 1. Stable Diffusion (Latent Diffusion)
*   <strong>代码实现</strong>: [stable_diffusion.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/stable_diffusion.py)
*   <strong>核心机制</strong>:
    1.  <strong>自编码器 (VAE)</strong>: 包含 Conv2d 组成的分级下采样编码器（将图像编码为隐变量 z<sub>0</sub>）与 ConvTranspose2d 组成的解码器。
    2.  <strong>条件编码 (CLIP Text Encoder)</strong>: 对输入的文本 Token 进行词向量转换和 Transformer 交互，生成文本 Embedding $c$。
    3.  <strong>前向加噪（DDPM Add Noise）</strong>: 根据预设的累计乘积因子 $\bar{\alpha}_t$ 在隐空间中对潜在变量注入指定步数 t 的高斯噪声。加噪公式为：
        <p align="center"><img src="images/eq1_ddpm_add_noise.png" width="40%" alt="DDPM Forward Process" /></p>
    4.  <strong>交叉注意力去噪 (Denoising UNet with Cross-Attention)</strong>: UNet 通过 Residual Block 处理隐变量，并使用 Cross-Attention 算子将文本 Embedding $c$ 的语义信息交叉融入空间特征中。
    5.  <strong>无分类器引导（Classifier-Free Guidance, CFG）</strong>: 在推理阶段将有文本条件的噪声预测与无条件噪声预测进行外推加权，增强生成图像的文本相关性与饱和度：
        <p align="center"><img src="images/eq2_cfg.png" width="55%" alt="Classifier-Free Guidance" /></p>

#### 代码调用示例
```python
import torch
from stable_diffusion import StableDiffusion

# 初始化隐空间扩散模型
model = StableDiffusion(cond_dim=128, latent_channels=4)

# 模拟输入: 1个 Batch 的高分辨率图像与分词后的文本引导
images = torch.randn(1, 3, 256, 256)
text_tokens = torch.randint(0, 1000, (1, 10))
timesteps = torch.tensor([450])  # 加噪步数

# 训练前向传播，计算噪声预测的均方误差(MSE)
loss = model(images, text_tokens, timesteps)
print("Diffusion training loss:", loss.item())

# 推理生成: 输入文本直接解算生成复原图
generated_img = model.generate(text_tokens, latent_shape=(1, 4, 32, 32), num_inference_steps=10)
print("Generated image size:", generated_img.shape)  # torch.Size([1, 3, 256, 256])
```

---

### 2. Diffusion Policy (RSS 2023)
*   <strong>代码实现</strong>: [diffusion_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/diffusion_policy.py)
*   <strong>核心机制</strong>:
    1.  <strong>多模态动作表示</strong>: 如下图所示，当人类演示遇到多叉路口分流时（Multimodal Traces），传统监督模仿学习（BCE/MSE）会学习两个峰值的平均数导致出错。动作扩散策略可以学习真实完整的双峰概率边界。
    2.  <strong>视觉观测编码器 (ObservationEncoder)</strong>: 提取摄像头图像的 2D 特征，并将其与当前机械臂的 proprioceptive 状态变量（如六轴关节角或末端坐标）拼接为高维观测条件嵌入 O<sub>cond</sub>。
    3.  <strong>1D 时间卷积 UNet (TemporalUNet1D)</strong>: 动作轨迹具有时间连续性。将待生成的动作序列建模为 $T_p \times D_a$（即预测未来 $T_p$ 步动作，每个动作维度为 $D_a$）。UNet 在 1D 时间轴上进行时序卷积去噪，并将 O<sub>cond</sub> 作为时间轴特征的偏置项或者缩放量。
    4.  <strong>退避视界控制（Receding Horizon Control）</strong>: 在真实机器人部署时，虽然模型一次性输出长达 $T_p$（如 16 步）的预测动作轨迹，但机器人仅执行前 $T_e$（如 8 步）动作，随后立即进行下一次视觉观测采集与去噪预测，以保证极佳的闭环纠错和动态避障能力。

<p align="center">
  <img src="images/diffusion_policy_teaser.png" width="85%" alt="Diffusion Policy Robotics Teaser" />
</p>
<p align="center">
  <img src="images/diffusion_policy_multimodal.png" width="70%" alt="Multimodal Action Trajectory Learning" />
</p>

#### 代码调用示例
```python
from diffusion_policy import DiffusionPolicy
import torch

# 初始化机器人姿态估计策略
policy = DiffusionPolicy(action_dim=2, state_dim=6, obs_dim=128)

camera_img = torch.randn(1, 3, 112, 112)
robot_state = torch.randn(1, 6)

# 推理: 从高斯动作噪声中生成 16 步长度的未来 2D 控制坐标序列
generated_actions = policy.predict_action(
    camera_img, robot_state, 
    action_shape=(1, 16, 2), num_inference_steps=10
)
print("Generated actions sequence shape:", generated_actions.shape)  # torch.Size([1, 16, 2])
```

---

### 3. Flow Matching Policy (ICLR 2023)
*   <strong>代码实现</strong>: [flow_matching_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/flow_matching_policy.py)
*   <strong>核心机制</strong>:
    1.  <strong>直线概率路径 (Straight Probability Flow)</strong>: CFM（Conditional Flow Matching）通过直接插值定义概率路径，构建纯直线流场来平滑过渡高斯随机分布 a<sub>0</sub> 和数据点 a<sub>1</sub>。轨迹路径与对应的速度场定义如下：
        <p align="center"><img src="images/eq3_cfm_path.png" width="55%" alt="Flow Matching Path" /></p>
    2.  <strong>速度场网络回归 (VelocityFieldNet1D)</strong>: 训练一个一维时序卷积网络来拟合这个瞬时速度向量。在时间 t 处输入插值状态 a<sub>t</sub>，直接回归速度。
    3.  <strong>常微分方程集成器 (ODE Integration Solver)</strong>: 推理时将求取噪声预测的随机采样替换为了基于常微分一阶 Euler 的累积积分过程。通过几步 Euler 更新向前推动动作状态。
    4.  <strong>向量场均方误差 Loss</strong>:
        <p align="center"><img src="images/eq4_cfm_loss.png" width="55%" alt="Flow Matching Loss" /></p>

#### 代码调用示例
```python
from flow_matching_policy import FlowMatchingPolicy
import torch

fm_policy = FlowMatchingPolicy(action_dim=2, state_dim=6, obs_dim=128)

camera_img = torch.randn(1, 3, 112, 112)
robot_state = torch.randn(1, 6)

# Euler 求解器集成: 仅需要 5 次 Euler 积分步骤就能获得直观顺滑的物理轨迹
actions_ode = fm_policy.predict_action(
    camera_img, robot_state, 
    action_shape=(1, 16, 2), num_euler_steps=5
)
print("ODE Integrated Action Trajectory:", actions_ode.shape)  # torch.Size([1, 16, 2])
```

---

## 快速开始与 Demo 运行

项目提供了一个完整的推理验证脚本 [run_demo.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/run_demo.py)。它将使用代表性的随机张量执行以上所有生成式与机器人策略流水线，并在终端输出输入与预测输出张量的详细尺寸信息。

运行 demo 命令：
```bash
python StableDiffusion/run_demo.py
```

---

## 公式与图示说明

*   <strong>静态公式</strong>: 所有的 Block Math 公式已经全部在本地使用 `matplotlib` 渲染为白底的高 DPI 图片并存放在 `images/` 中，以保证在暗色模式和不同的移动端浏览器上能瞬间且高保真地渲染显示。
*   <strong>论文图示</strong>: 已通过自动化脚本从官方开源库中获取：
    -   `diffusion_policy_teaser.png` 展示了 Diffusion Policy 在真实世界和模拟环境中的多样控制任务（如双臂协同、物体拨动等）。
    -   `diffusion_policy_multimodal.png` 形象对比了传统仿射克隆（容易在双峰均值处滑落失败）与动作扩散策略（能够拟合多峰概率密度从而保留完美演示的执行力）的区别。
