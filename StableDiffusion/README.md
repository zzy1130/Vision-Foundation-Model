# Stable Diffusion, Diffusion Policy & World Models (生成式基座与时空物理模拟)

本目录包含了基于 PyTorch 从零实现的 <strong>Stable Diffusion (Latent Diffusion Models)</strong>、机器人控制决策中的 <strong>Diffusion Policy</strong>、<strong>Flow Matching Policy</strong>，以及现代生成式基座的核心架构 <strong>DiT (Diffusion Transformer)</strong> 与 <strong>Video DiT (时空视频扩散 Transformer)</strong> 的完整设计与前向/推理逻辑。这些模型共同构成了现代生成式 AI 从“高维多模态媒体合成”走向“机器人动作轨迹决策”与“物理世界模型模拟（World Models）”的理论基石。

---

## 目录
1. [世界模型（World Models）与视频物理模拟](#1-世界模型world-models与视频物理模拟)
2. [LDM 与 VAE：隐空间压缩原理](#2-ldm-与-vae隐空间压缩原理)
3. [DDPM 与 DDIM：从随机演化到确定性加速](#3-ddpm-与-ddim从随机演化到确定性加速)
4. [无分类器引导（Classifier-Free Guidance, CFG）原理](#4-无分类器引导classifier-free-guidance-cfg原理)
5. [DiT (Diffusion Transformer) 架构变革与 AdaLN 调制](#5-dit-diffusion-transformer架构变革与-adaln-调制)
6. [Video DiT：3D VAE、时空 Patch 与文本注入（Sora 核心）](#6-video-dit3d-vae时空-patch与文本注入sora核心)
7. [机器人动作生成：Diffusion Policy 与 Flow Matching](#7-机器人动作生成diffusion-policy与-flow-matching)
8. [代码接口与 Demo 验证](#8-代码接口与-demo-验证)
9. [公式与图示对照](#9-公式与图示对照)

---

## 1. 世界模型（World Models）与视频物理模拟

在强化学习与自动驾驶领域，<strong>世界模型（World Models）</strong>是智能体在脑海中模拟真实物理世界运行机制的虚拟仿真器。它需要学习一个状态转移函数：
<p align="center"><strong>s<sub>t+1</sub> = T(s<sub>t</sub>, a<sub>t</sub>)</strong></p>
其中 s<sub>t</sub> 表示当前环境的状态，a<sub>t</sub> 表示智能体采取的控制动作，s<sub>t+1</sub> 表示物理世界受动作影响后的下一刻状态。
*   <strong>视频生成即物理模拟</strong>: 传统的物理引擎（如 Bullet, MuJoCo）需要人工编写复杂的运动方程，而以 OpenAI Sora 和 Google Genie 为代表的<strong>视频生成世界模型</strong>则直接将物理世界的演变建模为条件视频生成。
*   <strong>时空规律学习</strong>: 模型通过观察海量的视频数据，学习并理解重力、碰撞、惯性、流体力学以及刚体形变等物理规律。通过在时空特征上进行去噪，Video DiT 能够预测并合成极具真实感的未来视频序列，从而成为智能体进行无实物安全规控训练的“沙盒”。

---

## 2. LDM 与 VAE：隐空间压缩原理

直接在像素空间（Pixel Space）进行去噪（如在 512 × 512 的 RGB 图像上运行扩散过程）计算开销极其高昂，因为网络需要浪费大量参数去重构高频的细节冗余（如背景白噪、墙面纹理等），而这些冗余并不包含核心语义。
*   <strong>变分自编码器 (VAE)</strong>: <strong>Latent Diffusion Models (LDM)</strong> 提出通过一个预训练好的变分自编码器（VAE）将高维像素空间压缩到低维潜在特征空间（Latent Space）。VAE 的优化损失由重构误差和约束潜在变量分布的 KL 散度（Kullback-Leibler Divergence）共同构成：
    <p align="center"><img src="images/eq7_vae_loss.png" width="60%" alt="VAE Loss" /></p>
*   <strong>隐空间去噪</strong>: 隐特征 z<sub>0</sub> 的维度通常比原图降低 8 倍（如 512 × 512 × 3 图像被压缩为 64 × 64 × 4 隐变量），去除了大量感知冗余。扩散模型的加噪和去噪完全在这低维的 z 空间中运行，使得计算效率提升了数百倍。

---

## 3. DDPM 与 DDIM：从随机演化到确定性加速

*   <strong>DDPM (Denoising Diffusion Probabilistic Model)</strong>:
    -   <strong>正向过程（加噪）</strong>: 将高斯噪声逐步注入数据中。在任意时间步 t，加噪隐特征 z<sub>t</sub> 均可通过一步解析计算得出：
        <p align="center"><img src="images/eq1_ddpm_add_noise.png" width="40%" alt="DDPM Forward Process" /></p>
    -   <strong>反向过程（去噪）</strong>: 采用马尔可夫链（Markov Chain）退步逼近，预测每个时间步注入的噪声，目标函数（Noise Prediction Loss）定义为：
        <p align="center"><img src="images/eq8_ddpm_loss.png" width="60%" alt="DDPM Noise Prediction Loss" /></p>
        由于其随机性，采样去噪需要完整运行整个链条（如 1000 步），导致速度极慢。
*   <strong>DDIM (Denoising Diffusion Implicit Model)</strong>:
    -   <strong>确定性采样</strong>: DDIM 将正向过程推广至非马尔可夫链形式，构建了一条确定性的 ODE 去噪路径：
        <p align="center"><img src="images/eq5_ddim_step.png" width="70%" alt="DDIM Sampling Step" /></p>
    -   <strong>快速跳步</strong>: 因为去噪过程是确定性的，我们可以只在完整的 1000 步中均匀选取 20 步或 50 步进行迭代，在保持图像生成质量的同时大幅缩短推理时间。这对于需要高频输出动作轨迹的机器人系统至关重要。

---

## 4. 无分类器引导（Classifier-Free Guidance, CFG）原理

在文本生成图像或机器人基于视觉观测生成动作时，模型必须在样本多样性（Diversity）与条件贴合度（Fidelity）之间取得平衡。
*   <strong>训练方法</strong>: 在模型训练中，输入条件（如文本 Embedding c，或机器人观测 Embedding o）会以一定概率（通常为 10%--20%）被随机置为空值 ∅（即令其退化为无条件训练）。
*   <strong>推理外推</strong>: 推理阶段，网络会分别计算有条件下的预测噪声和无条件下的预测噪声。CFG 通过线性外推机制对条件噪声进行方向增强：
    <p align="center"><img src="images/eq2_cfg.png" width="55%" alt="Classifier-Free Guidance" /></p>
    其中 w 为引导因子。w 越大，生成的动作轨迹越紧密地贴合输入的视觉观测，动作变动范围越小，精度更高。

---

## 5. DiT (Diffusion Transformer) 架构变革与 AdaLN 调制
*   <strong>代码实现</strong>: [dit.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/dit.py)
*   <strong>核心机制</strong>:
    1.  <strong>Patchification（分块嵌入）</strong>: 将 2D Latent（如 32x32x4）切分为 p × p 的图像块（Patches），平坦化并映射为一维 Token 序列。
    2.  <strong>AdaLN (Adaptive Layer Normalization)</strong>: 传统的 Transformer 使用层归一化（LayerNorm），而 DiT 使用 AdaLN 将时间步和条件信息（如类别标签）注入每一层 Transformer Block 中。AdaLN 计算公式为：
        <p align="center"><img src="images/eq6_adaln.png" width="50%" alt="AdaLN Block" /></p>
        其中 $\gamma(y)$ 和 $\beta(y)$ 是由 timestep 和类别特征通过 MLP 回归得到的通道级缩放和偏移参数。
    3.  <strong>参数规模扩展（Scaling Up）</strong>: 彻底摆脱了 UNet 复杂的卷积 Skip Connection 限制，DiT 的参数规模和生成质量表现出极其完美的 Scaling Law。

---

## 6. Video DiT：3D VAE、时空 Patch 与文本注入（Sora 核心）
*   <strong>代码实现</strong>: [video_dit.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/video_dit.py)
*   <strong>核心机制</strong>:
    1.  <strong>3D 变分自编码器 (VAE3D)</strong>: 将视频帧序列（B, F, C, H, W）输入 3D 卷积层，在空间维度压缩 8x 的同时，在<strong>时间（帧）维度进行压缩（如 2x）</strong>，从而获得时空隐表征（B, F_lat, C_lat, H_lat, W_lat）。
    2.  <strong>时空分块（Spatiotemporal Patchification）</strong>: 提取 $p_t \times p_s \times p_s$（如 2 帧 $\times$ 2 像素 $\times$ 2 像素）的时空超像素块（3D Patches），映射为 1D 时空 Tokens。
    3.  <strong>时空联合自注意力（Spatiotemporal Self-Attention）</strong>: 所有的时空 Tokens 在 Transformer 中进行全局注意力交互，使得模型可以同时学习空间结构与跨帧的时间演变物理规律（World Model 的物理模拟基础）。
    4.  <strong>交叉注意力文本注入 (Text Injection)</strong>: 在每一层 Spatiotemporal DiT Block 中，插入一个专用的 Cross-Attention 层，让时空视频 Tokens 扮演 Query，去检索由 CLIP/T5 提取的文本 prompt 特征（Key/Value），将复杂的文本指令注入到视频生成的每一个像素时空轨道中。

<p align="center">
  <img src="images/pointmae_net.jpg" width="80%" style="display:none" alt="Placeholder" />
</p>

---

## 7. 机器人动作生成：Diffusion Policy 与 Flow Matching

在连续轨迹规划中，生成式模型用于建模高维非高斯动作分布。
*   <strong>动作去噪扩散（Diffusion Policy）</strong>: 将连续的动作轨迹序列（未来 T<sub>p</sub> 步）作为去噪目标，在 1D 时间轴上进行时序去噪，使得机器人能够在避障、抓取等复杂非线性路径规划中完全拟合人类演练的多峰动作分布。详细代码实现：[diffusion_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/diffusion_policy.py)。
*   <strong>低延迟直线流（Flow Matching Policy）</strong>: 使用直线向量场映射（Straight CFM）取代加噪过程中的弯曲漂移轨迹。推理阶段通过简单的 Euler 积分快速求解 ODE，以极少的推理步数（如 5--10 步）完成闭环解算。详细代码实现：[flow_matching_policy.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/flow_matching_policy.py)。

---

## 8. 代码接口与 Demo 运行

项目提供了一个一体化前向推理验证脚本 [run_demo.py](file:///Users/zhongzhiyi/Vision-Foundation-Model/StableDiffusion/run_demo.py)。它包含了所有模型（Stable Diffusion, Diffusion Policy, Flow Matching, DiT, Video DiT）的前向与推理测试。

运行测试命令：
```bash
python StableDiffusion/run_demo.py
```

---

## 9. 公式与图示对照

*   <strong>公式 1</strong>: [DDPM 隐空间单步加噪公式](images/eq1_ddpm_add_noise.png)
*   <strong>公式 2</strong>: [CFG 无分类器引导去噪外推](images/eq2_cfg.png)
*   <strong>公式 3</strong>: [流匹配（Flow Matching）插值概率路径](images/eq3_cfm_path.png)
*   <strong>公式 4</strong>: [流匹配速度场损失函数](images/eq4_cfm_loss.png)
*   <strong>公式 5</strong>: [DDIM 确定性迭代采样路径](images/eq5_ddim_step.png)
*   <strong>公式 6</strong>: [AdaLN（特征自适应归一化）调节算子](images/eq6_adaln.png)
*   <strong>公式 7</strong>: [VAE 编解码重构与 KL 约束损失](images/eq7_vae_loss.png)
*   <strong>公式 8</strong>: [DDPM 正向噪声预测损失](images/eq8_ddpm_loss.png)
*   <strong>公式 9</strong>: [Video DiT 3D 视频张量到 1D 时空 Tokens 投影映射](images/eq9_spatiotemporal_patch.png)
*   <strong>图示 1</strong>: [Diffusion Policy 机器人工作环境实拍](images/diffusion_policy_teaser.png)
*   <strong>图示 2</strong>: [模仿学习多模态轨迹与多峰概率重构对比](images/diffusion_policy_multimodal.png)
