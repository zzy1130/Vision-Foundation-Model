# Vision Foundation Model 学习项目

本项目致力于系统化学习和研究计算机视觉领域的各类**视觉基础模型（Vision Foundation Models, VFMs）**与**视觉语言模型（Vision-Language Models, VLMs）**。通过对这些模型的研究、对比、代码复现与实际应用，深入掌握多模态人工智能的核心技术。

## 学习路线图 (Roadmap)

本项目将按照技术演进脉络，逐步深入研究以下核心模型：

1. **CLIP 系列 (已启动)**
   - 探索从传统的 Contrastive Learning 到最新的 Sigmoid Loss 演进。
   - 研究模型包括：OpenAI CLIP, OpenCLIP, FLIP, CLIPA, EVA-CLIP, SigLIP。
   - 产出：中文技术总结文档、SigLIP 核心架构与损失函数复现、零分类与图文检索 Demo。
   
2. **多模态对齐与生成 (BLIP & BLIP-2 系列)**
   - 学习如何通过 Q-Former 对齐冻结的视觉编码器与大型语言模型 (LLM)。
   - 探索图文匹配 (ITM)、图文对比 (ITC) 和图文生成 (ITG) 三阶段预训练。

3. **视觉语言大模型 (LLaVA / MiniGPT 系列)**
   - 研究端到端多模态大模型的指令微调 (Instruction Tuning)。
   - 探索基于 Projection layer / MLP 的视觉特征映射。

4. **视觉自监督与分割大模型 (DINOv2 & SAM 系列)**
   - 学习无监督特征表示学习与高精度泛化分割。

---

## 当前阶段进度：CLIP 阶段

- [x] **环境初始化**：使用 `uv` 管理虚拟环境，并安装 `torch`、`torchvision`、`transformers` 等依赖。
- [x] **深度研究报告**：完成 CLIP 及其迭代款（FLIP, CLIPA, EVA-CLIP, SigLIP）的技术演进报告。
- [x] **代码复现**：从零实现最新的 **SigLIP**（Sigmoid Loss for Language-Image Pre-training）核心架构与 Loss 函数。
- [x] **任务实战**：编写演示脚本，调用预训练 SigLIP 模型完成**零样本图像分类（Zero-shot Classification）**与**图像-文本双向检索（Image-Text Retrieval）**。

*详细技术内容与代码实现请参考 [CLIP 学习目录](file:///Users/zhongzhiyi/Vision-Foundation-Model/CLIP/README.md)。*
