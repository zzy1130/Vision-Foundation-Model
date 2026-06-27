# Vision Foundation Model 项目开发与排版规范记忆文件

本项目是用于系统化学习和研究计算机视觉领域的各类**视觉基础模型（Vision Foundation Models, VFMs）**与**视觉语言模型（Vision-Language Models, VLMs）**的仓库。

本文件记录了在本项目开发、文档编写中总结出的核心排版与开发规范，以便后续的 AI 助手和开发者严格遵循，确保文档在 GitHub 页面及各端阅读器上的完美呈现。

---

## 1. 文档数学公式排版规范 (GitHub LaTeX/MathJax)

为了确保 README 文档在 GitHub 网页端（特别是中国大陆网络环境、开启了广告屏蔽插件或移动端浏览器上）能够 100% 成功加载且完美显示公式，须遵循以下规范：

### 1.1 公式块（Block Math）规范（必须遵循 `github-latex` Skill）
*   **解决方案**：**一律采用本地图片渲染方案**。由于 CodeCogs 等在线服务器极不稳定且常有 502/503 错误，对于新公式，必须通过本地 Python 脚本（利用 matplotlib / sympy）渲染出高分辨率的 PNG 图片（DPI 设为 300，设置白色底面以适应暗色模式），存放在对应子目录下的 `images/` 目录中，并在 README 中以相对路径引用（例如 `images/eq1_ssi.png`）。
*   **排版规范与工具**：项目创建了定制化的 `github-latex` Skill，所有公式的编写规范与渲染方式必须严格遵循此 Skill 的规则。先前已采用 CodeCogs 的旧文档公式若无修改，可保持原样以避免无谓改动，但所有新公式必须统一采用本地图片方案。
*   **LaTeX 编写禁忌**：
    *   **不要在公式中使用双竖线 `\|`**（表示范数）：Markdown 解析器会将 `\|` 误识别为表格的管道转义符，剥离斜杠从而破坏 LaTeX 语法。必须使用标准 LaTeX 命令 **`\Vert`** 代替。

### 1.2 内联公式（Inline Math）与下标规范
*   **痛点**：在内联公式 `$ ... $` 中如果存在多个带下划线的下标变量（例如 `$s_{i,j}$`、`$v_i$`、`$t_j$` 等），Markdown 解析器会优先将下划线成对解析为 *斜体标签*，导致公式源码被 HTML 标签截断而无法被数学引擎编译。
*   **解决方案**（二选一）：
    1.  **对于简单的变量、下标或希腊字母**：直接使用 Unicode 符号加 HTML 下标标签 `<sub>`，这样无需任何数学加载引擎，在所有设备上均能瞬间渲染。
        *   `$s_{i,j}$` ➡️ `s<sub>i,j</sub>`
        *   `$\tau$` ➡️ `τ`
        *   `$\beta$` ➡️ `β`
        *   `$O(N^2)$` ➡️ `O(N²)`
    2.  **对于必须使用内联 LaTeX 的场景**：使用反引号配合美元符号包裹（`$` `` `...` `` `$`），这能强制 Markdown 引擎跳过解析下划线，直接将源码传递给 MathJax。
        *   `$``y_{i,j} = 1``$`

---

## 2. 文本加粗规范 (Bold Typography)

*   **痛点**：在混合排版中，如果 Markdown 的双星号加粗 `**` 紧贴着中文汉字或中文括号（如 `将**视觉自监督**与` 或 `**自监督（如 MIM）**`），许多 Markdown 引擎（包括 GitHub）无法正确识别粗体边界，导致直接显示出星号。
*   **解决方案**：在涉及中文和括号的加粗排版中，一律使用 HTML 标签 **`<strong>加粗内容</strong>`** 代替 `**`。
    *   **正确**：`将 <strong>视觉自监督（如 Masked Image Modeling）</strong> 与多模态对齐相结合。`
    *   **避免**：`将**视觉自监督（如 Masked Image Modeling）**与多模态对齐相结合。`

---

## 3. 项目开发与依赖管理规范

1.  **虚拟环境**：本项目使用 `uv` 工具管理环境，所有环境依赖均装在根目录的 `.venv/` 中。
2.  **Git 追踪过滤**：
    *   在根目录的 `.gitignore` 中，必须忽略 `.venv/`、`__pycache__/`、`demo_images/`（测试推理时下载的临时图片）以及 IDE 配置文件。
    *   **严禁忽略** `CLIP/images/` 目录，这里存放了技术文档渲染所需的静态图（如 `openai_clip.png` 和 `siglip.png`），必须提交并推送到 Git 以供 README 显示。
3.  **代码复现要求**：
    *   各阶段模型（如第一阶段的 OpenAI CLIP, FLIP, CLIPA, EVA-CLIP, SigLIP）需在对应目录下提供纯 PyTorch 从零实现的模型代码文件（无需下载 pre-trained checkpoint，只需实现 forward 和 loss 计算以供学习模型架构）。
    *   必须在各目录的 README 中详细给出各个模型前向传播的正确调用代码框。

---

## 4. SAM 家族模型开发与空间维度对齐规范

1.  <strong>动态分辨率与特征图尺度对齐</strong>：提示编码器中对 Mask 进行下采样时，其输出分辨率必须与图像编码器提取的特征图维度完全一致（如 `img_feats.shape[-2:]`）。在 PyTorch 模型的前向传播中，避免硬编码空间尺寸（如 `64x64`），应动态传入 `feat_shape = (H_feat, W_feat)` 以确保模型能适应不同输入图像分辨率。
2.  <strong>分辨率无关的归一化算子 (GroupNorm)</strong>：在对特征进行下采样的模块（例如提示编码器中的 mask_downsampler）中，若特征的空间维度动态变化，严禁使用依赖固定空间维度的归一化算子，如 `nn.LayerNorm([channels, H, W])`。一律使用 <strong>`nn.GroupNorm(1, channels)`</strong> 或 `nn.BatchNorm2d` 替代，以确保前向传播在不同分辨率下均能正常计算。
3.  <strong>时序追踪状态与记忆单元重置</strong>：视频级别或时序跟踪模型（如 SAM 2）必须实现显式的记忆状态重置接口（例如 `reset_video_memory()`），并在处理不同视频流之前调用。在评估（eval）模式下，FIFO 记忆队列的长度控制和历史状态读取必须保持严格的维度一致性，防止发生跨序列内存污染或维度溢出。
4.  <strong>三维反投影 (Lifting) 中的几何变换</strong>：将 2D 像素坐标提升到三维空间（如 SAM 3D）时，必须通过外参 <strong>[R | T]</strong> 和内参 K 的精确逆矩阵（`torch.inverse(intrinsics)`）进行反投影，且深度图的取值必须严格为正数（可加极小偏置，如 `+ 1e-5`），以避免在 homogeneous 坐标系转换中出现奇异点或零分母。
