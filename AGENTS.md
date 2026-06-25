# Vision Foundation Model 项目开发与排版规范记忆文件

本项目是用于系统化学习和研究计算机视觉领域的各类**视觉基础模型（Vision Foundation Models, VFMs）**与**视觉语言模型（Vision-Language Models, VLMs）**的仓库。

本文件记录了在本项目开发、文档编写中总结出的核心排版与开发规范，以便后续的 AI 助手和开发者严格遵循，确保文档在 GitHub 页面及各端阅读器上的完美呈现。

---

## 1. 文档数学公式排版规范 (GitHub LaTeX/MathJax)

为了确保 README 文档在 GitHub 网页端（特别是中国大陆网络环境、开启了广告屏蔽插件或移动端浏览器上）能够 100% 成功加载且完美显示公式，须遵循以下规范：

### 1.1 公式块（Block Math）规范
*   **痛点**：GitHub 官方的 MathJax 渲染器脚本（加载自第三方 CDN 或 GitHub 资产域名）在中国大陆经常因 DNS 污染或 CDN 节流而无法加载；此外，uBlock Origin 等广告屏蔽插件也经常拦截 MathJax。这会导致标准的 `$$ ... $$` 无法渲染，直接显示为源码。
*   **解决方案**：一律将独立的公式块替换为 **CodeCogs 渲染的高清 SVG 矢量图片**。
*   **编写格式**：
    使用 HTML `<p align="center">` 标签包裹 `<img>`，并必须在 LaTeX 源码开头添加 `\bg_white`（URL 编码为 `%5Cbg_white%20`），以确保在 GitHub 的 **黑夜/暗色模式主题** 下，公式依然带有白色底板、清晰可读。
    *   **示例代码**：
        ```html
        <p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cbg_white%20%5Cmathcal%7BL%7D_%7B%5Ctext%7BSigLIP%7D%7D%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20..." alt="equation" /></p>
        ```
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
