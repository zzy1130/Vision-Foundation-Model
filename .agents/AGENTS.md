# Project Rules for Vision Foundation Model Study

This project is a dedicated repository for studying Vision Foundation Models (VFMs) and Vision-Language Models (VLMs) by implementing architectures from scratch in PyTorch.

When editing code, writing READMEs, or creating documentation in this repository, you must adhere to the following workspace rules:

## 1. Documentation & LaTeX Math Rendering Rules
To ensure all mathematical formulas render perfectly on GitHub web and mobile, and remain readable in both light/dark mode themes:
*   **Block Math (`$$ ... $$`) (Follow the `github-latex` Skill)**: Convert all block equations to centered SVG images hosted on CodeCogs. All math rendering must strictly follow the rules defined in our workspace `github-latex` skill.
    *   *Format*: `<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cbg_white%20URL_ENCODED_LATEX" alt="equation" /></p>`
    *   *Note*: The LaTeX string must be fully URL-encoded (specifically, encode `(`, `)` as `%28`, `%29`, `{`, `}` as `%7B`, `%7D`, etc.).
    *   *Do NOT* use double pipe `\|` (for norms) in LaTeX. Use `\Vert` instead to avoid markdown table/pipe parser conflicts.
*   **Inline Math**:
    *   For simple subscripts, variables, and Greek letters, use Unicode symbols and HTML `<sub>` tags to ensure instant rendering across all platforms without depending on MathJax/KaTeX.
        *   Example: `s<sub>i,j</sub> = v<sub>i</sub> · t<sub>j</sub>`, `λ`, `β`, `τ`.
    *   For complex inline expressions, use the backtick-dollar syntax: `$```...```$` to prevent markdown parser from treating underscores as italic markers.
        *   Example: `$``y_{i,j} = 1``$`

## 2. Bold Text Styling
*   Do NOT use standard `**` bold syntax immediately adjacent to Chinese characters or parentheses (e.g. `将**视觉自监督**与` or `**自监督（如 MIM）**`), as it fails to parse correctly in many Markdown parsers.
*   Instead, use HTML **`<strong>`** tags (e.g. `<strong>视觉自监督（如 MIM）</strong>`).

## 3. Environment & Git Tracking Rules
*   The project uses `uv` to manage the virtual environment at `.venv`.
*   `.venv/`, `__pycache__/`, and temporary image folders like `CLIP/demo_images/` must be ignored in `.gitignore`.
*   **Do NOT ignore `CLIP/images/`**. This folder contains documentation assets (diagrams) and must be tracked by Git.
*   For each implemented VFM variant, provide clean PyTorch class models in the directory (e.g. `siglip.py`, `flip.py`) and write detailed invocation snippets in the README.

## 4. SAM Family Implementation & Spatial Alignment Rules
*   **Dynamic Resolution & Spatial Dimensions**: The outputs of prompt mask downsamplers must dynamically align with the image features' spatial shape (e.g. `img_feats.shape[-2:]`). Avoid hardcoding target resolutions (e.g., `64x64`) inside the model; instead, dynamically pass the target feature map shape to prevent tensor shape mismatch errors during inference with arbitrary image inputs.
*   **Resolution-Agnostic Normalization (GroupNorm)**: Avoid normalization operators that require a fixed resolution such as `nn.LayerNorm([channels, H, W])` in modules processing feature maps of dynamic sizes. Use **`nn.GroupNorm(1, channels)`** or standard 2D batch normalization to ensure resolution-independent forward propagation.
*   **State Reset & Sequence Separation**: Always provide a public state reset interface (e.g., `reset_video_memory()`) for tracking-based architectures like SAM 2. The memory FIFO queues and historical lookup channels must be cleared between video sequences to prevent dimensions mismatch or cross-video prediction corruption.
*   **Geometric Inversion in 3D Backprojection**: When lifting 2D masks to 3D world space (e.g. SAM 3D), ensure intrinsics and extrinsics are inverted mathematically via `torch.inverse(intrinsics)` and translation shifting. Apply a small positive epsilon offset to depth values to prevent divide-by-zero or numeric instability at zero-depth regions.

