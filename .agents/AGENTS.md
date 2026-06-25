# Project Rules for Vision Foundation Model Study

This project is a dedicated repository for studying Vision Foundation Models (VFMs) and Vision-Language Models (VLMs) by implementing architectures from scratch in PyTorch.

When editing code, writing READMEs, or creating documentation in this repository, you must adhere to the following workspace rules:

## 1. Documentation & LaTeX Math Rendering Rules
To ensure all mathematical formulas render perfectly on GitHub web and mobile, and remain readable in both light/dark mode themes:
*   **Block Math (`$$ ... $$`)**: Convert all block equations to centered SVG images hosted on CodeCogs. Prepend `\bg_white` (URL encoded as `%5Cbg_white%20`) to the LaTeX code to add a white background for dark mode theme compatibility.
    *   *Format*: `<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cbg_white%20URL_ENCODED_LATEX" alt="equation" /></p>`
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
