---
name: github-latex
description: Standards and guidelines for rendering LaTeX mathematical formulas natively in GitHub Markdown documents.
---

# GitHub Native LaTeX Math Rendering standard

This skill defines the standard way to write and render mathematical equations and expressions in Markdown documents (`.md` files) hosted on GitHub. 

## 1. Core Principle: Native MathJax Delimiters

Instead of using unstable third-party rendering services (such as CodeCogs SVG/PNG APIs), which are prone to `503 Service Unavailable` or `SSL EOF` errors, all Markdown documents in this repository **must use GitHub's native MathJax support** (rolled out in May 2022).

### 1.1 Block Math (Display Mode)
For standalone, centered mathematical blocks, use double dollar signs (`$$`) on their own lines.

**Usage:**
```markdown
$$
\mathcal{L}_{\text{ssi}} = \frac{1}{N} \sum_{i} \left( \hat{d}_i^{\text{pred}} - \hat{d}_i^{\text{GT}} \right)^2
$$
```

*Note: Do not use HTML `<p align="center"><img>` tags for math blocks.*

### 1.2 Inline Math (Inline Mode)
For inline math variables, expressions, or subscripts within paragraphs:
1. For simple subscripts and variables (e.g., `s_{i,j}` or Greek letters `\beta`), prefer **Unicode symbols and HTML `<sub>` tags** (e.g., `s<sub>i,j</sub>` and `β`). This avoids relying on MathJax execution for simple text formatting.
2. For complex inline LaTeX expressions, wrap the formula with **backtick-dollar syntax** (`$``...```$`). This prevents the Markdown parser from treating LaTeX underscores (`_`) as italic markers.

**Usage:**
```markdown
Here is a complex variable $``y_{i,j} = \sigma(x)``$ inside a sentence.
```

---

## 2. Converting from CodeCogs to Native MathJax

When updating existing documents to the native standard, convert CodeCogs `<img>` tags directly into native LaTeX blocks or inline symbols.

### Example Conversion:
*   **Old CodeCogs Block:**
    ```html
    <p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cbg_white%20G%20%3D%20X%20X%5ET" alt="equation" /></p>
    ```
*   **New Native Block:**
    ```markdown
    $$
    G = X X^T
    $$
    ```

*Note: Strip `\bg_white` from the LaTeX code when converting to native blocks. GitHub automatically handles light/dark mode theme rendering for native MathJax.*
