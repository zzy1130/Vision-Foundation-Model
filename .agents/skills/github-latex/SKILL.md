---
name: github-latex
description: Standards and guidelines for rendering LaTeX mathematical formulas in GitHub Markdown using fully URL-encoded CodeCogs SVG images.
---

# GitHub LaTeX Math Rendering Standard (CodeCogs)

This skill defines the standard way to write and render mathematical equations and expressions in Markdown documents (`.md` files) in this repository.

## 1. Core Principle: Fully URL-Encoded CodeCogs SVG Images

To ensure math formulas render perfectly across all browsers (including mobile) and themes (light/dark mode) without relying on native MathJax loading (which can be blocked or slow in some regions), all block equations must be converted to centered SVG images hosted on CodeCogs.

### 1.1 Formatting HTML Image Tags
Use an HTML `<p align="center">` tag to center the formula, containing an `<img>` tag with the CodeCogs SVG endpoint:

```html
<p align="center"><img src="https://latex.codecogs.com/svg.latex?%5Cbg_white%20URL_ENCODED_LATEX" alt="equation" /></p>
```

*   `\bg_white` (encoded as `%5Cbg_white%20` or `%5Cbg_white&space;`): Prepend this to the LaTeX code to add a white background for readability under GitHub's dark/dimmed mode themes.
*   `alt="equation"`: Use standard alt text.

---

## 2. Crucial Requirement: Full URL Encoding

The LaTeX string passed to CodeCogs **must be fully URL-encoded**. 

### 2.1 Parentheses Encoding
A common pitfall is leaving raw parentheses `(` and `)` in the URL. Raw parentheses will often break the browser/GitHub image fetch parser, leading to `503 Service Unavailable` or broken image icons.
*   `(` **must** be encoded as `%28`
*   `)` **must** be encoded as `%29`

### 2.2 Other Special Characters
*   `{` must be encoded as `%7B`
*   `}` must be encoded as `%7D`
*   `+` must be encoded as `%2B`
*   Spaces must be encoded as `%20` or `&space;`
*   `\` must be encoded as `%5C`

### 2.3 Recommended Encoding Method
Use a Python script to reliably generate the URL:
```python
import urllib.parse
latex = r"\bg_white \mathcal{L}_{\text{ssi}} = \frac{1}{N} \sum_{i} \left( \hat{d}_i^{\text{pred}} - \hat{d}_i^{\text{GT}} \right)^2"
encoded = urllib.parse.quote(latex)
print(f"https://latex.codecogs.com/svg.latex?{encoded}")
```

---

## 3. Alternative & Robust Principle: Local Image Rendering

Due to third-party services like CodeCogs being unstable or frequently blocked, complex or new formulas should be rendered locally to high-resolution PNG images and stored in the subfolder's `images/` directory.

### 3.1 Local Rendering Python Script Template
Use the following Python snippet to render mathematical equations to PNGs using `matplotlib.mathtext` (which requires no external LaTeX installation):

```python
import os
import matplotlib.pyplot as plt

# Configure matplotlib for Computer Modern (LaTeX-style) math fonts
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

def render_latex(formula, save_path):
    # Set size and DPI
    fig = plt.figure(figsize=(8, 1.2), dpi=300)
    fig.patch.set_facecolor('white')  # Explicit white background for dark mode theme compatibility
    
    plt.text(0.5, 0.5, formula,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=14,
             color='black')
    
    plt.gca().axis('off')
    
    # Save with tight bounding box to crop margins
    plt.savefig(save_path, 
                bbox_inches='tight', 
                pad_inches=0.1, 
                facecolor=fig.get_facecolor(), 
                edgecolor='none')
    plt.close(fig)
```

### 3.2 HTML Insertion
Refer to the local image relative to the subfolder's root, specifying a standard width to ensure it scales nicely:
```html
<p align="center"><img src="images/eq1_ssi.png" width="55%" alt="SSI Loss" /></p>
```

