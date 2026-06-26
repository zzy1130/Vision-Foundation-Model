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
