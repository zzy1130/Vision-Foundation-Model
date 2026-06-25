import os
import requests

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {save_path}")
            return True
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    img_dir = "/Users/zhongzhiyi/Vision-Foundation-Model/DINO/images"
    os.makedirs(img_dir, exist_ok=True)
    
    assets = {
        # DINO-DETR Architecture Diagram
        "dino_detr_framework.png": "https://raw.githubusercontent.com/IDEA-Research/DINO/main/figs/framework.png",
        # DINO-DETR Logo
        "dino_detr_logo.png": "https://raw.githubusercontent.com/IDEA-Research/DINO/main/figs/dinosaur.png",
        # DINOv2 Overview/Architecture Figure 1 from arXiv HTML
        "dinov2_overview.jpg": "https://arxiv.org/html/2304.07193/new-figure-1.jpg",
        # DINOv3 Figure 1 from arXiv HTML (representing Gram anchoring or introduction)
        "dinov3_fig1.png": "https://arxiv.org/html/2508.10104/x1.png",
        # DINOv3 PCA visual comparison for features
        "dinov3_cat_pca_1.jpg": "https://arxiv.org/html/2508.10104/figures/introduction/new_cat_pca_1.lr.jpg",
        "dinov3_cat_pca_2.jpg": "https://arxiv.org/html/2508.10104/figures/introduction/new_cat_pca_2.lr.jpg",
        # DINOv3 Satellite comparison
        "dinov3_satellite_dinov2.png": "https://arxiv.org/html/2508.10104/figures/satellite/dinov2.png",
        "dinov3_satellite_dinov3.png": "https://arxiv.org/html/2508.10104/figures/satellite/dinov3.png"
    }
    
    for filename, url in assets.items():
        save_path = os.path.join(img_dir, filename)
        download_file(url, save_path)

if __name__ == "__main__":
    main()
