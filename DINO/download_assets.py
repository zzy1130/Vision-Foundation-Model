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
    
    # 1. DINO v1 Illustration GIF
    dino_v1_url = "https://raw.githubusercontent.com/facebookresearch/dino/main/.github/dino.gif"
    download_file(dino_v1_url, os.path.join(img_dir, "dino_illustration.gif"))
    
    # 2. DINOv2 PCA Feature Visualization Video/Asset
    # This is a public asset showing semantic segmentation of patch PCA features
    dinov2_url = "https://github.com/facebookresearch/dinov2/assets/60359573/f168823e-7922-415a-b429-578badf5c356"
    # Try downloading it (saving as .mp4 since github video assets are typically mp4)
    download_file(dinov2_url, os.path.join(img_dir, "dinov2_demo.mp4"))
    
    # 3. Grounding DINO Logo
    grounding_logo_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/grounding_dino_logo.png"
    download_file(grounding_logo_url, os.path.join(img_dir, "grounding_dino_logo.png"))
    
    # 4. Grounding DINO Architecture Diagram
    grounding_arch_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/arch.png"
    download_file(grounding_arch_url, os.path.join(img_dir, "grounding_dino_architecture.png"))
    
    # 5. Grounding DINO Detection Demo Image
    grounding_demo_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/cat_dog.jpeg"
    download_file(grounding_demo_url, os.path.join(img_dir, "grounding_dino_demo.jpg"))
    
    # 6. Grounding DINO Hero Figure Diagram
    grounding_hero_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/.asset/hero_figure.png"
    download_file(grounding_hero_url, os.path.join(img_dir, "grounding_dino_hero.png"))

if __name__ == "__main__":
    main()
