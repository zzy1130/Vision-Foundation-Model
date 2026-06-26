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
    img_dir = "/Users/zhongzhiyi/Vision-Foundation-Model/SAM/images"
    os.makedirs(img_dir, exist_ok=True)
    
    assets = {
        # SAM 1 Architecture Diagram
        "sam1_framework.png": "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/assets/model_diagram.png",
        # SAM 2 Architecture Diagram
        "sam2_framework.png": "https://raw.githubusercontent.com/facebookresearch/sam2/main/assets/model_diagram.png",
        # SAM 3 Architecture Diagram
        "sam3_framework.png": "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/model_diagram.png",
        # SAM 3D Architecture Diagram
        "sam3d_framework.png": "https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/doc/arch.png",
        # SAM 3D Visual Intro
        "sam3d_intro.png": "https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/doc/intro.png",
        # HQ-SAM Architecture Diagram
        "hqsam_framework.png": "https://raw.githubusercontent.com/SysCV/sam-hq/main/figs/sam_vs_hqsam_backbones.png",
        # MobileSAM Architecture Diagram
        "mobilesam_framework.jpg": "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/model_diagram.jpg"
    }
    
    for filename, url in assets.items():
        save_path = os.path.join(img_dir, filename)
        download_file(url, save_path)

if __name__ == "__main__":
    main()
