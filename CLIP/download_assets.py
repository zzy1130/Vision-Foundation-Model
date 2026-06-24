import os
import requests

def download_image(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        response = requests.get(url, timeout=15)
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
    os.makedirs("/Users/zhongzhiyi/Vision-Foundation-Model/CLIP/images", exist_ok=True)
    
    # 1. OpenAI CLIP Architecture Image
    clip_url = "https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png"
    download_image(clip_url, "/Users/zhongzhiyi/Vision-Foundation-Model/CLIP/images/openai_clip.png")
    
    # 2. SigLIP Loss Diagram or Comparison from a public URL
    # Let's try downloading a known SigLIP / PaliGemma diagram from Hugging Face
    siglip_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/paligemma/siglip.png"
    success = download_image(siglip_url, "/Users/zhongzhiyi/Vision-Foundation-Model/CLIP/images/siglip.png")
    if not success:
        # Fallback to another public diagram link if needed
        print("SigLIP image download failed, trying fallback...")
        fallback_url = "https://github.com/google-research/big_vision/raw/main/big_vision/configs/proj/image_text/README.md" # (not an image, just checking)
        # Let's find a fallback image link
        # We can also generate an image or use standard diagram or we can draw one or find another link.

if __name__ == "__main__":
    main()
