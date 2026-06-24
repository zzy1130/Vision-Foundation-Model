import os
import requests
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel

def download_image(url, filepath):
    print(f"Downloading sample image from {url}...")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filepath}")
            return True
        else:
            print(f"Failed to download image. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def main():
    # 1. Setup paths and directories
    base_dir = "/Users/zhongzhiyi/Vision-Foundation-Model/CLIP"
    images_dir = os.path.join(base_dir, "demo_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Image URLs
    image_sources = {
        "cat": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "dog": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "car": "https://images.unsplash.com/photo-1542282088-fe8426682b8f?w=500"
    }
    
    local_images = {}
    for name, url in image_sources.items():
        filepath = os.path.join(images_dir, f"{name}.jpg")
        success = download_image(url, filepath)
        if success:
            local_images[name] = filepath
        else:
            print(f"Skipping {name} due to download failure.")
            
    if not local_images:
        print("Error: No sample images could be downloaded. Exiting.")
        return

    # 2. Load SigLIP model from Hugging Face
    # We use google/siglip-base-patch16-224, which is a standard pre-trained SigLIP model
    model_name = "google/siglip-base-patch16-224"
    print(f"Loading pre-trained SigLIP model and processor: {model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    # 3. Task 1: Zero-shot Classification (零样本图像分类)
    print("\n" + "="*50)
    print("Task 1: Zero-shot Classification (零样本图像分类)")
    print("="*50)
    
    # Let's test with the cat image
    cat_img_path = local_images.get("cat")
    if cat_img_path:
        image = Image.open(cat_img_path)
        candidate_labels = [
            "a photo of a fluffy cat",
            "a photo of a cute dog",
            "a photo of a modern car",
            "a photo of a computer screen"
        ]
        
        print(f"Testing image: {cat_img_path}")
        print(f"Candidate labels: {candidate_labels}")
        
        # Preprocess inputs
        inputs = processor(text=candidate_labels, images=image, padding="max_length", return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # In SigLIP, logits are calculated directly as scale * sim + bias
            # We can use sigmoid or softmax over the candidate labels to show class probabilities
            logits_per_image = outputs.logits_per_image  # [1, num_classes]
            probs = torch.sigmoid(logits_per_image) # Sigmoid activation is the signature of SigLIP
            
            # Alternatively, we can normalize across the options using softmax for visualization
            softmax_probs = F.softmax(logits_per_image, dim=-1)
            
        print("\nResults (Sigmoid probabilities):")
        for label, prob in zip(candidate_labels, probs[0].tolist()):
            print(f"  - '{label}': {prob:.4f} (Sigmoid score)")
            
        print("\nResults (Normalized Softmax probabilities):")
        for label, prob in zip(candidate_labels, softmax_probs[0].tolist()):
            print(f"  - '{label}': {prob*100:.2f}% (Softmax confidence)")

    # 4. Task 2: Image-Text Retrieval / Similarity Matching (图文相似度检索)
    print("\n" + "="*50)
    print("Task 2: Image-Text Similarity Matrix (双向图文相似度检索)")
    print("="*50)
    
    # Load all images
    loaded_images = []
    image_names = list(local_images.keys())
    for name in image_names:
        loaded_images.append(Image.open(local_images[name]))
        
    texts = [
        "a chubby cat sitting down comfortably",
        "a playful dog sitting outdoors",
        "a sleek car driving on the road"
    ]
    
    print(f"Evaluating images: {image_names}")
    print(f"Evaluating texts: {texts}")
    
    inputs = processor(text=texts, images=loaded_images, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: shape [num_images, num_texts]
        # In transformers implementation of SigLIP, logits_per_image is calculated
        logits_per_image = outputs.logits_per_image  # [3, 3]
        
        # Calculate similarity scores (cosine similarity * scale + bias)
        # Using sigmoid to squash between 0 and 1
        scores = torch.sigmoid(logits_per_image)
        
    print("\nSimilarity Scores Matrix (Rows: Images, Columns: Texts):")
    print(f"{'Image':<10} | " + " | ".join([f"Text {i+1}" for i in range(len(texts))]))
    print("-" * 55)
    for i, name in enumerate(image_names):
        scores_str = " | ".join([f"{scores[i, j]:.4f}" for j in range(len(texts))])
        print(f"{name:<10} | {scores_str}")
        
    print("\nCross-matching analysis:")
    # Image to Text
    for i, name in enumerate(image_names):
        best_text_idx = scores[i].argmax().item()
        print(f"  Image '{name}' matches best with Text {best_text_idx+1}: '{texts[best_text_idx]}'")
        
    # Text to Image
    for j, text in enumerate(texts):
        best_img_idx = scores[:, j].argmax().item()
        print(f"  Text '{text}' matches best with Image: '{image_names[best_img_idx]}'")

if __name__ == "__main__":
    main()
