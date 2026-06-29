import torch
from stable_diffusion import StableDiffusion
from diffusion_policy import DiffusionPolicy
from flow_matching_policy import FlowMatchingPolicy
from dit import DiffusionTransformer
from video_dit import VideoDiT
from rdt import RDTRunner

def run_stable_diffusion_demo(device):
    print("\n--- Running Stable Diffusion (Latent Diffusion) Demo ---")
    model = StableDiffusion(cond_dim=128, latent_channels=4).to(device)
    model.eval()
    
    # Inputs: B=2, 3-channel RGB image (256x256), text prompt length=10
    images = torch.randn(2, 3, 256, 256, device=device)
    text_tokens = torch.randint(0, 1000, (2, 10), device=device)
    timesteps = torch.randint(0, 1000, (2,), device=device)
    
    # 1. Test training forward pass (noise prediction loss)
    loss = model(images, text_tokens, timesteps)
    
    # 2. Test generation loop
    with torch.no_grad():
        generated = model.generate(text_tokens, latent_shape=(2, 4, 32, 32), num_inference_steps=5)
        
    print(f"Input Images Shape: {images.shape}")
    print(f"Input Text Tokens Shape: {text_tokens.shape}")
    print(f"Training Noise Prediction Loss: {loss.item():.4f}")
    print(f"Generated Output Images Shape: {generated.shape}")
    print("Stable Diffusion forward and generation executed successfully!")


def run_diffusion_policy_demo(device):
    print("\n--- Running Diffusion Policy (RSS 2023) Demo ---")
    # Action dimension=2, robot state=6, observation feature=128
    model = DiffusionPolicy(action_dim=2, state_dim=6, obs_dim=128).to(device)
    model.eval()
    
    # Inputs:
    camera_img = torch.randn(2, 3, 112, 112, device=device)
    robot_state = torch.randn(2, 6, device=device)
    actions = torch.randn(2, 16, 2, device=device)
    timesteps = torch.randint(0, 100, (2,), device=device)
    
    # 1. Test training forward pass
    loss = model(camera_img, robot_state, actions, timesteps)
    
    # 2. Test action prediction (inference loop)
    with torch.no_grad():
        pred_actions = model.predict_action(camera_img, robot_state, action_shape=(2, 16, 2), num_inference_steps=10)
        
    print(f"Input Camera Images Shape: {camera_img.shape}")
    print(f"Input Robot Proprioceptive State Shape: {robot_state.shape}")
    print(f"Input Action Trajectory Shape: {actions.shape}")
    print(f"Training Trajectory Noise Loss: {loss.item():.4f}")
    print(f"Inference Generated Actions Shape: {pred_actions.shape}")
    print("Diffusion Policy executed successfully!")


def run_flow_matching_policy_demo(device):
    print("\n--- Running Flow Matching Policy (ICLR 2023) Demo ---")
    model = FlowMatchingPolicy(action_dim=2, state_dim=6, obs_dim=128).to(device)
    model.eval()
    
    # Inputs:
    camera_img = torch.randn(2, 3, 112, 112, device=device)
    robot_state = torch.randn(2, 6, device=device)
    actions = torch.randn(2, 16, 2, device=device)
    
    # 1. Test training forward pass (velocity regression loss)
    loss = model(camera_img, robot_state, actions)
    
    # 2. Test action prediction (ODE Euler integration loop)
    with torch.no_grad():
        pred_actions = model.predict_action(camera_img, robot_state, action_shape=(2, 16, 2), num_euler_steps=5)
        
    print(f"Input Camera Images Shape: {camera_img.shape}")
    print(f"Input Robot Proprioceptive State Shape: {robot_state.shape}")
    print(f"Input Action Trajectory Shape: {actions.shape}")
    print(f"Training Vector Field Velocity Loss: {loss.item():.4f}")
    print(f"ODE Integrated Actions Shape: {pred_actions.shape}")
    print("Flow Matching Policy executed successfully!")


def run_dit_demo(device):
    print("\n--- Running Diffusion Transformer (DiT) (ICCV 2023) Demo ---")
    model = DiffusionTransformer(input_size=32, patch_size=2, in_channels=4, hidden_size=128, num_heads=4, depth=2, num_classes=10).to(device)
    model.eval()
    
    # Inputs: B=2
    z_t = torch.randn(2, 4, 32, 32, device=device)
    t = torch.randint(0, 1000, (2, 1), device=device)
    class_labels = torch.randint(0, 10, (2,), device=device)
    
    with torch.no_grad():
        noise_pred = model(z_t, t, class_labels)
        
    print(f"Input Latents Shape: {z_t.shape}")
    print(f"Input Timesteps Shape: {t.shape}")
    print(f"Input Class Labels: {class_labels.tolist()}")
    print(f"Output Predicted Noise Shape: {noise_pred.shape}")
    print("DiT forward pass executed successfully!")


def run_video_dit_demo(device):
    print("\n--- Running Video Diffusion Transformer (Video DiT) Demo ---")
    model = VideoDiT(latent_shape=(8, 4, 16, 16), patch_size=(2, 2, 2), hidden_size=128, cond_dim=128, num_heads=4, depth=2).to(device)
    model.eval()
    
    # Inputs: B=2
    z_t = torch.randn(2, 8, 4, 16, 16, device=device)
    t = torch.randint(0, 1000, (2, 1), device=device)
    text_cond = torch.randn(2, 10, 128, device=device)
    
    # 1. Test Spatiotemporal VAE encode/decode
    raw_video = torch.randn(2, 16, 3, 128, 128, device=device)
    with torch.no_grad():
        latents_vae = model.vae_3d.encode(raw_video)
        reconstructed_video = model.vae_3d.decode(latents_vae)
        
    # 2. Test Video DiT forward pass
    with torch.no_grad():
        noise_pred = model(z_t, t, text_cond)
        
    print(f"Input Video Frames Shape: {raw_video.shape}")
    print(f"Encoded 3D VAE Latents Shape: {latents_vae.shape}")
    print(f"Decoded Video Frames Shape: {reconstructed_video.shape}")
    print(f"Video DiT Input Latents Shape: {z_t.shape}")
    print(f"Video DiT Predicted Noise Shape: {noise_pred.shape}")
    print("Video DiT executed successfully!")


def run_rdt_demo(device):
    print("\n--- Running Robotics Diffusion Transformer (RDT-1B) Demo ---")
    # action_dim=128 (unified action space), prediction horizon=64 steps
    # lang_token_dim=4096 (T5-XXL), img_token_dim=1152 (SigLIP), state_token_dim=128 (proprioception)
    model = RDTRunner(
        action_dim=128, pred_horizon=64,
        lang_token_dim=4096, img_token_dim=1152, state_token_dim=128,
        max_lang_cond_len=32, img_cond_len=196
    ).to(device)
    model.eval()
    
    # Mock inputs: B=2
    actions = torch.randn(2, 64, 128, device=device)
    t = torch.randint(0, 100, (2, 1), device=device)
    lang_cond = torch.randn(2, 32, 4096, device=device)
    img_cond = torch.randn(2, 196, 1152, device=device)
    state_traj = torch.randn(2, 1, 128, device=device)
    
    # 1. Test training forward pass (compute loss)
    loss = model(actions, t, lang_cond, img_cond, state_traj)
    
    # 2. Test action sampling
    with torch.no_grad():
        pred_actions = model.predict_action(lang_cond, img_cond, state_traj, num_inference_steps=5)
        
    print(f"Input Unified Actions Shape: {actions.shape}")
    print(f"Input T5 Language Cond Shape: {lang_cond.shape}")
    print(f"Input SigLIP Vision Cond Shape: {img_cond.shape}")
    print(f"Input Proprioception State Shape: {state_traj.shape}")
    print(f"Training Trajectory Loss: {loss.item():.4f}")
    print(f"Generated Actions Trajectory Shape: {pred_actions.shape}")
    print("RDT-1B model executed successfully!")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_stable_diffusion_demo(device)
    run_diffusion_policy_demo(device)
    run_flow_matching_policy_demo(device)
    run_dit_demo(device)
    run_video_dit_demo(device)
    run_rdt_demo(device)
    
    print("\nAll Generative, Policy, and Foundation model demos completed successfully!")
