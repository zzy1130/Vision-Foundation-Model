import torch
from stable_diffusion import StableDiffusion
from diffusion_policy import DiffusionPolicy
from flow_matching_policy import FlowMatchingPolicy

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
    # Latent shape: (B, C_latent, H_latent, W_latent) -> (2, 4, 32, 32)
    # Output generated image shape: (2, 3, 256, 256)
    with torch.no_grad():
        generated = model.generate(text_tokens, latent_shape=(2, 4, 32, 32), num_inference_steps=5)
        
    print(f"Input Images Shape: {images.shape}")
    print(f"Input Text Tokens Shape: {text_tokens.shape}")
    print(f"Training Noise Prediction Loss: {loss.item():.4f}")
    print(f"Generated Output Images Shape: {generated.shape}")
    print("Stable Diffusion forward and generation executed successfully!")


def run_diffusion_policy_demo(device):
    print("\n--- Running Diffusion Policy (RSS 2023) Demo ---")
    # Action dimension=2 (e.g. 2D end-effector position), robot state=6, observation feature=128
    model = DiffusionPolicy(action_dim=2, state_dim=6, obs_dim=128).to(device)
    model.eval()
    
    # Inputs:
    camera_img = torch.randn(2, 3, 112, 112, device=device)
    robot_state = torch.randn(2, 6, device=device)
    # Target action sequence: (B, T_horizon, D_action) -> 16 steps trajectory
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_stable_diffusion_demo(device)
    run_diffusion_policy_demo(device)
    run_flow_matching_policy_demo(device)
    
    print("\nAll Diffusion and Policy model demos completed successfully!")
