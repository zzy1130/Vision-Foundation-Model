import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy import ObservationEncoder

class VelocityFieldNet1D(nn.Module):
    """
    1D Temporal CNN to regress the velocity vector field for Flow Matching.
    Matches action trajectories along the time/horizon dimension.
    """
    def __init__(self, action_dim=2, obs_dim=128):
        super().__init__()
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )
        
        # 1D Temporal convolutions
        self.down_conv = nn.Conv1d(action_dim, 64, kernel_size=5, padding=2)
        self.cond_proj = nn.Linear(obs_dim, 64)
        
        self.mid_conv = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.up_conv = nn.Conv1d(64 + 64, 64, kernel_size=5, padding=2)
        self.out_conv = nn.Conv1d(64, action_dim, kernel_size=5, padding=2)

    def forward(self, action_t, t, obs_cond):
        """
        Predicts the velocity vector v_t at step t.
        Args:
            action_t: (B, T_horizon, D_action) current action trajectory state
            t: (B, 1) time variable in [0, 1]
            obs_cond: (B, D_obs) observation condition embedding
        """
        # Conv1D expects (B, D_action, T_horizon)
        x = action_t.transpose(1, 2)
        
        # Embed time t
        t_embed = self.time_mlp(t).unsqueeze(-1)  # (B, 64, 1)
        # Project observation condition
        cond_embed = self.cond_proj(obs_cond).unsqueeze(-1)  # (B, 64, 1)
        
        # Forward layers
        x_down = self.down_conv(x)
        x_down = x_down + t_embed + cond_embed
        
        x_mid = self.mid_conv(x_down)
        x_up = torch.cat([x_mid, x_down], dim=1)
        x_up = self.up_conv(x_up)
        
        out = self.out_conv(x_up)
        return out.transpose(1, 2)  # (B, T_horizon, D_action)


class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching Policy for robot control.
    Generates action trajectories using Conditional Flow Matching (CFM),
    constructing a straight-line probability flow from noise to target actions.
    """
    def __init__(self, action_dim=2, state_dim=6, obs_dim=128, sigma=1e-4):
        super().__init__()
        self.obs_encoder = ObservationEncoder(img_channels=3, state_dim=state_dim, obs_dim=obs_dim)
        self.velocity_net = VelocityFieldNet1D(action_dim=action_dim, obs_dim=obs_dim)
        self.sigma = sigma  # minimal noise scale to prevent singularity at t=0

    def forward(self, camera_img, robot_state, actions):
        """
        Training forward pass. Regresses the vector field using Conditional Flow Matching Loss.
        Formula: Loss = E_{t, a0, a1} || v_t(a_t) - (a_1 - a_0) ||^2
        where a_t = t * a_1 + (1 - (1 - sigma) * t) * a_0
        """
        B, T, D = actions.shape
        device = actions.device
        
        # 1. Encode observations
        obs_cond = self.obs_encoder(camera_img, robot_state)
        
        # 2. Sample random noise a0
        a0 = torch.randn_like(actions)  # (B, T, D)
        a1 = actions  # target action trajectory
        
        # 3. Sample time step t uniformly in [0, 1]
        t = torch.rand(B, 1, device=device)  # (B, 1)
        t_expand = t.unsqueeze(-1)  # (B, 1, 1)
        
        # 4. Compute interpolation path a_t (straight line path)
        # a_t = t * a1 + (1 - (1 - sigma)*t) * a0
        # This interpolates from (approx) a0 at t=0 to a1 at t=1
        a_t = t_expand * a1 + (1.0 - (1.0 - self.sigma) * t_expand) * a0
        
        # 5. Target velocity vector field (u_t = a1 - (1 - sigma) * a0)
        target_velocity = a1 - (1.0 - self.sigma) * a0
        
        # 6. Predict velocity field
        pred_velocity = self.velocity_net(a_t, t, obs_cond)
        
        # Loss: MSE between predicted and target velocity
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    def predict_action(self, camera_img, robot_state, action_shape=(1, 16, 2), num_euler_steps=10):
        """
        Inference loop. Generates action trajectory by integrating the velocity field
        from t=0 to t=1 using Euler integration.
        """
        device = camera_img.device
        B = camera_img.shape[0]
        
        # 1. Encode observation
        obs_cond = self.obs_encoder(camera_img, robot_state)
        
        # 2. Initialize from Gaussian noise at t=0
        a_t = torch.randn(B, *action_shape[1:], device=device)
        
        # 3. ODE Integration (Euler steps)
        dt = 1.0 / num_euler_steps
        for step in range(num_euler_steps):
            # Current time t
            t_val = step * dt
            t = torch.full((B, 1), t_val, device=device)
            
            with torch.no_grad():
                # Predict velocity
                v_t = self.velocity_net(a_t, t, obs_cond)
                # Euler update step
                a_t = a_t + dt * v_t
                
        return a_t
