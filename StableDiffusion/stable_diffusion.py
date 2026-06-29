import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Stable Diffusion.
    Encodes high-resolution images into low-resolution latents, 
    and decodes latents back to image space.
    """
    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        # Encoder: downsamples by 8x (e.g. 512x512 -> 64x64)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_channels * 2, kernel_size=3, stride=2, padding=1) # /8 (outputs mean and logvar)
        )
        
        # Decoder: upsamples by 8x
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),  # x8
            nn.Tanh()  # outputs in [-1, 1]
        )

    def encode(self, x):
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        # Reparameterization trick
        eps = torch.randn_like(mean)
        latent = mean + eps * std
        return latent

    def decode(self, z):
        return self.decoder(z)


class CLIPTextEncoderProxy(nn.Module):
    """
    CLIP Text Encoder Proxy. Maps tokenized text sequences to text embeddings.
    """
    def __init__(self, vocab_size=1000, embed_dim=128):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, S) text token indices
        Returns:
            prompt_embeds: (B, S, C_text) text conditioning features
        """
        x = self.token_embeddings(tokens)
        prompt_embeds = self.transformer(x)
        return prompt_embeds


class CrossAttention2D(nn.Module):
    """
    Cross-Attention Layer to inject text prompt embeddings into the spatial features of the UNet.
    """
    def __init__(self, in_channels, cond_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = in_channels // heads
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(cond_dim, in_channels)
        self.v_proj = nn.Linear(cond_dim, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, H, W) spatial feature map
            cond: (B, S, C_text) text prompt embeddings
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Project Query, Key, Value
        q = self.q_proj(x_flat)  # (B, HW, C)
        k = self.k_proj(cond)    # (B, S, C)
        v = self.v_proj(cond)    # (B, S, C)
        
        # Reshape to multi-head representation
        q = q.reshape(B, H * W, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, HW, head_dim)
        k = k.reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)      # (B, heads, S, head_dim)
        v = v.reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)      # (B, heads, S, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, HW, S)
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        context = context.transpose(1, 2).reshape(B, H * W, C)
        
        out = self.out_proj(context)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return out + x  # Residual connection


class UNet2D(nn.Module):
    """
    Denoising UNet used in Stable Diffusion.
    Predicts the noise added to the image latent, conditioned on step t and text embedding.
    """
    def __init__(self, latent_channels=4, cond_dim=128):
        super().__init__()
        # Time step embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )
        
        # Down blocks
        self.down_conv = nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1)
        self.down_attn = CrossAttention2D(64, cond_dim)
        
        # Bottleneck
        self.mid_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.mid_attn = CrossAttention2D(64, cond_dim)
        
        # Up blocks
        self.up_conv = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.up_attn = CrossAttention2D(64, cond_dim)
        
        self.out_conv = nn.Conv2d(64, latent_channels, kernel_size=3, padding=1)

    def forward(self, z_t, t, cond):
        """
        Args:
            z_t: (B, C_latent, H_latent, W_latent) image latent at step t
            t: (B, 1) timestep values in [0, T]
            cond: (B, S, C_text) text conditioning features
        """
        B, C, H, W = z_t.shape
        
        # Embed time step
        t_embed = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        
        # Down block
        x_down = self.down_conv(z_t)
        # Inject time embedding (scale/shift style)
        x_down = x_down + t_embed
        x_down = self.down_attn(x_down, cond)
        
        # Bottleneck
        x_mid = self.mid_conv(x_down)
        x_mid = self.mid_attn(x_mid, cond)
        
        # Up block (with skip connection from Down)
        x_up = torch.cat([x_mid, x_down], dim=1)
        x_up = self.up_conv(x_up)
        x_up = self.up_attn(x_up, cond)
        
        # Predict noise
        noise_pred = self.out_conv(x_up)
        return noise_pred


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Model) scheduler for Stable Diffusion.
    Manages alpha/beta schedules and noise removal steps.
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Adds noise to the original sample (forward diffusion process).
        Formula: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        device = original_samples.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        Removes noise from the sample (reverse diffusion process).
        """
        device = sample.device
        idx = timestep.item()
        
        beta = self.betas[idx].to(device)
        alpha = self.alphas[idx].to(device)
        alpha_cumprod = self.alphas_cumprod[idx].to(device)
        
        if idx > 0:
            alpha_cumprod_prev = self.alphas_cumprod[idx - 1].to(device)
            # Denoising variance
            variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        else:
            variance = torch.tensor(0.0, device=device)
            
        # Reconstruct sample mean
        pred_original_sample = (sample - torch.sqrt(1.0 - alpha_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        mean = (torch.sqrt(alpha_cumprod_prev if idx > 0 else torch.tensor(1.0, device=device)) * beta * pred_original_sample + 
                torch.sqrt(alpha) * (1.0 - (alpha_cumprod_prev if idx > 0 else torch.tensor(1.0, device=device))) * sample) / (1.0 - alpha_cumprod)
                
        noise = torch.randn_like(sample) if idx > 0 else torch.tensor(0.0, device=device)
        prev_sample = mean + torch.sqrt(variance) * noise
        return prev_sample


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Model) scheduler for accelerated deterministic sampling.
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        device = original_samples.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample, eta=0.0):
        """
        Performs a deterministic DDIM step (eta=0.0 means fully deterministic LDM sampling).
        Formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * noise_pred + sigma_t * noise
        """
        device = sample.device
        idx = timestep.item()
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        alpha_cumprod = alphas_cumprod[idx]
        alpha_cumprod_prev = alphas_cumprod[idx - 1] if idx > 0 else torch.tensor(1.0, device=device)
        
        # Calculate sigma_t based on eta
        sigma_t = eta * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)) * torch.sqrt(1.0 - alpha_cumprod / alpha_cumprod_prev)
        
        # 1. Predict x0 (reconstruct original latent)
        pred_original_sample = (sample - torch.sqrt(1.0 - alpha_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # 2. Reconstruct x_{t-1} direction pointing to x_t
        pred_sample_direction = torch.sqrt(1.0 - alpha_cumprod_prev - sigma_t**2) * model_output
        
        # 3. Combine to get x_{t-1}
        prev_sample = torch.sqrt(alpha_cumprod_prev) * pred_original_sample + pred_sample_direction
        
        if eta > 0.0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + sigma_t * noise
            
        return prev_sample


class StableDiffusion(nn.Module):
    """
    Stable Diffusion (Latent Diffusion) model.
    Integrates VAE, Text Encoder, and Denoising UNet for text-to-image synthesis.
    """
    def __init__(self, cond_dim=128, latent_channels=4):
        super().__init__()
        self.vae = VAE(in_channels=3, latent_channels=latent_channels)
        self.text_encoder = CLIPTextEncoderProxy(vocab_size=1000, embed_dim=cond_dim)
        self.unet = UNet2D(latent_channels=latent_channels, cond_dim=cond_dim)
        self.scheduler = DDPMScheduler()

    def forward(self, images, text_tokens, t):
        """
        Computes the denoising training loss.
        """
        # 1. Encode image to latent space
        latents = self.vae.encode(images)
        
        # 2. Encode text tokens
        cond = self.text_encoder(text_tokens)
        
        # 3. Sample random noise
        noise = torch.randn_like(latents)
        
        # 4. Add noise to latents (forward diffusion)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # 5. Predict noise (UNet reverse diffusion)
        t_float = t.float().view(-1, 1)
        noise_pred = self.unet(noisy_latents, t_float, cond)
        
        # Loss: MSE between actual noise and predicted noise
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def generate(self, text_tokens, latent_shape=(1, 4, 32, 32), num_inference_steps=50):
        """
        Generates an image from text conditioning using the reverse diffusion loop.
        """
        device = text_tokens.device
        B = text_tokens.shape[0]
        
        # 1. Encode text prompt
        cond = self.text_encoder(text_tokens)
        
        # 2. Initialize random noise in latent space
        latents = torch.randn(B, *latent_shape[1:], device=device)
        
        # 3. Denoising loop
        self.scheduler = DDPMScheduler(num_train_timesteps=num_inference_steps)
        for i in reversed(range(num_inference_steps)):
            timestep = torch.tensor([i], device=device)
            t_float = timestep.float().view(-1, 1).expand(B, -1)
            
            with torch.no_grad():
                noise_pred = self.unet(latents, t_float, cond)
                latents = self.scheduler.step(noise_pred, timestep, latents)
                
        # 4. Decode latent back to image space
        with torch.no_grad():
            generated_images = self.vae.decode(latents)
            
        return generated_images
