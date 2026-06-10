import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

NUM_BINS = 10  # discretize classifier probability [0,1] into 10 bins
# bin boundaries: 0.0, 0.1, 0.2, ..., 0.9
BOUNDARIES = torch.linspace(0, 1, NUM_BINS + 1)[1:-1]  # (9,) interior edges


def prob_to_bin(prob):
    """
    prob : (B,) float tensor of classifier probabilities in [0, 1]
    returns: (B,) long tensor of bin indices in [0, NUM_BINS-1]
    """
    return torch.bucketize(prob.cpu(), BOUNDARIES).to(prob.device)


class BinEmbedding(nn.Module):
    """
    Core conditioning module — mirrors the cGAN probability bin conditioning.
    Maps a bin index [0..NUM_BINS-1] or NULL token → (B, 1, 768).

    During training: bin comes from classifier(real_image)
    At counterfactual time: bin comes from the TARGET probability we want
    """
    def __init__(self, num_bins=NUM_BINS, emb_dim=768):
        super().__init__()
        self.null_token = num_bins          # index 10 = unconditional
        self.embed = nn.Embedding(num_bins + 1, emb_dim)

    def forward(self, bins):
        return self.embed(bins).unsqueeze(1)  # (B, 1, 768)


def get_transforms(image_size=512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_image(path, image_size=512):
    tfm = get_transforms(image_size)
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil(t):
    t = (t.squeeze(0).clamp(-1, 1) + 1) / 2
    t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


def ddim_inversion(latent, unet, scheduler, source_emb, device, num_steps):
    """
    DDIM Inversion: forward diffusion process to find noise trajectory.
    
    Args:
        latent: (1, C, H, W) latent representation
        unet: UNet2DConditionModel
        scheduler: DDIMScheduler
        source_emb: (1, 1, 768) source embedding
        device: torch device
        num_steps: number of diffusion steps
    
    Returns:
        trajectory: list of latents at each timestep
    """
    scheduler.set_timesteps(num_steps)
    trajectory, z = [latent], latent
    
    for t in scheduler.timesteps.flip(0):  # forward: t=0 → T
        noise_pred = unet(z, t.unsqueeze(0).to(device), encoder_hidden_states=source_emb).sample
        z = scheduler.step(noise_pred, t, z).prev_sample
        trajectory.append(z)
    
    return trajectory


def cfg_decode(z_noisy, start_step, unet, scheduler, null_emb, target_emb, 
               guidance_scale, device, num_steps):
    """
    Classifier-Free Guidance (CFG) decoding: reverse diffusion with guidance.
    
    Args:
        z_noisy: (1, C, H, W) noisy latent to start decoding from
        start_step: int, which step to start from in the scheduler
        unet: UNet2DConditionModel
        scheduler: DDIMScheduler
        null_emb: (1, 1, 768) unconditional embedding
        target_emb: (1, 1, 768) target condition embedding
        guidance_scale: float, CFG strength (7.5 typical)
        device: torch device
        num_steps: number of diffusion steps
    
    Returns:
        z: (1, C, H, W) decoded latent
    """
    scheduler.set_timesteps(num_steps)
    z = z_noisy
    
    for t in scheduler.timesteps[start_step:]:
        noise_pred = unet(
            torch.cat([z, z]),
            t.unsqueeze(0).expand(2).to(device),
            encoder_hidden_states=torch.cat([null_emb, target_emb])
        ).sample
        
        noise_uncond, noise_cond = noise_pred.chunk(2)
        z = scheduler.step(
            noise_uncond + guidance_scale * (noise_cond - noise_uncond), 
            t, z
        ).prev_sample
    
    return z