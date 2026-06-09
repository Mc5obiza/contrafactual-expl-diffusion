import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from utils import ClassEmbedding, IDX_TO_CLASS, load_image, tensor_to_pil

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = "/kaggle/working/cxr_sd_finetuned/checkpoint_epoch20"
PRETRAINED         = "runwayml/stable-diffusion-v1-5"
IMAGE_PATH         = "/kaggle/input/cxr-data-set/PNEUMONIA/person1_bacteria_1.jpeg"
SOURCE_LABEL       = 1        # 1 = PNEUMONIA
TARGET_LABEL       = 0        # 0 = NORMAL
IMAGE_SIZE         = 512
NUM_STEPS          = 50
GUIDANCE_SCALE     = 7.5
INVERSION_STRENGTH = 0.8      # 0=no change  1=full noise — 0.7-0.8 recommended
OUTPUT_DIR         = "/kaggle/working/counterfactuals"

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = Path(CHECKPOINT_DIR)

vae       = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device).eval()
unet      = UNet2DConditionModel.from_pretrained(ckpt / "unet").to(device).eval()
scheduler = DDIMScheduler.from_pretrained(PRETRAINED, subfolder="scheduler")
class_emb = ClassEmbedding().to(device).eval()
class_emb.load_state_dict(torch.load(ckpt / "class_emb.pt", map_location=device))

source_emb = class_emb(torch.tensor([SOURCE_LABEL], device=device))
target_emb = class_emb(torch.tensor([TARGET_LABEL], device=device))
null_emb   = class_emb(torch.tensor([class_emb.null_token], device=device))

# ── DDIM Inversion ────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_inversion(latent, source_emb):
    scheduler.set_timesteps(NUM_STEPS)
    trajectory, z = [latent], latent
    for t in scheduler.timesteps.flip(0):  # forward: t=0 → T
        noise_pred = unet(z, t.unsqueeze(0).to(device), encoder_hidden_states=source_emb).sample
        z = scheduler.step(noise_pred, t, z).prev_sample
        trajectory.append(z)
    return trajectory


# ── CFG Decoding ──────────────────────────────────────────────────────────────
@torch.no_grad()
def cfg_decode(z_noisy, start_step):
    scheduler.set_timesteps(NUM_STEPS)
    z = z_noisy
    for t in scheduler.timesteps[start_step:]:
        noise_pred = unet(
            torch.cat([z, z]), t.unsqueeze(0).expand(2).to(device),
            encoder_hidden_states=torch.cat([null_emb, target_emb])
        ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        z = scheduler.step(noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond), t, z).prev_sample
    return z


# ── Pipeline ──────────────────────────────────────────────────────────────────
original = load_image(IMAGE_PATH, IMAGE_SIZE).to(device)

with torch.no_grad():
    latent = vae.encode(original).latent_dist.sample() * vae.config.scaling_factor

inversion_step = int(INVERSION_STRENGTH * NUM_STEPS)
trajectory     = ddim_inversion(latent, source_emb)
z_counter      = cfg_decode(trajectory[inversion_step], start_step=NUM_STEPS - inversion_step)

with torch.no_grad():
    counterfactual = vae.decode(z_counter / vae.config.scaling_factor).sample

diff      = (counterfactual - original).abs()
diff_norm = diff / diff.max()

# ── Visualize ─────────────────────────────────────────────────────────────────
src_name, tgt_name = IDX_TO_CLASS[SOURCE_LABEL], IDX_TO_CLASS[TARGET_LABEL]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tensor_to_pil(original),      cmap="gray"); axes[0].set_title(f"Original ({src_name})");       axes[0].axis("off")
axes[1].imshow(tensor_to_pil(counterfactual), cmap="gray"); axes[1].set_title(f"Counterfactual ({tgt_name})"); axes[1].axis("off")
im = axes[2].imshow(diff_norm.squeeze(0).mean(0).cpu().numpy(), cmap="hot")
axes[2].set_title("Difference Map (Explanation)"); axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.suptitle(f"{src_name} → {tgt_name} | strength={INVERSION_STRENGTH}  guidance={GUIDANCE_SCALE}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/explanation_{src_name}_to_{tgt_name}.png", dpi=150)
plt.show()