import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torchvision.models import densenet121, DenseNet121_Weights
from utils import BinEmbedding, load_image, tensor_to_pil, IDX_TO_CLASS, prob_to_bin, NUM_BINS

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = "/kaggle/working/cxr_sd_finetuned/checkpoint_epoch20"
PRETRAINED         = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_PATH         = "/kaggle/input/cxr-data-set/PNEUMONIA/person1_bacteria_1.jpeg"
IMAGE_SIZE         = 512
NUM_STEPS          = 50
GUIDANCE_SCALE     = 7.5
INVERSION_STRENGTH = 0.8
# Target bin: which probability bin to decode toward.
# bin 0 = p(PNEUMONIA) ≈ 0.0  →  looks healthy
# bin 9 = p(PNEUMONIA) ≈ 1.0  →  looks most pneumonic
# For PNEUMONIA→NORMAL counterfactual, set TARGET_BIN to 0 or 1
TARGET_BIN         = 0
OUTPUT_DIR         = "/kaggle/working/counterfactuals"

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = Path(CHECKPOINT_DIR)

# Classifier — used to get source bin from original image
classifier = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
classifier.classifier = torch.nn.Linear(classifier.classifier.in_features, 1)
classifier = classifier.to(device).eval()
# classifier.load_state_dict(torch.load("/kaggle/working/classifier.pt"))

vae       = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device).eval()
unet      = UNet2DConditionModel.from_pretrained(ckpt / "unet").to(device).eval()
scheduler = DDIMScheduler.from_pretrained(PRETRAINED, subfolder="scheduler")
bin_emb   = BinEmbedding().to(device).eval()
bin_emb.load_state_dict(torch.load(ckpt / "bin_emb.pt", map_location=device))

# ── Get source bin from classifier ───────────────────────────────────────────
original = load_image(IMAGE_PATH, IMAGE_SIZE).to(device)
with torch.no_grad():
    source_prob = torch.sigmoid(classifier(original).squeeze())
    source_bin  = prob_to_bin(source_prob.unsqueeze(0))           # (1,)
    print(f"Source p(PNEUMONIA)={source_prob:.3f}  bin={source_bin.item()}  →  target bin={TARGET_BIN}")

target_bin_t = torch.tensor([TARGET_BIN], device=device)
null_bin_t   = torch.tensor([bin_emb.null_token], device=device)

source_emb = bin_emb(source_bin)      # (1, 1, 768)
target_emb = bin_emb(target_bin_t)    # (1, 1, 768)
null_emb   = bin_emb(null_bin_t)      # (1, 1, 768)

# ── DDIM Inversion ────────────────────────────────────────────────────────────
@torch.no_grad()
def ddim_inversion(latent):
    scheduler.set_timesteps(NUM_STEPS)
    trajectory, z = [latent], latent
    for t in scheduler.timesteps.flip(0):   # forward: t=0 → T
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
            torch.cat([z, z]),
            t.unsqueeze(0).expand(2).to(device),
            encoder_hidden_states=torch.cat([null_emb, target_emb])
        ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        z = scheduler.step(noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond), t, z).prev_sample
    return z


# ── Run pipeline ──────────────────────────────────────────────────────────────
with torch.no_grad():
    latent = vae.encode(original).latent_dist.sample() * vae.config.scaling_factor

inversion_step = int(INVERSION_STRENGTH * NUM_STEPS)
trajectory     = ddim_inversion(latent)
z_counter      = cfg_decode(trajectory[inversion_step], start_step=NUM_STEPS - inversion_step)

with torch.no_grad():
    counterfactual = vae.decode(z_counter / vae.config.scaling_factor).sample

# Difference map — the actual explanation
diff      = (counterfactual - original).abs()
diff_norm = diff / diff.max()

# ── Visualize ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tensor_to_pil(original),       cmap="gray"); axes[0].set_title(f"Original\np(PNEUMONIA)={source_prob:.2f}  bin={source_bin.item()}"); axes[0].axis("off")
axes[1].imshow(tensor_to_pil(counterfactual), cmap="gray"); axes[1].set_title(f"Counterfactual\ntarget bin={TARGET_BIN}");                          axes[1].axis("off")
im = axes[2].imshow(diff_norm.squeeze(0).mean(0).cpu().numpy(), cmap="hot")
axes[2].set_title("Difference Map\n(Explanation)"); axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.suptitle(f"Counterfactual Explanation | strength={INVERSION_STRENGTH}  guidance={GUIDANCE_SCALE}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/explanation_bin{source_bin.item()}_to_bin{TARGET_BIN}.png", dpi=150)
plt.show()