import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils import ClassEmbedding, get_transforms

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/cxr-data-set"
OUTPUT_DIR = "/kaggle/working/cxr_sd_finetuned"
PRETRAINED = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512
EPOCHS     = 20
BATCH_SIZE = 4
LR         = 1e-5
P_UNCOND   = 0.1
SAVE_EVERY = 5
LOG_EVERY  = 50

# ── Data — ImageFolder then filter to NORMAL/PNEUMONIA only ──────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

full_dataset = ImageFolder(DATA_DIR, transform=get_transforms(IMAGE_SIZE))

# Keep only NORMAL and PNEUMONIA indices
kept_classes = {"NORMAL", "PNEUMONIA"}
indices = [i for i, (_, lbl) in enumerate(full_dataset.samples)
           if full_dataset.classes[lbl] in kept_classes]
dataset = Subset(full_dataset, indices)
print(f"Using {len(dataset)} images from {kept_classes}")

val_size = max(1, int(0.1 * len(dataset)))
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# ── Models ────────────────────────────────────────────────────────────────────
vae             = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device)
unet            = UNet2DConditionModel.from_pretrained(PRETRAINED, subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED, subfolder="scheduler")
class_emb       = ClassEmbedding().to(device)

for param in vae.parameters():
    param.requires_grad = False
vae.eval()

# ── Optimizer ─────────────────────────────────────────────────────────────────
trainable    = list(unet.parameters()) + list(class_emb.parameters())
optimizer    = torch.optim.AdamW(trainable, lr=LR)
total_steps  = EPOCHS * len(train_loader)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
scaler       = torch.cuda.amp.GradScaler()

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    unet.train(); class_emb.train()
    train_losses = []

    for batch_idx, (pixel_values, labels) in enumerate(train_loader):
        pixel_values = pixel_values.to(device)
        labels       = labels.to(device)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            labels_in = labels.clone()
            labels_in[torch.rand(labels.size(0), device=device) < P_UNCOND] = class_emb.null_token

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=class_emb(labels_in)).sample
            loss       = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer); scaler.update(); lr_scheduler.step()
        train_losses.append(loss.item())

        if batch_idx % LOG_EVERY == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Validation
    unet.eval(); class_emb.eval()
    val_losses = []
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)
            latents      = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise        = torch.randn_like(latents)
            timesteps    = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred   = unet(noisy_latents, timesteps, encoder_hidden_states=class_emb(labels)).sample
            val_losses.append(F.mse_loss(noise_pred, noise).item())

    print(f"\nEpoch {epoch} | Train: {sum(train_losses)/len(train_losses):.4f}  Val: {sum(val_losses)/len(val_losses):.4f}\n")

    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        ckpt = Path(OUTPUT_DIR) / f"checkpoint_epoch{epoch}"
        ckpt.mkdir(exist_ok=True)
        unet.save_pretrained(ckpt / "unet")
        torch.save(class_emb.state_dict(), ckpt / "class_emb.pt")
        print(f"Saved → {ckpt}")