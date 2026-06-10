import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121, DenseNet121_Weights
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils import BinEmbedding, get_transforms, prob_to_bin

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/cxr-data-set"
OUTPUT_DIR = "/kaggle/working/cxr_sd_finetuned"
PRETRAINED = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_SIZE = 512
EPOCHS     = 20
BATCH_SIZE = 4
LR         = 1e-5
P_UNCOND   = 0.1   # CFG dropout probability
SAVE_EVERY = 5
LOG_EVERY  = 50

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Classifier — provides probability bins as conditioning signal ──────────────
# This is what replaces the raw class label from the previous version.
# The classifier runs on each real image and produces p(PNEUMONIA),
# which gets discretized into a bin and used to condition the UNet.
classifier = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
classifier.classifier = torch.nn.Linear(classifier.classifier.in_features, 1)
classifier = classifier.to(device).eval()
for param in classifier.parameters():
    param.requires_grad = False
# NOTE: ideally load a checkpoint trained on your CXR data here:
# classifier.load_state_dict(torch.load("/kaggle/working/classifier.pt"))

# ── Diffusion models ──────────────────────────────────────────────────────────
vae             = AutoencoderKL.from_pretrained(PRETRAINED, subfolder="vae").to(device)
unet            = UNet2DConditionModel.from_pretrained(PRETRAINED, subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED, subfolder="scheduler")
bin_emb         = BinEmbedding().to(device)

for param in vae.parameters():
    param.requires_grad = False
vae.eval()

# ── Data ──────────────────────────────────────────────────────────────────────
full_dataset = ImageFolder(DATA_DIR, transform=get_transforms(IMAGE_SIZE))
indices      = [i for i, (_, lbl) in enumerate(full_dataset.samples)
                if full_dataset.classes[lbl] in {"NORMAL", "PNEUMONIA"}]
dataset      = Subset(full_dataset, indices)

val_size     = max(1, int(0.1 * len(dataset)))
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
trainable    = list(unet.parameters()) + list(bin_emb.parameters())
optimizer    = torch.optim.AdamW(trainable, lr=LR)
total_steps  = EPOCHS * len(train_loader)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
scaler       = torch.cuda.amp.GradScaler()

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    unet.train(); bin_emb.train()
    train_losses = []

    for batch_idx, (pixel_values, _) in enumerate(train_loader):
        pixel_values = pixel_values.to(device)

        with torch.cuda.amp.autocast():
            # Get probability bins from classifier — this is the conditioning signal
            with torch.no_grad():
                logits = classifier(pixel_values).squeeze(1)       # (B,)
                probs  = torch.sigmoid(logits)                     # (B,) in [0,1]
                bins   = prob_to_bin(probs)                        # (B,) in [0,9]

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # CFG dropout — randomly replace bin with null token
            bins_in = bins.clone()
            bins_in[torch.rand(bins.size(0), device=device) < P_UNCOND] = bin_emb.null_token

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=bin_emb(bins_in)).sample
            loss       = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer); scaler.update(); lr_scheduler.step()
        train_losses.append(loss.item())

        if batch_idx % LOG_EVERY == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}  avg_prob: {probs.mean().item():.3f}")

    # Validation
    unet.eval(); bin_emb.eval()
    val_losses = []
    with torch.no_grad():
        for pixel_values, _ in val_loader:
            pixel_values  = pixel_values.to(device)
            logits        = classifier(pixel_values).squeeze(1)
            probs         = torch.sigmoid(logits)
            bins          = prob_to_bin(probs)
            latents       = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            noise         = torch.randn_like(latents)
            timesteps     = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred    = unet(noisy_latents, timesteps, encoder_hidden_states=bin_emb(bins)).sample
            val_losses.append(F.mse_loss(noise_pred, noise).item())

    print(f"\nEpoch {epoch} | Train: {sum(train_losses)/len(train_losses):.4f}  Val: {sum(val_losses)/len(val_losses):.4f}\n")

    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        ckpt = Path(OUTPUT_DIR) / f"checkpoint_epoch{epoch}"
        ckpt.mkdir(exist_ok=True)
        unet.save_pretrained(ckpt / "unet")
        torch.save(bin_emb.state_dict(), ckpt / "bin_emb.pt")
        print(f"Saved → {ckpt}")