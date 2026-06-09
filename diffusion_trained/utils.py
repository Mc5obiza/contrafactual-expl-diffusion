import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


class ClassEmbedding(nn.Module):
    """
    Maps {0=NORMAL, 1=PNEUMONIA, 2=NULL} → (B, 1, 768)
    Index 2 is the null token used during CFG dropout.
    """
    def __init__(self, num_classes=2, emb_dim=768):
        super().__init__()
        self.null_token = num_classes
        self.embed = nn.Embedding(num_classes + 1, emb_dim)

    def forward(self, labels):
        return self.embed(labels).unsqueeze(1)  # (B, 1, 768)


def get_transforms(image_size=512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_image(path, image_size=512):
    """Load a single image as a (1, 3, H, W) tensor in [-1, 1]."""
    tfm = get_transforms(image_size)
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil(t):
    """Convert a (1, 3, H, W) tensor in [-1, 1] to a PIL image."""
    t = (t.squeeze(0).clamp(-1, 1) + 1) / 2
    t = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
    from PIL import Image
    return Image.fromarray(t)