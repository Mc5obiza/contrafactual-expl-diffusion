import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class EncoderResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.learnable_sc = (in_dim != out_dim) or downsample
        self.conv_sc = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0)
        self.avgpool = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        if self.downsample:
            h = self.avgpool(h)

        sc = x
        if self.learnable_sc:
            sc = self.conv_sc(sc)
            if self.downsample:
                sc = self.avgpool(sc)
        return h + sc


class Encoder(nn.Module):
    def __init__(self, latent_dim = 100):
        super(Encoder, self).__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.block1 = EncoderResBlock(64, 128, downsample=True)
        self.block2 = EncoderResBlock(128, 256, downsample=True)
        self.block3 = EncoderResBlock(256, 512, downsample=True)
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x.unsqueeze(2).unsqueeze(3)
