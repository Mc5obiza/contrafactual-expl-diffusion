import torch
import torch.nn as nn
import torch.nn.functional as F
from from_scratch.models.encoder import Encoder
from from_scratch.models.classifier import Classifier


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.0)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, labels):
        out = self.bn(x)
        gamma, beta = self.embed(labels).chunk(2, dim=1)
        gamma = gamma.view(-1, out.size(1), 1, 1)
        beta = beta.view(-1, out.size(1), 1, 1)
        return gamma * out + beta


class ResBlockUp(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_dim, num_classes)
        self.cbn2 = ConditionalBatchNorm2d(out_dim, num_classes)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_sc = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0)

    def forward(self, x, labels):
        h = self.cbn1(x, labels)
        h = F.relu(h)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.cbn2(h, labels)
        h = F.relu(h)
        h = self.conv2(h)

        sc = self.upsample(x)
        sc = self.conv_sc(sc)
        return h + sc


class ResBlockDown(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.learnable_sc = (in_dim != out_dim) or downsample
        self.conv_sc = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.avgpool(h)

        sc = x
        if self.learnable_sc:
            sc = self.conv_sc(sc)
            if self.downsample:
                sc = self.avgpool(sc)
        return h + sc


class Generator(nn.Module):
    def __init__(self, in_dim=100, num_class=10):
        super().__init__()
        self.input = nn.ConvTranspose2d(in_dim, 1024, kernel_size=4, stride=1, padding=0)
        self.cbn0 = ConditionalBatchNorm2d(1024, num_class)
        self.block1 = ResBlockUp(1024, 512, num_class)
        self.block2 = ResBlockUp(512, 256, num_class)
        self.block3 = ResBlockUp(256, 128, num_class)
        self.block4 = ResBlockUp(128, 64, num_class)
        self.bn_out = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, labels):
        x = self.input(x)
        x = self.cbn0(x, labels)
        x = F.relu(x)
        x = self.block1(x, labels)
        x = self.block2(x, labels)
        x = self.block3(x, labels)
        x = self.block4(x, labels)
        x = F.relu(self.bn_out(x))
        x = torch.tanh(self.conv_out(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, num_class = 2):
        super().__init__()
        self.block1 = ResBlockDown(3, 64, downsample=True)
        self.block2 = ResBlockDown(64, 128, downsample=True)
        self.block3 = ResBlockDown(128, 256, downsample=True)
        self.block4 = ResBlockDown(256, 512, downsample=True)
        self.linear = nn.Linear(512, 1)
        self.embed = nn.Embedding(num_class, 512)

    def forward(self, x, labels):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h)
        h = h.sum(dim=(2, 3))
        out = self.linear(h).view(-1)
        proj = (self.embed(labels) * h).sum(dim=1)
        return out + proj
def calculate_gradient(disc,fake_image,image,labels):
    BATCH_SIZE, C, H, W = image.shape
    segma = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to("cuda")
    x_interpo = segma * image + (1-segma) * fake_image
    x_interpo.requires_grad_(True)
    output = disc(x_interpo,labels)
    gr  = torch.autograd.grad(
        inputs = x_interpo,
        outputs=output,
        retain_graph=True,
        create_graph=True,
        grad_outputs=torch.ones_like(output)
    )[0]
    gr = gr.view(gr.shape[0],-1)
    norm = torch.norm(gr,2,dim = 1)
    return torch.mean((norm -1)**2)
def disc_loss(real_images,labels,gen:Generator,disc:Discriminator,encoder:Encoder,LAMBDA_GRADIENT=10):
    z = encoder(real_images)
    fake_image = gen(z,labels)
    output_fake = disc(fake_image.detach(),labels)
    output_real = disc(real_images,labels)
    loss =- (torch.mean(output_real)-torch.mean(output_fake)) + LAMBDA_GRADIENT*calculate_gradient(disc,fake_image,real_images,labels)
    return loss
def gen_loss(images, source_labels, target_labels, target_cls,gen:Generator,disc:Discriminator,classifier:Classifier,encoder:Encoder,lambda_cgan=1, lambda_recon=10, lambda_cls=1):

    z = encoder(images)

    fake = gen(z, target_labels)
    loss_gan = -torch.mean(disc(fake, target_labels))

    recon = gen(z, source_labels)
    loss_recon = F.l1_loss(recon, images)

    pred = classifier(fake)
    loss_cls = F.binary_cross_entropy_with_logits(pred, target_cls)

    return lambda_cgan * loss_gan + lambda_recon * loss_recon + lambda_cls * loss_cls