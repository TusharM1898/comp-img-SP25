import torch
from torch import nn
from torchvision.models import vgg19


class ESRGANLoss(nn.Module):
    def __init__(self):
        super(ESRGANLoss, self).__init__()
        vgg = vgg19(weights="IMAGENET1K_V1").features[:36]  # Pretrained VGG19
        self.vgg_features = nn.Sequential(*vgg).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, sr, hr, real_out, fake_out):
        # Perceptual Loss
        sr_features = self.vgg_features(sr)
        hr_features = self.vgg_features(hr)
        perceptual_loss = self.mse_loss(sr_features, hr_features)

        # Adversarial Loss (Relativistic)
        adversarial_loss = torch.mean(torch.log(1 - real_out + fake_out))

        # Pixel Loss
        pixel_loss = self.mse_loss(sr, hr)

        # Total Loss
        return pixel_loss + 0.001 * adversarial_loss + 0.006 * perceptual_loss
