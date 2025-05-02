import torch
from torch import nn

# Residual-in-Residual Dense Block (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32, num_layers=3):
        super(RRDB, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels + i * growth_channels, growth_channels, kernel_size=3, padding=1)
            for i in range(num_layers)
        ])
        self.final_layer = nn.Conv2d(in_channels + num_layers * growth_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)  # Dense connections
        x = self.final_layer(x)
        return identity + x


# ESRGAN Generator
class ESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, num_blocks=23, num_features=64, growth_channels=32):
        super(ESRGANGenerator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            *[RRDB(num_features, growth_channels) for _ in range(num_blocks)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Conv2d(num_features, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.upsample(x)
        return torch.tanh(self.final_conv(x)) + 1


# Relativistic Discriminator
class ESRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(ESRGANDiscriminator, self).__init__()
        layers = []
        channels = base_channels

        for _ in range(6):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = channels
            channels *= 2

        layers.append(nn.AdaptiveAvgPool2d(1))  # Global average pooling
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels // 2, 1024))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(1024, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
