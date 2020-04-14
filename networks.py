import torch
import torch.nn as nn

from spectral import SpectralNorm
from self_attention import Self_Attn
from module import *

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            ConvBN(3, 64, 4, 2, 1),  # 64 x 64
            ConvBN(64, 128, 4, 2, 1),  # 32 x 32
            ConvBN(128, 256, 4, 2, 1),  # 16 x 16
            ConvBN(256, 512, 4, 2, 1),  # 8 x 8
            ConvBN(512, 512, 4, 2, 1),  # 4 x 4
        )

        self.mean = nn.Linear(512*4*4, latent_dim)
        self.logvar = nn.Linear(512*4*4, latent_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        latent_feature = self.features(x)
        mean = self.mean(latent_feature.view(batch_size, -1))
        logvar = self.logvar(latent_feature.view(batch_size, -1))
        # log乘以1/2 相当于 var开方
        std = torch.exp(logvar / 2)

        reparametrized_noise = torch.randn(
            (batch_size, self.latent_dim)).to(x.device)
        z = mean + std * reparametrized_noise

        return z, mean, logvar

    def get_feature(self, x):
        batch_size = x.size()[0]
        latent_feature = self.features(x)

        return latent_feature


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            ConvTrans(latent_dim, 512, 4),  # 4 x 4
            ConvTrans(512, 512, 4, 2, 1),  # 8x 8
            ConvTrans(512, 256, 4, 2, 1),  # 16 x 16
            ConvTrans(256, 128, 4, 2, 1),  # 32 x 32
            Self_Attn(128),
            ConvTrans(128, 64, 4, 2, 1),  # 64 x 64
            Self_Attn(64)
        )
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 128 x 128
            nn.Tanh()
        )

    def forward(self, latent):
        z = latent.view(latent.size(0), latent.size(1), 1, 1)
        out = self.layers(z)
        out = self.last_layer(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            Conv(3, 64, 4, 2, 1),  # 64 x 64
            Conv(64, 128, 4, 2, 1),  # 32 x 32
            Conv(128, 256, 4, 2, 1),  # 16 x 16
            Conv(256, 512, 4, 2, 1),  # 8 x 8
            Self_Attn(512),
            Conv(512, 512, 4, 2, 1),  # 4 x 4
            Self_Attn(512)
        )
        self.last_layer = nn.Conv2d(512, 1, 4)

    def forward(self, x):
        out = self.layers(x)
        out = self.last_layer(out)

        return out.squeeze()

    def similarity(self, x):
        features = self.layers(x)

        return features
