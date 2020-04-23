import torch
import torch.nn as nn

from spectral import SpectralNorm
from self_attention import Self_Attn
from module import *


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            ConvBN(3, 32, 4, 2, 1),  # 64 x 64
            ConvBN(32, 64, 4, 2, 1),  # 32 x 32
            ConvBN(64, 128, 4, 2, 1),  # 16 x 16
            ConvBN(128, 256, 4, 2, 1),  # 8 x 8
            ConvBN(256, 512, 4, 2, 1),  # 4 x 4
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
    def __init__(self, latent_dim=256):
        super(Generator, self).__init__()

        self.ConstantInput = nn.Parameter(torch.zeros(1, latent_dim, 4, 4))

        self.layers = nn.ModuleList(
            [
                # StyleConvBlock(latent_dim, 1024, 4, 1, 0),  # 4 x 4
                StyleConvBlock(latent_dim, 512, 3, 1, 1),  # 8 x 8
                StyleConvBlock(512, 256, 4, 2, 1),  # 16 x 16
                StyleConvBlock(256, 128, 4, 2, 1),  # 32 x 32
                StyleConvBlock(128, 64, 4, 2, 1),  # 64 x 64
                StyleConvBlock(64, 32, 4, 2, 1),  # 128 x 128
            ]

        )

        self.last_layer = nn.Sequential(
            ConvTrans(32, 3, 4, 2, 1),  # 128 x 128
            nn.Tanh()
        )

        self.attn1 = Self_Attn(64)
        self.attn2 = Self_Attn(32)

    def forward(self, latent):
        out = self.ConstantInput.repeat(latent.size(0), 1, 1, 1)
        #print(out)
        # print("len_self_layers: ", len(self.layers))
        for i in range(len(self.layers)):
            if (i == len(self.layers)-2) or i == (len(self.layers)-1):
                if i == (len(self.layers)-2):
                    out = self.layers[i](out, latent)
                    # print(out.shape)
                    out = self.attn1(out)
                if i == (len(self.layers)-1):
                    out = self.layers[i](out, latent)
                    out = self.attn2(out)
            else:
                out = self.layers[i](out, latent)
                print(out.mean())

        out = self.last_layer(out)
        
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            Conv(3, 32, 4, 2, 1),  # 64 x 64
            Conv(32, 64, 4, 2, 1),  # 64 x 64
            Conv(64, 128, 4, 2, 1),  # 32 x 32
            Conv(128, 256, 4, 2, 1),  # 16 x 16
            Self_Attn(256),
            Conv(256, 512, 4, 2,  1),  # 8 x 8
            Self_Attn(512)
            # Conv(512, 1024, 4, 2, 1),  # 4 x 4

        )
        self.last_layer = Conv(512, 1, 3, 1, 1)

    def forward(self, x):
        out = self.layers(x)
        out = self.last_layer(out)

        return out.squeeze()

    def similarity(self, x):
        features = self.layers(x)

        return features
