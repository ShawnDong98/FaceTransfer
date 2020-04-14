import torch
import torch.nn as nn

from math import sqrt

from spectral import SpectralNorm


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def ConvBN(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, out_channel,
                               kernel, stride, padding)),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2)
    )


def Conv(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, out_channel,
                               kernel, stride, padding)),
        nn.LeakyReLU(0.2)
    )


def ConvTrans(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
        SpectralNorm(
            nn.ConvTranspose2d(in_channel, out_channel,
                               kernel, stride, padding)
        ),
    )


class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class StyleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=1, padding=0, style_dim=256):
        super().__init__()

        self.conv = ConvTrans(in_channel, out_channel,
                              kernel_size, stride, padding)
        self.adain = AdaIN(out_channel, style_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv(input)
        out = self.lrelu(out)
        out = self.adain(out, style)

        return out
