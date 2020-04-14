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
        SpectralNorm(nn.ConvTranspose2d(
            in_channel, out_channel, kernel, stride, padding)),
        nn.InstanceNorm2d(out_channel),
        nn.LeakyReLU(0.2)
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = Conv(in_channel, out_channel, 3, 1, 1)
        self.conv2 = Conv(out_channel, out_channel, 4, 2, 1)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class ConvTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.convtrans1 = ConvTrans(in_channel, out_channel, 3, 1, 1)
        self.convtrans2 = ConvTrans(out_channel, out_channel, 4, 2, 1)

    def forward(self, input):
        out = self.convtrans1(input)
        out = self.convtrans2(out)

        return out
