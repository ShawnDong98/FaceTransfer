import torch
import torch.nn as nn

from math import sqrt

from spectral import SpectralNorm

def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate



class PixelShuffler(nn.Module):
    def __init__(self, size=(2, 2)):
        super().__init__()
        self.size = tuple(size)

    def forward(self, input):
        N, C, H, W = input.size()
        rH, rW = self.size
        oH, oW = H * rH, W * rW
        oC = C // (rH * rW)

        out = input.reshape(N, rH, rW, oC, H, W)
        out = out.permute(0, 3, 4, 1, 5, 2)
        out = out.reshape(N, oC, oH, oW)

        return out



def Conv(c_in, c_out, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True)
    )


def upscale(c_in, c_out, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out * 4, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(c_out * 4),
        nn.LeakyReLU(0.2, inplace=True),
        PixelShuffler()
    )




if __name__ == "__main__":
    t = torch.randn(3, 3, 2, 2)
    net = test()
    t1 = net(t)


    target = torch.ones((3, 5, 4, 4))
    optim = torch.optim.Adam(net.parameters(), 0.01, [0, 0.999])

    critic = nn.L1Loss()

    loss = critic(t1, target)

    optim.zero_grad()
    loss.backward()
    optim.step()

    for name, parms in net.named_parameters():
	    print('net-->name:', name, '-->grad_requirs:', parms.requires_grad, '--werms.datight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))




