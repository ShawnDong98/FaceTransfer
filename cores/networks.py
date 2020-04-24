import torch
import torch.nn as nn

import os
import os.path as osp
import sys

pwd_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(pwd_path)

from spectral import SpectralNorm
from self_attention import Self_Attn
from module import *
from dataloader import Data_Loader


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Conv(3, 128), # 32
            Conv(128, 256), # 16
            Conv(256, 512), # 8
            Conv(512, 1024), # 4
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1024 * 4 * 4, latent_dim), # 1024 x 4 x 4 -> 1024
            nn.Linear(latent_dim, 4 * 4 * 1024), # 1024 -> 4 x 4 x 1024
        )

        self.upscale = upscale(1024, 512)

    def forward(self, input):
        out = self.conv_layers(input)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        out = out.view(input.size(0), 1024, 4, 4)
        out = self.upscale(out)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            upscale(512, 256), # 8
            upscale(256, 128), # 16
            upscale(128, 64), # 32
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(64, 3, 5, 1, 2),
            nn.Tanh()
        )
        self.mask_layer = nn.Sequential(
            nn.Conv2d(64, 1, 5, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.layers(input)
        mask = self.mask_layer(out)
        out = self.last_layer(out)
        

        return mask, out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,  use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

if __name__ == "__main__":

    src_dataset = Data_Loader(64, 'D:\Deepfake\data\data_src\sqys\lyf', 4)
    src_loder = src_dataset.loader()

    src_loader = iter(src_loder)

    src_warp, src_target = next(src_loader)
    src_warp = src_warp.cuda()
    src_target = src_target.cuda()

    E = Encoder(128).cuda()
    D = Decoder().cuda()

    latent = E(src_warp)
    img = D(latent)

    critic = nn.L1Loss()
    optim = torch.optim.Adam(E.parameters(), 0.01, [0, 0.999])

    loss = critic(img, src_target)
    
    optim.zero_grad()
    loss.backward()
    optim.step()

    for name, parms in E.named_parameters():
	    print('net-->name:', name, '-->grad_requirs:', parms.requires_grad, '--werms.datight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))





