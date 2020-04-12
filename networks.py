import torch
import torch.nn as nn 
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F 

import numpy as np

from spectral import SpectralNorm
from self_attention import Self_Attn
from loss import *

import time
import datetime

from utils import *

import cv2

def ConvBN(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel, stride, padding)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
    )

def Conv(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel, stride, padding)),
            nn.LeakyReLU(0.2)
    )

def ConvTrans(in_channel, out_channel, kernel=4, stride=1, padding=0):
    return nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            ConvBN(3, 64, 4, 2, 1), # 64 x 64
            ConvBN(64, 128, 4, 2, 1), # 32 x 32
            ConvBN(128, 256, 4, 2, 1), # 16 x 16
            ConvBN(256, 512, 4, 2, 1), # 8 x 8
            ConvBN(512, 512, 4, 2, 1), # 4 x 4
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

        reparametrized_noise = torch.randn((batch_size, self.latent_dim)).to(x.device)
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
                ConvTrans(latent_dim, 512, 4), # 4 x 4
                ConvTrans(512, 512, 4, 2, 1), # 8x 8
                ConvTrans(512, 256, 4, 2, 1), # 16 x 16
                ConvTrans(256, 128, 4, 2, 1), # 32 x 32
                Self_Attn(128),
                ConvTrans(128, 64, 4, 2, 1), # 64 x 64
                Self_Attn(64)
        )
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1), # 128 x 128
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
            Conv(3, 64, 4, 2, 1), #64 x 64
            Conv(64, 128, 4, 2, 1), # 32 x 32
            Conv(128, 256, 4, 2, 1), # 16 x 16
            Conv(256, 512, 4, 2, 1), # 8 x 8
            Self_Attn(512), 
            Conv(512, 512, 4, 2, 1), # 4 x 4
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


import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated

from plot import *

class AE():
    def __init__(self, config, src_dataloader, dst_dataloader, device='cpu'):
        self.device = device
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.config = config

        self.net_init()
        self.src_dataloader = src_dataloader
        self.dst_dataloader = dst_dataloader

        self.e_optimizer = torch.optim.Adam(self.E.parameters(), 0.0001, [0.0, 0.9])
        self.g_src_optimizer = torch.optim.Adam(self.G_src.parameters(), 0.0001, [0.0, 0.9])
        self.d_src_optimizer = torch.optim.Adam(self.D_src.parameters(), 0.0004, [0.0, 0.9])

        self.g_dst_optimizer = torch.optim.Adam(self.G_dst.parameters(), 0.0001, [0.0, 0.9])
        self.d_dst_optimizer = torch.optim.Adam(self.D_dst.parameters(), 0.0004, [0.0, 0.9])

        # Create directories if not exist
        make_folder(self.config.model_save_path)
        make_folder(self.config.sample_path)


    def net_init(self):
        try:
            self.E = Encoder().to(self.device)
            self.E.load_state_dict(torch.load("./models/latest_E.pth"))
            self.G_src = Generator().to(self.device)
            self.G_src.load_state_dict(torch.load("./models/latest_G_src.pth"))
            self.D_src = Discriminator().to(self.device)
            self.D_src.load_state_dict(torch.load("./models/latest_D_src.pth"))
            self.G_dst = Generator().to(self.device)
            self.G_dst.load_state_dict(torch.load("./models/latest_G_dst.pth"))
            self.D_dst = Discriminator().to(self.device)
            self.D_dst.load_state_dict(torch.load("./models/latest_D_dst.pth"))
            self.state = torch.load("./models/state.pth")
            print("model loaded...")
        except:
            print("no pre-trained model...")
            self.E = Encoder().to(self.device)
            self.G_src = Generator().to(self.device)
            self.D_src = Discriminator().to(self.device)
            self.G_dst = Generator().to(self.device)
            self.D_dst = Discriminator().to(self.device)
            self.state = {
                    "iter": 0,
                    "rec_src_loss": [], 
                    "enc_src_loss": [], 
                    "g_src_loss": [], 
                    "d_src_loss": [],
                    "rec_dst_loss": [], 
                    "enc_dst_loss": [], 
                    "g_dst_loss": [], 
                    "d_dst_loss": [],
                    }

    def train(self):
        src_loader = iter(self.src_dataloader)
        dst_loader = iter(self.dst_dataloader)

        fixed_z = torch.randn(self.batch_size, self.latent_dim).to(self.device)

        # Start time
        start_time = time.time()

        

        for step in range(self.state['iter'], self.config.total_iter):
            self.E.train()
            self.G_src.train()
            self.D_src.train()

            self.G_dst.train()
            self.D_dst.train()

            #--------------------------------------------------
            # train src

            try:
                warp_img, target_img = next(src_loader)
                warp_img = warp_img.to(self.device)
                target_img = target_img.to(self.device)
            except:
                src_loader = iter(self.src_dataloader)
                warp_img, target_img = next(src_loader)
                warp_img = warp_img.to(self.device)
                target_img = target_img.to(self.device)
            
            # train Encoder
            latent_z, mean, logvar = self.E(warp_img)
            prior_loss = E_loss(mean, logvar)

            rec_img = self.G_src(latent_z)
            rec_src_loss = Rec_loss(self.D_src, target_img, rec_img)
            self.state['rec_src_loss'].append(float(rec_src_loss.cpu()))

            enc_src_loss = prior_loss + 5 * rec_src_loss
            self.state['enc_src_loss'].append(float(enc_src_loss.cpu()))
            self.e_optimizer.zero_grad()
            enc_src_loss.backward(retain_graph=True)
            self.e_optimizer.step()


            # train Generator
            g_loss = G_loss(self.D_src, self.G_src, rec_img, self.batch_size, self.latent_dim, self.device)
            g_src_loss = 15 * enc_src_loss + g_loss
            self.state['g_src_loss'].append(float(g_src_loss.cpu()))
            self.g_src_optimizer.zero_grad()
            g_src_loss.backward()
            self.g_src_optimizer.step()

            # train Discriminator
            d_src_loss = D_loss(self.D_src, self.G_src, target_img, rec_img, self.latent_dim)
            self.state['d_src_loss'].append(float(d_src_loss.cpu()))
            self.d_src_optimizer.zero_grad()
            d_src_loss.backward()
            self.d_src_optimizer.step()

            #---------------------------------------------------

            #---------------------------------------------------
            # train dst

            try:
                warp_img, target_img = next(dst_loader)
                warp_img = warp_img.to(self.device)
                target_img = target_img.to(self.device)
            except:
                dst_loader = iter(self.dst_dataloader)
                warp_img, target_img = next(dst_loader)
                warp_img = warp_img.to(self.device)
                target_img = target_img.to(self.device)
            
            # train Encoder
            latent_z, mean, logvar = self.E(warp_img)
            prior_loss = E_loss(mean, logvar)

            rec_img = self.G_dst(latent_z)
            rec_dst_loss = Rec_loss(self.D_dst, target_img, rec_img)
            self.state['rec_dst_loss'].append(float(rec_dst_loss.cpu()))

            enc_dst_loss = prior_loss + 5 * rec_dst_loss
            self.state['enc_dst_loss'].append(float(enc_dst_loss.cpu()))
            self.e_optimizer.zero_grad()
            enc_dst_loss.backward(retain_graph=True)
            self.e_optimizer.step()


            # train Generator
            g_loss = G_loss(self.D_dst, self.G_dst, rec_img, self.batch_size, self.latent_dim, self.device)
            g_dst_loss = 15 * rec_dst_loss + g_loss
            self.state['g_dst_loss'].append(float(g_dst_loss.cpu()))
            self.g_dst_optimizer.zero_grad()
            g_dst_loss.backward()
            self.g_dst_optimizer.step()

            # train Discriminator
            d_dst_loss = D_loss(self.D_dst, self.G_dst, target_img, rec_img, self.latent_dim)
            self.state['d_dst_loss'].append(float(d_dst_loss.cpu()))
            self.d_dst_optimizer.zero_grad()
            d_dst_loss.backward()
            self.d_dst_optimizer.step()

            #---------------------------------------------------

            if (step + 1) % 1 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], step [{}/{}],  rec_src_loss: {:.4f}, enc_src_loss: {:.4f}, G_src_loss: {:.4f}, D_src_loss: {:.4f}, rec_dst_loss: {:.4f}, enc_dst_loss: {:.4f}, G_dst_loss: {:.4f}, D_dst_loss: {:.4f}"
                .format(elapsed, step + 1, self.config.total_iter, rec_src_loss, enc_src_loss, g_src_loss, d_src_loss, rec_dst_loss, enc_dst_loss, g_dst_loss, d_dst_loss))

            if (step + 1) % 10 == 0:
                fake_images = self.G_src(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.config.sample_path, 'src_fake.png'))

                fake_images = self.G_dst(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.config.sample_path, 'dst_fake.png'))
            
            if (step+1) % 100 == 0:
                torch.save(self.E.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_E.pth'))
                torch.save(self.G_src.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_G_src.pth'))
                torch.save(self.D_src.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_D_src.pth'))
                torch.save(self.G_dst.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_G_dst.pth'))
                torch.save(self.D_dst.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_D_dst.pth'))
                self.state['iter'] = step + 1
                torch.save(self.state,
                           os.path.join(self.config.model_save_path, 'state.pth'))
                draw_lines()
                self.inference()

    def inference(self):
        self.E.eval()
        self.G_src.eval()
        self.D_src.eval()
        self.G_dst.eval()
        self.D_dst.eval()
        src_loader = iter(self.src_dataloader)
        dst_loader = iter(self.dst_dataloader)
        
        src_warp, src_target = next(src_loader)
        src_warp = src_warp.to(self.device)
        src_target = src_target.to(self.device)
        dst_warp, dst_target = next(dst_loader)
        dst_warp = dst_warp.to(self.device)
        dst_target = dst_target.to(self.device)

        #-----------------------------------------------------
        #src
        #------------------------------------------------------

        src_target = src_target[0].unsqueeze(0)

        z_src, mean_src, logver_src = self.E(src_target)

        rec_src = self.G_dst(z_src)

        src_target = src_target.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        rec_src = rec_src.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5

        src_img = np.concatenate((src_target, rec_src), axis=1)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

        #---------------------------------------------------------
        # dst
        #---------------------------------------------------------
        dst_target = dst_target[0].unsqueeze(0)

        z_dst, mean_dst, logver_dst = self.E(dst_target)

        rec_dst = self.G_src(z_dst)

        dst_target = dst_target.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
        rec_dst = rec_dst.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5

        dst_img = np.concatenate((dst_target, rec_dst), axis=1)
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)

        img = np.concatenate((src_img, dst_img), axis=0)
        img = img * 255
        cv2.imwrite("./samples/swap_img.jpg", img)

import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--total_iter', type=int, default=100000)

    #parser.add_argument('--image_path', type=str, default='/content/drive/My Drive/My Project/data1/')
    #parser.add_argument('--image_path', type=str, default='D:\Jupyter\GAN-zoo\data\own_data')
    parser.add_argument('--src_path', type=str, default='/content/datasets/shawndong98/face-transfer/lyf')
    parser.add_argument('--dst_path', type=str, default='/content/datasets/shawndong98/face-transfer/qq')
    # parser.add_argument('--src_path', type=str, default='D:\Deepfake\data\data_src\sqys\lyf')
    # parser.add_argument('--dst_path', type=str, default='D:\Deepfake\data\data_src\sqys\qq')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    return parser.parse_args()

from dataloader import Data_Loader

# path = 'D:\Jupyter\GAN-zoo\data\own_data'

config = get_config()


src_dataset = Data_Loader(128, config.src_path, config.batch_size)
src_loder = src_dataset.loader()


dst_dataset = Data_Loader(128, config.dst_path, config.batch_size)
dst_loder = dst_dataset.loader()

device = "cuda" if torch.cuda.is_available() else "cpu"


net = AE(config, src_loder, dst_loder, device)
net.train()
#net.inference()









