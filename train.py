import torch
import torch.nn as nn

import cv2
import time
import datetime
import argparse
import pynvml

from cores.utils import *
from cores.networks import Encoder, Decoder
from cores.losses import *
from cores.dataloader import Data_Loader



class AE():
    def __init__(self, config, src_dataloader, dst_dataloader, device='cpu'):
        self.device = device
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.config = config

        self.net_init()
        self.src_dataloader = src_dataloader
        self.dst_dataloader = dst_dataloader

        self.optim_E = torch.optim.Adam(self.E.parameters(), 0.0004, [0, 0.999])
        self.optim_G_src = torch.optim.Adam(self.G_src.parameters(), 0.0004, [0, 0.999])
        self.optim_G_dst = torch.optim.Adam(self.G_dst.parameters(), 0.0004, [0, 0.999])

        # Create directories if not exist
        make_folder(self.config.model_save_path)
        make_folder(self.config.sample_path)

    def net_init(self):
        try:
            self.E = Encoder(self.latent_dim).to(self.device)
            self.E.load_state_dict(torch.load("./models/latest_E.pth"))
            self.G_src = Decoder().to(self.device)
            self.G_src.load_state_dict(torch.load("./models/latest_G_src.pth"))
            self.G_dst = Decoder().to(self.device)
            self.G_dst.load_state_dict(torch.load("./models/latest_G_dst.pth"))
            self.state = torch.load("./models/state.pth")
            print("model loaded...")
        except:
            print("no pre-trained model...")
            self.E = Encoder(self.latent_dim).to(self.device)
            self.G_src = Decoder().to(self.device)
            self.G_dst = Decoder().to(self.device)
            self.state = {
                "iter": 0,
                "rec_src": [],
                "mask_src_loss": [],
                "rec_dst": [],
                "mask_dst_loss": [],
            }

    def train(self):
        src_loader = iter(self.src_dataloader)
        dst_loader = iter(self.dst_dataloader)

        

        for step in range(self.state['iter'], self.config.total_iter):
            self.E.train()
            self.G_src.train()
            self.G_dst.train()
    
            pynvml.nvmlInit()

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Start time
            start_time = time.time()

            # --------------------------------------------------
            # train src

            try:
                warp_src, src_img = next(src_loader)
                warp_src = warp_src.to(self.device)
                src_img = src_img.to(self.device)
            except:
                src_loader = iter(self.src_dataloader)
                warp_src, src_img = next(src_loader)
                warp_src = warp_src.to(self.device)
                src_img = src_img.to(self.device)

            # train Encoder
            latent_img = self.E(warp_src)

            mask_src, rec_src_img = self.G_src(latent_img)
            # print("rec_img_shape: ", rec_img.shape)
            
            rec_src = L1_Loss(rec_src_img, src_img)
            self.state["rec_src"].append(float(rec_src.cpu()))

            # mask loss
            mask_src_loss = Mask_Loss(mask_src)
            self.state["mask_src_loss"].append(float(mask_src_loss.cpu()))
            
            loss_src = rec_src + 1e-2 * mask_src_loss

            self.optim_E.zero_grad()
            self.optim_G_src.zero_grad()
            loss_src.backward()
            # for name, parms in self.E.named_parameters():
	        #     print('E-->name:', name, '-->grad_requirs:', parms.requires_grad, '--werms.datight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            self.optim_E.step()
            self.optim_G_src.step()
            
            
            # for name, parms in self.G_src.named_parameters():
            #     print('G_src-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            



            # ---------------------------------------------------

            # ---------------------------------------------------
            # train dst

            try:
                warp_dst, dst_img = next(dst_loader)
                warp_dst = warp_dst.to(self.device)
                dst_img = dst_img.to(self.device)
            except:
                dst_loader = iter(self.dst_dataloader)
                warp_dst, dst_img = next(dst_loader)
                warp_dst = warp_dst.to(self.device)
                dst_img = dst_img.to(self.device)

            # train Encoder
            latent_img = self.E(warp_dst)

            mask_dst, rec_dst_img = self.G_dst(latent_img)
            # print("rec_img_shape: ", rec_img.shape)
            
            rec_dst = L1_Loss(rec_dst_img, dst_img)
            self.state["rec_dst"].append(float(rec_dst.cpu()))

            # mask loss
            mask_dst_loss = Mask_Loss(mask_dst)
            self.state["mask_dst_loss"].append(float(mask_dst_loss.cpu()))
            
            loss_dst = rec_dst + 1e-2 * mask_dst_loss
            
            self.optim_E.zero_grad()
            self.optim_G_dst.zero_grad()
            loss_dst.backward()
            # for name, parms in self.E.named_parameters():
	        #     print('E-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            self.optim_E.step()
            self.optim_G_dst.step()

            
            # for name, parms in self.G_dst.named_parameters():
            #     print('G_dst-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))


            # ---------------------------------------------------

            if (step + 1) % 1 == 0:
                batch_time = time.time() - start_time
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print(meminfo.used)
                print("time [{:.4f}], mask_src_mean: {:.4f}, mask_dst_mean: {:.4f},"
                      .format(batch_time, mask_src.mean(), mask_dst.mean()))
                print("step [{}/{}],  rec_src: {:.4f}, mask_src_loss: {:.4f}, rec_dst: {:.4f}, mask_dst_loss: {:.4f}"
                      .format(step + 1, self.config.total_iter, rec_src, mask_src_loss, rec_dst, mask_dst_loss))

            if (step + 1) % 10 == 0:
                with torch.no_grad():
                    latent = self.E(src_img)
                    mask, Y = self.G_dst(latent)
                img = make_image(src_img, mask, Y)
                img.save("./samples/latest.jpg")

            if (step+1) % 100 == 0:
                torch.save(self.E.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_E.pth'))
                torch.save(self.G_src.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_G_src.pth'))
                torch.save(self.G_dst.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_G_dst.pth'))
                self.state['iter'] = step + 1
                torch.save(self.state,
                           os.path.join(self.config.model_save_path, 'state.pth'))
                draw_lines()
                self.inference()

    def inference(self):
        self.E.eval()
        self.G_src.eval()
        self.G_dst.eval()
        src_loader = iter(self.src_dataloader)
        dst_loader = iter(self.dst_dataloader)

        src_warp, src_target = next(src_loader)
        src_warp = src_warp.to(self.device)
        src_target = src_target.to(self.device)
        dst_warp, dst_target = next(dst_loader)
        dst_warp = dst_warp.to(self.device)
        dst_target = dst_target.to(self.device)

        # -----------------------------------------------------
        # src
        # ------------------------------------------------------
        src_target = src_target[:8]
        latent_img = self.E(src_target)

        mask_src, rec_src = self.G_dst(latent_img)

        src2dst = make_image(src_target, mask_src, rec_src)
        src2dst.save("./samples/src2dst.jpg")

        # ---------------------------------------------------------
        # dst
        # ---------------------------------------------------------
        dst_target = dst_target[:8]
        latent_img = self.E(dst_target)

        mask_dst, rec_dst = self.G_src(latent_img)

        dst2src = make_image(dst_target, mask_dst, rec_dst)
        dst2src.save("./samples/dst2src.jpg")


        self.E.train()
        self.G_src.train()
        self.G_dst.train()

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--total_iter', type=int, default=60000)

    parser.add_argument('--image_path', type=str, default='/content/datasets/shawndong98/aligncelebahqpix256')
    # parser.add_argument('--image_path', type=str, default='D:\Jupyter\GAN-zoo\data\own_data')
    parser.add_argument('--src_path', type=str,
                        default='/content/datasets/shawndong98/face-transfer/qq')
    parser.add_argument('--dst_path', type=str,
                        default='/content/datasets/shawndong98/face-transfer/lyf')
    # parser.add_argument('--src_path', type=str, default='D:\Deepfake\data\data_src\sqys\lyf')
    # parser.add_argument('--dst_path', type=str, default='D:\Deepfake\data\data_src\sqys\qq')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    return parser.parse_args()


# path = 'D:\Jupyter\GAN-zoo\data\own_data'
config = get_config()


src_dataset = Data_Loader(64, config.src_path, config.batch_size)
src_loder = src_dataset.loader()


dst_dataset = Data_Loader(64, config.dst_path, config.batch_size)
dst_loder = dst_dataset.loader()

device = "cuda" if torch.cuda.is_available() else "cpu"


net = AE(config, src_loder, dst_loder, device)
net.train()
# net.inference()

# if __name__ == "__main__":
#     G = Decoder()
#     for name, parms in G.named_parameters():
#         print(parms)