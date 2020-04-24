import torch
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

import pandas as pd
import numpy as np 
import os

import seaborn as sns
import matplotlib.pyplot as plt


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def tensor2var(x, grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1)/ 2
    return out.clamp_(0, 1)

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

def make_image(Xs, Xt, Y):
    Xs = make_grid(denorm(Xs).cpu(), nrow=Xs.shape[0])
    Xt = make_grid(denorm(Xt).cpu(), nrow=Xt.shape[0])
    Y = make_grid(denorm(Y).cpu(), nrow=Y.shape[0])
    return unloader(torch.cat((Xs, Xt, Y), dim=1))


def draw_lines():
    state = torch.load("./models/state.pth")
    
    df1 = pd.DataFrame(state['rec_src'], columns=['rec_src'])
    df2 = pd.DataFrame(state['mask_src_loss'], columns=['mask_src_loss'])
    f1=plt.figure()

    f1.add_subplot(121)
    sns.lineplot(data=df1)

    f1.add_subplot(122)
    sns.lineplot(data=df2)


    df11 = pd.DataFrame(state['rec_dst'], columns=['rec_dst'])
    df22 = pd.DataFrame(state['mask_dst_loss'], columns=['mask_dst_loss'])

    f2=plt.figure()
    f2.add_subplot(121)
    sns.lineplot(data=df11)

    f2.add_subplot(122)
    sns.lineplot(data=df22)


    f1.savefig("./plot/src_losses.jpg")
    f2.savefig("./plot/dst_losses.jpg")

    plt.close()


if __name__ == "__main__":
    pass