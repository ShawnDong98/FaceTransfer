import torch
import torch.nn as nn

def L1_Loss(rec_img, img):
    l1 = nn.L1Loss()
    loss = l1(rec_img, img)

    return loss


def Mask_Loss(mask):

    loss = torch.mean(torch.abs(mask))

    return loss