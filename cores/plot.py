import torch

import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt



def draw_lines():
    state = torch.load("./models/state.pth")
    
    df1 = pd.DataFrame(state['rec_src_loss'], columns=['rec_src_loss'])
    df2 = pd.DataFrame(state['enc_src_loss'], columns=['enc_src_loss'])
    df3 = pd.DataFrame(state['g_src_loss'], columns=['g_src_loss'])
    df4 = pd.DataFrame(state['d_src_loss'], columns=['d_src_loss'])

    f1=plt.figure()

    f1.add_subplot(221)

    sns.lineplot(data=df1)

    f1.add_subplot(222)
    sns.lineplot(data=df2)

    f1.add_subplot(223)
    sns.lineplot(data=df3)

    f1.add_subplot(224)
    sns.lineplot(data=df4)

    df11 = pd.DataFrame(state['rec_dst_loss'], columns=['rec_dst_loss'])
    df22 = pd.DataFrame(state['enc_dst_loss'], columns=['enc_dst_loss'])
    df33 = pd.DataFrame(state['g_dst_loss'], columns=['g_dst_loss'])
    df44 = pd.DataFrame(state['d_dst_loss'], columns=['d_dst_loss'])

    f2=plt.figure()

    f2.add_subplot(221)

    sns.lineplot(data=df11)

    f2.add_subplot(222)
    sns.lineplot(data=df22)

    f2.add_subplot(223)
    sns.lineplot(data=df33)

    f2.add_subplot(224)
    sns.lineplot(data=df44)


    f1.savefig("./plot/src_losses.jpg")
    f2.savefig("./plot/dst_losses.jpg")
