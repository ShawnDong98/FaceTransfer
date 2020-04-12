import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
            inputs: 
                x : input feature maps(B x C x W x H)
            returns:
                out: self attention value + input feature
                attention: B x N x N(N is Width*Height)
        '''
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # B x C x N
        energy = torch.bmm(proj_query, proj_key) # batch矩阵的乘法
        # dim=-1， 对列做softmax
        # B x N x N
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out