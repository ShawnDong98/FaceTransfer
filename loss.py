import torch

def E_loss(mean, logvar):
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)

    return prior_loss


def Rec_loss(D, x, rec_img):
    similarity_x = D.similarity(x)
    similarity_rec = D.similarity(rec_img)
    rec_loss = ((similarity_x - similarity_rec) ** 2) .mean()

    return rec_loss

def G_loss(D, G, rec_img, batch_size, latent_dim, device):
    z = torch.randn(batch_size, latent_dim).to(device)

    fake = G(z)
    G_out_fake = D(fake)

    G_out_rec = D(rec_img)
    G_loss = -(G_out_fake + G_out_rec).mean()

    return G_loss



def D_loss(D, G, x, rec_img, latent_dim):
    batch_size = x.size()[0]
    z = torch.randn((batch_size, latent_dim)).to(x.device)

    D_real = D(x)
    D_real_loss = torch.nn.ReLU()(1 - D_real).mean()

    fake = G(z)
    D_fake = D(fake.detach())
    D_fake_loss = torch.nn.ReLU()(1 + D_fake).mean()

    D_rec = D(rec_img.detach())
    D_rec_loss = torch.nn.ReLU()(1 + D_rec).mean()

    D_loss = D_real_loss + D_fake_loss + D_rec_loss

    return D_loss




