from torch import nn
import torch
class P_Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(P_Encoder, self).__init__()
        self.encoder_linear = nn.Sequential(
            nn.Linear(3,32),
            nn.ReLU(True),
            nn.Linear(32,64),
            nn.ReLU(True),
            nn.Linear(64,16)
        )
        self.linear2 = nn.Linear(16, latent_dims)
        self.linear3 = nn.Linear(16, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0
    def forward(self, x):
        x = self.encoder_linear(x)
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl =1/2 * (sigma + mu**2 - torch.log(sigma) - 1).sum()
        return z,mu,sigma

class P_Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(P_Decoder, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 16),
            nn.ReLU(True),
            nn.Linear(16,64),
           nn.ReLU(True),
            nn.Linear(64,32),
           nn.ReLU(True),
            nn.Linear(32,3),

        )
    def forward(self, z):
        z = self.decoder_linear(z)


        return z
class P_VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(P_VariationalAutoencoder, self).__init__()
        self.encoder = P_Encoder(latent_dims)
        self.decoder = P_Decoder(latent_dims)

    def forward(self, x):
        z,mu,sigma = self.encoder(x)
        return self.decoder(z)