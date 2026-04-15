from torch import nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
class Encoder(nn.Module):
    def __init__(self, latent_dims = 8):
        super(Encoder, self).__init__()

        self.encoder_linear = nn.Sequential(
           nn.Linear(20, 64),

            nn.ReLU(True),

             nn.Linear(64, 128),

            nn.ReLU(True),

            nn.Linear(128, 64),

             nn.ReLU(True),

            nn.Linear(64, 32)
        )

        self.linear2 = nn.Linear(32, latent_dims)
        self.linear3 = nn.Linear(32, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc# hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0



    def forward(self, inputs,conditions):
        x=torch.cat([inputs, conditions], dim=1)
        x= self.encoder_linear(x)

        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)

        self.kl =1/2 * (sigma + mu**2 - torch.log(sigma) - 1).sum()

        return z

class Decoder(nn.Module):
    def __init__(self,  latent_dims = 8):
        super(Decoder, self).__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims+3, 32),

            nn.ReLU(True),

            nn.Linear(32, 64),

            nn.ReLU(True),

            nn.Linear(64, 128),

            nn.ReLU(True),

            nn.Linear(128, 64),

            nn.ReLU(True),

            nn.Linear(64, 17),


        )

    def forward(self, inputs,conditions):
        x = torch.cat([inputs,conditions], dim=1)
        x = self.decoder_linear(x)
        return x
class C_VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims = 8):
        super(C_VariationalAutoencoder, self).__init__()
        self.encoder = Encoder( latent_dims)
        self.decoder = Decoder( latent_dims)
    def forward(self, x,conditions):
        z = self.encoder(x,conditions)
        z = self.decoder(z,conditions)
        return z
def normalize_top_n_values(tensor, n, threshold=0.001):
    # Apply thresholding
    threshold_tensor = torch.tensor(threshold, device=tensor.device)
    tensor = torch.where(tensor < threshold_tensor, torch.tensor(0.0, device=tensor.device), tensor)

    # Find the indices of the top n largest values along each row
    top_values, indices = torch.topk(tensor, n, dim=1)

    # Create a mask where only the top n values are kept
    mask = torch.zeros_like(tensor)
    mask.scatter_(1, indices, 1.0)

    # Apply the mask to zero out all values except the top n values
    tensor = tensor * mask


    # Normalize the top n values along each row
    row_sums = top_values.sum(dim=1, keepdim=True)
    normalized_top_values = top_values / row_sums

    # Round to 4 decimal places
    normalized_top_values_rounded = torch.round(normalized_top_values * 10**4) / (10**4)

    # If you need to ensure it's a float tensor:
    normalized_top_values_rounded = normalized_top_values_rounded.float()


    # Replace the original tensor values with normalized values at the top n indices
    tensor.scatter_(1, indices, normalized_top_values_rounded )

    return tensor

def normalize_filter(tensor, threshold=0.001):
    # Apply thresholding
    threshold_tensor = torch.tensor(threshold, device=tensor.device)/100
    tensor = torch.where(tensor < threshold_tensor, torch.tensor(0.0, device=tensor.device), tensor)

    non_zero = []
    for i in range(tensor.shape[0]):
        non_zero_count = torch.nonzero(tensor[i]).size(0)
        non_zero.append(non_zero_count)
    non_zero_tensor = torch.tensor(non_zero)

    result_tensor = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        # 找出 tensor[i] 中非零元素的索引
        non_zero_indices = tensor[i].nonzero().squeeze()

        # 将 tensor[i] 中非零元素加上 aa[i] 对应位置的值
        result_tensor[i, non_zero_indices] = tensor[i, non_zero_indices] + \
                                             ((1 - torch.sum(tensor, dim=1)) / non_zero_tensor)[i]


    tensor = result_tensor
    tensor = torch.where(tensor < threshold_tensor, torch.tensor(0.0, device=tensor.device), tensor)

    # Normalize the tensor
    # # Normalize the top n values along each row
    # row_sums = tensor.sum(dim=1, keepdim=True)
    # normalized_top_values =  tensor/ row_sums
    #
    # # Round to 4 decimal places
    # normalized_values_rounded = (torch.round(normalized_top_values * 10**4) / (10**4)).float()

    return tensor