import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import SEANetEncoder, SEANetDecoder
import random

class EncoderModel(nn.Module):
    def __init__(self, encoder_dim=30):
        super(EncoderModel, self).__init__()  
        self.encoder = SEANetEncoder(dimension=encoder_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z

class DecoderModel(nn.Module):
    def __init__(self, decoder_dim=30):
        super(DecoderModel, self).__init__()
        self.decoder = SEANetDecoder(dimension=decoder_dim)
        self.decoder_dim = decoder_dim
    
    def forward(self,z):
        
        z_normalized = F.normalize(z, p=2, dim=1)
        y = self.decoder(z_normalized)
        y = F.tanh(y)

        return y



