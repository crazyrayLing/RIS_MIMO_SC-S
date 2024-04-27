# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:02:47 2024

@author: 29803
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_rayleigh_channel(sample,Nt,Nr,num_RIS):
    H2 = torch.randn(sample,num_RIS,Nt, 2).to(device)  
    H2 = H2 / torch.sqrt(torch.tensor(2.0))
    H2_complex = H2[..., 0] + 1j * H2[..., 1]

    H3 = torch.randn(sample,num_RIS,Nr,2).to(device) 
    H3 = H3 / torch.sqrt(torch.tensor(2.0))
    H3_complex = H3[..., 0] + 1j * H3[..., 1]
    return H2,H3,H2_complex,H3_complex
    
def generate_rician_channel(sample,Nt,Nr,num_RIS):
    
    H2 = torch.randn(sample,num_RIS,Nt, 2).to(device)  # 将张量移动到GPU
    H2_complex = H2[..., 0] + 1j * H2[..., 1]
    H2_complex = (torch.sqrt(torch.tensor(1/2)) + torch.sqrt(torch.tensor(1/2))* H2_complex / torch.sqrt(torch.tensor(2.0))) 
    H2 = torch.cat((H2_complex.real.unsqueeze(-1), H2_complex.imag.unsqueeze(-1)), dim=-1).view(H2.shape)

    H3 = torch.randn(sample,num_RIS,Nr,2).to(device)  # 将张量移动到GPU
    H3_complex = H3[..., 0] + 1j * H3[..., 1]
    H3_complex = (torch.sqrt(torch.tensor(1/2)) + torch.sqrt(torch.tensor(1/2))* H3_complex / torch.sqrt(torch.tensor(2.0)))
    H3 = torch.cat((H3_complex.real.unsqueeze(-1), H3_complex.imag.unsqueeze(-1)), dim=-1).view(H3.shape)
    
    return H2,H3,H2_complex,H3_complex

    
