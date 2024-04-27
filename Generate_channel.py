# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:02:47 2024

@author: 29803
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_rayleigh_channel(sample,Nt,Nr,num_RIS):
    
    H2 = torch.randn(sample,num_RIS,Nt, 2).to(device)  # 将张量移动到GPU
    H2 = H2 / torch.sqrt(torch.tensor(2.0))
    
    H2_complex = H2[..., 0] + 1j * H2[..., 1]
    
    
    # print("h3_extended",h3_extended.shape)
    H3 = torch.randn(sample,num_RIS,Nr,2).to(device)  # 将张量移动到GPU
    H3 = H3 / torch.sqrt(torch.tensor(2.0))
    
    H3_complex = H3[..., 0] + 1j * H3[..., 1]
    
    
    # # print("h4_extended",h4_extended.shape)
    # H1 = torch.randn(sample,Nt, Nr, 2).to(device)  # 将张量移动到GPU
    # H1 = H1 / torch.sqrt(torch.tensor(2.0))
    
    # H1_complex = H1[..., 0] + 1j * H1[..., 1]
    
    
    return H2,H3,H2_complex,H3_complex
    
def generate_rician_channel(sample,Nt,Nr,num_RIS):
    
    H2 = torch.randn(sample,num_RIS,Nt, 2).to(device)  # 将张量移动到GPU
    H2_complex = H2[..., 0] + 1j * H2[..., 1]
    # print('H2:',H2[-1])
    # print('H2_complex:',H2_complex[-1])
    H2_complex = (torch.sqrt(torch.tensor(1/2)) + torch.sqrt(torch.tensor(1/2))* H2_complex / torch.sqrt(torch.tensor(2.0))) 
    H2 = torch.cat((H2_complex.real.unsqueeze(-1), H2_complex.imag.unsqueeze(-1)), dim=-1).view(H2.shape)
    # print('H2:',H2[-1])
    # print('H2_complex:',H2_complex[-1])
    # print("h3_extended",h3_extended.shape)
    H3 = torch.randn(sample,num_RIS,Nr,2).to(device)  # 将张量移动到GPU
    H3_complex = H3[..., 0] + 1j * H3[..., 1]
    
    H3_complex = (torch.sqrt(torch.tensor(1/2)) + torch.sqrt(torch.tensor(1/2))* H3_complex / torch.sqrt(torch.tensor(2.0)))
    H3 = torch.cat((H3_complex.real.unsqueeze(-1), H3_complex.imag.unsqueeze(-1)), dim=-1).view(H3.shape)
    # print("h4_extended",h4_extended.shape)
    H1 = torch.randn(sample,Nt, Nr, 2).to(device)  # 将张量移动到GPU
    H1 = H1 / torch.sqrt(torch.tensor(2.0))
    
    H1_complex = H1[..., 0] + 1j * H1[..., 1]
    
    
    return H1,H2,H3,H1_complex,H2_complex,H3_complex

def generate_Co_rayleigh_channel(sample,Nt,Nr,num_RIS):
    
    
    Phi_R_real = torch.eye(num_RIS).repeat(sample, 1, 1) * 0.5 + 0.5
    Phi_R_imag = torch.zeros_like(Phi_R_real)
    Phi_R = torch.complex(Phi_R_real, Phi_R_imag)
    Phi_T_real = torch.eye(Nt).repeat(sample, 1, 1) * 0.5 + 0.5
    Phi_T_imag = torch.zeros_like(Phi_T_real)
    Phi_T = torch.complex(Phi_T_real, Phi_T_imag)
    
    H2 = torch.randn(sample,num_RIS,Nt, 2).to(device)  # 将张量移动到GPU
    H2 = H2 / torch.sqrt(torch.tensor(2.0))
    
    H2_complex = H2[..., 0] + 1j * H2[..., 1]
    
    H2_complex = torch.sqrt(Phi_R) @ H2_complex @ torch.sqrt(Phi_T)
    
    
    # print("h3_extended",h3_extended.shape)
    Phi_R_real = torch.eye(Nr).repeat(sample, 1, 1) * 0.5 + 0.5
    Phi_R_imag = torch.zeros_like(Phi_R_real)
    Phi_R = torch.complex(Phi_R_real, Phi_R_imag)
    Phi_T_real = torch.eye(num_RIS).repeat(sample, 1, 1) * 0.5 + 0.5
    Phi_T_imag = torch.zeros_like(Phi_T_real)
    Phi_T = torch.complex(Phi_T_real, Phi_T_imag)
    
    H3 = torch.randn(sample,num_RIS,Nr,2).to(device)  # 将张量移动到GPU
    H3 = H3 / torch.sqrt(torch.tensor(2.0))
    
    H3_complex = H3[..., 0] + 1j * H3[..., 1]
    
    H3_complex = torch.sqrt(Phi_T) @ H3_complex @ torch.sqrt(Phi_R)
    # print("h4_extended",h4_extended.shape)
    H1 = torch.randn(sample,Nt, Nr, 2).to(device)  # 将张量移动到GPU
    H1 = H1 / torch.sqrt(torch.tensor(2.0))
    
    H1_complex = H1[..., 0] + 1j * H1[..., 1]
    
    
    return H1,H2,H3,H1_complex,H2_complex,H3_complex    
    
    