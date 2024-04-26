# -*- coding: utf-8 -*-
"""
@author: LRAY
"""
from generator import EncoderModel,DecoderModel

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
# import yaml
import numpy as np
import time
import argparse
from pathlib import Path
import soundfile as sf

from stft_loss import MultiResolutionSTFTLoss
from spec_loss import MultiResolutionLoss
from tqdm import tqdm
import pesq
import multiprocessing
from Generate_channel import generate_rayleigh_channel,generate_rician_channel
from RISmodel import RISModel,precoding_2x2,precoding_4x2
# from torch.multiprocessing import set_start_method
# set_start_method('fork')



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_data1 = np.load('path to your speech dataset for train')#
test_data = np.load('path to your speech dataset for test')
# (828470, 512)
print(input_data1.shape)

input_dataset = input_data1[0:63875]
input_dataset = input_dataset /(2**15)
test_data = test_data[0:2400]
test_data = test_data/(2**15)
test_data = test_data.reshape(-1)
test_data =test_data[0:1920*3600]
test_data = test_data.reshape(-1,9600)
batch_size = 64
input_data = input_dataset.reshape(-1)
input_data = input_data[0:39490*9600]
input_data = input_data.reshape(-1,9600)
num_workers = multiprocessing.cpu_count()  

train_loader = DataLoader(input_data, batch_size=batch_size,num_workers=num_workers)
#####load train dataset

test_loader = DataLoader(test_data, batch_size=32)
path = '2x2_N_16_Rayleigh_01'
print(path)    
    
Tn = 2 
Rn = 2 
NUM_RIS = 16

CR = 0.4 * 2
encoder_dim = int(CR * 320)    

Enc =  EncoderModel(encoder_dim=encoder_dim)
Dec =  DecoderModel(decoder_dim=encoder_dim)
Rismodel = RISModel(2*(Tn+Rn)*NUM_RIS,NUM_RIS)
chencoder = precoding_2x2((Tn+(Tn*Rn))*2,Tn*2)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, 0)

Enc.apply(init_weights)
Dec.apply(init_weights)
Rismodel.apply(init_weights)
chencoder.apply(init_weights)

Enc = Enc.to(device)
Dec = Dec.to(device)
Rismodel = Rismodel.to(device)
chencoder = chencoder.to(device)

optEnc = torch.optim.Adam(Enc.parameters(), lr=1e-4, betas=(0.5, 0.9))
optRis = torch.optim.Adam(Rismodel.parameters(), lr=1e-4, betas=(0.5, 0.9))
optDec = torch.optim.Adam(Dec.parameters(), lr=1e-4, betas=(0.5, 0.9))
optchen = torch.optim.Adam(chencoder.parameters(), lr=1e-4, betas=(0.5, 0.9))

epochs = 500

# Enc.load_state_dict(torch.load(f"{path}/SEAEnc.pt", map_location=torch.device('cpu')))
# Dec.load_state_dict(torch.load(f"{path}/SEADec.pt", map_location=torch.device('cpu')))
# Rismodel.load_state_dict(torch.load(f"{path}/Rismodel.pt", map_location=torch.device('cpu')))
# chencoder.load_state_dict(torch.load(f"{path}/chencoder.pt", map_location=torch.device('cpu')))
# optEnc.load_state_dict(torch.load(f"{path}/SEAoptEnc.pt", map_location=torch.device('cpu')))
# optDec.load_state_dict(torch.load(f"{path}/SEAoptDec.pt", map_location=torch.device('cpu')))
# optRis.load_state_dict(torch.load(f"{path}/optRis.pt", map_location=torch.device('cpu')))
# optchen.load_state_dict(torch.load(f"{path}/optchen.pt", map_location=torch.device('cpu')))
# print("load successfully!")
# path = '1x2_N_5_5'

def spectral_reconstruction_loss(x, G_x):
    
    s=[2**i for i in range(5,11)]
    hop=[2**i//4 for i in range(5,11)]
    stftloss = MultiResolutionSTFTLoss(fft_sizes=s,hop_sizes=hop,win_lengths=s,factor_sc=1, factor_mag=1).to(device)
    loss = stftloss(G_x.squeeze(1),x.squeeze(1))
    return loss

import matplotlib.pyplot as plt
import librosa.display

def generate_and_save_images(predictions,epoch, test_input,rec_loss,mse_loss,snr):
    test_input = test_input.cpu().numpy().reshape(-1)
    predictions = predictions.cpu().numpy().reshape(-1)   
    plt.figure(figsize=(12, 6))  
    predictions = np.array(predictions) 
    test_input = np.array(test_input)
    # Save original waveform
    sf.write("pic-audio_clean/original_waveform.wav", test_input, 16000, 'PCM_16')
    # Save predicted waveform
    sf.write(f"pic-audio_clean/predicted_epoch{epoch}.wav", predictions, 16000, 'PCM_16')
    print("save successfully!")
    print(np.mean(np.square(test_input - predictions)))
    return predictions,test_input
    
WINDOW_LENGTH = 1024
HOP_LENGTH = 256
no_improvement_count = 0
epochs_list = []
epochs_pesq = []
pesq_scores = []
rec_scores = []
mse_scores = []
best_pesq_score = float('-inf')  # 初始化最佳PESQ分数
for epoch in range(1, epochs + 1):
    
    Enc.train()
    Dec.train()
    Rismodel.train()
    chencoder.train()
    print(epoch)
    rec_mean=0.0
    mse_loss_mean = 0.0
    iterno = 0
    rec_loss = 0
    commit_loss = 0

    for itern, x_t in tqdm(enumerate(train_loader), total=len(train_loader)):
        iterno = itern
        sample = int(len(x_t.view(-1)) * CR  // (2*2))
        # print(sample)
        H2,H3,H2_complex,H3_complex = generate_rayleigh_channel(sample,Tn,Rn,NUM_RIS)
        H2 = H2.view(sample,-1)
        H3 = H3.view(sample,-1)
        H_concat = torch.cat((H2, H3), dim=1)
        # print("H_concat",H_concat.shape)
        Theta = Rismodel(H_concat) * torch.pi * 2
        # print("Theta",Theta.shape)
        # print("Theta",Theta[-1])
        complex_exp_theta = torch.complex(torch.cos(Theta), torch.sin(Theta))
        # print("complex_exp_theta",complex_exp_theta[-1])
        # print("complex_exp_theta",complex_exp_theta.shape)
        reflection_matrix = torch.diag_embed(complex_exp_theta)
        # print("diagonal_matrix",reflection_matrix.shape)
        # print("reflection_matrix",reflection_matrix[-1])
        x_t = x_t.to(device, dtype=torch.float32)
        x_t = x_t.unsqueeze(1)
        # print("x_t",x_t.shape)
        raw = random.randint(-10, 10)*2
        snr = torch.tensor(10**(raw / 10))
        std = torch.sqrt(torch.tensor(1 / (2*snr))).to(device)
        noise = std*torch.randn(sample,Rn,1,2).to(device)  # 将张量移动到GPU
        noise_complex = noise[..., 0] + 1j * noise[..., 1]
        token_origin = Enc(x_t.to(device))
        a,b,c = token_origin.shape
        token = token_origin.view(a,-1,2)
        # print('x_normalizedx',x_normalized[-1])
        # print('x_normalizedx',x_normalized.shape)
        # print('H2_complex',H2_complex.shape)
        # print('H3_complex',H3_complex.shape)
        H3_complex_conj = torch.conj(H3_complex.permute(0, 2, 1))
        H_eff =  H3_complex_conj @ reflection_matrix @ H2_complex
        # print('H_eff',H_eff.shape)
        H_eff_ri = torch.cat((H_eff.real, H_eff.imag), dim=2)
        # print('H_eff_ri',H_eff_ri.shape)
        H_eff_ri = H_eff_ri.view(sample,-1)
        token = token.view(-1,4)
        # print("token",token.shape)
        # token = torch.cat([token, token], dim=1)
        # print("token",token.shape)
        # print("token",token[-1])
        # print("H_eff_ri",H_eff_ri.shape)
        # print('x_normalized',x_normalized.shape)
        concatenated_tensor = torch.cat((H_eff_ri, token), dim=1)
        # print('concatenated_tensor',concatenated_tensor.shape)
        x_hat = chencoder(concatenated_tensor)
        # print(x_hat.shape)
        x_normalized = x_hat.view(a,-1,2)
        # print('x_normalized',x_normalized.shape)
        x_normalized = torch.nn.functional.normalize(x_normalized, p=2, dim=1) * torch.sqrt(torch.tensor(b*c//2 /2.0))
        # print('x_normalized',x_normalized.shape)
        complex_x = torch.complex(x_normalized[:, :, 0], x_normalized[:, :,1])
        complex_x = complex_x.view(-1,Tn,1)
        # complex_x = torch.cat([complex_x, complex_x], dim=1)
        # print('complex_x',complex_x.shape)
        # print('noise_complex',noise_complex.shape)

        y = H_eff @ complex_x + noise_complex
        # print('noise_complex',noise_complex.shape)
        # print('y',y.shape)
        # H_eff_conj = torch.conj(H_eff.permute(0, 2, 1))  
        # H_eff_pseudo_inverse = torch.matmul(torch.inverse(torch.matmul(H_eff_conj, H_eff)), H_eff_conj)
        # x_equ = torch.matmul(H_eff_pseudo_inverse, y)
        # # x_equ = torch.matmul(S_inv_complex,torch.matmul(U_conj_transpose,y))
        # # print('x_equ',x_equ[-1])
        y = torch.cat((y.real, y.imag), dim=2)
        # print('x_equ',x_equ[-1])
        # print('x_equ',x_equ.shape)
        x_equ = y.view(a,b,c)
        # print('x_equ',x_equ.shape)     
        x_pred_t = Dec(x_equ.to(device))

        # reconstruction_loss
        
        rec_loss = spectral_reconstruction_loss(x_t,x_pred_t)
        rec_mean+=rec_loss.item()
        mse_loss = F.mse_loss(x_pred_t, x_t)
        loss_g = rec_loss + mse_loss 
        mse_loss_mean+=mse_loss.item()
        optEnc.zero_grad()
        optDec.zero_grad()
        optRis.zero_grad()
        optchen.zero_grad()
        
        loss_g.backward()
        
        optEnc.step()
        optDec.step()
        optRis.step()
        optchen.step()
    
    rec_mean = rec_mean/(iterno+1)
    rec_scores.append(rec_mean)
    print("rec_meanloss {:.8f}".format(rec_mean))
    mse_loss_mean = mse_loss_mean/(iterno+1)
    
    mse_scores.append(mse_loss_mean)
    # print("rec_loss {:.8f}".format(rec_loss))
    print("mse_loss {:.8f}".format(mse_loss_mean))

    epochs_list.append(epoch)
    
    test_frequency = 5 if epoch > 250 else 10
    if epoch % test_frequency == 0 or epoch == 1:
        pesq_score_total = 0
        with torch.no_grad():
            for iterno, x_t in enumerate(test_loader):
                sample = int(len(x_t.view(-1)) * CR  // (2*2))
                # print(sample)
                H2,H3,H2_complex,H3_complex = generate_rayleigh_channel(sample,Tn,Rn,NUM_RIS)
                H2 = H2.view(sample,-1)
                H3 = H3.view(sample,-1)
                H_concat = torch.cat((H2, H3), dim=1)
                Theta = Rismodel(H_concat) * torch.pi * 2
                complex_exp_theta = torch.complex(torch.cos(Theta), torch.sin(Theta))
                reflection_matrix = torch.diag_embed(complex_exp_theta)
                x_t = x_t.to(device, dtype=torch.float32)
                x_t = x_t.unsqueeze(1)

                raw = 8
                print('SNR:',raw)
                snr = torch.tensor(10**(raw / 10))
                std = torch.sqrt(torch.tensor(1 / (2*snr))).to(device)
                noise = std*torch.randn(sample,Rn,1,2).to(device)  # 将张量移动到GPU
                noise_complex = noise[..., 0] + 1j * noise[..., 1]

                token_origin = Enc(x_t.to(device))
                a,b,c = token_origin.shape
                token = token_origin.view(a,-1,2)

                H3_complex_conj = torch.conj(H3_complex.permute(0, 2, 1))
                H_eff =  H3_complex_conj @ reflection_matrix @ H2_complex
                H_eff_ri = torch.cat((H_eff.real, H_eff.imag), dim=2)
                H_eff_ri = H_eff_ri.view(sample,-1)
                token = token.view(-1,4)
                # token = torch.cat([token, token], dim=1)

                
                concatenated_tensor = torch.cat((H_eff_ri, token), dim=1)
                x_hat = chencoder(concatenated_tensor)
                x_normalized = x_hat.view(a,-1,2)
                x_normalized = torch.nn.functional.normalize(x_normalized, p=2, dim=1) * torch.sqrt(torch.tensor(b*c//2 /2.0))
                complex_x = torch.complex(x_normalized[:, :, 0], x_normalized[:, :,1])
                complex_x = complex_x.view(-1,Tn,1)


                y = H_eff @ complex_x + noise_complex

                y = torch.cat((y.real, y.imag), dim=2)

                x_equ = y.view(a,b,c)
 
                x_pred_t = Dec(x_equ.to(device))
               
                re,origial = generate_and_save_images(x_pred_t,epoch,x_t,rec_mean,mse_loss_mean,raw)
                                
                pesq_score = pesq.pesq(16000, origial, re, 'wb')
                
                pesq_score_total +=pesq_score

                print(f"PESQ Score: {pesq_score}")
                
        pesq_score_total = pesq_score_total/(iterno+1)
        print(pesq_score_total)
        
        pesq_scores.append(pesq_score_total)
        epochs_pesq.append(epoch)
        
        
        print("best_pesq_score",best_pesq_score)
        
        if pesq_score_total > best_pesq_score:
            
            best_pesq_score = pesq_score_total
            
            no_improvement_count = 0
            torch.save(Enc.state_dict(), f"{path}/SEAEnc.pt")
            torch.save(Dec.state_dict(), f"{path}/SEADec.pt")
            torch.save(Rismodel.state_dict(), f"{path}/Rismodel.pt")
            torch.save(chencoder.state_dict(), f"{path}/chencoder.pt")
            
            torch.save(optRis.state_dict(), f"{path}/optRis.pt")
            torch.save(optEnc.state_dict(), f"{path}/SEAoptEnc.pt")
            torch.save(optDec.state_dict(), f"{path}/SEAoptDec.pt")
            torch.save(optchen.state_dict(), f"{path}/optchen.pt")
            # torch.save(residual_vq.state_dict(), 'residual_vq.pth')
            print("save successfully!")
            
        else:
            no_improvement_count += 1
            if no_improvement_count >= 2: 
                for param_group in optEnc.param_groups:
                    param_group['lr'] *= 0.8
                for param_group in optDec.param_groups:
                    param_group['lr'] *= 0.8
                # for param_group in optRis.param_groups:
                #     param_group['lr'] *= 0.5
                # for param_group in optchen.param_groups:
                #     param_group['lr'] *= 0.5
                print("Learning rate halved due to no improvement in validation performance.")
                no_improvement_count = 0  

fig, axes = plt.subplots(3, 1, figsize=(10, 15))


axes[0].plot(epochs_pesq, pesq_scores, label='PESQ Scores', color='blue', marker='+')
axes[0].set_xlabel('Epoches')
axes[0].set_ylabel('PESQ Score')
axes[0].legend()
axes[0].grid(True)

axes[0].text(0.5, 0.5, f'Max: {max(pesq_scores):.6f}\nMin: {min(pesq_scores):.6f}', transform=axes[0].transAxes, fontsize=10, verticalalignment='top')


axes[1].plot(epochs_list, rec_scores, label='REC Scores', color='green', marker='+')
axes[1].set_xlabel('Epoches')
axes[1].set_ylabel('REC Score')
axes[1].legend()
axes[1].grid(True)

axes[1].text(0.5, 0.5, f'Max: {max(rec_scores):.6f}\nMin: {min(rec_scores):.6f}', transform=axes[1].transAxes, fontsize=10, verticalalignment='top')

axes[2].plot(epochs_list, mse_scores, label='MSE Scores', color='red', marker='+')
axes[2].set_xlabel('Epoches')
axes[2].set_ylabel('MSE Score')
axes[2].legend()
axes[2].grid(True)

axes[2].text(0.5, 0.5, f'Max: {max(mse_scores):.6f}\nMin: {min(mse_scores):.6f}', transform=axes[2].transAxes, fontsize=10, verticalalignment='top')


plt.tight_layout()

# plt.show()

plt.savefig('2x2_N_16_Rayleigh_01.pdf')
        





