# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:37:37 2024

@author: 29803
"""

import torch
import torch.nn as nn

class RISModel(nn.Module):
    def __init__(self,input_num,RIS_NUM):
        super(RISModel, self).__init__()
        
        self.ris_model = nn.Sequential(
            nn.Linear(input_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, RIS_NUM),
            nn.Sigmoid()
            
        )
  
    def forward(self, x):
        
        ris_output = self.ris_model(x)
        return ris_output
    
class precoding_2x2(nn.Module):
    def __init__(self,input_num,output_num):
        super(precoding_2x2, self).__init__()
        
        self.chen_model = nn.Sequential(
            nn.Linear(input_num, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_num),
            nn.BatchNorm1d(output_num)     
        )
  
    def forward(self, x):
        
        output = self.chen_model(x)
        return output

class precoding_4x2(nn.Module):
    def __init__(self,input_num,output_num):
        super(precoding_4x2, self).__init__()
        
        SJY = 256
        
        self.chen_model = nn.Sequential(
            nn.Linear(input_num, SJY),
            nn.BatchNorm1d(SJY),
            nn.ReLU(),
            nn.Linear(SJY,SJY),
            nn.BatchNorm1d(SJY),
            nn.ReLU(),
            nn.Linear(SJY, output_num),
            nn.BatchNorm1d(output_num)     
        )
  
  
    def forward(self, x):
        
        output = self.chen_model(x)
        return output
    
class precoding_6x2(nn.Module):
    def __init__(self,input_num,output_num):
        super(precoding_6x2, self).__init__()
        
        # SJY = 256
        
        self.chen_model = nn.Sequential(
            nn.Linear(input_num, 256),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,384),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Linear(384, output_num),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(output_num)     
        )
  
  
    def forward(self, x):
        
        output = self.chen_model(x)
        return output

# Instantiate the RISModel
# model = RISModel()

# Define the variables MN_s, N_t, K_t, K, N_r with appropriate values before using the model
