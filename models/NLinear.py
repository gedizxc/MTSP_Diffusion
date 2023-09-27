# @Time    : 2023/6/10 11:28 上午
# @Author  : tang
# @File    : NLinear.py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Diffusion import Diffusion
from models.UNet import UNet
from layers.Revin import RevIN
import os
from Pretrain_diffusion.pre_main import Pretrain_diffusion
from statsmodels.tsa.seasonal import STL

class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.is_diffusion = configs.is_diffusion
        self.sample_num = configs.sample_num
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.sample_num = configs.sample_num
        self.device ="cuda:0" if torch.cuda.is_available() else "cpu"

        if configs.is_diffusion:
            self.UNet = UNet(num_steps=configs.noise_step, seq_len=self.seq_len ,channels=self.enc_in)  # Unet初始化给seq_input_len 的维度
            model_path = '{}_noise_step:{}'.format(configs.data, configs.noise_step)
            pre_path = os.path.join('../Pretrain_diffusion/Pretrain_checkpoints', model_path)

            Pretrain = Pretrain_diffusion(configs).to(self.device)
            self.Pre_model = Pretrain.model
            self.Pre_model.load_state_dict(torch.load(pre_path + '/' + 'checkpoint.pth',map_location='cpu'))


        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.dropout = nn.Dropout(configs.drop)

        self.mse = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.l1 =  nn.Linear(self.seq_len, self.seq_len)



    def forward(self, x,stl_data):
        B,S,C = x.shape
        seasonal,trend,resid = stl_data

        #STL:T->Diffusion
        trend = trend.reshape(B,S,C).to(self.device)
        Unet  = self.Pre_model.UNet
        orgin_trend = trend
        x_trend = torch.unsqueeze(trend, dim=1)  # 3->4维
        sample_x = self.Pre_model.sample(Unet, sample_num = self.sample_num,enc_out = x_trend) # 32 1 104 7
        x_trend = sample_x.view(B, -1, S, C)
        x_trend = torch.mean(x_trend, dim=1, keepdim=True)
        x_trend = torch.squeeze(x_trend, dim=1)
        x_trend = x_trend + orgin_trend #残差

        #Revin
        # x_trend = self.rev(x_trend, 'norm') if self.rev else x_trend
        # x_trend = self.dropout(x_trend)

        # STL:L
        x_resid = resid.reshape(B, S, C).to(self.device)


        # STL:S
        x_seasonal = seasonal.reshape(B,S,C).to(self.device) #2->3
        #NLinear
        x = x_trend
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        #x_trend = x_trend[:,:self.pred_len,:]
        res = x + x_seasonal[:,:self.pred_len,:]  #+trend[:,:self.pred_len,:]

        #revin
        # res =  self.rev(res, 'denorm') if self.rev else res

        return res  # [Batch, Output length, Channel]
