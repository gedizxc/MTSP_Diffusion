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
            pre_path = os.path.join('../Pretrain_diffusion/Pretrain_checkpoints/{}'.format(self.seq_len), model_path)

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

        # STL:L
        x_resid = resid.reshape(B, S, C).to(self.device)

        # STL:S
        x_seasonal = seasonal.reshape(B,S,C).to(self.device)

        #STL:T->Diffusion
        trend = trend.reshape(B,S,C).to(self.device)
        orgin_trend = trend

        Unet  = self.Pre_model.UNet
        trend = torch.unsqueeze(trend, dim=1)  # 3->4维
        sample_x = self.Pre_model.sample(Unet, sample_num = self.sample_num,enc_out = trend) # 32 1 104 7
        Diffusion_trend = sample_x.view(B, -1, S, C)
        Diffusion_trend = torch.mean(Diffusion_trend, dim=1, keepdim=True)
        Diffusion_trend = torch.squeeze(Diffusion_trend, dim=1)
        # 残差
        x_trend = Diffusion_trend + orgin_trend

        #NLinear
        x = x_trend #+x_seasonal
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

        res = x + x_seasonal[:,-self.pred_len:,:]  #+trend[:,:self.pred_len,:]



        return res  # [Batch, Output length, Channel]
