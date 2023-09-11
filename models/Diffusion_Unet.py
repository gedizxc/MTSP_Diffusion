# @Time    : 2023/7/21 9:15 上午
# @Author  : tang
# @File    : Diffusion_Unet.py
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from models.UNet import UNet
import logging
from models.UNet import UNet

class Diffusion_Unet(nn.Module):   #forward用来算pretrain 的mse，并提供sample功能
    def __init__(self, args, noise_steps=1, beta_start=1e-4, beta_end=0.02):
        super(Diffusion_Unet, self).__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.UNet = UNet(num_steps=self.noise_steps,seq_len = args.seq_len,channels=args.enc_in).to(self.device)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_forward(self, x, t): #正向一步到位
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None,None]

        M_1 = torch.zeros(t.shape[0],x.shape[1],x.shape[2],x.shape[3]).to(self.device)
        M_1[:x.shape[0]] = x
        Ɛ = torch.randn_like(M_1)
        x_t= sqrt_alpha_hat * M_1 +sqrt_one_minus_alpha_hat * Ɛ

        x_t = x_t.view(x.shape[0],-1,x.shape[-2],x.shape[-1])
        x_t = torch.mean(x_t,dim=1,keepdim=True)
        Ɛ = Ɛ.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        Ɛ = torch.mean(Ɛ, dim=1, keepdim=True)
        return x_t,Ɛ
    def noise_forward1(self, x, t): #正向一步到位
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None,None]



        Ɛ = torch.randn_like(x)
        x_t= sqrt_alpha_hat * x +sqrt_one_minus_alpha_hat * Ɛ

        x_t = x_t.view(x.shape[0],-1,x.shape[-2],x.shape[-1])
        x_t = torch.mean(x_t,dim=1,keepdim=True)
        Ɛ = Ɛ.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        Ɛ = torch.mean(Ɛ, dim=1, keepdim=True)
        return x_t,Ɛ


#bathsize：8，step：32    return：(32,1,1,1)*(8,1,24,7) + (32,1,1,1)*(8,1,24,7)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    #
    def sample(self, model, sample_num=1,enc_out = None, condition=None):
        logging.info(  f"Sampling {sample_num} new data....")
        with torch.no_grad():
            sample_shape = enc_out.shape[0]*10
            x = torch.randn((sample_shape,1, enc_out.shape[2],enc_out.shape[3])).to(self.device) #11  从高斯开始采样 #nsample
            for i in tqdm(reversed(range(1, 10)), position=0): #扩散步数
                t = (torch.ones(sample_shape) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None,None]
                alpha_hat = self.alpha_hat[t][:, None, None,None]
                beta = self.beta[t][:, None, None,None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                #逆向最核心的公式，由t推t-1
        x = (x.clamp(-1, 1) + 1) / 2      #压缩（0，1）
        return x

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        t = self.sample_timesteps(self.noise_steps).to(self.device)
        x_t, noise = self.noise_forward(x, t)
        predicted_noise = self.UNet(x_t, t)
        return noise,predicted_noise






