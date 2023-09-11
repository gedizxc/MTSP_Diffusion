# @Time    : 2023/6/15 9:49 上午
# @Author  : tang
# @File    : UNet.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# class UNet(nn.Module):
#     def __init__(self,configs):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16,1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#
#         self.linear1 =nn.Linear(1,configs.seq_len)
#         self.linear2 =nn.Linear(1,configs.enc_in)
#         self.embedding = nn.Linear(configs.enc_in, configs.d_model)
#         self.output = nn.Linear(configs.d_model, configs.enc_in)
#
# #pretrain sample
#     def forward(self, x, t):##(8,)    (8,24,7)
#         t = t.float().unsqueeze(1)      #目前t与x的batchsize一致比较好相加，如果模型扩散的能力长度变为50，t的shape就是（50,）,这里我是直接把它reshape成8么，该咋写？
#         t = self.linear1(t.unsqueeze(2))
#         t = self.linear2(t.unsqueeze(3))
#         x = x +t
#         x = self.embedding(x) #原来我想做成straight forward形式，但是这个Unet网络，最后一个维度必须是偶数，如果是奇数例如7，它decoder出来就变成6了，就少填充了一个，所以我做成了Enc-dec形式，如果直接sf形式，该咋写？
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = self.output(x)
#         return x


class UNet(nn.Module):
    def __init__(self, num_steps, seq_len, channels):
        super(UNet, self).__init__()
        self.num_units =  128

        self.linears = nn.ModuleList(
            [
                nn.Linear(seq_len * channels, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, seq_len * channels),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_steps, self.num_units),
                nn.Embedding(num_steps, self.num_units),
                nn.Embedding(num_steps, self.num_units),
            ]
        )

    def forward(self, x, t):
        B,_,S,C = x.shape
        x = x.view(B,-1)
        for idx, embedding_layer in enumerate(self.step_embeddings):  # 8 128
            t_embedding = embedding_layer(t)
            t_embedding = t_embedding.view(x.shape[0], -1,self.num_units)
            t_embedding = torch.mean(t_embedding, dim=1)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        x = x.view(B,1,S,C)
        return x

