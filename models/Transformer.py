# @Time    : 2023/6/9 4:33 下午
# @Author  : tang
# @File    : Transformer.py


import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer,Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp

from models.UNet import UNet
from layers.Revin import RevIN
import os
from Pretrain_diffusion.pre_main import Pretrain_diffusion



class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
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
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mse = nn.MSELoss()
        self.output_attention = False

        if configs.is_diffusion:
            self.UNet = UNet(num_steps=configs.noise_step, seq_len=self.seq_len,
                             channels=self.d_model)  # Unet初始化给seq_input_len 的维度
            model_path = '{}_noise_step:{}'.format(configs.data, configs.noise_step)
            pre_path = os.path.join('../Pretrain_diffusion/Pretrain_checkpoints', model_path)

            Pretrain = Pretrain_diffusion(configs).to(self.device)
            self.Pre_model = Pretrain.model
            self.Pre_model.load_state_dict(torch.load(pre_path + '/' + 'checkpoint.pth', map_location='cpu'))





        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, args=None):
        # 这个enc_self_mask就是batch_y(label+pre)
        enc_input = self.enc_embedding(x_enc, x_mark_enc)
        dec_input = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, attns = self.encoder(enc_input, attn_mask=enc_self_mask)
        if self.is_diffusion:
            B, S, C = enc_out.shape
            # Unet = self.Pre_model.UNet
            Unet =self.UNet
            orgin_enc_out= enc_out
            enc_out = torch.unsqueeze(enc_out, dim=1)  # 3->4维
            sample_x = self.Pre_model.sample(Unet, sample_num=self.sample_num, enc_out=enc_out)

            enc_out = sample_x.view(B, -1, S, C)
            enc_out = torch.mean(enc_out, dim=1, keepdim=True)
            enc_out = torch.squeeze(enc_out, dim=1)
            enc_out = enc_out + orgin_enc_out  # 残差





        dec_out = self.decoder(dec_input, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

