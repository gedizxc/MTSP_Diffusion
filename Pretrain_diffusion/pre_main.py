# @Time    : 2023/7/18 9:12 上午
# @Author  : tang
# @File    : pre_main.py

from models.Diffusion import Diffusion
from data_provider.data_factory import data_provider
import os
import torch
import torch.nn as nn
import time
from utils.tools import EarlyStopping,adjust_learning_rate
from torch import optim

from models.UNet import UNet
from models.Diffusion_Unet import Diffusion_Unet
import numpy as np




class Pretrain_diffusion(nn.Module):
    def __init__(self,args):
        super(Pretrain_diffusion, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.noise_steps = args.noise_step
        self.model = Diffusion_Unet(args,noise_steps=self.noise_steps)

    def _get_data(self,flag):
        data_set,data_loader = data_provider(self.args,flag)
        return data_set, data_loader

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device



def pretrain(args):
    Pretrain = Pretrain_diffusion(args)
    #Data
    train_data, train_loader = Pretrain._get_data(flag='train')
    if not Pretrain.args.train_only:
        vali_data, vali_loader = Pretrain._get_data(flag='val')
        test_data, test_loader = Pretrain._get_data(flag='test')
    model_path = '{}_noise_step:{}'.format(args.data,args.noise_step)
    path = os.path.join('./Pretrain_checkpoints/{}'.format(args.seq_len),model_path)
    if not os.path.exists(path):
        os.makedirs(path)

    model = Pretrain.model #diffusion
    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        model.train()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,stl_index) in enumerate(train_loader):
            # x:(batchsize,seq_len,channels),y:(batchsize,lab_len+pre_len,channels)
            iter_count += 1
            batch_x = batch_x.float().to(model.device)
            noise, predicted_noise = model(batch_x)
            loss = mse(noise,predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        vali_loss = prevali(mse,model,vali_loader)


        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)



def prevali(mse,model,vali_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,stl_index) in enumerate(vali_loader):
            batch_x = batch_x.float().to(model.device)
            noise, predicted_noise = model(batch_x)
            loss = mse(noise, predicted_noise)
            total_loss.append(loss.cpu())
    total_loss=np.average(total_loss)
    return total_loss


