# @Time    : 2023/6/9 4:23 下午
# @Author  : tang
# @File    : exp_main.py
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Transformer,  Linear, NLinear,DLinear
from utils.tools import EarlyStopping,adjust_learning_rate,visual,visual_stl
from utils.metrics import metric

from Pretrain_diffusion.pre_main import Pretrain_diffusion


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Linear': Linear,
            'NLinear':NLinear,
            'DLinear':DLinear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            pre_result = []
            gt = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,stl_data) in enumerate(train_loader):
                # x:(batchsize,seq_len,channels),y:(batchsize,lab_len+pre_len,channels)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #label len是有原值的，cat上pre_len的0
                # encoder - decoder

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x,stl_data)

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y, args = self.args)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)

                if self.args.features == 'M':
                  pre_result.append(outputs[0,:,-1].cpu().detach().numpy())
                  gt.append(batch_y[0, :, -1].cpu().detach().numpy())


                train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader,criterion)
                test_loss = self.vali(test_data, test_loader,criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            #draw pic
            pre_result = np.concatenate(pre_result, axis=0).reshape(-1,1)
            gt = np.concatenate(gt, axis=0).reshape(-1,1)

            folder_path = './visual/{}/train/'.format(self.args.data)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            train_result_path = 'pre_gt_epoch:{}'.format(str(epoch))
            visual(pre_result[0:720], gt[0:720], os.path.join(folder_path, train_result_path + '.pdf'))



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader,criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,stl_data) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x,stl_data)

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y,
                                                    args=self.args)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []

        #visual
        pre_result = []
        gt = []
        s = []
        t =[]
        l = []
        inputx = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,stl_data) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder


                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x,stl_data)
                else:

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, args =self.args)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                pre_result.append(pred[0,:,-1])
                gt.append(true[0,:,-1])

                s.append(stl_data[0][0,:,-1])
                t.append(stl_data[1][0,:,-1])
                l.append(stl_data[2][0,:,-1])
                inputx.append(batch_x[0, :, -1].detach().cpu().numpy())


        #visual pre_gt
        pre_result = np.concatenate(pre_result, axis=0)
        gt = np.concatenate(gt, axis=0)


        folder_path = './visual/{}/test/'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        visual(pre_result[0:200], gt[0:200], os.path.join(folder_path, 'pre_gt.pdf'))

        #visual stl
        s = np.concatenate(s, axis=0)[0:500]
        t = np.concatenate(t, axis=0)[0:500]
        l = np.concatenate(l, axis=0)[0:500]
        inputx = np.concatenate(inputx, axis=0)[0:500]

        folder_path = './visual/{}/test_STL/'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        visual_stl(s,t,l,inputx,os.path.join(folder_path, 'stl.pdf'))



        #result
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return