# -*- coding: utf-8 -*-
'''
 @Time : 2021/4/14 20:43
 @Author : xinming
 @FileName: trains.py
 @Email : xinming_365@163.com
 @Software: PyCharm
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
import os, sys, math
from torch.nn import functional as F
from cfg import Cfg
from QuantModel import Quant_CNN
from dataset import Fe_Dataset
import logging


def train(save_cp=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Fe_Dataset(Cfg.train_path, Cfg)
    test_dataset = Fe_Dataset(Cfg.test_path, Cfg)

    train_loader = DataLoader(train_dataset, batch_size=Cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Cfg.batch_size, shuffle=False)

    loss_func = nn.MSELoss()
    print(Cfg.n_classes)
    model = Quant_CNN(Cfg.n_classes)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    save_prefix = 'Quant_CNN'
    model.to(device=device)
    model.train()
    for epoch in range(Cfg.epochs):
        epoch_loss = 0
        print('epoch {}'.format(epoch))
        for i, (X, y) in enumerate(train_loader):
            # X = torch.from_numpy(X)
            x = X.to(device=device, dtype=torch.float32)
            # 转换 很重要！！
            y = y.to(device=device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()

            epoch_loss += loss.item()
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i, len(train_loader), loss.item()))

            optimizer.step()
            optimizer.zero_grad()

        if save_cp:
            os.makedirs(Cfg.checkpoints, exist_ok=True)
        save_path = os.path.join(Cfg.checkpoints, f'{save_prefix}_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)


def evaluate():
    pass


if __name__=='__main__':
    train()








