# -*- coding: utf-8 -*-
'''
 @Time : 2021/4/19 22:19
 @Author : xinming
 @FileName: predict.py
 @Email : xinming_365@163.com
 @Software: PyCharm
'''


from cfg import Cfg
import torch
import os
from torch.utils.data import DataLoader
from dataset import Fe_Dataset
import matplotlib.pyplot as plt
import numpy as np
from QuantModel import Quant_CNN, Easy_CNN



@torch.no_grad()
def predict(model, cfg):
    test_dataset = Fe_Dataset(cfg.train_path, cfg, train=True)
    test_loader = DataLoader(test_dataset, batch_size=Cfg.batch_size, shuffle=False)
    running_mse = 0
    for i, (X, y) in enumerate(test_loader):
        # X = torch.from_numpy(X)
        x = X.to(device=device, dtype=torch.float32)
        # 转换 很重要！！
        y = y.to(device=device, dtype=torch.float32)

        y_pred = model(x)
        # print(X, X.shape)
        # print(y, y_pred)
        squared_error = torch.mean((y - y_pred) ** 2)
        running_mse += squared_error
    mse = running_mse / len(test_loader)

    print(
        "Average accuracy: {:.4f}".format(mse))


@torch.no_grad()
def plot_test(model, cfg):
    test_dataset = Fe_Dataset(cfg.test_path, cfg, train=False)
    # test_dataset = Fe_Dataset(cfg.train_path, cfg, train=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    running_mse = 0

    test_data = []
    pred_test_data = []
    for i, (X, y) in enumerate(test_loader):
        x = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        y_pred = model(x)
        test_data.extend(np.squeeze(y.cpu().numpy()))
        pred_test_data.extend(np.squeeze(y_pred.cpu().detach().numpy()))
    x = np.arange(len(test_data))
    plt.plot(x, test_data, 'r', label='Test data')
    plt.plot(x, pred_test_data, 'g', label='Quant_CNN')
    plt.ylabel('Last Price')
    plt.xlabel('Numbers of Data')
    plt.xlim(left=0)
    plt.legend()
    pic_name = 'Fe_Dataset_1_test_easycnn' + '.png'
    # pic_name = 'Fe_Dataset_1_train_easycnn' + '.png'
    out_dir = './pictures'
    plt.savefig(os.path.join(out_dir, pic_name), dpi=500)
    # plt.xticks([])
    plt.show()




if __name__ == '__main__':
    cp_file = os.path.join(Cfg.checkpoints, Cfg.cp_file)
    # model = Quant_CNN(n_classes=1)
    model = Easy_CNN(n_classes=1)
    model.load_state_dict(torch.load(cp_file))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device=device)
    plot_test(model=model, cfg=Cfg)
    # predict(model=model, cfg=Cfg)
