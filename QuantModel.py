# -*- coding: utf-8 -*-
'''
 @Time : 2021/4/7 16:41
 @Author : xinming
 @FileName: QuantModel.py
 @Email : xinming_365@163.com
 @Software: PyCharm
'''

import torch
from torch import nn
import torch.nn.functional as F
from cfg import Cfg


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(nn.functional.softplus(x))
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation, bn=True, bias=False):
        super().__init__()

        pad = (kernel_size-1)//2
        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=True))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(num_features=out_channels))
        if activation == 'mish':
            self.conv.append(Mish())
        elif activation=='relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif activation=='leaky':
            self.conv.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif activation=='linear':
            pass
        else:
            print("activate function error !")

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x


class DownSample1(nn.Module):
    """
    input: number of channels:1 ; Input dimension(N, 1, H_in, W_in)
    output: number of channels :64; Output dimension(N, 64, H_out, W_out)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1, 32, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # Resblock, shortcut
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4

        x7 = self.conv7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8



class Quant_CNN(nn.Module):
    def __init__(self, n_classes):
        super(Quant_CNN, self).__init__()
        self.down1 = DownSample1()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'linear', bn=False, bias=True)
        self.linear = nn.Linear(64*16, n_classes)

    def forward(self, input):
        d1 = self.down1(input)
        x1 = self.conv1(d1)
        x2 = self.conv2(x1)
        x2 = torch.flatten(x2, start_dim=1)
        x3 = self.linear(x2)
        return x3


class Easy_CNN(nn.Module):
    def __init__(self, n_classes):
        super(Easy_CNN, self).__init__()
        self.conv1 = Conv_Bn_Activation(1, 64, 3, 1, 'linear', bn=False, bias=True)
        self.conv2 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky', bn=False, bias=True)
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'linear', bn=False, bias=True)
        self.linear = nn.Linear(4096, n_classes)

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # print(x3.shape)
        x4 = torch.flatten(x3, start_dim=1)
        # print(x4.shape)
        x5 = self.linear(x4)
        return x5


if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # model = Quant_CNN(Cfg.n_classes).to(device)
    model = Easy_CNN(Cfg.n_classes).to(device)
    print(model)


