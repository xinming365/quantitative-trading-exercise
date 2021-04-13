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


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * torch.tanh(nn.functional.softplus(x))
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation, bn=True):
        super().__init__()

        pad = (kernel_size-1)//2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=True))
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


class Quant_CNN(nn.Module):
    def __init__(self):
        super(Quant_CNN, self).__init__()
        self.conv1 = Conv_Bn_Activation(1, 64, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Activation(64, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(32, 64, 1, 1, 'mish')
        pass



