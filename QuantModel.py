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

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation, bn=True):
        super().__init__()

    def forward(self, x):
        pass


class Quant_CNN(nn.Module):
    def __init__(self):
        super(Quant_CNN, self).__init__()
        pass


