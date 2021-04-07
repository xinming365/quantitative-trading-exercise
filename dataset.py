# -*- coding: utf-8 -*-
'''
@Time :2021/04/07
@Author : xinming
'''
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from cfg import Cfg


class Fe_Dataset(Dataset):
    def __init__(self, data_path, cfg, train=True):
        super(Fe_Dataset, self).__init__()
        self.cfg=cfg
        self.train=train
        df_array = pd.read_csv(data_path).to_numpy()
        self.data= np.delete(df_array, 0, axis=1)

    def __len__(self):
        return len(self.data) - self.cfg.time_range +1 -self.cfg.t

    def __getitem__(self, index):
        time_range = self.cfg.time_range
        t = self.cfg.t
        label_i = self.cfg.label
        out_img = np.zeros([time_range, self.cfg.w, 1])
        start_i = index * time_range
        end_i = start_i + time_range

        img = self.data[start_i:end_i,:]
        out_label = self.data[end_i + t -1 , label_i]
        out_img = img
        return out_img, out_label


if __name__ == '__main__':

    # data_path = 'F:\GitHub\My_data_set\data\dataset.csv'
    dataset = Fe_Dataset(Cfg.test_path, Cfg)
    for i in range(10):
        out_img, out_label = dataset.__getitem__(i)
        print(out_img)
        print(out_label)

