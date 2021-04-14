# -*- coding: utf-8 -*-
'''
 @Time : 2021/4/7 15:54
 @Author : xinming
 @FileName: cfg.py
 @Email : xinming_365@163.com
 @Software: PyCharm
'''

from easydict import EasyDict
Cfg = EasyDict()
Cfg.time_range = 8
Cfg.width = 8
Cfg.w = Cfg.width
Cfg.t = 1
Cfg.label = 0
Cfg.train_path = 'F:\GitHub\My_data_set\data/train.csv'
Cfg.test_path = 'F:\GitHub\My_data_set\data/test.csv'
Cfg.n_classes = 1
Cfg.batch_size = 64
Cfg.learning_rate = 0.003
Cfg.decay = 0.0005
Cfg.momentum=0.949
Cfg.train_optimizer ='adam'
Cfg.train_epochs = 300
Cfg.epochs=5