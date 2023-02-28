# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/23
#@Author: TephrocactusHC
#@File: loss.py
#@Project: LeNet-5
#@Software: PyCharm
'''
import numpy as np
from tqdm import tqdm
def mean_val_loss(lenet_model, x_val, y_val):
    batch_size = 512
    return np.mean([lenet_model(x_val[i:i+batch_size, :, :, :], y_val[i:i+batch_size]) for i in range(0, x_val.shape[0], batch_size)])

def accuracy(lenet_model, x, y):
    batch_size = 512
    acc = []
    for i in tqdm(range(0, x.shape[0], batch_size)):
        pred = lenet_model(x[i:i+batch_size, :, :, :], None, 'test')
        acc.append(np.count_nonzero(y[i:i+batch_size] == pred) / pred.shape[0])
    return np.mean(acc)

def predictions(model, x):
    batch_size = 512
    res = []
    for i in tqdm(range(0, x.shape[0], batch_size)):
        pred = model(x[i:i+batch_size, :, :, :], None, 'test')
        res += list(pred)
    return res

