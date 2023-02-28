# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/23
#@Author: TephrocactusHC
#@File: tools.py
#@Project: LeNet-5
#@Software: PyCharm
'''
import numpy as np

# 根据输入输出形状初始化参数
def initialize(shape):
    miu, sigma = 0, 0.1
    b_shape = (1,1,1,shape[-1]) if len(shape) == 4 else (shape[-1],)
    weight = np.random.normal(miu, sigma,  shape)
    bias  = np.ones(b_shape)*0.01
    return weight, bias

# 为了解决(32,32)和(28,28)之间的牛马问题
def zero_pad(input, pad):
    return np.pad(input, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))   
