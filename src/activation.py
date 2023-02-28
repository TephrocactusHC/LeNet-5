# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/21
#@Author: TephrocactusHC
#@File: activation.py
#@Project: LeNet-5
#@Software: PyCharm
'''
import numpy as np
"""
损失函数们f(x)，和他们的导函数们f'(x)
第一个squashing function不太理解是什么东西，看完论文也没懂
此处借鉴了https://github.com/mattwang44/LeNet-from-Scratch的仓库内容，他加入了很多现代常用的损失函数
我都整合在了我这里，值得注意的是他那里面还提供了二阶导数，但是二阶导数根本不会被用到，不知道是什么意义
当然，最后只用到了relu(过于牛马)
"""
__loss_functions__ =  {
    'squashing' : (lambda x : 1.7159*np.tanh(2*x/3), lambda x : 1.14393*(1-np.power(np.tanh(2*x/3),2))),
    'mse' : (lambda x,y : np.mean(np.square(y - x)), lambda x,y : -np.mean(y - x, axis=1)),
    'sigmoid':(lambda x : 1 / (1 + np.exp(-x)), lambda x : np.exp(-x) / np.power((1+np.exp(-x)),2)),
    'relu' : (lambda x : np.maximum(x,0), lambda x : (x>0)*1 ),
    'tanh' : (lambda x: np.tanh(x), lambda x: 1/np.power(np.cosh(x),2)),
    'PRelu' : (lambda x: np.maximum(x,0.1*x), lambda x : np.where(x>0, 1,0.1)),
    'ELU' : (lambda x: np.where(x > 0, x, 0.5*(np.exp(x) - 1)),lambda x: np.where(x > 0, 1, np.where(x > 0, x, 0.5*(np.exp(x) - 1)+0.5)))
}


# 激活函数
class Activation(object):

#  这部分说实话默认哪个都无所谓的样子啊
    def __init__(self, name = 'tanh') -> None:
        super().__init__()
        self.name = name
        self.cache = None
        self.params = None
        self.in_shape, self.out_shape = None, None
        self.func, self.func_d = __loss_functions__[name]

    def __call__(self, input):
        # 函数调用
        output, self.cache = self.func(input), input
        return output
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape
        return self

    def __compute_gradients__(self, dout):
        # 算梯度，激活函数没有dW和db,只有dout，所以前两个是NONE
        return None, None, dout * self.func_d(self.cache)
    