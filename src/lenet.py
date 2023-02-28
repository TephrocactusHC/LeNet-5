# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/25
#@Author: TephrocactusHC
#@File: lenet.py
#@Project: LeNet-5
#@Software: PyCharm
'''
from layers import Conv2D, Pooling, FC, RBF
from optimizer import Adam_optimizer
from activation import Activation
import numpy as np
import pickle as pk

class LeNet5(object):

    def __init__(self, input_shape = (32, 32, 1), name='LeNet5') -> None:
        super().__init__()
        assert input_shape != None
        self.name = name
        self.layers = [
            Conv2D(6, (5,5)),
            Activation('relu'),
            Pooling(2),
            Conv2D(16, (5,5)),
            Activation('relu'),
            Pooling(2),
            Conv2D(120, (5,5)),
            Activation('relu'),
            FC(84),
            Activation('relu'),
            RBF(10)
        ]
        self.input_shape = input_shape
        prev_input_shape = input_shape
        for layer in self.layers:
           prev_input_shape = layer.init_layer(prev_input_shape).out_shape

    def total_params(self, params : 'list(tuple)') -> int:
        if not params: return 0
        return np.sum([np.prod(e.shape) for e in params])

    def compile_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
         self.optimizer = 'adam'
         for l in self.layers:
            if isinstance(l, Adam_optimizer): l.optimizer_adam(b1=b1, b2=b2, epsilon=epsilon, eta=eta)

    def __call__(self, input, label=None, mode = 'train'):
        o = input
        for layer in self.layers:
            if isinstance(layer, RBF): o = layer(o, label, mode)
            else:
                o = layer(o)
        return o

    def backward(self):
        dout = 1
        grads = {}
        for layer in reversed(self.layers):
            dW, db, dout = layer.__compute_gradients__(dout)
            if dW is None and db is None: continue
            grads[layer] = (dW, db)
        return grads

    def apply_gradients(self, gradients):
        for k,v in gradients.items():
            dW, db = v
            if isinstance(k, Adam_optimizer): k.optimzers[self.optimizer](dW , db)

    def model_save(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.params)
        with open('model.pickle', 'wb') as f:
            pk.dump(weights, f, pk.HIGHEST_PROTOCOL)
        print("Model saved successfully!")

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pk.load(f)
        for i, layer in enumerate(self.layers):
            layer.params = weights[i]
        print("Model loaded successfully!")