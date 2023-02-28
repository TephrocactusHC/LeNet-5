# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/22
#@Author: TephrocactusHC
#@File: layers.py
#@Project: LeNet-5
#@Software: PyCharm
'''
from RBF_bitmap import bitmap_initialize
from optimizer import Adam_optimizer
from tools import *

# 卷积层
class Conv2D(Adam_optimizer):
    """
    注意，卷积层实现了im2col
    这样可以避免四重for循环
    (h, w, f, b) x (b, k) 
    在算导数的时候也可以直接使用切片来计算
    """

    def __init__(self, filter_num, kernel_shape, stride = 1, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.in_shape, self.out_shape = None, None
        self.kernel_shape, self.filter_num = kernel_shape, filter_num
        self.params = None, None
        self.padding, self.stride = padding, stride
        
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (in_shape[0] - self.kernel_shape[0] + 1, in_shape[1] - self.kernel_shape[1] + 1 , self.filter_num)
        self.param_shape = self.kernel_shape + (self.in_shape[-1], self.filter_num)
        self.params = initialize(self.param_shape)
        return self

    def __call__(self, input):
        z = np.zeros((input.shape[0], ) + self.out_shape)
        input_pad = zero_pad(input, self.padding)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                slice = input_pad[:, h*self.stride:h*self.stride+self.kernel_shape[0], w*self.stride:w*self.stride+self.kernel_shape[1], :]  
                z[:, h, w, :] = np.tensordot(slice, self.params[0], axes=([1,2,3],[0,1,2])) + self.params[1] 
        # 缓存梯度值
        self.cache = input 
        return z
   
    def __compute_gradients__(self, dout):
        dinput = zero_pad(np.zeros(self.cache.shape), self.padding)
        # 算导数
        dW = np.zeros(self.params[0].shape) 
        db = np.zeros(self.params[1].shape) 
        cache_padded = zero_pad(self.cache, self.padding) 
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s, e = (h*self.stride, h*self.stride+self.kernel_shape[0]), (w*self.stride, w*self.stride+self.kernel_shape[1])
                slice = cache_padded[:,s[0]:s[1], e[0]:e[1], :] 
                dinput[:, s[0]:s[1], e[0]:e[1], :] += np.transpose(self.params[0] @ dout[:, h, w, :].T, (3,0,1,2))
                dW += np.matmul(np.transpose(slice, (1,2,3,0)), dout[:, h, w, :]) 
                db += np.sum(dout[:, h, w, :], axis=0)
        return dW, db, dinput


# 池化
class Pooling(Adam_optimizer):

    def __init__(self, kernel_shape, stride = 2, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.kernel_shape = kernel_shape
        self.params = None, None
        self.padding, self.stride = padding, stride
        self.in_shape, self.out_shape = None, None
    
    def init_layer(self, in_shape):
        assert len(in_shape) == 3
        self.in_shape = in_shape
        self.out_shape = ( (in_shape[0] - self.kernel_shape) // self.stride + 1, (in_shape[1] - self.kernel_shape + 1) //self.stride + 1 , in_shape[-1] )
        self.params = np.random.normal(0, 0.1, (1,1,1,self.in_shape[-1])),  np.random.normal(0, 0.1, (1,1,1,self.in_shape[-1])) 
        return self

    def __call__(self, input):
        o = np.zeros((input.shape[0], ) + self.out_shape) 
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                slice = input[:, h*self.stride:h*self.stride+self.kernel_shape, w*self.stride:w*self.stride+self.kernel_shape, :] 
                o[:, h, w, :] = np.average(slice, axis=(1,2))
        self.cache = (input, o)
        output = o * self.params[0] + self.params[1]
        return output

    def __compute_gradients__(self, dout):
        prev_input, out_ = self.cache 
        db = np.mean(dout, axis=(0,1,2), keepdims=True) 
        dW = np.mean(np.multiply(dout, out_), axis = (0,1,2), keepdims=True)
        dout_after = dout * self.params[0] 
        dinput = np.zeros(prev_input.shape)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s , e = (h*self.stride, h*self.stride+self.kernel_shape), (w*self.stride, w*self.stride+self.kernel_shape)
                da = dout_after[:, h, w, :][:,np.newaxis,np.newaxis,:]
                dinput[:, s[0]: s[1], e[0]: e[1], :] += np.repeat(np.repeat(da, self.kernel_shape, axis=1), self.kernel_shape, axis=2)/(self.kernel_shape * self.kernel_shape)
        return dW, db, dinput



# 全连接层
class FC(Adam_optimizer):
    """
    BP的内容
    """
    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def __call__(self, input):
        self.cache = (input.reshape((input.shape[0]), np.prod(list(input.shape)[1:])), input.shape)
        return  np.matmul(self.cache[0] , self.params[0]) + self.params[1]
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (np.product(self.in_shape), self.out_shape[0])
        self.params = initialize(self.param_shape)
        return self

    def __compute_gradients__(self, dout):
        return np.matmul(self.cache[0].T ,  dout), np.sum(dout.T, axis = 1), np.matmul(dout, self.params[0].T).reshape(self.cache[1])
    
   
#径向基层
class RBF(object):
    """
    和全连接很像。。。
    """
    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None

    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (1, 1)
        self.param_shape = (self.outputs, np.product(self.in_shape))
         # 这里用了那个bitmap，等于最后是和那些数字比
        self.params = (bitmap_initialize(), )
        assert self.param_shape == self.params[0].shape
        return self

    def __call__(self, input, label, mode = 'test'):
        if mode == 'test': return self.predict(input)
        self.cache = input , self.params[0][label, :]
        # 计算欧氏距离的牛马函数
        return np.sum(0.5 * np.sum((self.cache[0] - self.cache[1]) ** 2, axis = 1, keepdims=True))

    def predict(self, input):
        # 预测就比较简单，就是看哪个欧式距离最近，就是哪个类
        src_dst_sum = np.sum(np.square(input[:, np.newaxis, :] - np.array([self.params[0]] * input.shape[0])), axis=2)
        pred = np.argmin(src_dst_sum, axis = 1)
        return pred

    def __compute_gradients__(self, dout = 1):
        return None, None, dout * (self.cache[0] - self.cache[1])


    
