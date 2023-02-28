# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/22
#@Author: TephrocactusHC
#@File: optimizer.py
#@Project: LeNet-5
#@Software: PyCharm
'''
import numpy as np


#SGD优化器
class SGD_optimizer(object):
    def __init__(self) -> None:
        super().__init__()
        self.params = None, None
        self.optimzers =  {'SGD' : None }

    def opimizer_SGD(self, lr=0.001, reg = 0):
        self.lr,self.reg = lr,reg
        self.m_dw, self.m_db= None, None
        def SGD_update(dW, db):
            # 初始化参数
            if self.m_dw is None or self.m_db is None :
                self.m_dw, self.m_db =  np.zeros(dW.shape) , np.zeros(db.shape)
            if self.m_dw.shape != dW.shape or self.m_db.shape != db.shape:
                self.m_dw, self.m_db =  np.zeros(dW.shape) , np.zeros(db.shape)
            # 计算
            self.m_dw = self.lr*self.m_dw
            self.m_db = self.lr*self.m_db
            # 最后更新
            w = self.params[0] - (self.m_dw + self.reg*self.params[0])
            b = self.params[1] - (self.m_db + self.reg*self.params[1])
            # 这个好像挺简单的
            self.params = (w, b) 
        # 本来想写在一个类里边，但是没用abc库好像不能顺利继承，故放弃
        self.optimzers['SGD'] = SGD_update


#SGDMomentum放弃了，啥破玩意，现在都没人用了。。。


# Adam优化器
class Adam_optimizer(object):
    def __init__(self) -> None:
        super().__init__()
        self.params = None, None
        self.optimzers =  {'adam' : None }

    def optimizer_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
        self.beta1, self.beta2, self.epsilon, self.eta = b1, b2, epsilon, eta
        self.m_dw, self.m_db, self.v_dw, self.v_db = None, None, None, None
        def adam_update(dW, db, t = 1):
            # 初始化
            if self.m_dw is None or self.m_db is None or self.v_dw is None or self.v_db is None:
                self.m_dw, self.m_db, self.v_dw, self.v_db =  np.zeros(dW.shape) , np.zeros(db.shape), np.zeros(dW.shape), np.zeros(db.shape)
            if self.m_dw.shape != dW.shape or self.m_db.shape != db.shape:
                self.m_dw, self.m_db, self.v_dw, self.v_db =  np.zeros(dW.shape) , np.zeros(db.shape), np.zeros(dW.shape), np.zeros(db.shape)
            # 算个平均数
            self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dW
            self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
            self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dW**2)
            self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)
            # 正则化
            m_dw_corr = self.m_dw/(1-self.beta1**t)
            m_db_corr = self.m_db/(1-self.beta1**t)
            v_dw_corr = self.v_dw/(1-self.beta2**t)
            v_db_corr = self.v_db/(1-self.beta2**t)
            # 更新
            w = self.params[0] - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
            b = self.params[1] - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            # 最后设置参数
            self.params = (w, b) 
        # 然后设置
        self.optimzers['adam'] = adam_update
