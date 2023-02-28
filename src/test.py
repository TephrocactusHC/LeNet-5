# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/27
#@Author: TephrocactusHC
#@File: test.py
#@Project: LeNet-5
#@Software: PyCharm
'''
from lenet import LeNet5
from loss import *
from utils import *
import numpy as np

model = LeNet5()
model.compile_adam()
model.load_weights("model.pickle")

print("Now we are testing!")

X,X_test,y,y_test=handle()
acc=accuracy(model, X_test, y_test)
print("Testing acc is: ", acc)