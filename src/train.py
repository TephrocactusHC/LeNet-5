# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/27
#@Author: TephrocactusHC
#@File: train.py
#@Project: LeNet-5
#@Software: PyCharm
'''
from lenet import LeNet5
from loss import *
from utils import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1024)

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

np.random.seed(2022)
model = LeNet5()
model.compile_adam()

history = [[],[]]
x_train, x_val, y_train, y_val=main()
X,X_test,y,y_test=handle()
best_loss=999999999
print("Now we are training!")
for e in range(epochs):
    loss = []
    k = 0
    t = tqdm(range(0, x_train.shape[0], batch_size))
    for i in t:
        input = x_train[i : i + batch_size , :, :, :]
        labels = y_train[i : i + batch_size]
        loss.append(model(input, labels))
        grads = model.backward()
        model.apply_gradients(grads)
        k+=1
        if k % 50 == 0:
            history[0].append(np.mean(loss))
            history[1].append(mean_val_loss(model, x_val, y_val))
            val_acc = accuracy(model, x_val, y_val)
            t.set_description(f'epoch = {e + 1}, train_loss = {history[0][-1]}, val_loss = {history[1][-1]}, val_acc = {val_acc}')
            loss = []
            if history[1][-1] < best_loss:
                best_loss = history[1][-1]
                model.model_save()
print("Train Finished!!!")
"""
print("Now we are testing the model!")
test_acc = accuracy(model, X_test, y_test)
print("Test dataset acc is: ",test_acc)
"""
