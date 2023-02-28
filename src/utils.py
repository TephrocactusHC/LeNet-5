# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/12/24
#@Author: TephrocactusHC
#@File: utils.py
#@Project: LeNet-5
#@Software: PyCharm
'''
# 按理说可以直接用框架里的mnist_load()函数，但是为了避险吧，从网上找了一个
import struct #用来读取数据的
from tools import *

test_image_path = r'../MNIST/t10k-images-idx3-ubyte'
test_label_path = r'../MNIST/t10k-labels-idx1-ubyte'
train_image_path = r'../MNIST/train-images-idx3-ubyte'
train_label_path = r'../MNIST/train-labels-idx1-ubyte'
training_set = (train_image_path, train_label_path)
test_set = (test_image_path, test_label_path)


def load_dataset(dataset):
    (image, label) = dataset
    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (img, lbl)
# 正则化
# 如果不正则化，据说在十几轮的时候就会出现梯度爆炸
def normalize(image):
    image -= image.min()
    image = image / image.max()
    image = image * 1.275 - 0.1
    return image
def handle():
    (train_images, train_labels) = load_dataset(training_set)
    (test_images, test_labels) = load_dataset(test_set)
    X = normalize(zero_pad(train_images[:, :, :, np.newaxis], 2))
    X_test = normalize(zero_pad(test_images[:, :, :, np.newaxis], 2))
    y = train_labels
    y_test = test_labels
    return X,X_test,y,y_test

def train_test_split(X, y, ratio = 0.1):
    indices = list(np.random.choice(X.shape[0] , int(X.shape[0] * ratio), replace=False))
    left_over = list(set(range(X.shape[0])) - set(indices))
    return X[left_over, :, :, :], X[indices, : , :, :], y[left_over], y[indices]

def get_split(X,y):
    x_train, x_val, y_train, y_val = train_test_split(X, y, 0.1)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    return x_train, x_val, y_train, y_val


def main():
    X, X_test, y, y_test=handle()
    x_train, x_val, y_train, y_val = train_test_split(X, y, 0.1)
    return x_train, x_val, y_train, y_val
