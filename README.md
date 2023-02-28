# 关于项目
一个LeNet-5的python实现，只使用numpy而不使用框架。

## 框架结构
----------
    ├── MNIST #数据集
    ├── model #保存训练好的模型，防止因为重新训练导致模型被覆盖
    ├  ├── model.pickle #训练好的模型  
    ├── src  #代码
    ├  ├── activation.py #激活函数
    ├  ├── analysis.ipynb #结果展示
    ├  ├── layers.py #各个层的具体实现
    ├  ├── lenet.py #网络搭建
    ├  ├── loss.py #损失函数     
    ├  ├── model.pickle #训练好的模型     
    ├  ├── optimizer.py #优化器    
    ├  ├── RBF_bitmap.py #径向基层的bitmap     
    ├  ├── test.py #测试模型
    ├  ├── tools.py #一些其他函数     
    ├  ├── train.py #训练模型 
    ├  ├── utils.py # 一些其他函数                 
    └── README.md
-----------

## 运行环境

所需依赖库如下所示：
- numpy #主要实现LeNet5
- tqdm #训练过程可视化展示
- struct #数据读取
- pickle #模型保存和加载
- argparse #超参数设置，可使用命令行训练模型

关于各个库的具体版本，我这里没有安装pipreqs，无法只针对本项目生成requirements.txt，我猜应该都差不多吧:)

## 使用方法
### 训练模型

直接使用如下命令
```
cd src
python train.py --epochs 50 --batch_size 1024
```
来训练模型，超参数自己调节。我的超参数见实验报告。

epochs并不需要调的特别大，根据观察在20轮左右往往就能达到接近最优的结果了，后面就是在微弱地增长了。

关于batch_size，反正我的内存够大，能这么设置，如果内存不够大或者谨慎起见的话，还是建议设的小一些。我习惯于设为2的整数幂，应该是有论文论证过这一点的。

### 测试模型
```
cd src
python test.py
```
### 结果分析
可以打开analysis.ipynb进行查看

