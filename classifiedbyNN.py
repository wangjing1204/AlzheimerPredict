import torch
import common,os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
import torch.nn.functional as Fun
import matplotlib.pyplot as plt
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)# 定义隐藏层网络
        self.out = torch.nn.Linear(n_hidden, n_output)   # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hidden(x))      # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        x = self.out(x)                   # 输出层不用激活函数
        return x

def load_file():
    Xtrain = pd.read_csv('Xtrain.csv', index_col=0)
    Ytrain = pd.read_csv('Ytrain.csv', index_col=0)
    Xtest = pd.read_csv('Xtest.csv', index_col=0)
    Ytrain.astype(int)
    add = (Ytrain == 0).astype(int)
    # Ytrain = pd.concat([Ytrain,add],axis =1)
    Ytrain = np.array(Ytrain)
    Ytrain = Ytrain.ravel()
    return Xtrain,Ytrain,Xtest


def load_data(dir):
    filepaths = common.iterbrowse(dir)
    X = []
    Y = []
    for filep in filepaths:
        filepath,filename  = os.path.split(filep)
        if '-' not in filename:

            train_data = torch.load(filep).detach().numpy()

            x = train_data[:,:-1]
            x_mean = np.mean(x,axis=0)
            y = train_data[0,-1]

            X.append(x_mean)
            Y.append(y)

    return X,Y

def train(Xtrain,Ytrain,lr):
    input = torch.tensor(np.float64(Xtrain), dtype=torch.float32)
    # D = torch.Tensor(np.array(Xtrain))
    label = torch.LongTensor(Ytrain)
    net = Net(n_feature=30, n_hidden=20, n_output=2)  # n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化器选用随机梯度下降方式
    loss_func = torch.nn.CrossEntropyLoss()  # 对于多分类一般采用的交叉熵损失函数

    out = net(input)
    out.shape
    label.squeeze().shape

    # 4. 训练数据
    loss_list = []
    t_list = []
    for t in range(400):
        out = net(input)  # 输入input,输出out
        loss = loss_func(out,label.squeeze())  # 输出与label对比
        if t % 100 == 0:
            loss_list.append(loss.detach().numpy())
            t_list.append(t)
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 前馈操作
        optimizer.step()  # 使用梯度优化器


    return net, loss_list ,t_list,acuracy

def predict(net,Xtest,Ytest):
    Xtest = torch.tensor(np.float64(Xtest), dtype=torch.float32)
    label = torch.LongTensor(Ytest)

    predict_y = net(Xtest)
    target_y = np.array(label)
    acuracy = float((predict_y == target_y).astype(int).sum()) / float(target_y.size)
    print("ADcontest acuracy", acuracy)

    return predict_y,acuracy

def save_model(net,acuracy_0,acuracy):
    if acuracy > acuracy_0:
        acuracy_0 = acuracy
        print("save model")
        # 保存模型语句
        torch.save(net.state_dict(),"model.pth")
        return acuracy_0
    else:
        return acuracy_0
def plot_LearningCarve(list_lr,acuracy_list):
    fig, axes = plt.subplots(1,2
                           ,figsize=(8,4)
                           )
    for i,ax in enumerate(axes):
#         if i ==0:
        ax.plot(X[i],Y[i])
        ax.set_ylabel('acurracy', fontdict={"family": "Times New Roman", "size": 15})
#         ax1= ax.get_yticks()
#         else:
#             ax.plot(X[i],Y[i])
#             ax.set_yticks(ax1)
# def plot_line_loss(X,Y):
#     plt.plot(X,Y,label="loss")
#     plt.xlabel('Num of Iteration', fontdict={"family": "Times New Roman", "size": 15})
#     plt.ylabel('Loss', fontdict={"family": "Times New Roman", "size": 15})
#     plt.legend()
#     plt.show()
def plot_line_loss(X,Y):
    fig, axes = plt.subplots(1,1
                               ,figsize=(8,4)
                               )
    axes.plot(X,Y)
    axes.set_xlabel('Num of Iteration', fontdict={"family": "Times New Roman", "size": 15})
    axes.set_ylabel('Loss', fontdict={"family": "Times New Roman", "size": 5})

if __name__ =="__main__":
    Xtrain,Ytrain,Xtest = load_file()
    print(Xtrain.shape,Ytrain.shape,Xtest.shape)
    Xtrain, Ytrain, Xtest = load_file()
    print(Xtrain.shape, Ytrain.shape, Xtest.shape)




    #加载数据
    epochs = 10
    acuracy_0 = 0.9
    acuracy_list1 = []
    list_lr = []
    acuracy_list = []
    #初始化

    lr_list0 = np.linspace(0.001, 0.1, 30)
    lr_list1 = np.linspace(0.0001, 0.0015, 20)
    X = [lr_list0, lr_list1]
    Y = []

    for i in X:
        for lr in i:
            net, loss_list ,t_list,acuracy = nn(Xtrain, Ytrain, lr)
            acuracy_list.append(acuracy)
            acuracy_0 = save_model(net, acuracy_0, acuracy)  # 根据精准度保存模型
        Y.append(np.array(acuracy_list))
        acuracy_list = []
    print(predict(net, Xtest))
    plot_LearningCarve(X, Y)
    plot_line_loss(t_list,loss_list)
    print("Finish the best Acuracy is %.03f" % acuracy_0)
    print(predict(net,Xtest))




