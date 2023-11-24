from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils_loaddata import readXy_csv,shuffle
import pandas as pd
import numpy as np
import os

#进行网格搜索SVC模型
def SVC_grid_research(Xtrain,ytrain,Xtest,ytest):
    # 定义参数空间
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['auto',],
        'degree':[1,2,3,4,5],
        'kernel': ['poly','rbf','linear'],
        'cache_size':[100,200,300,500,1000,3000,5000]
    }

    # 定义模型
    model = SVC()

    # 进行网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=10)
    print('grid searching...')
    grid_search.fit(Xtrain, ytrain.ravel())

    # 输出最佳参数组合和得分
    print("SVC Best parameters:", grid_search.best_params_)
    print("SVC Best 10cv score:", grid_search.best_score_)

    # 预测验证集
    y_pred = grid_search.predict(Xtest)

    # 计算准确率
    accuracy = accuracy_score(ytest, y_pred)

    # 输出准确率
    print("SVC test Accuracy:", accuracy)

    return grid_search




def LR_grid_research(Xtrain, ytrain, Xtest, ytest):
    # 定义参数空间
    param_grid = {
        'C': [1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    # 定义模型
    model = LR(max_iter=2000)

    # 进行网格搜索
    grid_search = GridSearchCV(model, param_grid, cv=5)
    print('grid searching...')
    grid_search.fit(Xtrain, ytrain.ravel())

    # 输出最佳参数组合和得分
    print("LR Best parameters:", grid_search.best_params_)
    print("LR Best 10cv score:", grid_search.best_score_)

    # 预测验证集
    y_pred = grid_search.predict(Xtest)

    # 计算准确率
    accuracy = accuracy_score(ytest, y_pred)

    # 输出准确率
    print("LR test Accuracy:", accuracy)

    return grid_search




if __name__ == '__main__':
    #读取数据
    Xtrain,ytrain = readXy_csv('data/PAR/Xtrain.csv','data/PAR/Ytrain.csv')
    Xtest,ytest = readXy_csv('data/PAR/Xtest.csv','data/PAR/Ytest.csv')



    print(Xtrain.shape)
    print(Xtest.shape)

    #将训练集和测试集特征集合并
    # X = np.vstack((Xtrain,Xtest))
    # print(X.shape)


    #将所有数据的特征集进行归一化，并再切分为训练集和测试集
    # X = scale_X(X)
    # Xtrain = X[:166]
    # Xtest = X[166:]
    # print(Xtrain.shape)
    # print(Xtest.shape)

    #打乱训练集的顺序
    Xtrain,ytrain = shuffle(Xtrain,ytrain)

    # SVC_grid_research(Xtrain,ytrain,Xtest,ytest)
    LR_grid_research(Xtrain,ytrain,Xtest,ytest)




