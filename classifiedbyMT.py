import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,RobustScaler,MaxAbsScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit#分层洗牌分割交叉验证
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR

import torch,common,os
from collections import defaultdict


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


    return np.array(X),np.array(Y)



# 将X和Y进行shuffle
def shuffle(X,y):
    shuffle_index = np.random.permutation(len(X))
    X = X[shuffle_index]
    y = y[shuffle_index]
    return X,y




if __name__ == '__main__':
    train_dir = 'audio-embedding/AD2021_PAR+INV/20sec/train'
    test_dir = 'audio-embedding/AD2021_PAR+INV/20sec/test'

    X_train, Y_train = load_data(train_dir)
    X_test, Y_test = load_data(test_dir)

    print(X_train.shape)
    print(X_test.shape)

    # 将特征集进行归一化
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)
    #
    # #打乱训练集的顺序
    X_train, Y_train = shuffle(X_train, Y_train)

    # cls = SVC(kernel = "poly",gamma='auto',degree = 1,cache_size = 100,C=10)
    # cls = RFC(n_estimators = 100,random_state =1,max_features = 15)
    cls = LR(solver="sag", max_iter=2000)
    cls.fit(X_train, Y_train)
    accuracy = cls.score(X_test, Y_test)
    print("test Accuracy: %0.2f" % accuracy)

    scores = cross_val_score(cls, X_train, Y_train, cv=5)
    print("5-cv Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

