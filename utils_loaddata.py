import pandas as pd
import numpy as np
import common,os,torch

#读取eval-data并进行处理
def load_file(filepath='linguisticFeature/eval-data.xlsx'):
    dataframe = pd.read_excel(filepath)

    # mean = dataframe.iloc[-3,:]
    sd = dataframe.iloc[-2,1:]
    # median = dataframe.iloc[-1,:]
    print(dataframe.head())
    #根据删除方差低的参数
    sd = sd.sort_values(ascending=True)
    feature = sd.index.tolist()
    del_feature =feature[:11]
    print(del_feature)
    #加载处理好的Xtrain，Ytrain ,经测试，用处不大
    X = pd.read_csv('Xtrain.csv', index_col=0)
    Y = pd.read_csv('Ytrain.csv', index_col=0).astype(int)
    for i in del_feature:
        X.drop(columns=i)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y


# 加载语言特征数据集
def readXy_csv(Xfilepath,yfilepath):

    X = pd.read_csv(Xfilepath,index_col=0)
    y = pd.read_csv(yfilepath,index_col=0).astype(int)
    X = np.array(X)
    y = np.array(y)

    return X,y

#读取data2vec的embedding结果，embedding大小为768
def load_embeddingXy(dir,type='mean'):
    filepaths = common.iterbrowse(dir)
    X = []
    Y = []
    for filep in filepaths:
        filepath,filename  = os.path.split(filep)
        train_data = torch.load(filep).detach().numpy()

        x = train_data[:,:-1]
        x_mean = np.mean(x,axis=0)
        y = train_data[0,-1]
        if type=='mean':
            X.append(x_mean)
        else:
            X.append(x)
        Y.append(y)

    print('X shape is ',np.array(X).shape)
    print('Y shape is ',np.array(Y).shape)

    return np.array(X),np.array(Y)



# 将X和Y进行shuffle
def shuffle(X,y):
    shuffle_index = np.random.permutation(len(X))
    X = X[shuffle_index]
    y = y[shuffle_index]
    return X,y