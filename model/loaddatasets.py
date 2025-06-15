import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.utils import shuffle

def loadmagic():
    data = pd.read_csv(r"./data/magic04_X.csv", header=None).values
    label = pd.read_csv(r"./data/magic04_y.csv", header=None).values
    for i in label:
        if i[0] == -1:
            i[0] = 0
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((10, 30))
    x_S2 = np.dot(data, matrix1)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1, y_S2 = torch.Tensor(label), torch.Tensor(label)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=50)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=50)
    return x_S1, y_S1, x_S2, y_S2

def loadadult():
    df1 = pd.read_csv(r"C:\\Users\\mouni\\OneDrive\\Documents\\master\\these\\code\\OLD3S\\model\\data\\adult.data",  header=None, skipinitialspace=True)
    df1.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    le = preprocessing.LabelEncoder()
    categorical_cols = ['b','d','f','g','h','i','j','n','o']
    for col in categorical_cols:
        le.fit(df1[col])
        df1[col] = le.transform(df1[col])
    data = np.array(df1.iloc[:, :-1])
    label = np.array(df1.o)
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((14, 30))
    x_S2 = np.dot(data, matrix1)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1, y_S2 = torch.Tensor(label), torch.Tensor(label)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2
