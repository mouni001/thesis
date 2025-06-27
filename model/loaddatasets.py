import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.utils import shuffle
from river import datasets
from sklearn.preprocessing import StandardScaler

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

def loadcar():
    df = pd.read_csv('./data/car.data', header=None)
    categorical_cols = [0, 1, 2, 3, 4]  # all input columns
    le = preprocessing.LabelEncoder()

    # Encode all categorical input columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Encode the label column (column 5)
    df[5] = le.fit_transform(df[5])
    label = df[5].astype(int).values  # Force to integer array

    data = df.iloc[:, :-1].astype(float).values  # Convert to float before scaling

    data = preprocessing.scale(data)
    matrix = np.random.RandomState(1314).random((data.shape[1], 30))
    x_S2 = np.dot(data, matrix)

    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1 = y_S2 = torch.tensor(label, dtype=torch.long)  # Explicit tensor type

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)


    return x_S1, y_S1, x_S2, y_S2


def loadarrhythmia():
    df = pd.read_csv('./data/arrhythmia.data', header=None, na_values='?')
    df = df.dropna()

    data = df.iloc[:, :-1].values
    label = df.iloc[:, -1].values
    label = np.array([0 if x == 1 else 1 for x in label])  # binary

    data = preprocessing.scale(data)
    matrix = np.random.RandomState(1314).random((data.shape[1], 30))
    x_S2 = np.dot(data, matrix)

    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1 = y_S2 = torch.Tensor(label)

    # Shuffle
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)


    return x_S1, y_S1, x_S2, y_S2



def loadthyroid():
    df = pd.read_csv('./data/new-thyroid.data', header=None)

    data = df.iloc[:, :-1].values
    label = df.iloc[:, -1].values
    label = np.array([0 if x == 1 else 1 for x in label])  # binary

    data = preprocessing.scale(data)
    matrix = np.random.RandomState(1314).random((data.shape[1], 30))
    x_S2 = np.dot(data, matrix)

    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1 = y_S2 = torch.Tensor(label)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)

    return x_S1, y_S1, x_S2, y_S2




def loadinsects():
    df = pd.read_csv('./data/INSECTS.csv', header=None)

    data = df.iloc[:, :-1].values.astype(np.float32)
    label = df.iloc[:, -1].values

    # Encode labels into 0...C-1
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(label)

    # ✅ Convert to binary: class 0 stays 0, all others become 1
    label = np.array([0 if x == 0 else 1 for x in label])
    print("Labels:", np.unique(label, return_counts=True))


    # Normalize features
    data = preprocessing.scale(data)

    # Define drift point (e.g., 80% into the stream)
    drift_point = int(len(data) * 0.8)

    # Pre-drift data
    x1 = data[:drift_point]
    y1 = label[:drift_point]

    # Post-drift data
    x2 = data[drift_point:]
    y2 = label[drift_point:]

    # Project x2 into new feature space (simulate feature evolution)
    matrix = np.random.RandomState(1314).random((x2.shape[1], 30))
    x2_proj = np.dot(x2, matrix)

    x_S1 = torch.sigmoid(torch.tensor(x1, dtype=torch.float32))
    x_S2 = torch.sigmoid(torch.tensor(x2_proj, dtype=torch.float32))

    y_S1 = torch.tensor(y1, dtype=torch.long)
    y_S2 = torch.tensor(y2, dtype=torch.long)

    # Shuffle each stream independently
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)

    return x_S1, y_S1, x_S2, y_S2