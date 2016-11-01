import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def encodeData(df, catFeaList):
    le = LabelEncoder()
    X = df.values
    ohe = OneHotEncoder(categorical_features=catFeaList)
    for i in range(X.shape[1]):
        X[:, i] = le.fit_transform(X[:, i])
    H = ohe.fit_transform(X).toarray()
    return H

# read the data
data_src = '../dataFile/agaricus-lepiota.data'
# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df = pd.read_csv(data_src)
# data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

H = encodeData(df, range(df.shape[1]))
print(H)




