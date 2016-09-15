import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def constructH(data):
    return



#data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
data_src = '../dataFile/agaricus-lepiota.data'
df = pd.read_csv(data_src, header=None)

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df.tail()

dMat = df.iloc[:].values
# definition of V and E
# compute the column number of E matrix
cols = sum (list(map(lambda xs : len(np.unique(xs)), zip(*dMat)))[1:])
print(cols)