import numpy as np
import pandas as pd

import operation.matConstruction as mc

data_src = '../dataFile/agaricus-lepiota.data'
testNum = 20

#data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
df = pd.read_csv(data_src, header=None)

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df.tail()
dMat = df.iloc[:].values

# definition of V and E
# compute the column number of E matrix
fList = list(enumerate(map(lambda xs : list(set(xs)), zip(*dMat))))
fList = list(map(list, fList))
fList = np.concatenate(list(map(lambda xs : list(map(lambda x : x + str(xs[0]),xs[1])), fList))[1:])
print (fList)







