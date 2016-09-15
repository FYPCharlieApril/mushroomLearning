import numpy as np
import pandas as pd
import random
import src.operation.matConstruction as mc

testNum = 20

data_src = '../dataFile/agaricus-lepiota.data'
#data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
df = pd.read_csv(data_src, header=None)

ranList = [random.randint(0,1000) for _ in range(testNum)]

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df.tail()
dMat = df.iloc[:].values

# here we randomly pick testNum data for experiments
testCases = []
for i in range(testNum):
    testCases.append(dMat[ranList[i]].tolist())
testCases = np.matrix(testCases)

# definition of V and E
# compute the column number of E matrix
fList = list(enumerate(map(lambda xs : list(set(xs)), zip(*dMat))))
fList = list(map(list, fList))
fList = np.concatenate(list(map(lambda xs : list(map(lambda x : x + str(xs[0]),xs[1])), fList))[1:])
hMat = mc.constructH(testCases,fList)
print (hMat)






