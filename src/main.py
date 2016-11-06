import numpy as np
import pandas as pd
import src.hyper_graph as hg

Header = np.array(['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

data_src = '../dataFile/agaricus-lepiota.data'
# data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df = pd.read_csv(data_src, header=None, na_values=['?'])
df.columns = Header

h = hg.hyper_graph(weight=[], dMat=[], df=df, catFeaList=range(22))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = h.hMat, h.y
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.9963, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
y_train = list(y_train)
from sklearn.linear_model import Perceptron

print("Perception")
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train, y_train)

y_pred = ppn.predict(X_test)

print('We pick %d instances from %d as training set' %  (X_train.shape[0], X.shape[0]))
print('Learning accuracy is: %2f' % (1-(y_test != y_pred).sum()/y_test.shape[0]))

from sklearn.neighbors import KNeighborsClassifier
print("KNN")
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('We pick %d instances from %d as training set' %  (X_train.shape[0], X.shape[0]))
print('Learning accuracy is: %2f' % (1-(y_test != y_pred).sum()/y_test.shape[0]))

