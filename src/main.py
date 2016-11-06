import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#np.set_printoptions(threshold=np.inf)

def encodeData(df, catFeaList):
    le = LabelEncoder()
    label_mapping = {'e': 1, 'p': 0}
    df['label'] = df['label'].map(label_mapping)
    df = df.dropna()
    y, X = df.values[:, 0], df.values[:, 1:]
    ohe = OneHotEncoder(categorical_features=catFeaList)
    for i in range(X.shape[1]):
        X[:, i] = le.fit_transform(X[:, i])
    H = ohe.fit_transform(X).toarray()
    return H, y

Header = np.array(['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

# read the data
data_src = '../dataFile/agaricus-lepiota.data'
# data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df = pd.read_csv(data_src, header=None, na_values=['?'])
df.columns = Header

X, y = encodeData(df, range(22))
print(X)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
y_train = np.array(y_train)
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)