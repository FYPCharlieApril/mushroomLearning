import numpy as np
import pandas as pd
#import subgradient_method
from src.subgradient_method import subgradient_method
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.nan)

Header = np.array(['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape',  'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

train_size_arr = [20,40,60,80,100,120,140,160,180,200]
#data_src = '../dataFile/agaricus-lepiota.data'
data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df = pd.read_csv(data_src, header=None, na_values=['?'])
df = df.dropna(axis=1)
df.columns = Header
df['label'] = df['label'].map({'e': 1, 'p': -1})
X, y = df.values[:, 1:] , df.values[:, 0]

y_ind = np.matrix(list(enumerate(y)))
X_train, X_test, y_train, y_test = train_test_split(X, y_ind, train_size=40)
ind_test, y_test = y_test[:, 0], y_test[:, 1]
ind_train, y_train = y_train[:, 0], y_train[:, 1]

st = subgradient_method(X, y, ind_train)
fn = st.fit_predict()
print("True predict:", np.array(np.where(fn == y)).shape[1]/y.shape[0])

