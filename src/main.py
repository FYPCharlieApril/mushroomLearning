import numpy as np
import pandas as pd
from src.hyper_graph import hyper_graph
from src.subgradient_method import subgradient_method

np.set_printoptions(threshold=np.nan)

Header = np.array(['label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape',  'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
          'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])

train_size_arr = [20,40,60,80,100,120,140,160,180,200]
data_src = '../dataFile/agaricus-lepiota.data'
# data_src = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'

# df is the data of the mushrooms, size 8123 * 23, with 1 colume of label and 22 for feature, 8123 data slots
df = pd.read_csv(data_src, header=None, na_values=['?'])
df = df.dropna(axis=1)
df.columns = Header
h = hyper_graph(weight=np.array([1] * df.shape[0]), head=None, tail=None, df=df, \
                   catFeaList=range(df.shape[1]-1), label_mapping={'e': 1, 'p': -1})

from sklearn.model_selection import train_test_split

result = []
X,y = h.hMat, h.y
y =np.matrix(list(enumerate(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20)

ind_test, y_test = y_test[:, 0], y_test[:, 1]
ind_train, y_train = y_train[:, 0], y_train[:, 1]

f = np.array([0] * df.shape[0])
f[ind_train] = y_train
st = subgradient_method(h)
fn = st.markov_operator(f)

#print (len(np.where((fn!=1)&(fn!=-1)&(fn!=0))[0]))