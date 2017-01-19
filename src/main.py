import numpy as np
import pandas as pd
from src.subgradient_method import subgradient_method
from src.total_variance_method import total_variance_method
from sklearn.model_selection import train_test_split
from src.hyper_graph import hyper_graph

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
#df['label'] = df['label'].map({'e': 1, 'p': 0})
X, y = df.values[:, 1:] , df.values[:, 0]
result = []
y_ind = np.matrix(list(enumerate(y)))
X_train, X_test, y_train, y_test = train_test_split(X, y_ind, train_size=200)
ind_test, y_test = y_test[:, 0], y_test[:, 1]
ind_train, y_train = y_train[:, 0], y_train[:, 1]

f = np.array([0] * X.shape[0])
f[ind_train] = y[ind_train]
acc = 0
'''
for p in range(5):
    print("currently computing the", p+1, "-th iteration.")
    st = subgradient_method(X, f, ind_train, parallel=4)
    fn = st.fit_predict()
    this_acc = np.array(np.where(fn == y)).shape[1]/y.shape[0]
    acc += this_acc
    print(this_acc)
acc /= 5
print("True prediction rate:", acc)
result.append(acc)
'''
st = total_variance_method(sigma=0.07,
                           seta=1,
                           tau=0.07,
                           lamda=0.00001,
                           weight=[1]*f.shape[0],
                           X=X,
                           f=f,
                           y_train_ind=ind_train)

fn = st.fit_predict()
acc = np.array(np.where(fn == y)).shape[1]/y.shape[0]
print("True prediction rate:", acc)
'''

#following are the supervised learning methods
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

acc_list = []
X = hyper_graph(weight=np.array([1] * X.shape[0]),
                 head=None,
                 tail=None,
                 X=X,
                 catFeaList=range(X.shape[1])).hMat

for test_size in train_size_arr:
    this_acc = 0
    for _ in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=test_size)
        y_train = list(y_train)
        y_test = list(y_test)
        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB()

        gnb.fit(X_train, y_train)

        y_pred = gnb.predict(X_test)
        this_acc += accuracy_score(y_test, y_pred)
    this_acc /= 20
    acc_list.append(this_acc)
    print('Accuracy: %.2f' % this_acc)
print(acc_list)
'''