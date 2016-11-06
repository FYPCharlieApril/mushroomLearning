from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class hyper_graph:
    def __init__(self, weight, dMat, df, catFeaList):
        self.weight = weight
        self.direction = dMat
        self.constructor(df=df, catFeaList=catFeaList)

    def constructor(self, df, catFeaList):
        le = LabelEncoder()
        label_mapping = {'e': 1, 'p': -1}
        df['label'] = df['label'].map(label_mapping)
        df = df.dropna()
        y, X = df.values[:, 0], df.values[:, 1:]
        ohe = OneHotEncoder(categorical_features=catFeaList)
        for i in range(X.shape[1]):
            X[:, i] = le.fit_transform(X[:, i])
        hMat = ohe.fit_transform(X).toarray()
        self.hMat = hMat
        self.y = y
