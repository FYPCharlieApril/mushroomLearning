from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class hyper_graph:
    def __init__(self, weight, head, tail, df, catFeaList, label_mapping):
        self.weight = weight
        self.head=head
        self.tail=tail
        self.constructor(df=df, catFeaList=catFeaList,label_mapping=label_mapping)

    def constructor(self, df, catFeaList, label_mapping):
        le = LabelEncoder()
        if label_mapping is not None:
            df['label'] = df['label'].map(label_mapping)
        df = df.dropna()
        y, X = df.values[:, 0], df.values[:, 1:]
        ohe = OneHotEncoder(categorical_features=catFeaList)
        for i in range(X.shape[1]):
            X[:, i] = le.fit_transform(X[:, i])
        hMat = ohe.fit_transform(X).toarray()
        self.hMat = hMat
        self.y = y
