from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class hyper_graph:
    def __init__(self, weight, head, tail, df, catFeaList, label_mapping):
        self.weight = weight
        self.constructor(df=df, catFeaList=catFeaList, label_mapping=label_mapping)
        if head is not None:
            self.head = head
        else:
            self.head = self.hMat
        if tail is not None:
            self.tail = tail
        else:
            self.tail = self.hMat

    def constructor(self, df, catFeaList, label_mapping):
        le = LabelEncoder()
        if label_mapping is not None:
            df['label'] = df['label'].map(label_mapping)

        y, X = df.values[:, 0], df.values[:, 1:]
        ohe = OneHotEncoder(categorical_features=catFeaList)
        for i in catFeaList:
            X[:, i] = le.fit_transform(X[:, i])
        hMat = ohe.fit_transform(X).toarray()
        self.hMat = hMat
        self.y = y
