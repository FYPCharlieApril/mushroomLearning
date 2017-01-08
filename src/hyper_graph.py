from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class hyper_graph:
    def __init__(self, weight, head, tail, X, y, catFeaList):
        self.weight = weight
        self.y, self.hMat = self.constructor(X=X, y=y, catFeaList=catFeaList)
        if head is not None:
            self.head = head
        else:
            self.head = self.hMat
        if tail is not None:
            self.tail = tail
        else:
            self.tail = self.hMat

    def constructor(self, y, X, catFeaList):
        le = LabelEncoder()
        ohe = OneHotEncoder(categorical_features=catFeaList)
        for i in catFeaList:
            X[:, i] = le.fit_transform(X[:, i])
        hMat = ohe.fit_transform(X).toarray()
        return y, hMat
