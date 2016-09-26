"""Created By Chen Jiali"""
import numpy as np
import random

class TotalVariation(object):

    """Perceptron classifier.

        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, d_mat, testSize):
        # here we randomly pick testNum data for experiments
        self.constructH(self, d_mat)
        print (self.h_mat)

    # Following is the function to construct the H matrix, return type: np.matrix
    def constructH(self, d_mat):
        fList = list(map(list, list(enumerate(map(lambda xs: list(set(xs)), zip(*d_mat))))))
        fList = np.concatenate(list(map(lambda xs: list(map(lambda x: x + str(xs[0]), xs[1])), fList))[1:])
        r, c = d_mat.shape;
        self.h_mat = []
        for i in range(r):
            h_arr = np.zeros(len(fList))
            for j in range(c - 1):
                # here we add the label with feature index to make the feature unique
                f = d_mat.item((i, j + 1)) + str(j + 1)
                h_arr[np.where(fList == f)[0][0]] = 1
            self.h_mat.append(h_arr)
        return self.h_mat
