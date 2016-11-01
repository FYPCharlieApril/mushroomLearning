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
        return
