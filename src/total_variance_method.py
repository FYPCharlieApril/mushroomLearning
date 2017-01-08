import numpy as np
from src import hyper_graph

def __init__(self, sigma, seta, tau):
    self.seta = seta
    self.sigma = sigma
    self.tau = tau

def fit_predict(self, X, y):
    pass

def pdhg_wh2(self, hMat, y):
    f, f_ = 0, 0
    dul_gap = 0
    edge_num = hMat.shape[1]
    for _ in range(100):
        for i in range(edge_num):
            # preparation
            e = hMat[:][i]

            # step 1

            # step 2
            # step 3
            pass


