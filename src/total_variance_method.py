import numpy as np
from src import hyper_graph

def __init__(self, sigma, seta, tau):
    self.seta = seta
    self.sigma = sigma
    self.tau = tau

def fit_predict(self, X, y):
    pass


def proximal(self, x_, y, gf, s1=None):
    if gf == "square_norm":
        prox = 1/(1+self.tau) * (x_+self.tau * y)
    if gf == "indicator":
        if s1 is None:
            print("Need to input s1 for this method")
            return
        else:
            v = 1/(1+self.tau) * (x_+self.tau * y)
            dino = np.norm(v)
            dino = max([dino, 1])
            prox = (x_ + self.tau * s1) / dino
    return prox

def lovazs_extension(self, set_func):
    pass


def ratio_dca(self):
    pass


def pdhg(self):
    pass

def total_variation(self, h):
    pass