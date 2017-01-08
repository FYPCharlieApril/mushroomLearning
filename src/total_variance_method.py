import numpy as np
from src import hyper_graph

def __init__(self, sigma, seta, tau):
    self.seta = seta
    self.sigma = sigma
    self.tau = tau

def fit_predict(self, X, y):
    pass

def pdhg_wh2(self, hMat, y):
    #dul_gap = 0
    m = hMat.shape[0]
    n = hMat.shape[1]
    f, f_ = [0] * m, [0] * m
    alpha_arr=[]
    for k in range(100):
        K_e_arr = []
        m_e_arr = []
        # step 1
        for i in range(n):
            if k == 0:
                e = hMat[:][i]
                one_indeces = np.where(e == 1)[0]
                m_e = one_indeces.shape[0]
                K_e = np.matrix([[0] * m_e] * m)
                for i, one_index in enumerate(one_indeces):
                    K_e[i][one_index] = 1
                alpha_arr.append(np.array([0]*m_e))
                K_e_arr.append(K_e)
                m_e_arr.append(m_e)
            else:
                tmp = alpha_arr[i] + self.sigma * np.dat(K_e_arr[i], f_)
                alpha_arr[i] = tmp - proximal(tmp)

        # step 2
        delta = np.array([0]*m)
        for i in range(n):
            delta = delta + np.dot(K_e_arr[i].T, alpha_arr[i])
        old_f = f
        x_ = f - self.tau * delta
        f = 1 / (1 + self.tau) * (x_ + self.tau * y)

        # step 3
        f_ = f + self.seta * (f - old_f)

def proximal(self, alpha):
    return 0


