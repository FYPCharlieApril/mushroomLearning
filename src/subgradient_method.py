import numpy as np
from numpy import linalg as LA
from src.hyper_graph import hyper_graph
import time


class subgradient_method:
    def __init__(self, X, y, y_train_ind):
        self.hg = self.construct_h_mat(X, y, y_train_ind)
        self.start_time = time.time()
        self.end_time = self.start_time
        self.y_train_ind = np.array(y_train_ind)
        self.y_un_ind = []
        for i in range(y.shape[0]):
            if i not in y_train_ind:
                self.y_un_ind.append(i)
        self.y_un_ind = np.array(self.y_un_ind)
        self.y = y

    def construct_h_mat(self, X, y, y_train_ind):
        hg = hyper_graph(weight=np.array([1] * X.shape[0]),
                         head=None,
                         tail=None,
                         X=X,
                         y=y,
                         catFeaList=range(X.shape[1]))

        self.f = np.array([0] * X.shape[0])
        self.f[y_train_ind] = y[y_train_ind]
        return hg

    def markov_operator(self,  f):
        # here we compute A and W
        v_size, e_size = self.hg.hMat.shape[0], self.hg.hMat.shape[1]
        W = np.zeros((v_size, v_size))
        A = np.zeros((v_size, v_size))
        head, tail = self.hg.head, self.hg.tail
        for e in range(e_size):
            e_tail = np.where(tail[:, e] == 1)[0]
            e_head = np.where(head[:, e] == 1)[0]
            f_e_tail = zip(e_tail, f[e_tail])
            f_e_head = zip(e_head, f[e_head])
            u_can, v_can = max(f_e_tail,key=lambda x:x[1]), min(f_e_head,key=lambda x:x[1])
            if u_can[1] - v_can[1] > 0:
                u = u_can[0]
                v = v_can[0]
                A[u, v] = A[u, v] + self.hg.weight[e]
                A[v, u] = A[u, v]

        for u in self.y_un_ind:
            W[u, u] = sum(A[u, :])
        # following are teh procedures of computing the Markov operator, here the projection matrix we use
        # are the one with all entries to be 1.
        f_out = (W-A).dot(f)
        f_out[self.y_train_ind] = self.y[self.y_train_ind]
        return f_out

    def sgm(self,f):
        t = 0
        f_iter, f_last = f, f
        #f_iter = self.markov_operator(f_iter)

        #while(abs(LA.norm(f_iter - f_last)) < 100):
        while (t < 500):
            print("Current step:", t)
            gn = self.markov_operator(f_iter)
            f_iter = f_iter - (0.9/LA.norm(gn)) * gn
            f_iter[self.y_train_ind] = self.y[self.y_train_ind]
            t += 1
        self.end_time = time.time()
        return f_iter


    def fit_predict(self):
        f = self.f
        f_index = np.array(list(enumerate(f)))
        self.f_star = f

        f_p = np.zeros(f.size)+1
        f_p[self.y_train_ind] = self.y[self.y_train_ind]
        f_n = np.zeros(f.size)-1
        f_n[self.y_train_ind] = self.y[self.y_train_ind]

        f_p = self.sgm(f_p)
        f_n = self.sgm(f_n)

        f_avg = 0.5 * (f_p + f_n)
        threshold = sum(f_avg[self.y_un_ind])/len(self.y_un_ind)
        for i in self.y_un_ind:
            f_avg[i] = 1 if f_avg[i] > threshold else -1
        return f_avg



