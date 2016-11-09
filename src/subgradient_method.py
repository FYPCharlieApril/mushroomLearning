import numpy as np
from numpy import linalg as LA
import math

class subgradient_method:
    def __init__(self, hg):
        self.hg = hg

    def markov_operator(self,  f):
        # here we compute A and W
        v_size, e_size = self.hg.hMat.shape[0], self.hg.hMat.shape[1]
        f_index = np.array(list(enumerate(f)))
        L = np.where((f_index[:, 1] == 1) | (f_index[:, 1] == -1))[0]
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

        for i, u in f_index:
            W[i, i] = A[u, :].sum()
        # following are teh procedures of computing the Markov operator, here the projection matrix we use
        # are the one with all entries to be 1.
        f_out = (W-A).dot(f)
        f_out[L] = f[L]
        return f_out

    def sgm(self,f):
        f_index = np.array(list(enumerate(f)))
        L = np.where((f_index[:, 1] == 1) | (f_index[:, 1] == -1))[0]
        t = 0
        f_iter, f_last = f, f
        f_iter = self.markov_operator(f_iter)

        #while(abs(LA.norm(f_iter - f_last)) < 100):
        while (t < 200):
           gn = self.markov_operator(f_iter)
           f_last = f_iter
           f_iter = f_iter - (0.84/LA.norm(gn)) * gn
           f_iter[L] = f[L]
           t += 1
        return f_iter


    def semisupervised(self,f):
        f_index = np.array(list(enumerate(f)))
        L = np.where((f_index[:, 1] == 1) | (f_index[:, 1] == -1))[0]

        f_p = np.zeros(f.size)+1
        f_p[L] = f[L]

        f_n = np.zeros(f.size)-1
        f_n[L] = f[L]

        f_p = self.sgm(f_p)
        f_n = self.sgm(f_n)


        f_avg = 1/2(f_p + f_n)
        U = np.where((f_index[:, 1] != 1) & (f_index[:, 1] != -1))[0]
        threshold = sum(f_avg[U])/len(U)

        for u in f_avg:
            if(u>=threshold):
                u = 1
            else:
                u = -1
        f_avg[L] = f[L]
        return f_avg

