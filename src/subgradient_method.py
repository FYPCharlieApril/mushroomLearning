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
        W = np.array([[0] * v_size] * v_size)
        head, tail = self.hg.head, self.hg.tail
        A = np.zeros((v_size, v_size))
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

        t = 1
        no_N = (f==0).sum()
        stepsize = math.sqrt(no_N)/30
        f_iter= f
        f_iter = self.markov_operator(f_iter)
        #t += 1

        while(t < stepsize):
           gn = self.markov_operator(f_iter)
           f_iter = f_iter - (stepsize/LA.norm(gn))* gn
           t += 1
        return f_iter

'''
    def semisupervised(self,f):
        no_N = (f==0).sum()
        fn_p = np.zeros(no_N)+1
        fn_n = np.zeros(no_N)-1
        fn_p = self.sgm(fn_p)
        fn_n = self.sgm(fn_n)
        fn_avg = 1/2(fn_p + fn_n)
        threshold = sum(fn_avg)/no_N

 #       for u in no_N:
'''