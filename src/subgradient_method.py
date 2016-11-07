import numpy as np

class subgradient_method:
    def __init__(self, hg):
        self.hg = hg

    def markov_operator(self,  f):
        # here we compute A and W
        W = []
        L = []
        hMat = self.hg.hMat
        v_size, e_size = hMat.shape[0], hMat.shape[1]
        N = np.array(list(enumerate(f)))
        N = np.where((N[:, 1] != 1) & (N[:, 1] != -1))[0]
        head, tail = self.hg.head, self.hg.tail
        A = np.matrix([[0]*v_size] * v_size)
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

        for i, u in enumerate(N):
            W.append(A[u, :].sum())

        # following are teh procedures of computing the Markov operator, here the projection matrix we use
        # are the one with all entries to be 1.
        for i in range(N.shape[0]):
            L.append([W[i]]*v_size)
        L = np.matrix(L)
        col_sum = np.array(np.sum(A, axis=0))[0]
        R = np.matrix([col_sum] * N.shape[0])
        return (L-R).dot(f)

    def sgm(self):
        print("TO BE DONE")