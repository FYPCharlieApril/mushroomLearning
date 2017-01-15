import numpy as np
from operator import itemgetter
import bisect
from src.hyper_graph import hyper_graph

class total_variance_method:
    def __init__(self, sigma, seta, tau, lamda, weight, X, f, y_train_ind):
        self.hg = self.construct_h_mat(X)
        self.seta = seta
        self.sigma = sigma
        self.tau = tau
        self.lamda = lamda
        self.weight = weight
        self.f_star = f
        self.y_train_ind = y_train_ind

    def construct_h_mat(self, X):
        hg = hyper_graph(weight=np.array([1] * X.shape[0]),
                         head=None,
                         tail=None,
                         X=X,
                         catFeaList=range(X.shape[1]))
        return hg

    def fit_predict(self):
        return self.pdhg_wh2(self.hg)

    def pdhg_wh2(self, hMat):
        m = hMat.shape[0]
        n = hMat.shape[1]
        f, f_ = [0] * m, [0] * m
        alpha_arr = []
        for k in range(100):
            K_e_arr = []
            # step 1
            for i in range(n):
                if k == 0:
                    # when k = 0, it is the step for initialization for K_e
                    e = hMat[:][i]
                    one_indeces = np.where(e == 1)[0]
                    m_e = one_indeces.shape[0]               # this is the number of vertices for edge e
                    K_e = np.matrix([[0] * m_e] * m)        # K matrix for the particular edge e
                    for i, one_index in enumerate(one_indeces):
                        K_e[i][one_index] = 1
                    alpha_arr = self.f_star[one_indeces]
                    #alpha_arr.append(np.array([0]*m_e))
                    K_e_arr.append(K_e)
                else:
                    tmp = alpha_arr[i] + self.sigma * np.dot(K_e_arr[i], f_)
                    alpha_arr[i] = tmp - self.proximal(tmp, self.weight[i])

            # step 2
            delta = np.array([0]*m)
            for i in range(n):
                delta = delta + np.dot(K_e_arr[i].T, alpha_arr[i])
            old_f = f
            x_ = f - self.tau * delta
            f = 1 / (1 + self.tau) * (x_ + self.tau * self.f_star)

            # step 3
            f_ = f + self.seta * (f - old_f)
        return f


    def proximal(self, alpha, we):
        #step 1:
        m = len(alpha)
        mu = self.lamda * we / self.sigma
        index_alpha = list(enumerate(alpha))
        sorted_alpha = sorted(index_alpha, key=itemgetter(1))

        #step 2:
        r = max(sorted_alpha, key=itemgetter(1))[1]
        s = min(sorted_alpha, key=itemgetter(1))[1]

        #step 3:
        p = m-self.find_pos(sorted_alpha, r)+1
        q = self.find_pos(sorted_alpha, s)
        murs = 2 * mu * (r - s)
        deltaE1r = sum(np.array(sorted_alpha[m - p:p][1]) - r)
        while deltaE1r < murs and q+1 <= m-p:

            #step 4:
            r_old = r
            r = sorted_alpha[m-p][1]
            s = s + p / q * (r_old-r)
            p = m-self.find_pos(sorted_alpha, r)+1
            q = self.find_pos(sorted_alpha, s)

            # loop condition computation
            murs = 2 * mu * (r - s)
            deltaE1r = sum(np.array(sorted_alpha[m-p:p][1]) - r)

        #step 6:
        tmp1 = sum(sorted_alpha[m-p:m][1])
        tmp2 = sum(sorted_alpha[0:q][1])
        s = (2*mu*tmp1/(p+2*mu) + tmp2) / (q + 2*mu - 4*mu*mu/(tmp1+2*mu))
        r = ((q+2*mu)*s-tmp2) / (2*mu)

        #step 7:
        for i in range(m):
            if alpha[i] >= r:
                alpha[i] = r
            elif alpha[i] <= s:
                alpha[i] = s
        return alpha


    def find_pos(sorted_alpha, r):
        alpha = zip(*sorted_alpha)[1]
        n = bisect.bisect_left(alpha, r)
        return n
