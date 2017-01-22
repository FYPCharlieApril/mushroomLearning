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
        self.y_un_ind = []
        for i in range(f.shape[0]):
            if i not in y_train_ind:
                self.y_un_ind.append(i)
        self.y_un_ind = np.array(self.y_un_ind)

    def construct_h_mat(self, X):
        hg = hyper_graph(weight=np.array([1] * X.shape[0]),
                         head=None,
                         tail=None,
                         X=X,
                         catFeaList=range(X.shape[1]))
        return hg

    def fit_predict(self):
        return self.pdhg_wh2(self.hg)

    def pdhg_wh2(self, hg):
        hMat = hg.hMat
        m = hMat.shape[0]
        n = hMat.shape[1]
        f, f_ = np.array([0] * m), np.array([0] * m)
        alpha_arr = []
        K_e_arr = []
        for k in range(200):
            print("Current Step:", k+1)
            # step 1
            for i in range(n):
                if k == 0:
                    # when k = 0, it is the step for initialization for K_e
                    e = hMat[:, i]
                    one_indeces = np.where(e == 1)[0]
                    this_alpha = self.f_star[one_indeces]
                    alpha_arr.append(this_alpha)
                    K_e_arr.append(one_indeces)
                else:
                    # dot product to be optimized
                    #print(f_[K_e_arr[i]])
                    #print(alpha_arr[i])
                    tmp = alpha_arr[i] + self.sigma * f_[K_e_arr[i]]
                    alpha_arr[i] = tmp - self.proximal(tmp, self.weight[i])

            # step 2
            delta = np.array([0]*m)
            for i in range(n):
                for ind, k_e in enumerate(K_e_arr[i]):
                    delta[k_e] += alpha_arr[i][ind]

            old_f = f
            x_ = f - self.tau * delta
            f = 1 / (1 + self.tau) * (x_ + self.tau * self.f_star)
            f[self.y_train_ind] = self.f_star[self.y_train_ind]
            # step 3
            f_ = f + self.seta * (f - old_f)

        threshold = sum(f[self.y_un_ind]) / len(self.y_un_ind)
        for i in self.y_un_ind:
            f[i] = 1 if f[i] > threshold else -1
        return f


    def proximal(self, alpha, we):
        #step 1:
        m = len(alpha)
        mu = self.lamda * we / self.sigma
        index_alpha = list(enumerate(alpha))
        sorted_alpha = sorted(index_alpha, key=itemgetter(1))
        #print(sorted_alpha)
        #step 2:
        r = max(sorted_alpha, key=itemgetter(1))[1]
        s = min(sorted_alpha, key=itemgetter(1))[1]

        #step 3:
        p = m - self.find_p(sorted_alpha, r)
        q = self.find_q(sorted_alpha, s) + 1
        murs = 2 * mu * (r - s)
        tar = np.array(list(zip(*sorted_alpha[m - p:m]))[1])
        deltaE1r = sum(tar - r)
        while deltaE1r < murs and q <= m-p-1:
            #step 4:
            r_old = r
            r = sorted_alpha[m-p-1][1]
            s = s + p / q * (r_old-r)
            p = m - self.find_p(sorted_alpha, r)
            q = self.find_q(sorted_alpha, s) + 1

            # loop condition computation
            murs = 2 * mu * (r - s)
            tar = np.array(list(zip(*sorted_alpha[m - p:m]))[1])
            deltaE1r = sum(tar - r)

        #step 6:
        tar = np.array(list(zip(*sorted_alpha[m - p:m]))[1])
        tmp1 = sum(tar)
        tar = np.array(list(zip(*sorted_alpha[0 : q]))[1])
        tmp2 = sum(tar)
        s = (2*mu*tmp1/(p+2*mu) + tmp2) / (q + 2*mu - 4*mu*mu/(tmp1+2*mu))
        r = ((q+2*mu)*s-tmp2) / (2*mu)

        #step 7:
        for i in range(m):
            if alpha[i] >= r:
                alpha[i] = r
            elif alpha[i] <= s:
                alpha[i] = s
        return alpha


    def find_p(self, sorted_alpha, r):
        alpha = list(zip(*sorted_alpha))[1]
        n = bisect.bisect_left(alpha, r)
        return n

    def find_q(self, sorted_alpha, s):
        alpha = list(zip(*sorted_alpha))[1]
        n = bisect.bisect_right(alpha, s)
        return n-1