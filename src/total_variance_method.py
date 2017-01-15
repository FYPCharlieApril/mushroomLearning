import numpy as np
from operator import itemgetter
import bisect
from src import hyper_graph

def __init__(self, sigma, seta, tau, lamda, weight):
    self.seta = seta
    self.sigma = sigma
    self.tau = tau
    self.lamda = lamda
    self.weight = weight

def fit_predict(self, X, f):
    hg = hyper_graph(weight=np.array([1] * X.shape[0]),
                     head=None,
                     tail=None,
                     X=X,
                     catFeaList=range(X.shape[1]))


def pdhg_wh2(self, hMat, y):
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
                alpha_arr[i] = tmp - proximal(tmp, self.weight[i])

        # step 2
        delta = np.array([0]*m)
        for i in range(n):
            delta = delta + np.dot(K_e_arr[i].T, alpha_arr[i])
        old_f = f
        x_ = f - self.tau * delta
        f = 1 / (1 + self.tau) * (x_ + self.tau * y)

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
    r = max(sorted_alpha, key=itemgetter(1))
    s = min(sorted_alpha, key=itemgetter(1))

    #step 3:
    p = findp(sorted_alpha, m, r)
    q = findq(sorted_alpha, m, s)
    deltaE1r = 0
    murs = 2 * mu * (r-s)
    for i in range(m-p+1, m+1):
        deltaE1r += (sorted_alpha[i][1]-r)
    while deltaE1r < murs and q+1 < m-p:

        #step 4:
        r_old = r
        r = sorted_alpha[m-p][1]
        s = s + p / q * (r_old-r)
        p = findp(sorted_alpha, m, r)
        q = findq(sorted_alpha, s)

    #step 6:
    tmp1 = 0
    for i in range(m-p+1, m+1):
        tmp1 += (sorted_alpha[i][1])
    tmp2 = 0
    for i in range(1, q+1):
        tmp2 += (sorted_alpha[i][1])
    s = (2*mu*tmp1/(p+2*mu) + tmp2) / (q + 2*mu - 4*mu*mu/(tmp1+2*mu))
    r = ((q+2*mu)*s-tmp2) / (2*mu)

    #step 7:
    for i in range(m):
        if alpha[i] >= r:
            alpha[i] = r
        elif alpha[i] <= s:
            alpha[i] = s
    return alpha

def findp(sorted_alpha, m, r):
    alpha = zip(*sorted_alpha)[1]
    n = bisect.bisect_left(alpha, r)
    p = m-n+1
    return p

def findq(sorted_alpha, s):
    alpha = zip(*sorted_alpha)[1]
    q = bisect.bisect_left(alpha, s)
    return q