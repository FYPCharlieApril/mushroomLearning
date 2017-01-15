import numpy as np
from numpy import linalg as LA
from src.hyper_graph import hyper_graph
import time
import threading

class subgradient_method:
    def __init__(self, X, f, y_train_ind, parallel=1):
        self.hg = self.construct_h_mat(X)
        self.start_time = time.time()
        self.end_time = self.start_time
        self.y_train_ind = np.array(y_train_ind)
        self.y_un_ind = []
        for i in range(f.shape[0]):
            if i not in y_train_ind:
                self.y_un_ind.append(i)
        self.y_un_ind = np.array(self.y_un_ind)
        self.f = f
        self.f_star = f
        self.parallel = parallel
        self.threads = []
        self.task_list = []
        print("The task will run in", self.parallel, "threads parallelly")

    def construct_h_mat(self, X):
        hg = hyper_graph(weight=np.array([1] * X.shape[0]),
                         head=None,
                         tail=None,
                         X=X,
                         catFeaList=range(X.shape[1]))
        return hg

    def compute_delta(self, e_list, f, f_out, threadLock):
        # this function is used for parallel computing
        # here e is the index of this edge
        for e in e_list:
            head, tail = self.hg.head, self.hg.tail
            e_tail = np.where(tail[:, e] == 1)[0]
            e_head = np.where(head[:, e] == 1)[0]
            f_e_tail = zip(e_tail, f[e_tail])
            f_e_head = zip(e_head, f[e_head])
            u_can, v_can = max(f_e_tail, key=lambda x: x[1]), min(f_e_head, key=lambda x: x[1])
            if u_can[1] - v_can[1] > 0:
                u = u_can[0]
                v = v_can[0]
                d = self.hg.weight[e] * (f[u] - f[v])
                threadLock.acquire()
                f_out[u] = f_out[u] + d
                threadLock.release()

    def markov_operator(self,  f):
        # here we compute A and W
        v_size, e_size = self.hg.hMat.shape[0], self.hg.hMat.shape[1]
        f_out = np.array([0]*v_size)
        threads = []
        #total_task = range(e_size)
        # parallel task dealing
        threadLock = threading.Lock()
        for task in self.task_list:
            t = threading.Thread(target=self.compute_delta, args=(task, f, f_out, threadLock))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # un-parallel version
        #_ = self.compute_delta(total_task, f, f_out, threadLock)

        f_out[self.y_train_ind] = self.f_star[self.y_train_ind]
        return f_out

    def sgm(self,f):
        t = 0
        f_iter, f_last = f, f
        e_size = self.hg.hMat.shape[1]
        total_task = np.array(range(e_size))
        for i in range(self.parallel):
            self.task_list.append(np.where(total_task % self.parallel == i)[0])
        while (t < 500):
            print("Current step:", t+1)
            gn = self.markov_operator(f_iter)
            f_iter = f_iter - (0.9/LA.norm(gn)) * gn
            f_iter[self.y_train_ind] = self.f_star[self.y_train_ind]
            t += 1

        self.end_time = time.time()
        print("Time used to run:", self.end_time-self.start_time)
        return f_iter

    def fit_predict(self):
        f = self.f

        f_p = np.zeros(f.size)+1
        f_p[self.y_train_ind] = self.f_star[self.y_train_ind]
        f_n = np.zeros(f.size)-1
        f_n[self.y_train_ind] = self.f_star[self.y_train_ind]

        f_p = self.sgm(f_p)
        f_n = self.sgm(f_n)

        f_avg = 0.5 * (f_p + f_n)
        threshold = sum(f_avg[self.y_un_ind])/len(self.y_un_ind)
        for i in self.y_un_ind:
            f_avg[i] = 1 if f_avg[i] > threshold else -1
        return f_avg



