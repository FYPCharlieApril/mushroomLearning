import src.hyper_graph

class subgradient_method:
    def __init__(self, hg):
        self.hg = hg

    def markov_operator(self, hg, f):
        hMat = hg.hMat
        V, E = hMat.shape[0], hMat.shape[1]
        w = hg.weight

        A = [[0]*V] * V

    def sgm(self):
        print("TO BE DONE")