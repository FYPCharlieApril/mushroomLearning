class subgradient_method:
    def __init__(self, hg):
        self.hg = hg

    def markov_operator(self,  f):
        hMat = self.hg.hMat
        V, E = hMat.shape[0], hMat.shape[1]
        w = self.hg.weight
        A = [[0]*V] * V


    def sgm(self):
        print("TO BE DONE")