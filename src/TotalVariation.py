import numpy as np

class TotalVariation(object):
    """define function for total variation implementation"""
    # Following is the function to construct the H matrix, return type: np.matrix
    def constructH(testcases, dMat):
        fList = list(map(list, list(enumerate(map(lambda xs: list(set(xs)), zip(*dMat))))))
        fList = np.concatenate(list(map(lambda xs: list(map(lambda x: x + str(xs[0]), xs[1])), fList))[1:])
        r, c = dMat.shape;
        hMat = []
        for i in range(r):
            hArr = np.zeros(len(fList))
            for j in range(c - 1):
                # here we add the label with feature index to make the feature unique
                f = dMat.item((i, j + 1)) + str(j + 1)
                hArr[np.where(fList == f)[0][0]] = 1
            hMat.append(hArr)
        return hMat

    def constructDv(dMat):
        # TODO
        return

    def constructW(dMat):
        # TODO
        return

    def constructDe(dMat):
        # TODO
        return
