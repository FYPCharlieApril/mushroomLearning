import numpy as np

# Following is the function to construct the H matrix, return type: np.matrix
def constructH(dMat, fList):
    r, c = dMat.shape;
    hMat = []
    for i in range(r):
        hArr = np.zeros(len(fList))
        for j in range(c-1):
            # here we add the label with feature index to make the feature unique
            f = dMat.item((i, j+1)) + str(j+1)
            hArr[np.where(fList == f)[0][0]]  = 1
        hMat.append(hArr)
    return hMat

def constructDv(dMat):
    #TODO
    return

def constructW(dMat):
    #TODO
    return