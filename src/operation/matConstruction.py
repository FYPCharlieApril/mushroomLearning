import numpy as np

def constructH(dMat, fList):
    #TODO
    r, c = dMat.shape;
    hMat = []
    for i in range(r):
        hArr = np.zeros(len(fList))
        for j in range(c-1):
            f = dMat[i][j+1] + str(j+1)
            hArr[np.where(fList == f)[0][0]]  = 1
        hMat.append(hArr)
    return hMat

def constructDv(dMat):
    #TODO
    return

def constructW(dMat):
    #TODO
    return