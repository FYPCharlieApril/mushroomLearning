import numpy as np

def constructH(dMat, fList, hMat, cols):
    #TODO
    for dSlide in dMat:
        thisList = [0] * cols
        for fea in dSlide:
            index = np.argwhere(fList == fea)

    return

def constructDv(dMat):
    #TODO
    return

def constructW(dMat):
    #TODO
    return