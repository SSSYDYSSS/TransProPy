from numpy import *
def auc(tlofe, ne, n0, n1):
    lpp = 0
    lnp = 0
    flag = 0
    aac = 0
    for i in range(-1, -size(tlofe) - 1, -1):
        if tlofe[i] == ne:
            if flag == 1:
                aac += lnp * lpp
                flag = 0
                lpp = 0
            lnp += 1
        else:
            if flag == 0:
                flag = 1
            lpp += 1
    aac += lnp * lpp
    auc = (n0 * n1 - aac) / (n0 * n1)
    return auc
