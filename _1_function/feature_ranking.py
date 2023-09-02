from numpy import *
from TransProPy._1_function import auc
def feature_ranking(f, c, max_rank, pos, neg, n0, n1):
    f_auc = []
    f_no = [str(i) for i in range(shape(f)[0])]
    f_mtf = full((shape(f)[0], shape(f)[1]), False)
    f_ne = []
    fl = shape(f_no)[0]
    # print('fl ', fl)
    for j in range(fl):
        argfv = argsort(f[j])
        slofe = c[argfv]
        ne = slofe[0]
        a = auc(slofe, ne, n0, n1)
        if a < 0.5:
            if slofe[0] == slofe[-1]:
                a = 1 - a
                if ne == pos:
                    ne = neg
                else:
                    ne = pos
        f_auc.append(a)
        f_ne.append(ne)

        ml = 1
        mr = 1
        for i in range(1, size(slofe)):
            if slofe[i] == slofe[0]:
                ml += 1
            else:
                break
        for i in range(-2, -size(slofe), -1):
            if slofe[i] == slofe[-1]:
                mr += 1
            else:
                break
        mr = size(slofe) - mr

        if slofe[0] == slofe[-1]:
            if not slofe[0] == ne:
                ml = 0
            else:
                mr = size(slofe)
        f_mtf[j][argfv[ml:mr]] = True
        # print(f_auc)
    arg_auc = argsort(-array(f_auc))
    FName = array(f_no)[arg_auc]
    Fvalue = array(f)[arg_auc]
    Fauc = array(f_auc)[arg_auc]
    Fne = array(f_ne)[arg_auc]
    FmTF = array(f_mtf)[arg_auc]
    # print('SORT VALUE', Fvalue)
    # print('SORT M', FmTF)
    # print('SORT NAME', FName)
    # print('SORT AUC', Fauc)

    kk = 0
    slen = 0
    Fmcount = ones((len(FmTF[0])))
    Fmcount = Fmcount.astype(bool)
    for i in range(fl):
        if Fauc[i] < 0.5:
            kk += 1
        Fmcount &= FmTF[i]
        if True in Fmcount:
            slen += 1
    # print('Totally ', kk, ' features with auc under 0.5')

    for i in range(fl):
        if Fauc[i] < 0.5:
            continue
        for j in range(i + 1, fl):
            if Fauc[j] < 0.5:
                continue
            nflg = 0
            if not ((FmTF[i] & FmTF[j]) == FmTF[i]).all():
                if not ((FmTF[i] & FmTF[j]) == FmTF[j]).all():
                    nflg = 1
            if nflg == 0:
                if FmTF[i].sum() <= FmTF[j].sum():
                    Fauc[j] = -2
                else:
                    Fauc[i] = -2
                    break
    ii = []
    gg = []
    for i in range(fl):
        if Fauc[i] > 0.5:
            ii.append(i)
        else:
            gg.append(i)
    arg_auc_gg = argsort(-array(f_auc)[gg])
    gg = [str(FName[i]) for i in array(gg)[arg_auc_gg]]
    # print('Totally ' + str(fl - len(ii)) + ' features are covered and removed.')
    # print(gg)
    FName = FName[ii]
    Fvalue = Fvalue[ii]
    Fauc = Fauc[ii]
    Fne = Fne[ii]
    FmTF = FmTF[ii]

    over = 0
    if max_rank > len(ii):
        over = max_rank - len(ii)
        max_rank = len(ii)


    # start ranking
    rankset = []  # store unique features
    ranklist = []  # with overlap
    order = 0
    while len(rankset) < max_rank:
        ## start selection
        rnk = 2
        mv_auc = Fauc[order]
        fs = [FName[order]]
        cpms = FmTF[order]
        fl = shape(FName)[0]

        while mv_auc != 1:
            ft = 0
            temp = 0
            for j in range(fl):
                if FName[j] not in fs:
                    tmpFmTF = cpms & FmTF[j]
                    if not ((FmTF[j] & cpms) == cpms).all():
                        mauc = 0
                        for g in fs + [FName[j]]:
                            fval = Fvalue[argwhere(FName == g)[0][0]][tmpFmTF]
                            stwlofe = array(c)[tmpFmTF]
                            argfv = argsort(fval)
                            slofe = stwlofe[argfv]
                            tauc = auc(slofe, Fne[argwhere(FName == g)[0][0]], n0, n1)
                            mauc += tauc
                        tmpauc = mauc / rnk
                        if tmpauc > mv_auc:
                            mv_auc = tmpauc
                            ft = j
                            temp = Fauc[j]
                        elif tmpauc == mv_auc and Fauc[j] > temp:
                            ft = j
                            temp = Fauc[j]

            if mv_auc == -2 or ft == 0:
                break
            fs.append(FName[ft])
            cpms = cpms & FmTF[ft]
            rnk += 1
            # print('\nRank-' + str(rnk - 1) + ' mvAUC: ' + str(mv_auc) + '  Feature set:', fs)

        for i in fs:
            ranklist.append(i)
            if i not in rankset:
                rankset.append(i)

        order += 1

    if over != 0:
        ranklist = ranklist + list(gg)[:over]
        rankset = rankset + list(gg)[:over]

    return FName, Fauc, rankset, ranklist