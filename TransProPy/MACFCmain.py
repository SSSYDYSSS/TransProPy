from numpy import *
from TransProPy.UtilsFunction1.LoadData import load_data
from TransProPy.UtilsFunction1.FeatureRanking import feature_ranking
from TransProPy.UtilsFunction1.PrintResults import print_results
from collections import Counter

def MACFCmain(max_rank, lable_name, data_path='../data/gene_tpm.csv', label_path='../data/tumor_class.csv'):
    """
    1.1_feature_ranking_modle.
    Applying the MACFC selection for relevant feature genes in classification.
    --------------------------------------------------------------------------

    Parameters:
    max_rank : int
        The total number of gene combinations you want to obtain.
    -------------------------------------------------------------

    Returns:
    fr : List of strings
        representing ranked features.
    fre1 : Dictionary
        feature names as keys and their frequencies as values.
    frequency : List of tuples
        feature names and their frequencies.
    len(FName) : Integer
        count of AUC values greater than 0.5.
    FName : Array of strings
        feature names after ranking with AUC > 0.5.
    Fauc : Array of floats
        AUC values corresponding to the ranked feature names.
    ---------------------------------------------------------

    References:
    Su,Y., Du,K., Wang,J., Wei,J. and Liu,J. (2022) Multi-variable AUC for sifting complementary features and its biomedical application. Briefings in Bioinformatics, 23, bbac029.
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # 加载 UCI 数据
    f, c = load_data(lable_name, data_path, label_path)

    pos, neg = set(c)
    n0, n1 = list(c).count(pos), list(c).count(neg)

    FName, Fauc, fr, fre = feature_ranking(f, c, max_rank, pos, neg, n0, n1)  # 注意，这里把 n0 和 n1 作为参数传递了

    fre1 = dict(Counter(fre))
    fre2 = {key: value for key, value in fre1.items() if value > 1}
    frequency = sorted(fre2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    print_results(fr, fre1, frequency, len(FName), FName, Fauc)
    return(fr, fre1, frequency, len(FName), FName, Fauc)






