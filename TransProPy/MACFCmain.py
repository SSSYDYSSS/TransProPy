from numpy import *
from TransProPy.UtilsFunction1.LoadData import load_data
from TransProPy.UtilsFunction1.FeatureRanking import feature_ranking
from TransProPy.UtilsFunction1.PrintResults import print_results
from collections import Counter

def MACFCmain(max_rank, lable_name, threshold, data_path='../data/gene_tpm.csv', label_path='../data/tumor_class.csv'):
    """
    1.1_feature_ranking_modle.
    Applying the MACFC selection for relevant feature genes in classification.
    --------------------------------------------------------------------------
    Parameters:
    max_rank: int
        The total number of gene combinations you want to obtain.
    lable_name: string
        For example: gender, age, altitude, temperature, quality, and other categorical variable names.
    data_path: string
        For example: '../data/gene_tpm.csv'
        Please note: Preprocess the input data in advance to remove samples that contain too many missing values or zeros.
    label_path: string
        For example: '../data/tumor_class.csv'
        Please note: The input sample categories must be in a numerical binary format, such as: 1,2,1,1,2,2,1.
        In this case, the numerical values represent the following classifications: 1: male; 2: female.
    threshold: float
        For example: 0.9
        The set threshold indicates the proportion of non-zero value samples to all samples in each feature.
    --------------------------------------------------------------------------------------------------------
    Returns:
    fr: list of strings
        representing ranked features.
    fre1: dictionary
        feature names as keys and their frequencies as values.
    frequency: list of tuples
        feature names and their frequencies.
    len(FName): integer
        count of AUC values greater than 0.5.
    FName: array of strings
        feature names after ranking with AUC > 0.5.
    Fauc: array of floats
        AUC values corresponding to the ranked feature names.
    ---------------------------------------------------------

    References:
    Su,Y., Du,K., Wang,J., Wei,J. and Liu,J. (2022) Multi-variable AUC for sifting complementary features and its biomedical application. Briefings in Bioinformatics, 23, bbac029.
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # 加载 UCI 数据
    f, c = load_data(lable_name, threshold, data_path, label_path)

    pos, neg = set(c)
    n0, n1 = list(c).count(pos), list(c).count(neg)

    FName, Fauc, fr, fre = feature_ranking(f, c, max_rank, pos, neg, n0, n1)  # 注意，这里把 n0 和 n1 作为参数传递了

    fre1 = dict(Counter(fre))
    fre2 = {key: value for key, value in fre1.items() if value > 1}
    frequency = sorted(fre2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    print_results(fr, fre1, frequency, len(FName), FName, Fauc)
    return(fr, fre1, frequency, len(FName), FName, Fauc)






