from numpy import *
from TransProPy.UtilsFunction1.LoadData import load_data
from TransProPy.UtilsFunction1.NewFeatureRanking import new_feature_ranking
from TransProPy.UtilsFunction1.PrintResults import print_results
from collections import Counter

def New_MACFCmain(max_rank, lable_name, threshold, data_path='../data/gene_tpm.csv', label_path='../data/tumor_class.csv'):
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
    high_auc_features: list of tuples
        This list contains tuples of feature indices and their corresponding AUC values, where the AUC value is greater than 0.95. Each tuple consists of the feature's index in string format and its AUC value as a float. This signifies that these features are highly predictive, with a strong ability to distinguish between different classes in the classification task.
    fr: list of strings
        representing ranked features.
    fre1: dictionary
        feature names as keys and their frequencies as values.
    frequency: list of tuples
        feature names and their frequencies.
        The frequency outputs a list sorted by occurrence frequency (in descending order). This list includes only those elements from the dictionary fre1 (which represents the counted frequencies of elements in the original data) that have an occurrence frequency greater than once, along with their frequencies.
    len(FName): integer
        count of AUC values greater than 0.5.
    FName: array of strings
        feature names after ranking with AUC > 0.5.
    Fauc: array of floats
        AUC values corresponding to the ranked feature names.
    ---------------------------------------------------------
    References:
    - Su,Y., Du,K., Wang,J., Wei,J. and Liu,J. (2022) Multi-variable AUC for sifting complementary features and its biomedical application. Briefings in Bioinformatics, 23, bbac029.
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # load data
    f, c = load_data(lable_name, threshold, data_path, label_path)

    pos, neg = set(c)
    n0, n1 = list(c).count(pos), list(c).count(neg)

    high_auc_features, FName, Fauc, fr, fre = new_feature_ranking(f, c, max_rank, pos, neg, n0, n1)  # Note that here n0 and n1 are passed as parameters.

    fre1 = dict(Counter(fre))
    fre2 = {key: value for key, value in fre1.items() if value > 1}
    frequency = sorted(fre2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    # print_results(fr, fre1, frequency, len(FName), FName, Fauc)
    return(high_auc_features, fr, fre1, frequency, len(FName), FName, Fauc)


