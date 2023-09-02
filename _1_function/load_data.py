from pandas import *
from numpy import *
from TransProPy._1_function import auto_norm

def load_data(lable_name):
    """
    Data Reading and Transformation.
        Data normalization for constant value
        Extract matrix data and categorical data.
    ---------------------------------------------
    Parameters:
    lable_name : string
        For example: gender, age, altitude, temperature, quality, and other categorical variable names.
    ---------------------------------------------------------------------------------------------------
    Returns:
    transpose(f) : ndarray
        A transposed feature-sample matrix.
    c : ndarray
        A NumPy array containing classification labels.
    ---------------------------------------------------------
    """
    data = read_csv('../data/gene_tpm.csv', header=0, index_col=0)
    data = data.transpose()
    lable = read_csv('../data/tumor_class.csv', header=0, index_col=0)
    lable = lable[lable_name]
    data = merge(data, lable, left_index=True, right_index=True)
    values = unique(data.values, axis=0)
    f = auto_norm(values[:, :-1])  # data normalization for constant value
    c = values[:, -1]
    # return feature*sample matrix and class labels
    return transpose(f), c
