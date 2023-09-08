from pandas import *
from numpy import *
import os
from TransProPy.UtilsFunction1.AutoNorm import auto_norm

def load_data(lable_name, data_path='../data/gene_tpm.csv', label_path='../data/tumor_class.csv'):
    """
    Data Reading and Transformation.
        Data normalization for constant value
        Extract matrix data and categorical data.
    ---------------------------------------------
    Parameters:
    lable_name : string
        For example: gender, age, altitude, temperature, quality, and other categorical variable names.
    data_path : String
        For example: '../data/gene_tpm.csv'
        Please note: Preprocess the input data in advance to remove samples that contain too many missing values or zeros.
        The input data matrix should have genes as rows and samples as columns.
    label_path : String
        For example: '../data/tumor_class.csv'
        Please note: The input sample categories must be in a numerical binary format, such as: 1,2,1,1,2,2,1.
        In this case, the numerical values represent the following classifications: 1: male; 2: female.
    ---------------------------------------------------------------------------------------------------
    Returns:
    transpose(f) : ndarray
        A transposed feature-sample matrix.
    c : ndarray
        A NumPy array containing classification labels.
    ---------------------------------------------------------
    """
    # Check if the data files exist at the given paths
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The data file was not found at '{data_path}'. Please ensure it's in the correct location.")

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"The label file was not found at '{label_path}'. Please ensure it's in the correct location.")

    # Continue with the rest of your function
    data = read_csv(data_path, header=0, index_col=0)
    data = data.transpose()
    lable = read_csv(label_path, header=0, index_col=0)
    lable = lable[lable_name]
    data = merge(data, lable, left_index=True, right_index=True)
    values = unique(data.values, axis=0)
    f = auto_norm(values[:, :-1])  # data normalization for constant value; assuming you have auto_norm function defined somewhere
    c = values[:, -1]

    return transpose(f), c