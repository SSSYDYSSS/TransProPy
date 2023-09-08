from numpy import *
def auto_norm(data):
    # data:（sample,feature）
    """
    Normalization Function
        The auto_norm function is designed to normalize a two-dimensional array (matrix). The purpose of normalization is generally to bring all features into the same numerical range, facilitating subsequent analysis or model training.
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Parameters:
    data : ndarray
        Order Requirements for Input Data：
        1.This function does indeed have specific requirements for the row and column order of the input matrix data. Rows should represent individual samples, and columns should represent different features. In other words, each row vector represents a sample containing multiple features.
        2.Each column of the matrix will be independently normalized, so different features should be placed in separate columns.
    -----------------------------------------------------------------------------------------------------------------------------
    Returns:
    norm_data:ndarray
        It is the normalized data.
    ------------------------------
    """
    mins = data.min(0)
    maxs = data.max(0)
    ranges = maxs - mins
    row = data.shape[0]
    norm_data = data - tile(mins, (row, 1))
    norm_data = norm_data / tile(ranges, (row, 1))
    return norm_data
