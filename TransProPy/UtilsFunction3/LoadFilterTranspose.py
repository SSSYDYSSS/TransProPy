import pandas as pd

def load_filter_transpose(threshold, data_path='../data/gene_tpm.csv'):
    """
    Remove samples with high zero expression.
    -----------------------------------------
    Parameters
    data_path: string
        For example: '../data/gene_tpm.csv'
        Please note: The input data matrix should have genes as rows and samples as columns.
    threshold: float
        For example: 0.9
        The set threshold indicates the proportion of non-zero value samples to all samples in each feature.
    --------------------------------------------------------------------------------------------------------
    Return
        X: pandas.core.frame.DataFrame
    -----------------------------------
    """
    data = pd.read_csv(data_path, index_col=0, header=0)
    # Calculate the count of non-zero values in each row.
    non_zero_counts = data.astype(bool).sum(axis=1)
    # Set a threshold indicating the proportion of gene expressions that are zeros.
    # threshold = 0.9
    # Filter rows based on the threshold.
    X = data[non_zero_counts / data.shape[1] > threshold]
    X = X.transpose()
    # Return the result.
    return X
