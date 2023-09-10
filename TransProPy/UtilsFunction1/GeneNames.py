import os
from pandas import read_csv, merge


def gene_name(data_path='../data/gene_tpm.csv'):
    """
    Extract gene_names data.
    ------------------------
    Parameters:
    data_path: string
        For example: '../data/gene_tpm.csv'
        Please note: Preprocess the input data in advance to remove samples that contain too many missing values or zeros.
        The input data matrix should have genes as rows and samples as columns.
    ---------------------------------------------------------------------------
    Return:
    gene_name: list
    ---------------------------------------------------------------------------
    """
    # Check if the data files exist at the given paths
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The data file was not found at '{data_path}'. Please ensure it's in the correct location.")

    # Load data and labels
    data = read_csv(data_path, header=0, index_col=0)  # Assuming row names are gene names
    # Get the gene names directly from the row names (assuming row names are gene names)
    gene_names = data.index.tolist()

    return gene_names