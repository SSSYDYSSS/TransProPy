import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(gene_data_path, class_data_path, class_name, test_size=0.2, random_state=42, threshold=0.9):
    """
    Reads the gene expression and class data, processes it, and splits it into training and testing sets.

    Parameters:
    - gene_data_path (str): Path to the CSV file containing the gene expression data.
        For example: '../data/gene_tpm.csv'
    - class_data_path (str): Path to the CSV file containing the class data.
        For example: '../data/tumor_class.csv'
    - class_name (str): The name of the class column in the class data.
    - test_size (float, optional): The proportion of the data to be used as the testing set. Default is 0.2.
    - random_state (int, optional): The seed used by the random number generator. Default is 42.
    - threshold (float, optional): The threshold used to filter out rows based on the proportion of non-zero values. Default is 0.9.

    Returns:
    - train_data (pd.DataFrame): The training data.
    - test_data (pd.DataFrame): The testing data.
    """

    # Reading the data
    X = pd.read_csv(gene_data_path, index_col=0, header=0)
    y = pd.read_csv(class_data_path, index_col=0, header=0)

    # Finding common sample names between X(column names) and y(row names)
    common = X.columns.intersection(y.index)

    # Filtering out low-quality data
    non_zero_counts = X.astype(bool).sum(axis=1)
    X = X[non_zero_counts / X.shape[1] > threshold]

    # Keeping only the common samples in X and y
    X = X.loc[:, common]
    y = y.loc[common]

    # Transposing X and merging it with the specified column from y
    X = X.transpose()
    Y = y[class_name]
    data = pd.merge(X, Y, left_index=True, right_index=True)

    # Splitting the data into training and validation sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data
