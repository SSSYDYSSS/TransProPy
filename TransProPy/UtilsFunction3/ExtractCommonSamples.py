def extract_common_samples(X, Y):
    """
    Extracts common samples (rows) from two DataFrames based on their indices.

    Parameters:
    X (pd.DataFrame): First DataFrame.
    Y (pd.DataFrame): Second DataFrame.

    Returns:
    pd.DataFrame, pd.DataFrame: Two DataFrames containing only the rows that are common in both.
    """
    # Find common indices
    common_indices = X.index.intersection(Y.index)

    # Filter both DataFrames to keep only common rows
    X_common = X.loc[common_indices]
    Y_common = Y.loc[common_indices]

    return X_common, Y_common