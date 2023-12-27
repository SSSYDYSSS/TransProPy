# TransProPy.UtilsFunction3.load_and_preprocess_data.py

from TransProPy.UtilsFunction3.LoadFilterTranspose import load_filter_transpose
from TransProPy.UtilsFunction3.LoadEncodeLabels import load_encode_labels
from TransProPy.UtilsFunction3.ExtractCommonSamples import extract_common_samples


def load_and_preprocess_data(feature_file, label_file, label_column, threshold):
    """
    Load and preprocess the data.

    Parameters:
    - feature_file: str, path to the feature data file.
    - label_file: str, path to the label data file.
    - label_column: str, column name of the labels in the label file.
    - threshold: float, threshold for filtering in load_filter_transpose function.

    Returns:
    - X: DataFrame, preprocessed feature data.
    - Y: ndarray, preprocessed label data.
    """
    X = load_filter_transpose(threshold, feature_file)  # Load and filter features
    Y = load_encode_labels(label_file, label_column)  # Load and encode labels
    X, Y = extract_common_samples(X, Y)  # Extract common samples
    Y = Y.values.ravel()  # Flatten Y to 1D array
    return X, Y
