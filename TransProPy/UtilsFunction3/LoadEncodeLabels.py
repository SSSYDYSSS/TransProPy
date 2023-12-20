from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_encode_labels(file_path, column_name):
    """
    Reads a CSV file containing labels and encodes categorical labels in the specified column to numeric labels.

    Parameters:
    file_path (str): Path to the CSV file containing labels.
    column_name (str): Name of the column to be encoded.

    Returns:
    pd.DataFrame: A DataFrame containing the encoded numeric labels.
    """

    # Load the data
    y = pd.read_csv(file_path, index_col=0, header=0)

    # Check if the specified column exists in the DataFrame
    if column_name not in y.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")

    # Create an instance of LabelEncoder
    le = LabelEncoder()

    # Apply LabelEncoder to the specified column
    y_encoded = le.fit_transform(y[column_name]) # Many Scikit-learn models require Y to be numerical. Therefore, if Y is categorical, use the fit_transform method of LabelEncoder to convert the character labels of Y into integers.

    # Convert the encoded labels back to a DataFrame
    Y = pd.DataFrame(y_encoded, index=y.index, columns=[column_name])

    return Y


