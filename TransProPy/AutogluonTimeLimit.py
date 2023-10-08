from autogluon.tabular import TabularDataset, TabularPredictor
from TransProPy.UtilsFunction2.splitdata import split_data
def Autogluon_TimeLimit(gene_data_path, class_data_path, label_column, test_size, threshold, random_feature=None, time_limit=120, random_state=42):
    """
    2.1_autogluon_time-limit.
    Trains a model using AutoGluon on provided data path and returns feature importance and model leaderboard.

    Parameters:
    - gene_data_path (str): Path to the gene expression data CSV file.
        For example: '../data/gene_tpm.csv'
    - class_data_path (str): Path to the class data CSV file.
         For example: '../data/tumor_class.csv'
    - label_column (str): Name of the column in the dataset that is the target label for prediction.
    - test_size (float): Proportion of the data to be used as the test set.
    - threshold (float): The threshold used to filter out rows based on the proportion of non-zero values.
    - random_feature (int, optional): The number of random feature to select. If None, no random feature selection is performed. Default is None.
    - time_limit (int, optional): Time limit for training in seconds. Default is 120.
    - random_state (int): The seed used by the random number generator. Default is 42.

    Returns:
    - importance (dict): DataFrame containing feature importance.
    - leaderboard (DataFrame): DataFrame containing model performance on the test data.
    """
    train_data, test_data = split_data(gene_data_path, class_data_path, class_name=label_column, test_size=test_size, random_state=random_state, threshold=threshold, random_feature=random_feature)
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    # Train the model using AutoGluon
    predictor = TabularPredictor(label=label_column).fit(train_data, time_limit=time_limit)

    # Get the feature importance
    importance = predictor.feature_importance(test_data, subsample_size=None)

    # Get the leaderboard of models
    leaderboard = predictor.leaderboard(test_data)

    return importance, leaderboard





