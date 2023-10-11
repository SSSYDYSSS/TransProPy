from autogluon.tabular import TabularDataset, TabularPredictor
from TransProPy.UtilsFunction2.splitdata import split_data

def AutoGluon_SelectML(gene_data_path, class_data_path, label_column, test_size, threshold, hyperparameters=None, random_feature=None, num_bag_folds=None, num_stack_levels=None, time_limit=120, random_state=42):
    """
    Trains a model using AutoGluon on provided data path and returns feature importance and model leaderboard.
    ----------------------------------------------------------------------------------------------------------
    Parameters:
    - gene_data_path (str): Path to the gene expression data CSV file.
        For example: '../data/gene_tpm.csv'
    - class_data_path (str): Path to the class data CSV file.
        For example: '../data/tumor_class.csv'
    - label_column (str): Name of the column in the dataset that is the target label for prediction.
    - test_size (float): Proportion of the data to be used as the test set.
    - threshold (float): The threshold used to filter out rows based on the proportion of non-zero values.
    - hyperparameters (dict, optional): Dictionary of hyperparameters for the models.
        For example: {'GBM': {}, 'RF': {}}
    - random_feature (int, optional): The number of random feature to select. If None, no random feature selection is performed. Default is None.
    - num_bag_folds (int, optional):
       *Please note: This parameter annotation source can be referred to the documentation link in References.
        Number of folds used for bagging of models. When `num_bag_folds = k`, training time is roughly increased by a factor of `k` (set = 0 to disable bagging).
        Disabled by default (0), but we recommend values between 5-10 to maximize predictive performance.
        Increasing num_bag_folds will result in models with lower bias but that are more prone to overfitting.
        `num_bag_folds = 1` is an invalid value, and will raise a ValueError.
        Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
        To further improve predictions, avoid increasing `num_bag_folds` much beyond 10 and instead increase `num_bag_sets`.
        default = None
    - num_stack_levels (int, optional):
       *Please note: This parameter annotation source can be referred to the documentation link in References.
        Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of `num_stack_levels+1` (set = 0 to disable stack ensembling).
        Disabled by default (0), but we recommend values between 1-3 to maximize predictive performance.
        To prevent overfitting, `num_bag_folds >= 2` must also be set or else a ValueError will be raised.
        default = None
    - time_limit (int, optional): Time limit for training in seconds.
        Default is 120.
    - random_state (int, optional): The seed used by the random number generator.
        Default is 42.
    --------------------------------------------------------------------------------------------
    Returns:
    - importance (DataFrame): DataFrame containing feature importance.
    - leaderboard (DataFrame): DataFrame containing model performance on the test data.
    -----------------------------------------------------------------------------------
    References:
    Scientific Publications:
    - AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data (Arxiv, 2020)
    - Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation (NeurIPS, 2020)
    - Multimodal AutoML on Structured Tables with Text Fields (ICML AutoML Workshop, 2021)
    Articles:
    - AutoGluon for tabular data: 3 lines of code to achieve top 1% in Kaggle competitions (AWS Open Source Blog, Mar 2020)
    - Accurate image classification in 3 lines of code with AutoGluon (Medium, Feb 2020)
    - AutoGluon overview & example applications (Towards Data Science, Dec 2019)
    Documentation:
    - https://auto.gluon.ai/0.1.0/api/autogluon.predictor.html?highlight=num_bag_folds
    --------------------------------------------------------------------------------
    """

    train_data, test_data = split_data(gene_data_path, class_data_path, class_name=label_column, test_size=test_size, random_state=random_state, threshold=threshold, random_feature=random_feature)
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    # Train the model using AutoGluon
    predictor = TabularPredictor(label=label_column).fit(train_data, hyperparameters=hyperparameters, time_limit=time_limit, num_bag_folds=num_bag_folds, num_stack_levels=num_stack_levels)

    # Get the feature importance
    importance = predictor.feature_importance(test_data, subsample_size=None)

    # Get the leaderboard of models
    leaderboard = predictor.leaderboard(test_data)

    return importance, leaderboard

