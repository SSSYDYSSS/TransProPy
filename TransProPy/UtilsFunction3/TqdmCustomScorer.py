from sklearn.metrics import make_scorer, accuracy_score
from tqdm import tqdm

def tqdm_custom_scorer(n_iter=10, n_cv=5):
    """
    This function creates a custom scorer for use in model evaluation processes like
    RandomizedSearchCV. It integrates a progress bar to track the evaluation process.

    Parameters:
    n_iter (int): Number of iterations for the search process. Default is 10.
    n_cv (int): Number of cross-validation splits. Default is 5.

    Returns:
    function: A custom scorer function that can be used with model evaluation methods.
    """

    # Initialize a tqdm progress bar with a total count based on the number of iterations and CV splits
    pbar = tqdm(total=n_iter * n_cv, desc='RandomizedSearchCV progress')

    # Define an inner function that will be used as the scorer
    def custom_scorer(y_true, y_pred):
        """
        Inner function to calculate the accuracy score and update the progress bar.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels by the model.

        Returns:
        float: The accuracy score.
        """
        # Calculate the accuracy score
        score = accuracy_score(y_true, y_pred)
        # Update the progress bar
        pbar.update()
        return score

    # Return the custom scorer function
    return custom_scorer




