from sklearn.metrics import accuracy_score
import logging

def logging_custom_scorer(n_iter=10, n_cv=5):
    """
    Creates a custom scorer function for use in model evaluation processes.
    This scorer logs the accuracy score each time it is called.

    Parameters:
    n_iter (int): Number of iterations for the search process. Default is 10.
    n_cv (int): Number of cross-validation splits. Default is 5.

    Returns:
    function: A custom scorer function that logs the accuracy score each time it is called.
    """

    def custom_scorer(y_true, y_pred):
        """
        Inner function to calculate the accuracy score and log it.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels by the model.

        Returns:
        float: The accuracy score.
        """
        # Calculate the accuracy score
        score = accuracy_score(y_true, y_pred)
        # Log the score
        logging.info(f"One scoring iteration completed, accuracy: {score}")
        return score

    return custom_scorer
