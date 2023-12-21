from sklearn.metrics import accuracy_score
import logging
import time

def logging_custom_scorer(n_iter=10, n_cv=5):
    """
    Creates a custom scorer function for use in model evaluation processes.
    This scorer logs both the accuracy score and the time taken for each call.

    Parameters:
    n_iter (int): Number of iterations for the search process. Default is 10.
    n_cv (int): Number of cross-validation splits. Default is 5.

    Returns:
    function: A custom scorer function that logs the accuracy score and time taken for each call.
    """

    # Initialize the time for the first call
    last_time = time.time()

    def custom_scorer(y_true, y_pred):
        """
        Inner function to calculate the accuracy score, log it, and measure the time taken.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels by the model.

        Returns:
        float: The accuracy score.
        """
        nonlocal last_time  # Reference the last_time from the outer scope

        # Record the current time and calculate the elapsed time since the last call
        current_time = time.time()
        elapsed = current_time - last_time
        last_time = current_time  # Update last_time for the next call

        # Calculate the accuracy score
        score = accuracy_score(y_true, y_pred)

        # Log the accuracy and the time taken for this scoring iteration
        logging.info(f"One scoring iteration completed, accuracy: {score}, time taken: {elapsed:.2f} seconds")

        return score

    return custom_scorer

