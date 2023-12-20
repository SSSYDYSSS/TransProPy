import logging
from tqdm import tqdm

def setup_logging_and_progress_bar(n_iter, n_cv):
    """
    Set up logging and initialize a tqdm progress bar.

    Parameters:
    n_iter (int): Number of iterations for RandomizedSearchCV.
    n_cv (int): Number of cross-validation folds.

    Returns:
    tqdm object: An initialized tqdm progress bar.
    """

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        filename='progress.log',
                        filemode='w')

    # Calculate total iterations
    total_iterations = n_iter * n_cv

    # Initialize and return tqdm progress bar
    pbar = tqdm(total=total_iterations, desc='RandomizedSearchCV Progress')
    return pbar

