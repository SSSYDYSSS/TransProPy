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

    # Configure basic logging - this time, without filename and filemode
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s')

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler('progress.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

    # Create a stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))

    # Get the default logger and add the two handlers to it
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Calculate total iterations
    total_iterations = n_iter * n_cv

    # Initialize and return tqdm progress bar
    pbar = tqdm(total=total_iterations, desc='RandomizedSearchCV Progress')
    return pbar
