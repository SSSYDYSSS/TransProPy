import threading
import time
from scipy.stats import reciprocal, randint
from TransProPy.UtilsFunction3.LoadAndPreprocessData import load_and_preprocess_data
from TransProPy.UtilsFunction3.SetupFeatureSelection import setup_feature_selection
from TransProPy.UtilsFunction3.TrainModel import train_model
from TransProPy.UtilsFunction3.ExtractAndSaveResults import extract_and_save_results
from TransProPy.UtilsFunction3.SetupLoggingAndProgressBar import setup_logging_and_progress_bar
from TransProPy.UtilsFunction3.UpdateProgressBar import update_progress_bar

def auto_feature_selection(data_file, label_file, label_col, threshold, show_plot, show_progress, n_iter=2, n_cv=5, n_jobs=9, save_path='../data/', sleep_interval=1, use_tkagg=False):
    """
    Run the complete analysis pipeline from data loading to training and result extraction.

    Parameters:
    - data_file: str, path to the feature data file.
    - label_file: str, path to the label data file.
    - label_col: str, name of the label column.
    - threshold: float, threshold for data preprocessing.
    - show_plot: bool, whether to display plot.
    - show_progress: bool, whether to show progress bar.
    - n_iter: int, number of iterations for RandomizedSearchCV.
    - n_cv: int, number of folds for cross-validation.
    - n_jobs: int, number of parallel jobs for RandomizedSearchCV.
    - save_path: str, path to save results.
    - sleep_interval: int, interval time in seconds for progress bar update.
    - use_tkagg: bool, whether to use 'TkAgg' backend for matplotlib. Generally, choose False when using in PyCharm IDE, and choose True when rendering file.qmd to an HTML file.
    """

    # Load and preprocess data
    X, Y = load_and_preprocess_data(data_file, label_file, label_col, threshold)

    # Set up feature selection
    feature_selection = setup_feature_selection()

    # Define parameters for RandomizedSearchCV
    parameters = {
        'feature_selection__rfecv__estimator__svm__C': reciprocal(0.001, 1000),
        'feature_selection__rfecv__estimator__tree__max_depth': randint(2, 10),
        'feature_selection__rfecv__estimator__tree__min_samples_split': randint(2, 10),
        'feature_selection__rfecv__estimator__gbm__learning_rate': reciprocal(0.01, 0.2),
        'feature_selection__rfecv__estimator__gbm__n_estimators': randint(100, 500),
        'feature_selection__rfecv__step': randint(10, 150),
        'feature_selection__rfecv__min_features_to_select': randint(10, 1000),
        'feature_selection__selectkbest__k': randint(10, 200),
        'stacking__final_estimator__C': reciprocal(0.001, 1000)  # Parameter for logistic regression in stacking classifier
    }

    # Train the model
    clf = train_model(X, Y, feature_selection, parameters, n_iter, n_cv, n_jobs)

    # Define a function to run RandomizedSearchCV
    def run_randomized_search():
        clf.fit(X, Y)

    # Initialize tqdm progress bar and logging
    if show_progress:
        progress_bar = setup_logging_and_progress_bar(n_iter, n_cv)
        search_thread = threading.Thread(target=run_randomized_search)
        search_thread.start()

        # Update the progress bar in the main thread
        while search_thread.is_alive():
            update_progress_bar(progress_bar)
            time.sleep(sleep_interval)

        # Ensure RandomizedSearchCV completes
        search_thread.join()
    else:
        run_randomized_search()

    # Extract and save results
    extract_and_save_results(clf, X, save_path, show_plot, use_tkagg)

