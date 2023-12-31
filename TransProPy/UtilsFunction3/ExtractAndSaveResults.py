# TransProPy.UtilsFunction3.extract_and_save_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TransProPy.UtilsFunction3.PrintBoxedText import print_boxed_text
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

def extract_and_save_results(
        clf,
        X,
        Y,
        save_path,
        n_cv,
        show_plot=False,
        use_tkagg=False):
    """
    Extract and save various results from the trained model.

    Parameters:
    - clf: trained model (RandomizedSearchCV object).
    - X: DataFrame, feature data used for training.
    - save_path: str, base path for saving results.
    - show_plot: bool, whether to display the plot.
    - use_tkagg: bool, whether to use 'TkAgg' backend for matplotlib. Generally, choose True when using in PyCharm IDE, and choose False when rendering file.qmd to an HTML file.
    """

    # Setting the matplotlib backend to 'TkAgg' if specified
    if use_tkagg:
        import matplotlib
        matplotlib.use('TkAgg')

    # Extracting cross-validation results
    cv_results = clf.cv_results_
    mean_test_scores = cv_results['mean_test_score'] # Calculate the average test score for each iteration
    n_iterations = len(mean_test_scores)

    # Plotting and saving the accuracy per iteration figure
    plt.figure(figsize=(4, 3), facecolor='#f0f8fe')
    plt.plot(range(1, n_iterations + 1), mean_test_scores, marker='o')
    plt.title('Model Accuracy per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Test Accuracy')
    plt.grid(True, color='#11479c', alpha=0.2)
    # Get the current axes (ax), and set the background color of the plot area to white
    ax = plt.gca()
    ax.set_facecolor('#e1f0fb')
    # Call tight_layout to automatically adjust the layout
    plt.tight_layout()
    plt.savefig(save_path + "Model_Accuracy_per_Iteration_figure.pdf", format='pdf')
    # Optionally display the plot
    if show_plot:
        plt.show()

    # plotting the ROC curve
    # Predict probabilities
    y_probas = cross_val_predict(clf.best_estimator_, X, Y, cv=StratifiedKFold(n_splits=n_cv), method='predict_proba')
    # Take the probability of the positive class
    y_scores = y_probas[:, 1]
    # Calculate values for the ROC curve
    fpr, tpr, thresholds = roc_curve(Y, y_scores)
    # Calculate AUC value
    roc_auc = roc_auc_score(Y, y_scores)

    # Plot and save the ROC curve
    plt.figure(figsize=(4, 3), facecolor='#f0f8fe')
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for a random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    # Get the current axes (ax), and set the background color of the plot area to white
    ax = plt.gca()
    ax.set_facecolor('#e1f0fb')
    # Call tight_layout to automatically adjust the layout
    plt.tight_layout()
    plt.savefig(save_path + "ROC_Curve_figure.pdf", format='pdf')
    # Optionally display the plot
    if show_plot:
        plt.show()


    # Extracting feature selection results
    feature_union = clf.best_estimator_.named_steps['feature_selection']
    rfecv = feature_union.transformer_list[0][1]
    selectkbest = feature_union.transformer_list[1][1]
    selected_features_rfecv = rfecv.support_
    selected_features_selectkbest = selectkbest.get_support()
    # Print selected features
    print_boxed_text("Features selected by RFECV:")
    print(X.columns[selected_features_rfecv])
    print_boxed_text("Features selected by SelectKBest:")
    print(X.columns[selected_features_selectkbest])

    # Combining and saving selected features
    combined_selected_features = np.logical_or(selected_features_rfecv, selected_features_selectkbest)
    combined_features_df = pd.DataFrame({'Feature': X.columns[combined_selected_features]})
    combined_features_df.to_csv(save_path + 'combined_features.csv', index=False)
    print_boxed_text(f"Total number of selected features: {combined_features_df.shape[0]}")

    # Extracting and saving EnsembleForRFE feature importances
    ensemble_for_rfe = feature_union.transformer_list[0][1].estimator_
    feature_importances_ensemble = ensemble_for_rfe.feature_importances_
    importances_ensemble = zip(X.columns[selected_features_rfecv], feature_importances_ensemble)
    sorted_importances_ensemble = sorted(importances_ensemble, key=lambda x: x[1], reverse=True)
    df_importances_ensemble = pd.DataFrame(sorted_importances_ensemble, columns=['Feature', 'Importance'])
    df_importances_ensemble.to_csv(save_path + 'ensemble_importances.csv', index=False)
    print_boxed_text("Feature Importances from EnsembleForRFE:")
    print(df_importances_ensemble)

    # Extracting and saving SelectKBest scores
    selectkbest_scores = selectkbest.scores_[selected_features_selectkbest]
    scores_selectkbest = zip(X.columns[selected_features_selectkbest], selectkbest_scores)
    sorted_scores_selectkbest = sorted(scores_selectkbest, key=lambda x: x[1], reverse=True)
    df_scores_selectkbest = pd.DataFrame(sorted_scores_selectkbest, columns=['Feature', 'Score'])
    df_scores_selectkbest.to_csv(save_path + 'selectkbest_scores.csv', index=False)
    print_boxed_text("Scores from SelectKBest:")
    print(df_scores_selectkbest)