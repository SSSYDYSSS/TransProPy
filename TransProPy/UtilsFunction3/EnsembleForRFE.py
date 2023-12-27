from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class EnsembleForRFE(BaseEstimator):
    """
    Ensemble estimator for recursive feature elimination.

    Parameters:
    - svm_C: float, regularization parameter for SVM.
    - tree_max_depth: int, maximum depth of the decision tree.
    - tree_min_samples_split: int, minimum number of samples required to split an internal node.
    - gbm_learning_rate: float, learning rate for gradient boosting.
    - gbm_n_estimators: int, number of boosting stages to be run for gradient boosting.
    """

    def __init__(self, svm_C=1.0, tree_max_depth=None,
                 tree_min_samples_split=2, gbm_learning_rate=0.1,
                 gbm_n_estimators=100):
        # Save passed parameters as class attributes
        self.svm_C = svm_C
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_split = tree_min_samples_split
        self.gbm_learning_rate = gbm_learning_rate
        self.gbm_n_estimators = gbm_n_estimators

        # Initialize individual models with the specified parameters
        self.svm = SVC(kernel="linear", probability=True, C=self.svm_C)
        self.tree = DecisionTreeClassifier(max_depth=self.tree_max_depth,
                                           min_samples_split=self.tree_min_samples_split)
        self.gbm = GradientBoostingClassifier(learning_rate=self.gbm_learning_rate,
                                              n_estimators=self.gbm_n_estimators)

        self.feature_importances_ = None # Initialize feature importances attribute

    def fit(self, X, y):
        """
        Fit the individual models and compute aggregated feature importances.

        Parameters:
        - X: DataFrame, Feature dataset with shape (n_samples, n_features).
        - y: ndarray, 1-D array of target values with shape (n_samples,).

        Returns:
        - self: object, Instance of the model.
        """
        # Fit individual models
        self.svm.fit(X, y)
        self.tree.fit(X, y)
        self.gbm.fit(X, y)

        # Calculate feature importances and store as attributes
        svm_importances = np.abs(self.svm.coef_[0])
        tree_importances = self.tree.feature_importances_
        gbm_importances = self.gbm.feature_importances_

        # Average feature importances
        self.feature_importances_ = (svm_importances + tree_importances + gbm_importances) / 3
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X using a soft voting mechanism.

        Parameters:
        - X: DataFrame, Input features.

        Returns:
        - Predicted class labels.
        """
        # Get the probability predictions from individual models
        probabilities = np.array([self.svm.predict_proba(X),
                                  self.tree.predict_proba(X),
                                  self.gbm.predict_proba(X)])

        # Average probabilities for soft voting
        avg_prob = np.mean(probabilities, axis=0)

        # Predict class labels based on the highest probability
        return np.argmax(avg_prob, axis=1)

    def set_params(self, **params):
        """
        Set parameters for the ensemble estimator. This will be used by hyperparameter
        optimization methods like RandomizedSearchCV to update the parameters of the
        individual models.

        Parameters:
        - **params: Keyword arguments for parameter names and values.
        """
        # Update the parameter values based on provided keyword arguments
        for key, value in params.items():
            if key in ['svm_C', 'tree_max_depth',
                       'tree_min_samples_split', 'gbm_learning_rate',
                       'gbm_n_estimators']:
                setattr(self, key, value)

        # Re-initialize the models with the updated parameters
        self.svm = SVC(kernel="linear", probability=True, C=self.svm_C)
        self.tree = DecisionTreeClassifier(max_depth=self.tree_max_depth,
                                           min_samples_split=self.tree_min_samples_split)
        self.gbm = GradientBoostingClassifier(learning_rate=self.gbm_learning_rate,
                                              n_estimators=self.gbm_n_estimators)
        return self
