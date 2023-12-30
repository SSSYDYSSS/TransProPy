# TransProPy.UtilsFunction3.train_model.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from TransProPy.UtilsFunction3.LoggingCustomScorer import logging_custom_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def train_model(X, Y, feature_selection, parameters, n_iter, n_cv, n_jobs=9):
    """
    Set up and run the model training process.

    Parameters:
    - X: DataFrame, feature data.
    - Y: ndarray, label data.
    - feature_selection: FeatureUnion, the feature selection process.
    - parameters: dict, parameters for RandomizedSearchCV.
    - n_iter: int, number of iterations for RandomizedSearchCV.
    - n_cv: int, number of cross-validation folds.
    - n_jobs: int, number of jobs to run in parallel (default is 9).

    Returns:
    - clf: RandomizedSearchCV object after fitting.
    """
    feature_selection_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('feature_selection', feature_selection),
        ('stacking', StackingClassifier(
            estimators=[
                ('svm', SVC(probability=True)),
                ('dt', DecisionTreeClassifier()),
                ('gbm', GradientBoostingClassifier())
            ],
            final_estimator=LogisticRegression()))
    ])

    clf = RandomizedSearchCV(
        feature_selection_pipeline,
        parameters,
        cv=StratifiedKFold(n_splits=n_cv),
        scoring=make_scorer(logging_custom_scorer(n_iter=n_iter, n_cv=n_cv)),
        n_iter=n_iter,
        random_state=0,
        error_score='raise',
        n_jobs=n_jobs  # Use the customizable n_jobs parameter
    )
    return clf
