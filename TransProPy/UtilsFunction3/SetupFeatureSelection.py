# TransProPy.UtilsFunction3.setup_feature_selection.py

from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from TransProPy.UtilsFunction3.EnsembleForRFE import EnsembleForRFE


def setup_feature_selection():
    """
    Set up the feature selection process.

    Returns:
    - feature_selection: FeatureUnion, combined feature selection process.
    """
    ensemble_estimator = EnsembleForRFE()
    rfecv = RFECV(estimator=ensemble_estimator, cv=StratifiedKFold(5), scoring='accuracy')
    selectkbest = SelectKBest(score_func=mutual_info_classif)
    return FeatureUnion([("rfecv", rfecv), ("selectkbest", selectkbest)])
