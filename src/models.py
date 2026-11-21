"""Model definitions: Isolation Forest scoring, SMOTE + XGBoost classifier, tuning, thresholds.

``SmoteXGB`` lives here (not in a training script) so a saved model can be loaded from any
entry point, such as the Streamlit app. xgboost and imbalanced-learn are imported lazily so the
rest of the package works without them.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, RandomizedSearchCV

from .data import SEED

SEARCH_SPACE = {
    "sampling_strategy": [0.02, 0.05, 0.1, 0.2, 0.5],
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 1, 2],
    "reg_lambda": [1, 5, 10],
}


def make_iforest(seed):
    return IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=-1)


def iforest_scores(A_tr, A_te, seed=SEED, n_folds=5):
    """Anomaly scores (higher = more anomalous).

    Training rows get out-of-fold scores so the feature is not fitted on the rows it describes;
    test rows are scored by a forest fitted on the whole training set only.
    """
    oof = np.zeros(len(A_tr))
    for a, b in KFold(n_folds, shuffle=True, random_state=seed).split(A_tr):
        m = make_iforest(seed).fit(A_tr[a])
        oof[b] = -m.score_samples(A_tr[b])
    full = make_iforest(seed).fit(A_tr)
    return oof, -full.score_samples(A_te), full


class SmoteXGB(ClassifierMixin, BaseEstimator):
    """XGBoost with optional SMOTE applied inside ``fit`` (so only to training data)."""

    def __init__(self, use_smote=True, sampling_strategy=0.1, k_neighbors=5,
                 n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=1, gamma=0, reg_lambda=1,
                 random_state=SEED, n_jobs=-1):
        self.use_smote = use_smote
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from imblearn.over_sampling import SMOTE
        from xgboost import XGBClassifier

        X, y = np.asarray(X), np.asarray(y)
        if self.use_smote:
            X, y = SMOTE(sampling_strategy=self.sampling_strategy,
                         k_neighbors=self.k_neighbors,
                         random_state=self.random_state).fit_resample(X, y)
        self.model_ = XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            importance_type="gain", n_estimators=self.n_estimators,
            max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, gamma=self.gamma,
            reg_lambda=self.reg_lambda, random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def tune_hybrid(Xh_tr, y_tr, cv, n_iter=12):
    """Randomised search scored on PR-AUC; SMOTE runs inside every training fold."""
    search = RandomizedSearchCV(SmoteXGB(), SEARCH_SPACE, n_iter=n_iter, scoring="average_precision",
                                cv=cv, n_jobs=1, random_state=SEED, verbose=1)
    search.fit(Xh_tr, y_tr)
    return search


def threshold_for_fpr(y_ref, p_ref, max_fpr):
    """Threshold so that about max_fpr of legitimate reference transactions are flagged."""
    return float(np.quantile(p_ref[np.asarray(y_ref) == 0], 1 - max_fpr))
