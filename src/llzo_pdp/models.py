"""Fixed-protocol training and portable prediction bundles for M2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .preprocess import ImputationResult


def split_ids(y: pd.Series, seed: int, stratified: bool = True) -> dict[str, list[int]]:
    """Split test first, then validation; preserve ID order for exact reruns."""
    if not y.index.is_unique or y.isna().any():
        raise ValueError("Labels must have unique IDs and no missing values")
    train_val, test = train_test_split(y.index, test_size=0.15, random_state=seed,
                                     stratify=y if stratified else None)
    train, val = train_test_split(train_val, test_size=0.15/0.85, random_state=seed,
                                 stratify=y.loc[train_val] if stratified else None)
    return {"train": list(train), "val": list(val), "test": list(test)}


def make_estimator(name: str, seed: int):
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1)
    if name == "CB":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, thread_count=1, verbose=False,
                                  allow_writing_files=False)
    if name == "AB":
        return AdaBoostClassifier(random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def safe_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Keep model-internal names independent of each library's name restrictions."""
    return X.set_axis([f"f{i}" for i in range(X.shape[1])], axis=1)


def fit_grid(X: pd.DataFrame, y: pd.Series, name: str, seed: int, grid: dict) -> GridSearchCV:
    if not X.index.equals(y.index):
        raise ValueError("Feature rows and labels must be aligned")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Model inputs must be finite after imputation")
    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Five-fold tuning requires at least five rows of each class")
    search = GridSearchCV(make_estimator(name, seed), grid,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          scoring="f1_weighted", n_jobs=1, refit=True, error_score="raise")
    return search.fit(safe_columns(X), y)


def classification_metrics(y_true, y_pred, p_high) -> dict:
    y_true, y_pred, p_high = map(np.asarray, (y_true, y_pred, p_high))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {"n": len(y_true), "true_0": int((y_true == 0).sum()),
            "true_1": int((y_true == 1).sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_true, p_high) if two_classes else np.nan,
            "auc_reason": "" if two_classes else "single_class_test",
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@dataclass
class ModelBundle:
    estimator: object
    imputation: ImputationResult
    feature_names: list[str]
    context: dict

    @property
    def classes_(self):
        return self.estimator.classes_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not set(self.feature_names).issubset(X.columns):
            raise ValueError("Input is missing required features")
        return self.imputation.transform(X.loc[:, self.feature_names])

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(safe_columns(self.transform(X)))

    def predict(self, X: pd.DataFrame):
        return np.asarray(self.estimator.predict(safe_columns(self.transform(X)))).reshape(-1)
