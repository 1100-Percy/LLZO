"""Row filtering, deterministic numeric MICE-like imputation and thresholds."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning


def filter_rows(raw_features: pd.DataFrame) -> tuple[pd.Index, pd.DataFrame]:
    """Count raw selected columns once, before descriptors and one-hot expansion."""
    missing = raw_features.isna().sum(axis=1)
    log = pd.DataFrame({"record_id": raw_features.index, "n_missing": missing.to_numpy(),
                        "kept": (missing <= 3).to_numpy()})
    return raw_features.index[missing <= 3], log


@dataclass
class ImputationResult:
    X: pd.DataFrame
    imputer: IterativeImputer
    numeric_columns: list[str]
    fit_ids: list[int]
    convergence_warnings: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform held-out or diagnostic rows without refitting."""
        result = X.copy()
        result[self.numeric_columns] = self.imputer.transform(X[self.numeric_columns])
        return result


def impute_features(X: pd.DataFrame, numeric_columns: list[str], mode: str,
                    seed: int, train_ids=None) -> ImputationResult:
    """A0 fits all supplied kept rows; A1 requires explicit training record IDs."""
    if mode not in ("A0", "A1"):
        raise ValueError(f"Unknown imputation mode: {mode}")
    if mode == "A1" and train_ids is None:
        raise ValueError("A1 requires train_ids")
    ids = list(X.index if mode == "A0" else train_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(X.index):
        raise ValueError("Invalid training IDs")
    train = X.loc[ids, numeric_columns]
    if train.isna().all().any():
        raise ValueError("All-missing training column cannot be imputed")
    imputer = IterativeImputer(random_state=seed, max_iter=10, tol=1e-3,
                               initial_strategy="mean", imputation_order="ascending",
                               sample_posterior=False, skip_complete=False,
                               keep_empty_features=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        imputer.fit(train)
    convergence_warnings = []
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            convergence_warnings.append(str(warning.message))
        else:
            warnings.warn(str(warning.message), warning.category)
    result = ImputationResult(X.copy(), imputer, list(numeric_columns), ids,
                              convergence_warnings)
    result.X = result.transform(X)
    return result


def threshold_table(sigma: pd.Series, main: float, benchmark: float) -> pd.DataFrame:
    """Choose the nearest attainable high fraction; ties prefer fewer high labels."""
    if sigma.empty or sigma.isna().any() or not np.isfinite(sigma).all():
        raise ValueError("Thresholds require finite observed conductivity values")
    ratio = 21 / 33
    candidates = sorted(sigma.unique())
    ratio_threshold = min(candidates, key=lambda t: (abs((sigma >= t).mean() - ratio), -t))
    thresholds = {"T_main": main, "T_median_kept": float(sigma.median()),
                  "T_1e4": benchmark, "T_ratio": ratio_threshold}
    return pd.DataFrame([{
        "threshold_name": name, "threshold": threshold,
        "n_high": int((sigma >= threshold).sum()),
        "high_fraction": float((sigma >= threshold).mean()),
        "ratio_target": ratio, "ratio_error": float((sigma >= threshold).mean() - ratio),
        "ratio_exact": bool(np.isclose((sigma >= threshold).mean(), ratio, atol=1e-12, rtol=0)),
    } for name, threshold in thresholds.items()])
