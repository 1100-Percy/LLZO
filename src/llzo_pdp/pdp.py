"""Brute virtual-input predictions and frozen-rule PDP window extraction."""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles


_MAX_BATCH_ROWS = 50_000
_RESERVED_COLUMNS = {"bg_record_id", "grid_value", "grid_value_2", "p_high"}
UpdateFunction = Callable[[pd.Series, float], pd.Series]


class ProbabilityModel(Protocol):
    classes_: np.ndarray

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return array


def make_grid(values: Sequence[float], kind: str = "G1") -> np.ndarray:
    """Use frozen G1/G2 rules; a constant feature has one grid point (D-018)."""
    array = _finite_vector(values, "Grid values")
    unique = np.unique(array)
    if kind == "G1":
        if len(unique) < 100:
            return unique
        # sklearn's default uses these public SciPy quantiles, not np.percentile.
        endpoints = mquantiles(array, prob=(0.05, 0.95), axis=0)
        if np.allclose(endpoints[0], endpoints[1]):
            raise ValueError("G1 percentiles are too close to build the grid")
        return np.linspace(endpoints[0], endpoints[1], 100)
    if kind == "G2":
        return unique if len(unique) == 1 else np.linspace(unique[0], unique[-1], 60)
    raise ValueError(f"Unknown grid kind: {kind}")


def _validate_background(model: ProbabilityModel, X_bg: pd.DataFrame,
                         features: Sequence[str]) -> np.ndarray:
    if not isinstance(X_bg, pd.DataFrame) or X_bg.empty:
        raise ValueError("Background must be a nonempty DataFrame")
    if not X_bg.index.is_unique or X_bg.index.to_frame().isna().to_numpy().any():
        raise ValueError("Background record IDs must be unique and nonmissing")
    if not X_bg.columns.is_unique or _RESERVED_COLUMNS.intersection(X_bg.columns):
        raise ValueError("Background feature names must be unique and not reserved")
    if any(feature not in X_bg.columns for feature in features):
        raise ValueError("Unknown feature in background")
    values = X_bg.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Background features must be finite after imputation")
    classes = np.asarray(getattr(model, "classes_", []))
    if (classes.shape != (2,) or classes.dtype.kind not in "biuf"
            or not np.array_equal(classes, [0, 1])):
        raise ValueError("Model classes must be [0, 1] in that order")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model must expose predict_proba probabilities")
    return values


def _probabilities(model: ProbabilityModel, virtual: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(virtual), dtype=float)
    if (probabilities.shape != (len(virtual), 2)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Model probabilities must be finite normalized binary probabilities")
    return probabilities[:, 1]


def _evaluate(model: ProbabilityModel, X_bg: pd.DataFrame, features: Sequence[str],
              grids: Sequence[np.ndarray], update_fn: Optional[UpdateFunction]) -> pd.DataFrame:
    values = _validate_background(model, X_bg, features)
    if update_fn is not None and not callable(update_fn):
        raise ValueError("update_fn must be callable")
    n_bg = len(X_bg)
    n_grid = len(grids[0]) * (len(grids[1]) if len(grids) == 2 else 1)
    positions = [X_bg.columns.get_loc(feature) for feature in features]
    chunks = []
    for start in range(0, n_grid * n_bg, _MAX_BATCH_ROWS):
        offsets = np.arange(start, min(start + _MAX_BATCH_ROWS, n_grid * n_bg))
        bg_positions = offsets % n_bg
        grid_positions = offsets // n_bg
        if len(grids) == 2:
            grid_values = [grids[0][grid_positions // len(grids[1])],
                           grids[1][grid_positions % len(grids[1])]]
        else:
            grid_values = [grids[0][grid_positions]]
        virtual_values = values[bg_positions].copy()
        if update_fn is None:
            for column, assigned in zip(positions, grid_values):
                virtual_values[:, column] = assigned
        else:
            for i, (bg_position, value) in enumerate(zip(bg_positions, grid_values[0])):
                updated = update_fn(X_bg.iloc[int(bg_position)].copy(deep=True), float(value))
                if not isinstance(updated, pd.Series) or not updated.index.equals(X_bg.columns):
                    raise ValueError("update_fn must return a Series with the same feature schema")
                try:
                    updated_values = updated.to_numpy(dtype=float)
                except (TypeError, ValueError) as error:
                    raise ValueError("update_fn must return finite numeric features") from error
                if not np.isfinite(updated_values).all():
                    raise ValueError("update_fn must return finite numeric features")
                virtual_values[i] = updated_values
        virtual = pd.DataFrame(virtual_values, columns=X_bg.columns)
        output = {"bg_record_id": X_bg.index.to_numpy()[bg_positions],
                  "grid_value": grid_values[0]}
        if len(grids) == 2:
            output["grid_value_2"] = grid_values[1]
        output["p_high"] = _probabilities(model, virtual)
        saved_columns = X_bg.columns if update_fn is not None else features
        for column in saved_columns:
            output[column] = virtual[column].to_numpy()
        chunks.append(pd.DataFrame(output))
    return pd.concat(chunks, ignore_index=True)


def ice(model: ProbabilityModel, X_bg: pd.DataFrame, feature: str,
        grid: Sequence[float], update_fn: Optional[UpdateFunction] = None) -> pd.DataFrame:
    """Predict each grid/background pair, ordered by grid then background row.

    Inputs must already be imputed. Without update_fn only the modified column
    is saved; unchanged columns can be joined from X_bg using bg_record_id.
    With update_fn all returned feature values are saved to retain linked edits.
    Neither the background nor the model is transformed or fitted here.
    """
    axis = _finite_vector(grid, "grid")
    return _evaluate(model, X_bg, [feature], [axis], update_fn)


def pdp_2d(model: ProbabilityModel, X_bg: pd.DataFrame, f1: str, f2: str,
           grid1: Sequence[float], grid2: Sequence[float]) -> pd.DataFrame:
    """Return unaveraged virtual rows in grid1, grid2, background order."""
    if f1 == f2:
        raise ValueError("Two-dimensional PDP requires distinct features")
    grids = [_finite_vector(grid1, "grid1"), _finite_vector(grid2, "grid2")]
    return _evaluate(model, X_bg, [f1, f2], grids, None)


def extract_window(pd_curve: pd.DataFrame, rule: Mapping[str, object]) -> dict:
    """Extract contiguous high grid runs without interpolation or smoothing.

    Missing probabilities split segments. The lower/upper pair is the envelope;
    segments retains gaps. Filtering by row/source support is done upstream.
    """
    try:
        amplitude_flat = float(rule["amplitude_flat"])
        level = float(rule["level"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Window rule requires numeric amplitude_flat and level") from error
    if (not np.isfinite(amplitude_flat) or not 0 <= amplitude_flat <= 1
            or not np.isfinite(level) or not 0 <= level <= 1):
        raise ValueError("Window rule values must be finite within [0, 1]")
    if (not isinstance(pd_curve, pd.DataFrame) or not pd_curve.columns.is_unique
            or not {"grid_value", "p_high"}.issubset(pd_curve.columns)):
        raise ValueError("Curve requires unique grid_value and p_high columns")
    grid = pd_curve.grid_value.to_numpy(dtype=float)
    probability = pd_curve.p_high.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError("Curve grid must be sorted, unique, and finite")
    valid = ~np.isnan(probability)
    if (not np.isfinite(probability[valid]).all()
            or ((probability[valid] < 0) | (probability[valid] > 1)).any()):
        raise ValueError("Curve probabilities must be in [0, 1] or missing")
    result = {"lower": np.nan, "upper": np.nan, "peak_x": np.nan,
              "amplitude": np.nan, "n_segments": 0, "status": "insufficient",
              "segments": []}
    if not valid.any():
        return result
    minimum = float(probability[valid].min())
    maximum = float(probability[valid].max())
    amplitude = maximum - minimum
    result.update(peak_x=float(grid[np.nanargmax(probability)]), amplitude=amplitude)
    if amplitude < amplitude_flat:
        result["status"] = "flat"
        return result
    high = valid & (probability >= minimum + level * amplitude)
    transitions = np.diff(np.r_[False, high, False].astype(int))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1) - 1
    segments = [[float(grid[start]), float(grid[end])] for start, end in zip(starts, ends)]
    result.update(lower=segments[0][0], upper=segments[-1][1], n_segments=len(segments),
                  status="split" if len(segments) > 1 else "single", segments=segments)
    return result
