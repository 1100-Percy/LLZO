import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.tree import DecisionTreeClassifier


RULE = {"amplitude_flat": 0.05, "level": 0.8,
        "min_points_per_gridpoint": 5, "min_sources_per_gridpoint": 2}


def sample(n=151):
    rng = np.random.RandomState(11)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["Li_comp", "Zr_comp", "a b"],
                     index=pd.Index(np.arange(n) * 2 + 10, name="record_id"))
    y = ((X.Li_comp + 0.3 * X.Zr_comp) > 0).astype(int)
    return X, y


def fitted(name, X, y):
    model = (DecisionTreeClassifier(max_depth=4, random_state=0) if name == "DT"
             else RandomForestClassifier(n_estimators=7, max_depth=4, random_state=0,
                                         n_jobs=1))
    return model.fit(X, y)


@pytest.mark.parametrize("n", [99, 100, 151])
def test_g1_matches_sklearn_default_grid(n):
    from llzo_pdp.pdp import make_grid
    X, y = sample(n)
    reference = partial_dependence(fitted("DT", X, y), X, [0], method="brute",
                                   response_method="predict_proba")
    np.testing.assert_array_equal(make_grid(X.Li_comp), reference.grid_values[0])


def test_g1_discrete_and_g2_observed_range():
    from llzo_pdp.pdp import make_grid
    np.testing.assert_array_equal(make_grid([3, 1, 3, 0, 1]), [0, 1, 3])
    np.testing.assert_array_equal(make_grid([3, 1, 0], "G2"), np.linspace(0, 3, 60))
    np.testing.assert_array_equal(make_grid([2, 2], "G1"), [2])
    np.testing.assert_array_equal(make_grid([2, 2], "G2"), [2])


@pytest.mark.parametrize("values,kind", [([], "G1"), ([1, np.nan], "G1"),
    ([1, np.inf], "G2"), ([[1, 2]], "G1"), ([1, 2], "unknown")])
def test_grid_rejects_invalid_values(values, kind):
    from llzo_pdp.pdp import make_grid
    with pytest.raises(ValueError):
        make_grid(values, kind)


def test_g1_rejects_indistinguishable_percentile_endpoints():
    from llzo_pdp.pdp import make_grid
    with pytest.raises(ValueError, match="percentiles"):
        make_grid(np.linspace(1, 1 + 1e-8, 101))


@pytest.mark.parametrize("name", ["DT", "RF"])
def test_ice_means_match_sklearn_brute_and_preserve_inputs(name):
    from llzo_pdp.pdp import ice, make_grid
    X, y = sample()
    original = X.copy(deep=True)
    model = fitted(name, X, y)
    reference = partial_dependence(model, X, [0], method="brute",
                                   response_method="predict_proba")
    grid = make_grid(X.Li_comp)
    result = ice(model, X, "Li_comp", grid)
    assert list(result) == ["bg_record_id", "grid_value", "p_high", "Li_comp"]
    np.testing.assert_array_equal(result.bg_record_id, np.tile(X.index, len(grid)))
    np.testing.assert_array_equal(result.grid_value, np.repeat(grid, len(X)))
    np.testing.assert_array_equal(result.Li_comp, result.grid_value)
    actual = result.groupby("grid_value", sort=True).p_high.mean().to_numpy()
    np.testing.assert_allclose(actual, reference.average[0], atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(X, original)


def test_update_fn_preserves_coupled_columns_and_does_not_mutate_background():
    from llzo_pdp.pdp import ice
    X, y = sample(12)
    original = X.copy(deep=True)
    model = fitted("DT", X, y)

    def linked(row, value):
        row["Li_comp"] = value
        row["Zr_comp"] = 2 * value - row["a b"]
        return row

    result = ice(model, X, "Li_comp", [-1.0, 2.0], update_fn=linked)
    assert list(result) == ["bg_record_id", "grid_value", "p_high"] + list(X)
    np.testing.assert_array_equal(result.Li_comp, result.grid_value)
    np.testing.assert_allclose(result.Zr_comp, 2 * result.grid_value - result["a b"])
    np.testing.assert_array_equal(result["a b"], np.tile(X["a b"], 2))
    np.testing.assert_array_equal(result.p_high, model.predict_proba(result[list(X)])[:, 1])
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("name", ["DT", "RF"])
def test_2d_means_match_sklearn_brute_and_keep_both_virtual_inputs(name):
    from llzo_pdp.pdp import pdp_2d
    X, y = sample(12)
    original = X.copy(deep=True)
    model = fitted(name, X, y)
    reference = partial_dependence(model, X, [(0, 1)], method="brute",
                                   response_method="predict_proba")
    g1, g2 = reference.grid_values
    result = pdp_2d(model, X, "Li_comp", "Zr_comp", g1, g2)
    assert list(result) == ["bg_record_id", "grid_value", "grid_value_2", "p_high",
                            "Li_comp", "Zr_comp"]
    np.testing.assert_array_equal(result.bg_record_id, np.tile(X.index, len(g1) * len(g2)))
    np.testing.assert_array_equal(result.Li_comp, np.repeat(g1, len(g2) * len(X)))
    np.testing.assert_array_equal(result.Zr_comp, np.tile(np.repeat(g2, len(X)), len(g1)))
    actual = result.groupby(["grid_value", "grid_value_2"]).p_high.mean().to_numpy()
    np.testing.assert_allclose(actual.reshape(len(g1), len(g2)), reference.average[0],
                               atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(X, original)


def test_predictions_are_batched_and_batch_boundaries_preserve_order(monkeypatch):
    import llzo_pdp.pdp as pdp
    X, y = sample(12)
    underlying = fitted("DT", X, y)

    class RecordingPredictor:
        classes_ = np.array([0, 1])

        def __init__(self):
            self.batch_sizes = []

        def predict_proba(self, frame):
            self.batch_sizes.append(len(frame))
            return underlying.predict_proba(frame)

    predictor = RecordingPredictor()
    monkeypatch.setattr(pdp, "_MAX_BATCH_ROWS", 25)
    result = pdp.ice(predictor, X, "Li_comp", [-2, 0, 2])
    assert predictor.batch_sizes == [25, 11]
    reconstructed = X.loc[result.bg_record_id].reset_index(drop=True)
    reconstructed["Li_comp"] = result.grid_value
    np.testing.assert_array_equal(result.p_high, underlying.predict_proba(reconstructed)[:, 1])
    predictor.batch_sizes.clear()
    two = pdp.pdp_2d(predictor, X, "Li_comp", "Zr_comp", [-1, 1], [-2, 2])
    assert predictor.batch_sizes == [25, 23]
    reconstructed = X.loc[two.bg_record_id].reset_index(drop=True)
    reconstructed["Li_comp"] = two.grid_value
    reconstructed["Zr_comp"] = two.grid_value_2
    np.testing.assert_array_equal(two.p_high, underlying.predict_proba(reconstructed)[:, 1])


@pytest.mark.parametrize("bad_update", [
    lambda row, value: row.drop("a b"),
    lambda row, value: row.iloc[::-1],
    lambda row, value: row.to_numpy(),
    lambda row, value: row.mask(row.index == "Li_comp", np.nan),
])
def test_update_fn_rejects_invalid_return(bad_update):
    from llzo_pdp.pdp import ice
    X, y = sample(12)
    with pytest.raises(ValueError, match="update_fn"):
        ice(fitted("DT", X, y), X, "Li_comp", [0], update_fn=bad_update)


@pytest.mark.parametrize("case", ["duplicate_ids", "missing_ids", "duplicate_columns",
                                  "missing", "empty", "reserved"])
def test_ice_rejects_invalid_background(case):
    from llzo_pdp.pdp import ice
    X, y = sample(12)
    model = fitted("DT", X, y)
    if case == "duplicate_ids":
        X.index = [1] * len(X)
    elif case == "missing_ids":
        X.index = [np.nan] + list(X.index[1:])
    elif case == "duplicate_columns":
        X.columns = ["Li_comp", "Li_comp", "a b"]
    elif case == "missing":
        X.iloc[0, 0] = np.nan
    elif case == "empty":
        X = X.iloc[:0]
    else:
        X = X.rename(columns={"a b": "p_high"})
    with pytest.raises(ValueError):
        ice(model, X, "Li_comp", [0])


def test_engines_reject_unknown_features_and_invalid_grids():
    from llzo_pdp.pdp import ice, pdp_2d
    X, y = sample(12)
    model = fitted("DT", X, y)
    with pytest.raises(ValueError, match="feature"):
        ice(model, X, "unknown", [0])
    for bad in [[], [np.nan], [[0]]]:
        with pytest.raises(ValueError, match="grid"):
            ice(model, X, "Li_comp", bad)
    with pytest.raises(ValueError, match="distinct"):
        pdp_2d(model, X, "Li_comp", "Li_comp", [0], [0])


@pytest.mark.parametrize("classes", [[1, 0], [0], [-1, 1], ["0", "1"]])
def test_ice_rejects_class_order_not_binary_zero_one(classes):
    from llzo_pdp.pdp import ice
    X, y = sample(12)
    model = fitted("DT", X, y)
    model.classes_ = np.asarray(classes)
    with pytest.raises(ValueError, match="classes"):
        ice(model, X, "Li_comp", [0])


@pytest.mark.parametrize("result", [np.array([[np.nan, 0.5]]), np.array([[1.1, -0.1]]),
                                     np.array([[0.5, 0.7]]), np.array([[0.5]])])
def test_ice_rejects_invalid_prediction_probabilities(result):
    from llzo_pdp.pdp import ice
    X, _ = sample(12)

    class InvalidPredictor:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.repeat(result, len(frame), axis=0)

    with pytest.raises(ValueError, match="probabilit"):
        ice(InvalidPredictor(), X, "Li_comp", [0])


@pytest.mark.parametrize("heights,expected", [
    ([0.2, 0.21, 0.22], ("flat", [], 2, 0.02)),
    ([0, 0.8, 1, 0.8, 0], ("single", [[1, 3]], 2, 1)),
    ([1, 0, 1], ("split", [[0, 0], [2, 2]], 0, 1)),
    ([0, 1, np.nan, 1, 0], ("split", [[1, 1], [3, 3]], 1, 1)),
    ([np.nan, 0, 1, np.nan], ("single", [[2, 2]], 2, 1)),
    ([0, 0.05], ("single", [[1, 1]], 1, 0.05)),
    ([0, 0.049], ("flat", [], 1, 0.049)),
    ([0.7], ("flat", [], 0, 0)),
])
def test_window_frozen_rule_and_disconnected_segments(heights, expected):
    from llzo_pdp.pdp import extract_window
    curve = pd.DataFrame({"grid_value": np.arange(len(heights)), "p_high": heights})
    original = curve.copy(deep=True)
    result = extract_window(curve, RULE)
    status, segments, peak, amplitude = expected
    assert result["status"] == status
    assert result["segments"] == segments
    assert result["n_segments"] == len(segments)
    assert result["peak_x"] == peak
    assert result["amplitude"] == pytest.approx(amplitude)
    if segments:
        assert result["lower"] == segments[0][0]
        assert result["upper"] == segments[-1][1]
    else:
        assert np.isnan(result["lower"]) and np.isnan(result["upper"])
    pd.testing.assert_frame_equal(curve, original)


@pytest.mark.parametrize("heights", [[], [np.nan], [np.nan, np.nan]])
def test_window_all_missing_is_insufficient(heights):
    from llzo_pdp.pdp import extract_window
    result = extract_window(pd.DataFrame({"grid_value": np.arange(len(heights)),
                                          "p_high": heights}), RULE)
    assert result["status"] == "insufficient"
    assert result["segments"] == [] and result["n_segments"] == 0
    assert all(np.isnan(result[k]) for k in ["lower", "upper", "peak_x", "amplitude"])


@pytest.mark.parametrize("grid,heights", [([1, 0], [0, 1]), ([0, 0], [0, 1]),
    ([0, np.nan], [0, 1]), ([0, 1], [0, np.inf]), ([0, 1], [0, 1.1])])
def test_window_rejects_invalid_curves(grid, heights):
    from llzo_pdp.pdp import extract_window
    with pytest.raises(ValueError):
        extract_window(pd.DataFrame({"grid_value": grid, "p_high": heights}), RULE)


@pytest.mark.parametrize("rule", [{}, {"amplitude_flat": -0.1, "level": 0.8},
    {"amplitude_flat": 0.05, "level": 1.1}, {"amplitude_flat": np.nan, "level": 0.8}])
def test_window_rejects_invalid_rules(rule):
    from llzo_pdp.pdp import extract_window
    with pytest.raises(ValueError, match="rule"):
        extract_window(pd.DataFrame({"grid_value": [0, 1], "p_high": [0, 1]}), rule)
