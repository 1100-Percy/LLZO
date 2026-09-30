import numpy as np
import pandas as pd
import pytest


def test_filter_uses_raw_missing_cells_before_category_encoding():
    from llzo_pdp.features import build_features
    from llzo_pdp.io import load_raw
    from llzo_pdp.preprocess import filter_rows
    result = build_features(load_raw(), "V_raw")
    kept, log = filter_rows(result.raw_features)
    assert len(kept) == 218
    assert log.loc[~log.kept, "record_id"].tolist() == [8, 38, 170, 188, 189, 190, 191, 192, 193]
    sample = pd.DataFrame([[1, np.nan, np.nan, np.nan, 2],
                           [np.nan, np.nan, np.nan, np.nan, 2]], index=[1, 2])
    assert filter_rows(sample)[0].tolist() == [1]


def test_a1_is_train_only_and_preserves_one_hot_and_observed_values():
    from llzo_pdp.preprocess import impute_features
    X = pd.DataFrame({"a": [1., 2., 3., 4., np.nan, 100.],
                      "b": [2., 4., 6., 8., 10., np.nan],
                      "category_A": [1, 0, 1, 0, 1, 0]}, index=range(10,16))
    training = [10, 11, 12, 13, 14]
    first = impute_features(X, ["a", "b"], "A1", 0, training)
    changed = X.copy()
    changed.loc[15, "a"] = 1e8
    second = impute_features(changed, ["a", "b"], "A1", 0, training)
    pd.testing.assert_frame_equal(first.X.loc[training], second.X.loc[training])
    assert first.fit_ids == training
    pd.testing.assert_series_equal(first.X.category_A, X.category_A)
    assert not first.X.isna().any().any()
    assert np.array_equal(first.X.a[X.a.notna()], X.a.dropna())
    a0 = impute_features(X, ["a", "b"], "A0", 0)
    assert a0.fit_ids == list(X.index)
    assert a0.convergence_warnings
    pd.testing.assert_frame_equal(a0.X, impute_features(X, ["a", "b"], "A0", 0).X)
    with pytest.raises(ValueError, match="train"):
        impute_features(X, ["a", "b"], "A1", 0)
    with pytest.raises(ValueError, match="training"):
        impute_features(X, ["a", "b"], "A1", 0, [999])
    with pytest.raises(ValueError, match="mode"):
        impute_features(X, ["a", "b"], "A2", 0)


def test_threshold_ratio_reports_nearest_attainable_fraction_with_ties():
    from llzo_pdp.preprocess import threshold_table
    values = pd.Series([1., 1., 2., 3., 4.])
    table = threshold_table(values, main=2., benchmark=3.).set_index("threshold_name")
    assert table.loc["T_main", "n_high"] == 3
    assert table.loc["T_median_kept", "threshold"] == 2
    assert table.loc["T_ratio", "threshold"] == 2
    assert table.loc["T_ratio", "high_fraction"] == 0.6
    assert not table.loc["T_ratio", "ratio_exact"]
