import pickle

import numpy as np
import pandas as pd
import pytest


def sample():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a b", "c", "d"], index=range(10,90))
    return X, pd.Series(np.arange(len(X)) % 2, index=X.index)


def test_split_is_disjoint_complete_reproducible_and_matches_m1():
    from llzo_pdp.models import split_ids
    from llzo_pdp.io import load_raw, TARGET_COL
    from llzo_pdp.features import build_features
    from llzo_pdp.preprocess import filter_rows
    from llzo_pdp.meta import ROOT
    raw = load_raw()
    kept, _ = filter_rows(build_features(raw, "V_raw").raw_features)
    y = (raw.loc[kept, TARGET_COL] >= 2.69e-4).astype(int)
    saved = pd.read_csv(ROOT / "outputs/M1/split_ids.csv")
    for seed in range(5):
        result = split_ids(y, seed)
        assert {k: len(v) for k,v in result.items()} == {"train":152,"val":33,"test":33}
        assert set().union(*map(set,result.values())) == set(kept)
        assert sum(map(len,result.values())) == len(kept)
        for kind, ids in result.items():
            expected = saved.query('data_version == "V_raw" and seed == @seed and split == @kind').record_id
            assert list(ids) == list(expected)
        assert y.loc[result["test"]].sum() in (16,17)
    unstratified = split_ids(y,0,stratified=False)
    assert len(unstratified["test"]) == 33


def test_metrics_use_probability_auc_and_handle_single_class():
    from llzo_pdp.models import classification_metrics
    m = classification_metrics([0,0,1,1], [0,1,0,1], [.1,.8,.4,.9])
    assert m["accuracy"] == .5
    assert m["f1_weighted"] == .5
    assert m["roc_auc"] == .75
    assert [m[k] for k in ("tn","fp","fn","tp")] == [1,1,1,1]
    one = classification_metrics([1,1], [1,0], [.9,.2])
    assert np.isnan(one["roc_auc"])
    assert one["auc_reason"] == "single_class_test"


@pytest.mark.parametrize("name,grid", [
    ("DT", {"max_depth":[2,3]}), ("RF", {"n_estimators":[3],"max_depth":[2]}),
    ("LGBM", {"n_estimators":[3],"min_child_samples":[2]}),
    ("CB", {"iterations":[3],"depth":[2]}), ("AB", {"n_estimators":[3]}),
])
def test_grid_and_bundle_roundtrip_exclude_evaluation_labels(name, grid):
    from llzo_pdp.models import fit_grid, ModelBundle
    from llzo_pdp.preprocess import impute_features
    X,y = sample()
    X.iloc[1,0] = np.nan
    imp = impute_features(X,list(X),"A0",0)
    train = X.index[:60]
    search = fit_grid(imp.X.loc[train], y.loc[train], name,0,grid)
    assert search.scoring == "f1_weighted"
    assert search.cv.n_splits == 5 and search.cv.shuffle
    assert search.best_estimator_.n_features_in_ == len(X.columns)
    bundle = ModelBundle(search.best_estimator_,imp,list(X),{"train_ids":list(train)})
    restored = pickle.loads(pickle.dumps(bundle))
    np.testing.assert_allclose(bundle.predict_proba(X),restored.predict_proba(X),atol=0,rtol=0)
    assert np.array_equal(restored.predict(X),bundle.predict(X))
    assert restored.feature_names == list(X)
    with pytest.raises(ValueError, match="features"):
        restored.predict(X.drop(columns="c"))


def test_fit_rejects_missing_values_or_misaligned_labels():
    from llzo_pdp.models import fit_grid
    X,y=sample()
    with pytest.raises(ValueError,match="aligned"):
        fit_grid(X,y.iloc[::-1],"DT",0,{"max_depth":[2]})
    X.iloc[0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        fit_grid(X,y,"DT",0,{"max_depth":[2]})
