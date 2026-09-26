"""Data facts verified independently before the project started.

RULE FOR AGENTS: never edit an expected value in this file to make a test pass.
If a test fails, the loader or your code changed behaviour -> investigate, and if
you believe the fact itself is wrong, write the evidence to DECISIONS.md and stop
that milestone (status BLOCKED) instead of editing the test.
"""
import numpy as np
import pandas as pd
import pytest

from llzo_pdp.io import load_raw, TARGET_COL
from llzo_pdp import chemistry as ch


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_size_and_sources(df):
    assert len(df) == 227
    assert df.source_id.nunique() == 53
    assert sorted(set(range(1, 56)) - set(df.source_id)) == [23, 52]
    assert df.groupby("source_id").size().max() == 20


def test_missing_composition(df):
    assert df.record_id[df.Composition.isna()].tolist() == [8, 38, 170, 188, 189, 190, 191, 192, 193]


def test_charge_balance(df):
    r = ch.charge_residual_raw(df)
    assert r.notna().sum() == 218
    assert (r.abs() < ch.TOL).sum() == 204
    bad = sorted(df.record_id[r.abs() >= ch.TOL].tolist())
    assert bad == [21, 42, 43, 44, 45, 102, 107, 112, 117, 143, 178, 179, 180, 181]


def test_site_sto_vs_dopant_columns(df):
    """Site totals used as model features disagree with the dopant columns here."""
    X = df.rename(columns={"Li Composition": "Li_comp"})
    rf = ch.charge_residual_features(X, ch.site_charge_table(df))
    rr = ch.charge_residual_raw(df)
    diff = (rf - rr).abs()
    assert sorted(df.record_id[diff > 0.011].tolist()) == [11, 29, 37, 71]


def _ga_single(df):
    return df[(df.dopant_1 == "Ga") & (df["# of Dopants"] == 1)]


def test_ga_single_is_on_stoichiometric_line(df):
    ga = _ga_single(df)
    assert len(ga) == 55 and ga.source_id.nunique() == 12
    assert np.allclose(ga["Li Composition"], 7 - 3 * ga.x_1, atol=1e-6)
    # g=0.20 dominates; g>=0.35 only from two sources, both sintered at 1100 C
    assert (ga.x_1.round(2) == 0.20).sum() == 28
    hi = ga[ga.x_1 >= 0.35]
    assert hi.source_id.nunique() == 2 and set(hi["Sintering Temperature"]) == {1100}
    assert (ga.source_id == 54).sum() == 18


def test_ta_single_is_on_stoichiometric_line(df):
    ta = df[(df.dopant_1 == "Ta") & (df["# of Dopants"] == 1)]
    assert len(ta) == 37 and ta.source_id.nunique() == 11
    ok = ta.dropna(subset=["Li Composition"])
    assert np.allclose(ok["Li Composition"], 7 - ok.x_1, atol=1e-6)
    zr_bad = ok.record_id[(ok.Zr_comp - (2 - ok.x_1)).abs() > 1e-6].tolist()
    assert zr_bad == [178]


def test_ga_containing_count(df):
    has_ga = (df[["dopant_1", "dopant_2", "dopant_3"]] == "Ga").any(axis=1)
    assert has_ga.sum() == 86


def test_target_distribution(df):
    s = df[TARGET_COL]
    assert s.notna().all()
    assert abs(s.median() - 2.63e-4) / 2.63e-4 < 0.01
    assert (s >= 2.69e-4).sum() == 111


def test_li_above_7(df):
    ids = sorted(df.record_id[df["Li Composition"] > 7].tolist())
    assert ids == [21, 30, 33, 44, 169, 205]
    # the only record near Li=7.2 is Gd0.2 with sigma below the paper's median threshold
    near = df[(df["Li Composition"] - 7.2).abs() < 0.1]
    assert near.record_id.tolist() == [30] and near[TARGET_COL].iloc[0] < 2.69e-4


def test_low_zr_comp_supported_by_entry_errors(df):
    low = df[df.Zr_comp < 1.0]
    assert sorted(low.record_id.tolist()) == [19, 178, 179, 180, 181]


def test_feature_identical_groups(df):
    excl = {"source_no", "doi", "Author", "source_id", "record_id", TARGET_COL,
            "Bulk ionic conductivity", "GB ionic conductivity", "Activation Energy eV",
            "Electronic conductivity", "Battery Capacity(mAh/g)", "Number of cycles",
            "%retention", "Relative Density", "Crystal Structure", "Lattice Parameter"}
    cols = [c for c in df.columns if c not in excl]
    key = pd.Series(["|".join(map(str, row)) for row in df[cols].itertuples(index=False)], index=df.index)
    dup = key[key.duplicated(keep=False)]
    assert len(dup) == 21 and dup.nunique() == 8
    assert set(df.loc[dup.index, "source_id"]) == {48, 54}
