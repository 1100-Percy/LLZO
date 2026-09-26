"""Unit tests for linked composition updates (must stay green)."""
import numpy as np
import pandas as pd

from llzo_pdp.io import load_raw
from llzo_pdp import chemistry as ch


def _feat(df):
    return df.rename(columns={"Li Composition": "Li_comp"})[
        ["Li_comp", "La_comp", "Zr_comp", "Li_site_sto", "La_site_sto", "Zr_site_sto"]]


def test_linked_update_preserves_charge_on_all_consistent_records():
    df = load_raw()
    X, st = _feat(df), ch.site_charge_table(df)
    r0 = ch.charge_residual_features(X, st)
    ok = r0.abs() < ch.TOL
    n_checked = 0
    for rid in X.index[ok]:
        for site in ("Li", "La", "Zr"):
            q = st.at[rid, f"qbar_{site}"]
            if np.isnan(q):
                continue
            x0 = X.at[rid, f"{site}_site_sto"]
            z = ch.linked_update(X.loc[rid], site, x0 + 0.1, q)
            r1 = ch.charge_residual_features(z.to_frame().T, st.loc[[rid]])
            assert abs(float(r1.iloc[0]) - float(r0[rid])) < 1e-9
            n_checked += 1
    assert n_checked > 150


def test_ga_path_reproduces_li_equals_7_minus_3g():
    row = pd.Series({"Li_comp": 7.0, "La_comp": 3.0, "Zr_comp": 2.0,
                     "Li_site_sto": 0.0, "La_site_sto": 0.0, "Zr_site_sto": 0.0})
    z = ch.linked_update(row, "Li", 0.25, 3.0)
    assert np.isclose(z.Li_comp, 7 - 3 * 0.25)


def test_ta_path_reproduces_li_and_zr():
    row = pd.Series({"Li_comp": 7.0, "La_comp": 3.0, "Zr_comp": 2.0,
                     "Li_site_sto": 0.0, "La_site_sto": 0.0, "Zr_site_sto": 0.0})
    z = ch.linked_update(row, "Zr", 0.6, 5.0)
    assert np.isclose(z.Li_comp, 6.4) and np.isclose(z.Zr_comp, 1.4)
