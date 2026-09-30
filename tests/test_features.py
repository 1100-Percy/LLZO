import numpy as np
import pandas as pd
import pytest

from llzo_pdp.io import load_raw


def build(df=None, version="V_raw"):
    from llzo_pdp.features import build_features
    return build_features(load_raw() if df is None else df, version)


def test_paper_descriptor_examples_and_one_hot():
    raw = load_raw()
    result = build(raw)
    ga = raw.index[(raw.dopant_1 == "Ga") & (raw["# of Dopants"] == 1)]
    assert np.allclose(result.X.loc[ga, "Li_site_elnv"], 1.81)
    assert (result.X.loc[ga, "Zr_site_elnv"] == 0).all()
    assert result.X.at[30, "Zr_site_elnv"] == pytest.approx(1.20)
    assert "Li_dop_Al" in result.X
    assert "Li_dop_missing" in result.X
    assert "Measurement Temperature" in result.X
    assert "Ionic Conductivity(Total)" not in result.X
    assert "source_id" not in result.X


def test_corrections_are_cellwise_audited_and_do_not_mutate_input():
    df = load_raw()
    original = df.copy(deep=True)
    raw, corr = build(df), build(df, "V_corr")
    pd.testing.assert_frame_equal(df, original)
    assert raw.X.at[178, "Zr_comp"] == 0.4
    assert raw.X.at[178, "La_comp"] == 0
    assert corr.X.at[178, "Zr_comp"] == 1.6
    assert corr.X.at[178, "La_comp"] == 3
    assert corr.X.at[143, "La_comp"] == 2.875
    assert corr.X.at[29, "Li_site_sto"] == 0.4
    assert corr.X.at[11, "Li_dop_Al"] == 1
    assert corr.X.at[11, "Li_site_elnv"] == 1.61
    assert corr.site_qbar.at[11, "qbar_Li"] == 3
    assert raw.X.columns.tolist() == corr.X.columns.tolist()
    assert len(corr.audit_log) == 9
    assert set(corr.audit_log.issue_id) == {"K01", "K02", "K05", "K07", "K08"}
    assert raw.audit_log.empty
    for rid in [102, 107, 112, 117, 37, 71]:
        pd.testing.assert_series_equal(raw.X.loc[rid], corr.X.loc[rid])


def test_unassigned_dopants_are_not_inferred_from_element():
    for version in ("V_raw", "V_corr"):
        result = build(version=version)
        ids = [8, 188, 189, 190, 191, 192, 193]
        if version == "V_raw":
            ids.append(11)
        assert (result.X.loc[ids, ["Li_site_elnv", "Zr_site_elnv"]] == 0).all().all()
        assert (result.site_qbar.loc[ids, "n_unassigned"] > 0).all()


def test_weighted_descriptors_and_site_specific_lookup():
    df = load_raw().loc[[30]].copy()
    df.loc[30, ["dopant_1", "x_1", "dopant_2", "x_2", "Zr_dop", "Li_dop"]] = ["Ta", 0.1, "Nb", 0.3, "Ta/Nb", 0]
    result = build(df)
    assert result.X.at[30, "Zr_site_elnv"] == pytest.approx((0.1*1.5 + 0.3*1.6)/0.4)
    table = result.descriptor_table.query('site == "Zr"').iloc[0]
    assert table.x == pytest.approx(0.4)
    df.loc[30, ["dopant_1", "dopant_2", "Zr_dop", "La_dop"]] = ["Y", "Y", "Y", 0]
    assert build(df).X.at[30, "Zr_site_irad"] == pytest.approx(0.9)
    df.loc[30, ["Zr_dop", "La_dop"]] = [0, "Y"]
    table = build(df).descriptor_table.set_index("site")
    assert table.at["La", "irad"] == pytest.approx(1.019)


def test_unknown_version_or_property_is_an_error():
    with pytest.raises(ValueError, match="version"):
        build(version="other")
    df = load_raw().loc[[30]].copy()
    df.loc[30, ["dopant_1", "Zr_dop"]] = ["Unknown", "Unknown"]
    with pytest.raises(ValueError, match="propert"):
        build(df)
