from pathlib import Path
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def entry():
    return runpy.run_path(str(ROOT / 'scripts/run_m3.py'))


def test_panel_plan_covers_paper_and_own_features():
    module = entry()
    own = {model: ['Li_comp', 'Zr_comp', 'Zr_site_sto', 'Li_site_sto', 'Sintering Time']
           for model in ('DT', 'RF', 'LGBM', 'CB')}
    panels = module['panel_plan'](own)
    assert panels.panel_id.is_unique
    assert panels.groupby('figure').size().to_dict() == {'fig7': 20, 'fig9': 16, 'fig12': 10, 'own_top5': 20}
    assert (panels.query('figure == "fig9"').feature == 'Li_comp').all()
    assert set(panels.query('figure == "fig12"').cohort) == {'ga'}


def test_agreement_uses_disconnected_sets_not_envelope():
    compare = entry()['compare_target']
    curve = pd.DataFrame({'grid_value': [0., 1., 2., 3., 4.], 'p_high': [1., 1., 0., 1., 1.]})
    window = {'status': 'split', 'segments': [[0., 1.], [3., 4.]], 'peak_x': 0.}
    assert compare(curve, window, {'kind': 'range', 'value': [0., 4.]})[0] == 'partial'
    assert compare(curve, window, {'kind': 'range', 'value': [2., 2.]})[0] == 'disagree'
    assert compare(curve, window, {'kind': 'peak', 'value': 3.})[0] == 'partial'
    assert compare(curve, window, {'kind': 'peak', 'value': -1.})[0] == 'not_comparable'
    assert compare(curve, window, {'kind': 'peak', 'value': 0.})[0] == 'agree'
    assert compare(curve, {'status': 'flat', 'segments': [], 'peak_x': 0.}, {'kind': 'range', 'value': [0., 4.]})[0] == 'not_comparable'


def test_virtual_rows_reconstruct_actual_predictions_and_sources(tmp_path):
    module = entry()
    from sklearn.tree import DecisionTreeClassifier
    X = pd.DataFrame({'Li_comp': [1., 2., 3., 4.], 'Zr_comp': [0., 1., 0., 1.]}, index=[10, 20, 30, 40])
    model = DecisionTreeClassifier(random_state=0).fit(X, [0, 0, 1, 1])
    panel = {'panel_id': 'test', 'figure': 'fig7', 'cohort': 'full', 'model': 'DT', 'feature': 'Li_comp', 'feature_2': ''}
    sources = pd.Series([2, 2, 3, 3], index=X.index)
    long = module['make_virtual'](panel, model, X, sources)
    assert len(long) == len(X) * X.Li_comp.nunique()
    reconstructed = X.loc[long.bg_record_id].copy()
    reconstructed['Li_comp'] = long.grid_value.to_numpy()
    np.testing.assert_allclose(model.predict_proba(reconstructed)[:, 1], long.p_high)
    np.testing.assert_array_equal(long.bg_source_id, sources.loc[long.bg_record_id])
    assert long[['r_charge', 'r_site_Zr', 'r_site_La', 'chem_label', 'support_label']].isna().all().all()
    assert set(long.scheme) == {'A'}


def test_prepare_preserves_existing_generation_fingerprint(tmp_path, monkeypatch):
    module = entry()
    main = module['main']
    monkeypatch.setitem(main.__globals__, 'OUTPUT', tmp_path)
    monkeypatch.setitem(main.__globals__, 'load_inputs', lambda: ({}, {}))
    monkeypatch.setitem(main.__globals__, 'panel_plan', lambda own: pd.DataFrame())
    monkeypatch.setitem(main.__globals__, 'prepare', lambda inputs, panels: None)
    monkeypatch.setattr('sys.argv', ['run_m3.py', '--prepare'])
    manifest = tmp_path / 'input_manifest.json'
    manifest.write_text('{"previous_generation": "preserve"}\n')
    pd.DataFrame([{'key': 'estimate_minutes', 'value': 0}]).to_csv(tmp_path / 'budget.csv', index=False)
    before = manifest.read_bytes()
    main()
    assert manifest.read_bytes() == before
