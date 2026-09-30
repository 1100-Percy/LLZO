"""Generate M3 brute PDP panels and CSV-driven window comparisons, offline."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from llzo_pdp.features import build_features
from llzo_pdp.io import load_raw
from llzo_pdp.meta import write_meta
from llzo_pdp.models import safe_columns
from llzo_pdp.pdp import ice, pdp_2d, make_grid, extract_window

OUTPUT = ROOT / 'outputs/M3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, pd.DataFrame):
        if path.suffix == '.parquet':
            value.to_parquet(path, index=False)
        else:
            value.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    metadata(path)


def metadata(path):
    write_meta(path, seed=0, data_version='V_raw', script='scripts/run_m3.py',
               imputation='A0', threshold='T_main', background='B_train', grid='G1')


class PreparedPredictor:
    """Use the saved estimator on fully imputed virtual vectors."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.classes_ = bundle.classes_

    def predict_proba(self, X):
        if list(X) != self.bundle.feature_names:
            raise ValueError('Virtual feature order differs from fitted model')
        return self.bundle.estimator.predict_proba(safe_columns(X))


def load_inputs():
    manifest = json.loads((ROOT / 'outputs/M2/input_manifest.json').read_text())
    for name, expected in manifest['inputs'].items():
        actual = version(name.split(':', 1)[1]) if name.startswith('package:') else digest(ROOT / name)
        if actual != expected:
            raise ValueError(f'M2 input changed: {name}')
    raw = load_raw()
    original = build_features(raw, 'V_raw').X
    inputs, own = {}, {}
    for cohort, models in (('full', ['DT', 'RF', 'LGBM', 'CB']), ('ga', ['CB', 'AB'])):
        for model in models:
            path = ROOT / f'outputs/M2/models/{cohort}/{model}_V_raw_A0_T_main_seed0.pkl'
            meta = json.loads(path.with_name(path.name + '.meta.json').read_text())
            if digest(path) != meta['sha256']:
                raise ValueError(f'Model hash changed: {path}')
            bundle = pickle.loads(path.read_bytes())
            job = ROOT / f'outputs/M2/jobs/{cohort}_{model}_V_raw_A0_T_main_seed0'
            marker = json.loads((job / 'complete.json').read_text())
            for name, expected in marker['hashes'].items():
                if digest(ROOT / 'outputs/M2' / name) != expected:
                    raise ValueError(f'M2 output hash changed: {name}')
            splits = pd.read_csv(job / 'split_ids.csv')
            ids = splits.query('split == "train"').record_id.tolist()
            if ids != bundle.context['train_ids']:
                raise ValueError('Saved train IDs disagree')
            X = bundle.imputation.X.loc[ids, bundle.feature_names].copy()
            predictor = PreparedPredictor(bundle)
            np.testing.assert_allclose(predictor.predict_proba(X), bundle.predict_proba(original.loc[ids]), atol=1e-12, rtol=0)
            inputs[(cohort, model)] = (predictor, X, raw.loc[ids, 'source_id'], original.loc[ids, bundle.feature_names])
            if cohort == 'full':
                own[model] = pd.read_csv(job / 'importance.csv').sort_values('rank').head(5).feature.tolist()
    return inputs, own


def panel_plan(own):
    paper = yaml.safe_load((ROOT / 'paper_targets.yaml').read_text())
    rows = []
    for i, (model, features) in enumerate(paper['top5_features_fig7'].items()):
        for j, feature in enumerate(features):
            rows.append(dict(panel_id=f'fig7_{chr(97 + 5*i + j)}', figure='fig7', cohort='full', model=model, feature=feature, feature_2=''))
    for letter, spec in paper['two_way_fig9']['panels'].items():
        rows.append(dict(panel_id=f'fig9_{letter}', figure='fig9', cohort='full', model=spec['model'], feature=paper['two_way_fig9']['x_axis'], feature_2=spec['y']))
    for i, (model, features) in enumerate(paper['ga_subset_fig12']['panels'].items()):
        for j, feature in enumerate(features):
            rows.append(dict(panel_id=f'fig12_{chr(97 + 5*i + j)}', figure='fig12', cohort='ga', model=model, feature=feature, feature_2=''))
    for model, features in own.items():
        for rank, feature in enumerate(features, 1):
            rows.append(dict(panel_id=f'own_{model}_{rank}', figure='own_top5', cohort='full', model=model, feature=feature, feature_2=''))
    return pd.DataFrame(rows)


def make_virtual(panel, predictor, X, sources):
    f1, f2 = panel['feature'], panel['feature_2']
    grid = make_grid(X[f1], 'G1')
    if f2:
        long = pdp_2d(predictor, X, f1, f2, grid, make_grid(X[f2], 'G1'))
    else:
        long = ice(predictor, X, f1, grid)
    for key in ('panel_id', 'model', 'feature', 'feature_2'):
        long[key] = panel[key]
    long['bg_source_id'] = long.bg_record_id.map(sources)
    for key, value in dict(data_version='V_raw', scheme='A', background='B_train', grid='G1', seed=0, imputation='A0', threshold='T_main').items():
        long[key] = value
    for column in ('r_charge', 'r_site_Zr', 'r_site_La', 'chem_label', 'support_label'):
        long[column] = np.nan
    return long


def aggregate_curve(long, two_dimensional):
    keys = ['grid_value'] + (['grid_value_2'] if two_dimensional else [])
    return long.groupby(keys, sort=True).agg(p_high=('p_high', 'mean'), n=('p_high', 'size'), n_sources=('bg_source_id', 'nunique')).reset_index()


def compare_target(curve, window, target):
    """D-020: set overlap on the fixed grid, with no fitted tolerances."""
    kind = target['kind']
    if kind == 'qualitative':
        return 'not_comparable', '无预先声明的可操作数值判据'
    if window['status'] in ('flat', 'insufficient'):
        return 'not_comparable', '平坦或数据不足，不存在可比较窗口'
    grid = curve.grid_value.to_numpy()
    selected = np.zeros(len(grid), dtype=bool)
    for lower, upper in window['segments']:
        selected |= (grid >= lower) & (grid <= upper)
    if kind == 'peak':
        value = target['value']
        if not grid.min() <= value <= grid.max():
            return 'not_comparable', '论文峰位在本研究网格范围外'
        closest = int(np.argmin(abs(grid - value)))
        if window['peak_x'] == grid[closest]:
            return 'agree', '论文峰位的最近网格点与首个峰位一致'
        return ('partial', '论文峰位的最近网格点仅落在窗口内') if selected[closest] else ('disagree', '论文峰位的最近网格点不在窗口内')
    if kind == 'range':
        expected = (grid >= target['value'][0]) & (grid <= target['value'][1])
    elif kind == 'greater':
        expected = grid > target['value']
    else:
        raise ValueError(f'Unknown comparison kind: {kind}')
    if not expected.any():
        return 'not_comparable', '论文目标没有覆盖当前网格点'
    if np.array_equal(expected, selected):
        return 'agree', '目标与窗口的网格点集合相同'
    return ('partial', '目标与窗口有交集但集合不同') if (expected & selected).any() else ('disagree', '目标与窗口的网格点集合无交集')


def targets_for(panel, paper):
    feature, model, figure = panel['feature'], panel['model'], panel['figure']
    if figure == 'fig9':
        statements = paper['two_way_fig9']['claims']
        targets = [dict(kind='qualitative', value=statements[0])]
        if panel['panel_id'] == 'fig9_c':
            targets.append(dict(kind='qualitative', value=statements[1]))
        if panel['panel_id'] in ('fig9_c', 'fig9_d'):
            targets.append(dict(kind='qualitative', value=statements[2]))
        return targets
    if figure == 'fig12':
        spec = paper['ga_subset_fig12']
        targets = []
        shape = spec['shapes'].get(f'{model}_{feature}')
        if shape:
            targets.append(dict(kind='qualitative', value=shape))
        if feature == 'Sintering Time':
            targets.append(dict(kind='peak', value=spec['text']['sintering_time_peak_h']))
        if feature == '1st Calcination Temp':
            targets.append(dict(kind='qualitative', value=f"calcination_T={spec['text']['calcination_T']}; 未明确窗口或峰位判据"))
        return targets or [dict(kind='qualitative', value='paper_targets 未给出该面板的数值窗口')]
    if figure == 'own_top5':
        return [dict(kind='qualitative', value='本模型自行选取的特征面板，不对应论文面板')]
    spec = paper['windows_text'].get(feature, {})
    targets = [dict(kind=kind, value=spec[kind]) for kind in ('range', 'peak') if kind in spec]
    if feature == 'Sintering Time' and model == 'DT':
        targets.append(dict(kind='range', value=spec['DT']))
    if f'{model}_peak' in spec:
        targets.append(dict(kind='peak', value=spec[f'{model}_peak']))
    if 'rule' in spec:
        rule = spec['rule'].split()
        if len(rule) == 2 and rule[0] == '>':
            targets.append(dict(kind='greater', value=float(rule[1])))
    return targets or [dict(kind='qualitative', value=json.dumps(spec, ensure_ascii=False))]


def plot_panel(panel, curve, raw, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    if panel['feature_2']:
        matrix = curve.pivot(index='grid_value_2', columns='grid_value', values='p_high')
        artist = ax.pcolormesh(matrix.columns.to_numpy(), matrix.index.to_numpy(), matrix.to_numpy(), shading='nearest', cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(artist, ax=ax, label='Mean P(high conductivity)')
        ax.set_ylabel(panel['feature_2'])
        observed_y = raw[panel['feature_2']].dropna()
        ax.plot(np.full(len(observed_y), .015), observed_y, '_', color='black', alpha=.35, transform=ax.get_yaxis_transform(), markersize=5)
    else:
        ax.plot(curve.grid_value, curve.p_high, color='#176b87', linewidth=1.8)
        ax.set_ylim(-.03, 1.03)
        ax.set_ylabel('Mean P(high conductivity)')
    observed = raw[panel['feature']].dropna()
    ax.plot(observed, np.full(len(observed), .015), '|', color='black', alpha=.35, transform=ax.get_xaxis_transform(), markersize=5)
    ax.set_xlabel(panel['feature'])
    ax.set_title(f"{panel['panel_id']} / {panel['model']} / {panel['cohort']}\nV_raw / A0 / T_main / seed 0 / B_train / G1", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    metadata(path)


def prepare(inputs, panels):
    probes = []
    for (cohort, model), (predictor, X, _, _) in inputs.items():
        count = 20000
        batch = X.iloc[np.arange(count) % len(X)].copy()
        start = time.perf_counter()
        predictor.predict_proba(batch)
        elapsed = time.perf_counter() - start
        subset = panels.query('cohort == @cohort and model == @model')
        evaluations = sum(len(X) * len(make_grid(X[row.feature], 'G1')) * (len(make_grid(X[row.feature_2], 'G1')) if row.feature_2 else 1) for row in subset.itertuples())
        probes.append(dict(cohort=cohort, model=model, probe_rows=count, seconds=elapsed, virtual_rows=evaluations,
                           estimated_seconds=elapsed/count*evaluations + 2*len(subset)))
    save(pd.DataFrame(probes), OUTPUT / 'runtime_probe.csv')
    probe = pd.read_csv(OUTPUT / 'runtime_probe.csv')
    save(pd.DataFrame([dict(key='estimated_minutes', value=probe.estimated_seconds.sum()/60),
                       dict(key='conservative_minutes', value=probe.estimated_seconds.sum()*3/60),
                       dict(key='virtual_rows', value=probe.virtual_rows.sum())]), OUTPUT / 'budget.csv')


def run_tests():
    with tempfile.TemporaryDirectory() as temp:
        junit = Path(temp) / 'tests.xml'
        subprocess.run([sys.executable, '-m', 'pytest', '-q', f'--junitxml={junit}'], cwd=ROOT, check=True)
        suite = ET.parse(junit).getroot().find('testsuite')
        save(pd.DataFrame([dict(key=k, value=int(suite.attrib[k])) for k in ('tests', 'failures', 'errors', 'skipped')]), OUTPUT / 'test_validation.csv')


def verify(panels, inputs):
    windows = pd.read_csv(OUTPUT / 'windows.csv', float_precision='round_trip')
    comparisons = pd.read_csv(OUTPUT / 'agreement_vs_paper.csv')
    assert set(comparisons.panel_id) == set(panels.panel_id)
    assert set(windows.panel_id) == set(panels.query('feature_2 == ""').panel_id)
    virtual_count = 0
    for panel in panels.to_dict('records'):
        long = pd.read_parquet(OUTPUT / 'virtual' / f"{panel['panel_id']}.parquet")
        curve = pd.read_csv(OUTPUT / 'panels' / f"{panel['panel_id']}.csv", float_precision='round_trip')
        recomputed = aggregate_curve(long, bool(panel['feature_2']))
        np.testing.assert_allclose(curve.to_numpy(), recomputed.to_numpy(), atol=1e-12, rtol=0)
        predictor, X, sources, _ = inputs[(panel['cohort'], panel['model'])]
        keys = ['grid_value'] + (['grid_value_2'] if panel['feature_2'] else [])
        expected = len(X) * len(make_grid(X[panel['feature']], 'G1'))
        if panel['feature_2']:
            expected *= len(make_grid(X[panel['feature_2']], 'G1'))
        assert len(long) == expected
        assert not long.duplicated(['bg_record_id'] + keys).any()
        assert (curve.n == len(X)).all()
        assert set(long.bg_record_id) == set(X.index)
        np.testing.assert_array_equal(long[panel['feature']], long.grid_value)
        if panel['feature_2']:
            np.testing.assert_array_equal(long[panel['feature_2']], long.grid_value_2)
        else:
            window = extract_window(curve, yaml.safe_load((ROOT / 'protocol_v1.yaml').read_text())['window_rule'])
            saved = windows[windows.panel_id == panel['panel_id']].iloc[0]
            assert json.loads(saved.segments) == window['segments']
            assert saved.status == window['status']
            for name in ('lower', 'upper', 'peak_x', 'amplitude', 'n_segments'):
                np.testing.assert_allclose(saved[name], window[name], atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(long.bg_source_id, sources.loc[long.bg_record_id])
        background_path = OUTPUT / 'backgrounds' / f"{panel['cohort']}_{panel['model']}.csv"
        saved_background = pd.read_csv(background_path, float_precision='round_trip').set_index('bg_record_id')
        np.testing.assert_array_equal(saved_background.index, X.index)
        np.testing.assert_array_equal(saved_background.bg_source_id, sources.to_numpy())
        np.testing.assert_array_equal(saved_background.loc[:, X.columns].to_numpy(), X.to_numpy())
        sampled = long.iloc[np.unique(np.linspace(0, len(long)-1, min(101, len(long)), dtype=int))]
        reconstructed = saved_background.loc[sampled.bg_record_id, X.columns].copy()
        reconstructed[panel['feature']] = sampled.grid_value.to_numpy()
        if panel['feature_2']:
            reconstructed[panel['feature_2']] = sampled.grid_value_2.to_numpy()
        np.testing.assert_allclose(predictor.predict_proba(reconstructed)[:, 1], sampled.p_high, atol=1e-12, rtol=0)
        virtual_count += len(long)
        assert (OUTPUT / 'panels' / f"{panel['panel_id']}.png").exists()
    for path in OUTPUT.rglob('*'):
        if path.is_file() and not path.name.endswith('.meta.json') and path.name != 'run.log':
            meta = json.loads(path.with_name(path.name + '.meta.json').read_text())
            assert digest(path) == meta['sha256'], str(path)
    save(pd.DataFrame([dict(key='panels', value=len(panels)), dict(key='paper_panels', value=int((panels.figure != 'own_top5').sum())),
                       dict(key='own_panels', value=int((panels.figure == 'own_top5').sum())), dict(key='virtual_rows', value=virtual_count),
                       dict(key='window_rows', value=len(windows)), dict(key='comparison_rows', value=len(comparisons))]), OUTPUT / 'validation.csv')


def report():
    facts = pd.read_csv(OUTPUT / 'validation.csv').set_index('key').value
    tests = pd.read_csv(OUTPUT / 'test_validation.csv').set_index('key').value
    windows = pd.read_csv(OUTPUT / 'windows.csv', float_precision='round_trip')
    comparisons = pd.read_csv(OUTPUT / 'agreement_vs_paper.csv')
    counts = comparisons.groupby(['figure', 'agreement']).size().reset_index(name='count')
    save(counts, OUTPUT / 'agreement_summary.csv')
    lines = ['# M3 完成检查', '', '## 完成内容', '',
             f"已生成论文指定面板 {facts['paper_panels']} 个及模型自身前五特征面板 {facts['own_panels']} 个，共保存 {facts['virtual_rows']} 条虚拟输入。每个面板均有 CSV、PNG 和元数据。",
             f"单变量窗口记录 {facts['window_rows']} 条，对照记录 {facts['comparison_rows']} 条。二维定性叙述保留原文，未强行转成单变量窗口。",
             f"完整测试 {tests['tests']} 项通过；失败 {tests['failures']}，错误 {tests['errors']}，跳过 {tests['skipped']}。",
             '', '## 论文数值对照', '', '以下为声明判据下的逐项对照，不表示材料可行性或数据支持。一个面板可有多条论文陈述。', '',
             '| 图组 | 对照结果 | 陈述数 |', '|---|---|---:|']
    for row in pd.read_csv(OUTPUT / 'agreement_summary.csv').itertuples():
        lines.append(f'| {row.figure} | {row.agreement} | {row.count} |')
    lines += ['', '## 人类需要看', '',
              '- panels/fig7_a.png、panels/fig7_f.png：Li_comp 曲线与原始训练观测 rug；同名 CSV 是概率曲线来源。',
              '- windows.csv：所有单变量窗口及 segments；lower/upper 仅为包络，分裂窗口不得当作连续区间。',
              '- agreement_vs_paper.csv：论文陈述、判据、不可比较原因及对应面板。',
              '- panels/fig9_c.png、panels/fig9_d.png 与 virtual/：双变量概率面和逐条虚拟输入，后续可重建。',
              '- backgrounds/、panel_manifest.csv、validation.csv：插补后完整背景、原始观测、产物覆盖与校验依据。',
              '', '## 未解决问题与边界', '',
              '网格、背景、窗口及对照的实现选择见 DECISIONS.md；不能称为复现了原文未公开的实现。',
              'A0 在全部保留行拟合插补，本阶段未训练新模型。原论文只有定性叙述或不明确数值含义的目标保持 not_comparable。',
              '化学残差和支持标签尚未计算，虚拟输入中的相关列留空。解释曲线变化不能推出预测错误或材料不可能存在。',
              '下一会话从 M4 开始；本次未执行化学与支持诊断。']
    checkpoint = ROOT / 'reports/checkpoints/M3.md'
    checkpoint.write_text('\n'.join(lines) + '\n')
    metadata(checkpoint)
    claims = []
    for name in ('validation.csv', 'test_validation.csv'):
        for row in pd.read_csv(OUTPUT / name).itertuples():
            claims.append(dict(claim_id=f'M3-{name}-{row.key}', statement=f'M3 {row.key}', artifact_path=f'outputs/M3/{name}', key=f'key={row.key}|value', value=row.value))
    for row in pd.read_csv(OUTPUT / 'agreement_summary.csv').itertuples():
        claims.append(dict(claim_id=f'M3-agreement-{row.figure}-{row.agreement}', statement='声明判据下的论文陈述对照数', artifact_path='outputs/M3/agreement_summary.csv', key=f'figure={row.figure}|agreement={row.agreement}|count', value=row.count))
    for row in windows.to_dict('records'):
        for key in ('lower', 'upper', 'peak_x', 'amplitude', 'n_segments'):
            if pd.notna(row[key]):
                claims.append(dict(claim_id=f"M3-window-{row['panel_id']}-{key}", statement=f"单变量窗口 {row['panel_id']} {key}", artifact_path='outputs/M3/windows.csv', key=f"panel_id={row['panel_id']}|{key}", value=row[key]))
    path = ROOT / 'reports/claims.csv'
    existing = pd.read_csv(path)
    pd.concat([existing[~existing.claim_id.str.startswith('M3-')], pd.DataFrame(claims)], ignore_index=True).to_csv(path, index=False)
    metadata(path)



def input_manifest():
    names = ['protocol_v1.yaml', 'paper_targets.yaml', 'config/known_issues.csv',
             'config/dopant_properties.csv', 'data/raw/Machine_Learning_Guided_Design_dataset.xlsx',
             'scripts/run_m3.py']
    names += [f'src/llzo_pdp/{name}.py' for name in
              ('__init__', 'io', 'chemistry', 'features', 'preprocess', 'models', 'meta', 'pdp')]
    names += [f'outputs/M2/models/{cohort}/{model}_V_raw_A0_T_main_seed0.pkl'
              for cohort, models in (('full', ['DT', 'RF', 'LGBM', 'CB']), ('ga', ['CB', 'AB'])) for model in models]
    manifest = {name: digest(ROOT / name) for name in names}
    decisions = (ROOT / 'DECISIONS.md').read_text().splitlines()
    for number in ('D-018', 'D-019', 'D-020', 'D-021'):
        entry = [line for line in decisions if line.startswith(f'- {number} |')]
        if len(entry) != 1:
            raise ValueError(f'Missing or duplicated study choice: {number}')
        manifest[f'decision:{number}'] = hashlib.sha256(entry[0].encode()).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    inputs, own = load_inputs()
    panels = panel_plan(own)
    protocol = yaml.safe_load((ROOT / 'protocol_v1.yaml').read_text())
    if args.verify:
        if json.loads((OUTPUT / 'input_manifest.json').read_text()) != input_manifest():
            raise ValueError('M3 source/input fingerprint changed; regenerate artifacts')
        verify(panels, inputs)
        run_tests()
        report()
        return
    prepare(inputs, panels)
    if args.prepare:
        print(pd.read_csv(OUTPUT / 'budget.csv').to_string(index=False), flush=True)
        return
    save(input_manifest(), OUTPUT / 'input_manifest.json')
    save(panels, OUTPUT / 'panel_plan.csv')
    started = time.perf_counter()
    save({'status': 'running', 'pid': os.getpid(), 'completed_panels': 0}, OUTPUT / 'run_state.json')
    for (cohort, model), (_, X, sources, original) in inputs.items():
        save(X.rename_axis('bg_record_id').reset_index().assign(bg_source_id=sources.to_numpy()), OUTPUT / 'backgrounds' / f'{cohort}_{model}.csv')
        save(original.rename_axis('bg_record_id').reset_index().assign(bg_source_id=sources.to_numpy()), OUTPUT / 'backgrounds' / f'{cohort}_{model}_observed.csv')
    windows, comparisons, records = [], [], []
    paper = yaml.safe_load((ROOT / 'paper_targets.yaml').read_text())
    for i, panel in enumerate(panels.to_dict('records'), 1):
        if time.perf_counter() - started > protocol['compute_budget']['max_wall_hours_per_milestone'] * 3600:
            raise TimeoutError('M3 wall-time budget exceeded; inspect before applying protocol fallback')
        predictor, X, sources, original = inputs[(panel['cohort'], panel['model'])]
        long = make_virtual(panel, predictor, X, sources)
        curve = aggregate_curve(long, bool(panel['feature_2']))
        virtual_path = OUTPUT / 'virtual' / f"{panel['panel_id']}.parquet"
        curve_path = OUTPUT / 'panels' / f"{panel['panel_id']}.csv"
        save(long, virtual_path)
        save(curve, curve_path)
        plot_panel(panel, curve, original, curve_path.with_suffix('.png'))
        window = dict(status='not_univariate', segments=[])
        if not panel['feature_2']:
            window = extract_window(curve, protocol['window_rule'])
            windows.append({**panel, **window, 'segments': json.dumps(window['segments']), 'data_version': 'V_raw', 'scheme': 'A', 'background': 'B_train', 'grid': 'G1'})
        for number, target in enumerate(targets_for(panel, paper), 1):
            agreement, reason = compare_target(curve, window, target)
            comparisons.append({**panel, 'target_id': number, 'target_kind': target['kind'], 'target_value': json.dumps(target['value'], ensure_ascii=False), 'agreement': agreement, 'criterion': 'D-020', 'reason': reason})
        records.append({**panel, 'virtual_rows': len(long), 'grid_points': len(curve), 'background_rows': len(X),
                        'virtual_path': str(virtual_path.relative_to(ROOT)), 'curve_path': str(curve_path.relative_to(ROOT)),
                        'background_path': f"outputs/M3/backgrounds/{panel['cohort']}_{panel['model']}.csv",
                        'rug_path': f"outputs/M3/backgrounds/{panel['cohort']}_{panel['model']}_observed.csv"})
        save({'status': 'running', 'pid': os.getpid(), 'completed_panels': i, 'total_panels': len(panels)}, OUTPUT / 'run_state.json')
        print(f"[{i}/{len(panels)}] {panel['panel_id']}: {len(long)} virtual inputs", flush=True)
    save(pd.DataFrame(windows), OUTPUT / 'windows.csv')
    save(pd.DataFrame(comparisons), OUTPUT / 'agreement_vs_paper.csv')
    save(pd.DataFrame(records), OUTPUT / 'panel_manifest.csv')
    verify(panels, inputs)
    run_tests()
    report()
    save({'status': 'complete', 'completed_panels': len(panels), 'elapsed_seconds': time.perf_counter()-started}, OUTPUT / 'run_state.json')
    print('M3 COMPLETE; panels, virtual inputs and model reconstruction verified.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        save({'status': 'failed', 'pid': os.getpid(), 'traceback': traceback.format_exc()}, OUTPUT / 'run_state.json')
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if (OUTPUT / 'run.log').exists():
            metadata(OUTPUT / 'run.log')
