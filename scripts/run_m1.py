"""Generate M1 datasets and a CSV-backed checkpoint without training models."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from llzo_pdp.chemistry import charge_residual_features, site_residuals
from llzo_pdp.features import build_features
from llzo_pdp.io import load_raw, TARGET_COL
from llzo_pdp.meta import write_meta
from llzo_pdp.preprocess import filter_rows, impute_features, threshold_table


def verify_claim(root, claim):
    """Resolve exact CSV equality filters and compare a scalar claim."""
    frame = pd.read_csv(root / claim.artifact_path)
    parts = claim.key.split("|")
    for part in parts[:-1]:
        column, value = part.split("=", 1)
        frame = frame[frame[column].astype(str) == value]
    if len(frame) != 1:
        raise ValueError(f"Non-unique claim key: {claim.claim_id}")
    actual = frame.iloc[0][parts[-1]]
    if not np.isclose(float(actual), float(claim.value), atol=1e-9, rtol=0):
        raise ValueError(f"Claim mismatch: {claim.claim_id}")


def run(destination=ROOT):
    destination = Path(destination)
    output = destination / "outputs/M1"
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load((ROOT / "protocol_v1.yaml").read_text())
    seeds = protocol["split_and_tuning"]["seeds"]
    issues = pd.read_csv(ROOT / "config/known_issues.csv", keep_default_na=False)
    k12 = list(map(int, issues.set_index("issue_id").at["K12", "record_ids"].split(";")))
    protected = pd.read_csv(ROOT / "outputs/M0/protected_files.csv")
    for item in protected.itertuples():
        if hashlib.sha256((ROOT / item.path).read_bytes()).hexdigest() != item.sha256:
            raise ValueError(f"Protected file changed: {item.path}")
    raw = load_raw()

    def save(frame, name, version="V_raw+V_corr", seed=seeds):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=True)
        else:
            frame.to_csv(path, index=False)
        write_meta(path, seed=seed, data_version=version, script="scripts/run_m1.py")

    summary, descriptors, qbars, filters, diagnostics, runs, splits = [], [], [], [], [], [], []
    schemas, audit = [], None
    for version in protocol["data"]["versions"]:
        result = build_features(raw, version)
        kept, filtering = filter_rows(result.raw_features)
        filters.append(filtering.assign(data_version=version))
        descriptors.append(result.descriptor_table.assign(data_version=version))
        qbars.append(result.site_qbar.reset_index().assign(data_version=version))
        if version == "V_corr":
            audit = result.audit_log
        table = threshold_table(raw.loc[kept, TARGET_COL],
                                protocol["target"]["threshold_main"],
                                protocol["target"]["threshold_variants"]["T_1e4"])
        table["data_version"] = version
        table["n_raw"] = len(raw)
        table["n_kept"] = len(kept)
        table["n_dropped"] = len(raw) - len(kept)
        table["n_sources_raw"] = raw.source_id.nunique()
        table["n_sources_kept"] = raw.loc[kept, "source_id"].nunique()
        summary.append(table)
        save(result.X.reset_index(), f"features_{version}.csv", version)
        save(result.raw_features.reset_index(), f"raw_filter_features_{version}.csv", version)
        schemas.extend({"data_version": version, "feature": col,
                        "kind": "numeric" if col in result.numeric_columns else "one_hot"}
                       for col in result.X)
        label = (raw.loc[kept, TARGET_COL] >= protocol["target"]["threshold_main"]).astype(int)
        for seed in seeds:
            train_val, test = train_test_split(kept, test_size=0.15, stratify=label, random_state=seed)
            train, val = train_test_split(train_val, test_size=0.15/0.85,
                                         stratify=label.loc[train_val], random_state=seed)
            for name, ids in (("train", train), ("val", val), ("test", test)):
                splits.extend({"data_version": version, "seed": seed, "threshold_name": "T_main",
                               "split": name, "record_id": int(rid),
                               "source_id": int(raw.at[rid, "source_id"]), "label": int(label.at[rid])}
                              for rid in ids)
            for mode in ("A0", "A1"):
                imputed = impute_features(result.X.loc[kept], result.numeric_columns,
                                          mode, seed, train)
                name = f"matrices/{version}_{mode}_T_main_seed{seed}.parquet"
                save(imputed.X, name, version, seed)
                runs.append({"data_version": version, "imputation": mode, "seed": seed,
                             "n_fit": len(imputed.fit_ids), "n_transformed": len(imputed.X),
                             "n_iter": imputed.imputer.n_iter_,
                             "convergence_warning": bool(imputed.convergence_warnings),
                             "warning": "; ".join(imputed.convergence_warnings),
                             "artifact_path": f"outputs/M1/{name}"})
                # Excluded K12 rows are transformed for diagnostics only.
                X_diag = imputed.transform(result.X.loc[k12])
                diag = X_diag[protocol["features"]["composition"]].copy()
                diag["r_charge"] = charge_residual_features(X_diag, result.site_qbar.loc[k12])
                diag = diag.join(site_residuals(X_diag))
                diag["n_unassigned"] = result.site_qbar.loc[k12, "n_unassigned"]
                diag["used_in_model"] = diag.index.isin(kept)
                diag["eligible_for_chemistry_checks"] = False
                diag["scope"] = "excluded_rows_transform_only"
                diag["source_id"] = raw.loc[k12, "source_id"]
                diagnostics.append(diag.reset_index().assign(data_version=version, imputation=mode, seed=seed))

    expected = {(rid, field, issue.issue_id)
                for issue in issues.query('action_V_corr == "correct"').itertuples()
                for rid in map(int, issue.record_ids.split(";"))
                for field in issue.field.split(";")}
    actual = set(audit[["record_id", "field", "issue_id"]].itertuples(index=False, name=None))
    if actual != expected or len(audit) != len(expected):
        raise ValueError("Audit does not match expanded correct issues")
    summary = pd.concat(summary, ignore_index=True)
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    runs = pd.DataFrame(runs)
    save(audit, "audit_log.csv", "V_corr")
    save(pd.concat(descriptors, ignore_index=True), "descriptor_table.csv")
    save(pd.concat(qbars, ignore_index=True), "site_qbar.csv")
    save(pd.concat(filters, ignore_index=True), "row_filter.csv")
    save(summary, "dataset_summary.csv")
    save(diagnostics, "imputed_chemistry.csv")
    save(runs, "imputation_runs.csv")
    save(pd.DataFrame(splits), "split_ids.csv")
    save(pd.DataFrame(schemas), "feature_schema.csv")
    save(raw[["record_id", "source_id", TARGET_COL]], "record_targets.csv", "V_raw")
    validation = pd.DataFrame([
        {"key": "correct_issues", "value": len(issues.query('action_V_corr == "correct"'))},
        {"key": "expected_audit_cells", "value": len(expected)},
        {"key": "actual_audit_cells", "value": len(audit)},
        {"key": "imputation_runs", "value": len(runs)},
        {"key": "convergence_warnings", "value": int(runs.convergence_warning.sum())},
        {"key": "k12_records", "value": len(k12)},
        {"key": "diagnostic_rows", "value": len(diagnostics)},
        {"key": "k12_used_in_model", "value": int(diagnostics.used_in_model.sum())},
        {"key": "k12_max_abs_charge", "value": float(diagnostics.r_charge.abs().max())},
        {"key": "protocol_status_frozen", "value": int(protocol["status"] == "frozen")},
        {"key": "protected_files_match", "value": 1},
    ])
    save(validation, "validation.csv")
    inputs = [ROOT / "protocol_v1.yaml", ROOT / "config/known_issues.csv",
              ROOT / "config/dopant_properties.csv", ROOT / "pyproject.toml",
              *sorted((ROOT / "src/llzo_pdp").glob("*.py")),
              *sorted((ROOT / "tests").glob("*.py")), ROOT / "scripts/run_m1.py",
              *sorted((ROOT / "data/raw").glob("*"))]
    save(pd.DataFrame([{"path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in inputs if p.is_file()]), "input_manifest.csv")
    generate_checkpoint(destination)


def generate_checkpoint(destination):
    """Read saved CSVs for every reported numeric result and register each claim."""
    output = destination / "outputs/M1"
    summary = pd.read_csv(output / "dataset_summary.csv")
    validation = pd.read_csv(output / "validation.csv").set_index("key").value
    claims_path = destination / "reports/claims.csv"
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame(
        columns=["claim_id", "statement", "artifact_path", "key", "value"])
    claims = claims[~claims.claim_id.str.startswith("M1-", na=False)]
    new_claims = []

    def register(suffix, statement, artifact, key, value):
        new_claims.append(dict(claim_id=f"M1-{suffix}", statement=statement,
                               artifact_path=f"outputs/M1/{artifact}", key=key, value=value))
        return value

    lines = ["# M1 数据版本、特征构建与协议冻结", "",
             "完成 V_raw/V_corr、按位点加权描述符、类别编码、原始列过滤、A0/A1 数值插补及排除记录诊断。",
             "所有实现细节标注为本研究选择，见 DECISIONS.md；本阶段未训练模型或查看 PDP 曲线。", "",
             "## 数据与阈值（从 CSV 生成）", ""]
    for version, sub in summary.groupby("data_version", sort=False):
        row = sub[sub.threshold_name == "T_main"].iloc[0]
        key = f"data_version={version}|threshold_name=T_main"
        for col in ("n_raw", "n_kept", "n_dropped", "n_sources_kept"):
            register(f"{version}-{col}", f"{version} {col}", "dataset_summary.csv", f"{key}|{col}", row[col])
        lines.append(f"- {version}：原始 {int(row.n_raw)} 条，保留 {int(row.n_kept)} 条，"
                     f"删除 {int(row.n_dropped)} 条，保留来源 {int(row.n_sources_kept)} 个。")
    lines += ["", "| 数据版本 | 阈值 | 数值 | 高导数 | 高导比例 |", "|---|---|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        key = f"data_version={row.data_version}|threshold_name={row.threshold_name}"
        for col in ("threshold", "n_high", "high_fraction", "ratio_error"):
            register(f"{row.data_version}-{row.threshold_name}-{col}",
                     f"{row.data_version} {row.threshold_name} {col}",
                     "dataset_summary.csv", f"{key}|{col}", getattr(row, col))
        lines.append(f"| {row.data_version} | {row.threshold_name} | {row.threshold:.10g} | "
                     f"{row.n_high} | {row.high_fraction:.8f} |")
    for key, value in validation.items():
        register(key, key, "validation.csv", f"key={key}|value", value)
    test_path = output / "test_validation.csv"
    if test_path.exists():
        tests = pd.read_csv(test_path).set_index("key").value
        for key, value in tests.items():
            register(f"test-{key}", f"测试验收 {key}", "test_validation.csv", f"key={key}|value", value)
        lines += ["", f"完整测试：{tests['tests']} 项，失败 {tests['failures']}，"
                  f"错误 {tests['errors']}，跳过 {tests['skipped']}。"]
    lines += ["", "## 审计与诊断", "",
              f"- correct 问题 {int(validation['correct_issues'])} 项，展开期望单元格 "
              f"{int(validation['expected_audit_cells'])} 个，实际审计 {int(validation['actual_audit_cells'])} 个。",
              f"- 插补运行 {int(validation['imputation_runs'])} 组；收敛警告 "
              f"{int(validation['convergence_warnings'])} 组，见 imputation_runs.csv。",
              f"- K12 共 {int(validation['k12_records'])} 条，生成排除记录诊断 "
              f"{int(validation['diagnostic_rows'])} 行；进入模型的诊断行 "
              f"{int(validation['k12_used_in_model'])} 行。",
              f"- 诊断中最大 |r_charge| 为 {validation['k12_max_abs_charge']:.8g}。这是独立插补后、"
              "依照声明的名义假设计算的数值残差；缺少位点分配和 qbar 的记录不作有效化学判定，也不能由此推断材料不可能存在。",
              "", "## 人类需要看", "",
              "- audit_log.csv：每个修正单元格的前后值及问题编号。",
              "- row_filter.csv 与 dataset_summary.csv：删除记录及不同阈值的标签变化。",
              "- imputed_chemistry.csv：K12 只做排除记录诊断，不纳入模型。",
              "- imputation_runs.csv、split_ids.csv：插补警告和 A1 训练范围；未来 CV 需重新在训练 fold 内拟合。",
              "- DECISIONS.md 与协议：冻结状态、实现选择及环境限制。",
              "", "## 未解决问题与下次入口", "",
              "T_ratio 的目标比例在当前保留集无法精确达到，最近可达比例与误差已完整保存；未调整主阈值。",
              "lightgbm、catboost 在当前运行时缺失，后续需遵守离线约束处理。",
              "下次从 M2 开始；本会话到此停止。"]
    checkpoint = destination / "reports/checkpoints/M1.md"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("\n".join(lines) + "\n")
    added = pd.DataFrame(new_claims)
    claims = pd.concat([claims, added], ignore_index=True) if not claims.empty else added
    claims.to_csv(claims_path, index=False)
    for claim in claims.itertuples():
        verify_claim(destination, claim)
    for path in (checkpoint, claims_path):
        write_meta(path, seed=None, data_version="V_raw+V_corr", script="scripts/run_m1.py")


if __name__ == "__main__":
    run()
    with tempfile.TemporaryDirectory() as temp:
        junit = Path(temp) / "tests.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
                       cwd=ROOT, check=True)
        suite = ET.parse(junit).getroot().find("testsuite")
        test_path = ROOT / "outputs/M1/test_validation.csv"
        pd.DataFrame([{"key": key, "value": int(suite.attrib[key])}
                      for key in ("tests", "failures", "errors", "skipped")]).to_csv(test_path, index=False)
        write_meta(test_path, seed=None, data_version="V_raw+V_corr", script="scripts/run_m1.py")
    generate_checkpoint(ROOT)
