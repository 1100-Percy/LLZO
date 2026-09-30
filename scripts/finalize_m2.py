"""Generate the M2 completion checkpoint from verified CSV artifacts."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llzo_pdp.meta import write_meta

OUTPUT = ROOT / "outputs/M2"


def metadata(path):
    write_meta(path, seed=[0, 1, 2, 3, 4], data_version="V_raw+V_corr",
               script="scripts/finalize_m2.py")


def main():
    completion = pd.read_csv(OUTPUT / "completion_summary.csv").set_index("key").value
    tests = pd.read_csv(OUTPUT / "test_validation.csv").set_index("key").value
    plan = pd.read_csv(OUTPUT / "task_plan.csv")
    state = json.loads((OUTPUT / "run_state.json").read_text())
    assert state["status"] == "complete"
    assert completion["completed_jobs"] == state["completed_jobs"] == len(plan)
    assert completion["all_reload_verified"] == 1
    assert tests[["failures", "errors", "skipped"]].sum() == 0
    rows = []
    for filename in ("metrics.csv", "ga_metrics.csv"):
        frame = pd.read_csv(OUTPUT / filename)
        main_test = frame.query('split == "test" and data_version == "V_raw" and threshold_name == "T_main"')
        for (cohort, model), group in main_test.groupby(["cohort", "model"], sort=True):
            row = {"cohort": cohort, "model": model, "method": "holdout_seed_mean",
                   "runs": len(group), "n_test_min": group.n.min(), "n_test_max": group.n.max()}
            for metric in ("accuracy", "f1_weighted", "roc_auc"):
                row[metric] = group[metric].mean()
                row[metric + "_min"] = group[metric].min()
                row[metric + "_max"] = group[metric].max()
            rows.append(row)
    groups = pd.read_csv(OUTPUT / "group_cv.csv")
    for row in groups.query('split == "pooled_oof"').itertuples():
        rows.append({"cohort": "full", "model": row.model, "method": row.method,
                     "runs": len(groups.query('method == @row.method and model == @row.model and split == "test"')),
                     "n_test_min": row.n, "n_test_max": row.n,
                     **{metric: getattr(row, metric) for metric in ("accuracy", "f1_weighted", "roc_auc")}})
    path = OUTPUT / "review_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata(path)
    summary = pd.read_csv(path)
    claims_path = ROOT / "reports/claims.csv"
    claims = pd.read_csv(claims_path)
    claims = claims[~claims.claim_id.str.startswith("M2-review-")]
    new_claims = []
    for row in summary.to_dict("records"):
        key = "|".join(f"{column}={row[column]}" for column in ("cohort", "model", "method"))
        for column in summary.columns[3:]:
            if pd.notna(row[column]):
                new_claims.append({"claim_id": f"M2-review-{row['cohort']}-{row['model']}-{row['method']}-{column}",
                                   "statement": f"主阈值指标汇总 {key} {column}",
                                   "artifact_path": "outputs/M2/review_summary.csv",
                                   "key": f"{key}|{column}", "value": row[column]})
    pd.concat([claims, pd.DataFrame(new_claims)], ignore_index=True).to_csv(claims_path, index=False)
    metadata(claims_path)
    checkpoint = ROOT / "reports/checkpoints/M2.md"
    text = checkpoint.read_text().split("## 人类需要看")[0]
    text = text.replace("# M2 阶段检查", "# M2 完成检查")
    text = text.replace("全部计算和重载验证完成；待会话更新里程碑状态并提交", "全部计算与本次重载验收完成")
    lines = [text.rstrip(), "", f"实际完成 {int(completion['completed_jobs'])} 个任务，全部模型重载预测及指标核对通过。",
             f"分组验证中 {int(completion['single_class_test_folds'])} 个模型测试折只有单一类别，ROC-AUC 留空并注明原因。",
             "", "## 主阈值结果", "",
             "主留出结果为协议种子的均值；分组结果为 pooled OOF。它们使用不同评估划分，只并列展示，不作模型排名。",
             "数值读取自 outputs/M2/review_summary.csv；原始逐种子、逐折结果及混淆矩阵保留在对应指标表。", "",
             "| 数据 | 模型 | 评估方式 | accuracy | weighted F1 | ROC-AUC |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.cohort} | {row.model} | {row.method} | {row.accuracy:.4f} | {row.f1_weighted:.4f} | {row.roc_auc:.4f} |")
    lines += ["", "## 人类需要看", "",
              "- threshold_check.csv、main_seed_class_distribution.csv：主阈值与其他协议阈值的全部类别计数，不分层只作诊断。",
              "- review_summary.csv、metrics.csv、ga_metrics.csv：汇总与逐种子结果，包含测试集混淆矩阵。",
              "- group_cv.csv：分层随机、来源分组和逐来源留出结果；单类别折及 pooled OOF 需分别理解。",
              "- importance_vs_paper.csv：模型内置重要性与论文特征对照，不据此改变设置。",
              "- completion_summary.csv、jobs/*/complete.json：完成状态与重载校验依据。",
              "", "## 未解决问题与边界", "",
              "A0 使用全部保留行拟合插补，分组测试也不能称为无泄漏；A1 属后续敏感性分析。",
              "所有实现选择沿用 DECISIONS.md；模型得分和重要性不构成有利成分窗口可靠性的结论。",
              "本次结束于 M2 验收与提交。下一会话按 PLAN.md 从 M3 开始。"]
    checkpoint.write_text("\n".join(lines) + "\n")
    metadata(checkpoint)


if __name__ == "__main__":
    main()
