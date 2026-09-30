"""Smoke-test installed model libraries on synthetic data, without running M2."""
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from llzo_pdp.meta import write_meta


def main():
    output = ROOT / "outputs/feedback_dependencies"
    output.mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame(np.arange(60, dtype=float).reshape(20, 3), columns=["a", "b", "c"])
    y = np.arange(len(X)) % 2
    models = {
        "lightgbm": LGBMClassifier(n_estimators=3, min_child_samples=1,
                                   n_jobs=1, random_state=0, verbosity=-1),
        "catboost": CatBoostClassifier(iterations=3, depth=2, random_seed=0,
                                       thread_count=1, verbose=False, allow_writing_files=False),
    }
    records = []
    for name, model in models.items():
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities.sum(axis=1), 1)
        records.append({"package": name, "version": version(name), "smoke_passed": 1,
                        "n_synthetic_rows": len(X), "scope": "synthetic_only"})
    environment = output / "environment.csv"
    pd.DataFrame(records).to_csv(environment, index=False)
    with tempfile.TemporaryDirectory() as temp:
        junit = Path(temp) / "tests.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
                       cwd=ROOT, check=True)
        suite = ET.parse(junit).getroot().find("testsuite")
        tests = pd.DataFrame([{"key": key, "value": int(suite.attrib[key])}
                              for key in ("tests", "failures", "errors", "skipped")])
    validation = output / "test_validation.csv"
    tests.to_csv(validation, index=False)
    for path in (environment, validation):
        write_meta(path, seed=0, data_version="synthetic_only", script="scripts/check_model_dependencies.py")
    installed = pd.read_csv(environment)
    results = pd.read_csv(validation).set_index("key").value
    report = ROOT / "reports/checkpoints/feedback_dependencies.md"
    report.write_text(
        "# 人类反馈处理：模型依赖\n\n"
        "根据 HUMAN_FEEDBACK.md 的明确授权，安装指定模型库及所需运行依赖。\n\n"
        + "\n".join(f"- {r.package} {r.version}：导入、合成数据训练与概率输出检查通过。"
                    for r in installed.itertuples())
        + f"\n- 完整测试：{results['tests']} 项，失败 {results['failures']}，"
        f"错误 {results['errors']}，跳过 {results['skipped']}。\n\n"
        "初次导入 LightGBM 缺少 libomp.dylib，安装 Homebrew libomp 后重新检查通过。\n"
        "Python 包安装在本 checkout 的 .venv；libomp 安装于 Homebrew 管理目录，环境二进制不提交 Git。\n"
        "这些检查仅使用合成数据；M2 尚未开始，冻结协议与研究代码保持不变。\n\n"
        "重跑：`.venv/bin/python scripts/check_model_dependencies.py`。\n"
        "GitHub 推送与反馈完成状态以 HUMAN_FEEDBACK.md、PROGRESS.md 为准。\n"
    )
    claims_path = ROOT / "reports/claims.csv"
    claims = pd.read_csv(claims_path)
    claims = claims[~claims.claim_id.str.startswith("HF-dependencies-")]
    added = [{"claim_id": f"HF-dependencies-{r.package}",
              "statement": f"{r.package} 合成数据检查通过", "artifact_path": str(environment.relative_to(ROOT)),
              "key": f"package={r.package}|smoke_passed", "value": r.smoke_passed}
             for r in installed.itertuples()]
    added.extend({"claim_id": f"HF-dependencies-{r.key}", "statement": f"安装后完整测试 {r.key}",
                  "artifact_path": str(validation.relative_to(ROOT)), "key": f"key={r.key}|value",
                  "value": r.value} for r in tests.itertuples())
    pd.concat([claims, pd.DataFrame(added)], ignore_index=True).to_csv(claims_path, index=False)
    for path in (report, claims_path):
        write_meta(path, seed=0, data_version="synthetic_only", script="scripts/check_model_dependencies.py")


if __name__ == "__main__":
    main()
