"""Offline environment and provenance checkpoint; run from any directory."""
from pathlib import Path
import hashlib
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from llzo_pdp.meta import write_meta


def main():
    output = ROOT / "outputs/M0"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for requirement in (ROOT / "requirements.txt").read_text().splitlines():
        name = requirement.split(">=")[0]
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "MISSING"
        records.append({"package": name, "installed_version": installed})
    pd.DataFrame(records).to_csv(output / "environment.csv", index=False)
    protected = [*sorted((ROOT / "data/raw").glob("*")),
                 ROOT / "src/llzo_pdp/io.py", ROOT / "src/llzo_pdp/chemistry.py",
                 ROOT / "tests/test_data_facts.py", ROOT / "tests/test_chemistry.py"]
    pd.DataFrame([{"path": str(p.relative_to(ROOT)),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in protected if p.is_file()]).to_csv(output / "protected_files.csv", index=False)
    with tempfile.TemporaryDirectory() as temp:
        junit = Path(temp) / "results.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"],
                       cwd=ROOT, check=True)
        suite = ET.parse(junit).getroot().find("testsuite")
        checks = [{"key": k, "value": int(suite.attrib[k])}
                  for k in ("tests", "failures", "errors", "skipped")]
    pd.DataFrame(checks).to_csv(output / "validation.csv", index=False)
    for path in output.glob("*.csv"):
        write_meta(path, seed=None, data_version="raw_read_only", script="scripts/run_m0.py")
    values = pd.read_csv(output / "validation.csv").set_index("key").value
    env = pd.read_csv(output / "environment.csv")
    missing = env.loc[env.installed_version == "MISSING", "package"].tolist()
    checkpoint = ROOT / "reports/checkpoints/M0.md"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        "# M0 环境与骨架\n\n"
        "已建立离线虚拟环境入口、Makefile、元数据工具及原始文件校验清单。\n\n"
        f"验收：测试 {values['tests']} 项，失败 {values['failures']}，错误 {values['errors']}，"
        f"跳过 {values['skipped']}；数字读取 outputs/M0/validation.csv。\n\n"
        "人类需要看：\n"
        "- outputs/M0/environment.csv：依赖版本与缺包状态。\n"
        "- outputs/M0/protected_files.csv：原始数据和已核实代码、测试的校验值。\n"
        "- DECISIONS.md：离线环境选择及后续模型依赖限制。\n\n"
        f"未解决问题：缺少 {', '.join(missing) or '无'}；本阶段不使用这些模型包，未安装。\n"
    )
    claims_path = ROOT / "reports/claims.csv"
    claims = pd.read_csv(claims_path)
    claims = claims[~claims.claim_id.str.startswith("M0-", na=False)]
    added = pd.DataFrame([{
        "claim_id": f"M0-{r['key']}", "statement": f"环境验收 {r['key']}",
        "artifact_path": "outputs/M0/validation.csv", "key": f"key={r['key']}|value",
        "value": r["value"],
    } for r in checks])
    pd.concat([claims, added], ignore_index=True).to_csv(claims_path, index=False)
    for path in (checkpoint, claims_path):
        write_meta(path, seed=None, data_version="raw_read_only", script="scripts/run_m0.py")


if __name__ == "__main__":
    main()
