import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_m1_artifacts_preserve_exclusions_provenance_and_claims(tmp_path):
    entry = runpy.run_path(str(ROOT / "scripts/run_m1.py"))
    entry["run"](tmp_path)
    output = tmp_path / "outputs/M1"
    summary = pd.read_csv(output / "dataset_summary.csv")
    assert len(summary) == 8
    assert set(summary.n_kept) == {218}
    assert set(summary.n_dropped) == {9}
    audit = pd.read_csv(output / "audit_log.csv")
    assert len(audit) == 9
    checks = pd.read_csv(output / "imputed_chemistry.csv")
    assert len(checks) == 2 * 2 * 5 * 9
    assert not checks.eligible_for_chemistry_checks.any()
    assert not checks.used_in_model.any()
    assert np.isfinite(checks.r_charge).all()
    for row in pd.read_csv(output / "imputation_runs.csv").itertuples():
        matrix = pd.read_parquet(tmp_path / row.artifact_path)
        assert not set(checks.record_id) & set(matrix.index)
        assert not matrix.isna().any().any()
        assert "record_id" not in matrix.columns
        assert "source_id" not in matrix.columns
    splits = pd.read_csv(output / "split_ids.csv")
    for _, group in splits.groupby(["data_version", "seed"]):
        assert group.record_id.nunique() == len(group) == 218
        assert group.groupby("split").size().to_dict() == {"test": 33, "train": 152, "val": 33}
    for path in output.rglob("*"):
        if path.is_file() and not path.name.endswith(".meta.json"):
            metadata = json.loads(path.with_name(path.name + ".meta.json").read_text())
            assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    claims = pd.read_csv(tmp_path / "reports/claims.csv")
    assert claims.claim_id.is_unique
    for claim in claims.itertuples():
        entry["verify_claim"](tmp_path, claim)
