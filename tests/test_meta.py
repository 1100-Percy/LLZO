import hashlib
import json
from datetime import datetime

import pytest


def test_metadata_tracks_artifact_and_run(tmp_path):
    from llzo_pdp.meta import write_meta

    artifact = tmp_path / "example.csv"
    artifact.write_text("a\n1\n")
    sidecar = write_meta(artifact, seed=0, data_version="V_raw", script="test_meta.py")
    meta = json.loads(sidecar.read_text())
    assert len(meta["git_hash"]) == 40
    assert meta["protocol_version"] == 1
    assert meta["seed"] == 0
    assert meta["data_version"] == "V_raw"
    assert meta["script"] == "test_meta.py"
    assert meta["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert datetime.fromisoformat(meta["timestamp"]).tzinfo is not None


def test_metadata_requires_run_context(tmp_path):
    from llzo_pdp.meta import write_meta

    artifact = tmp_path / "example.csv"
    artifact.write_text("a\n1\n")
    with pytest.raises(ValueError, match="seed.*data_version.*script"):
        write_meta(artifact)
