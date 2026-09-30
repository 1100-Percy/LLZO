"""Write provenance next to generated artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def write_meta(path: str | Path, **kw) -> Path:
    """Record the generating checkout, protocol, run context and artifact hash."""
    required = ("seed", "data_version", "script")
    if any(key not in kw for key in required):
        raise ValueError("Metadata requires seed, data_version, script")
    path = Path(path)
    protocol = ROOT / "protocol_v1.yaml"
    metadata = dict(kw)
    metadata.update(
        git_hash=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_dirty=bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        protocol_version=yaml.safe_load(protocol.read_text())["version"],
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    sidecar = path.with_name(path.name + ".meta.json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return sidecar
