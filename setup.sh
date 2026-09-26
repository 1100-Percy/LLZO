#!/usr/bin/env bash
# Run ONCE with network access before starting Codex (sandboxed agent runs may have no network).
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
git init -q 2>/dev/null || true
git add -A && git commit -qm "M-1: scaffold, verified data facts, chemistry utils" || true
echo "Setup OK. Now start Codex with prompts/00_bootstrap.md"
