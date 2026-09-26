"""Raw data loading for the Abraham et al. (2026) LLZO dataset.

Design rules
------------
* The raw xlsx is never modified. All corrections happen in later, logged steps.
* `record_id` = Excel sheet row number (header is row 1, first data row is 2).
  It is the only key used to refer to a record anywhere in the project.
* `source_id` = the paper number in the first column, forward-filled
  (it is only written on the first row of each paper in the raw sheet).
* The three (Dopant, Dopant Composition) pairs are renamed to
  dopant_1/x_1, dopant_2/x_2, dopant_3/x_3. All other columns keep their
  original names (stripped of surrounding whitespace).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_XLSX = Path(__file__).resolve().parents[2] / "data" / "raw" / "Machine_Learning_Guided_Design_dataset.xlsx"

TARGET_COL = "Ionic Conductivity(Total)"
MEAS_T_COL = "Temperature(Ionic conductivity)"

NUMERIC_COLS = [
    "%excess Li", "Li Composition", "La_comp", "Zr_comp",
    "Li_site_sto", "La_site_sto", "Zr_site_sto",
    "x_1", "x_2", "x_3", "# of Dopants",
    "1st Calcination Temp", "1st Calcination Time",
    "Ball milling speed", "Ball milling time", "# of ball milling",
    "Cold  Pressure", "Sintering Temperature", "Sintering Time",
    "Relative Density", TARGET_COL, MEAS_T_COL,
]


def _dedupe_header(raw_header: list) -> list[str]:
    names, seen = [], {}
    dop_i = 0
    xcomp_i = 0
    for pos, h in enumerate(raw_header):
        h = "" if (h is None or (isinstance(h, float) and np.isnan(h))) else str(h).strip()
        if pos == 0:
            names.append("source_no")
            continue
        if pos == 2:
            names.append("doi")
            continue
        if h == "Dopant":
            dop_i += 1
            names.append(f"dopant_{dop_i}")
            continue
        if h == "Dopant Composition":
            xcomp_i += 1
            names.append(f"x_{xcomp_i}")
            continue
        if h == "":
            h = f"unnamed_{pos}"
        if h in seen:
            seen[h] += 1
            h = f"{h}__{seen[h]}"
        else:
            seen[h] = 0
        names.append(h)
    return names


def load_raw(path: str | Path = RAW_XLSX) -> pd.DataFrame:
    """Load the sheet exactly as recorded, plus record_id / source_id."""
    df = pd.read_excel(path, header=None)
    header = list(df.iloc[0])
    df = df.iloc[1:].copy()
    df.columns = _dedupe_header(header)
    df["record_id"] = df.index + 1  # Excel row number
    df = df.dropna(how="all", subset=[c for c in df.columns if c != "record_id"])
    df["source_id"] = pd.to_numeric(df["source_no"], errors="coerce").ffill().astype(int)
    df["Author"] = df["Author"].ffill()
    for c in ("dopant_1", "dopant_2", "dopant_3", "Li_dop", "La_dop", "Zr_dop"):
        df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("record_id", drop=False)
