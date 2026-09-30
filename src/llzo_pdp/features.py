"""Protocol features with explicit corrections and site-specific descriptors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .chemistry import dopant_long, site_charge_table
from .meta import ROOT

AUDIT_COLUMNS = ["record_id", "field", "old", "new", "issue_id"]


@dataclass
class FeatureResult:
    X: pd.DataFrame
    site_qbar: pd.DataFrame
    raw_features: pd.DataFrame
    descriptor_table: pd.DataFrame
    audit_log: pd.DataFrame
    numeric_columns: list[str]


def build_features(df: pd.DataFrame, version: str) -> FeatureResult:
    """Build without mutating raw data; return audit and pre-one-hot filter inputs."""
    if version not in ("V_raw", "V_corr"):
        raise ValueError(f"Unknown data version: {version}")
    if not df.index.is_unique or not np.array_equal(df.index, df.record_id):
        raise ValueError("Index must contain unique record_id values")
    protocol = yaml.safe_load((ROOT / "protocol_v1.yaml").read_text())
    spec = protocol["features"]
    rename = spec["rename"]
    inverse = {value: key for key, value in rename.items()}
    data = df.copy(deep=True)
    audit = []
    if version == "V_corr":
        issues = pd.read_csv(ROOT / "config/known_issues.csv", keep_default_na=False)
        for issue in issues.query('action_V_corr == "correct"').itertuples(index=False):
            fields = issue.field.split(";")
            values = issue.suspected_value.split(";")
            if len(fields) != len(values):
                raise ValueError(f"Malformed correction: {issue.issue_id}")
            for rid in map(int, issue.record_ids.split(";")):
                if rid not in data.index:
                    continue
                for field, value in zip(fields, values):
                    column = inverse.get(field, field)
                    old = data.at[rid, column]
                    new = value if field in spec["one_hot"] else float(value)
                    audit.append(dict(record_id=rid, field=field, old=old, new=new,
                                      issue_id=issue.issue_id))
                    data.at[rid, column] = new

    properties = pd.read_csv(ROOT / "config/dopant_properties.csv").set_index(["element", "site"])
    long = dopant_long(data)
    rows = []
    for rid in data.index:
        for site in ("Li", "La", "Zr"):
            sub = long[(long.record_id == rid) & (long.site == site)]
            weighted = np.zeros(3)
            total = float(sub.x.sum())
            for dopant in sub.itertuples(index=False):
                key = (dopant.element, site)
                if key not in properties.index:
                    raise ValueError(f"Missing dopant properties: {key}, record {rid}")
                if dopant.x < 0:
                    raise ValueError(f"Negative dopant amount: record {rid}")
                weighted += dopant.x * properties.loc[key, [
                    "pauling_en", "shannon_radius_A", "atomic_mass"
                ]].to_numpy(dtype=float)
            mean = weighted / total if total > 0 else np.zeros(3)
            rows.append(dict(record_id=rid, site=site,
                             elements="/".join(sorted(set(sub.element))), x=total,
                             elnv=mean[0], irad=mean[1], A=mean[2]))
    descriptors = pd.DataFrame(rows)
    base_numeric = spec["composition"] + spec["counts"] + spec["synthesis_numeric"]
    raw_columns = [inverse.get(c, c) for c in base_numeric] + spec["one_hot"]
    raw_features = data[raw_columns].copy()
    for column in raw_columns:
        if column not in spec["one_hot"]:
            raw_features[column] = pd.to_numeric(raw_features[column], errors="raise")
    X = raw_features[[inverse.get(c, c) for c in base_numeric]].rename(columns=rename).astype(float)
    for feature in spec["descriptors"]:
        site, _, prop = feature.split("_")
        X[feature] = descriptors[descriptors.site == site].set_index("record_id")[prop].reindex(X.index)
    numeric_columns = list(X.columns)
    for column in spec["one_hot"]:
        categories = raw_features[column].map(
            lambda value: "missing" if pd.isna(value) else str(value).strip()
        )
        for category in sorted(set(categories) | {"missing"}):
            X[f"{column}_{category}"] = (categories == category).astype(int)
    return FeatureResult(X, site_charge_table(data), raw_features, descriptors,
                         pd.DataFrame(audit, columns=AUDIT_COLUMNS), numeric_columns)
