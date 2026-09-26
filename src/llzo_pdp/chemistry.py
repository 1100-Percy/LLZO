"""Stoichiometry utilities for doped LLZO (nominal composition model).

Nominal model (declared assumption, see protocol_v1.yaml -> chemistry):
    Li_{Li} + Σ(Li-site dopants) | La_{La} + Σ(La-site dopants) | Zr_{Zr} + Σ(Zr-site dopants) | O12
Charge balance (O fixed at 12):
    Li + 3*La + 4*Zr + Σ_d q_d * x_d = 24
Site sums:
    Zr + Zr_site_sto = 2,  La + La_site_sto = 3
Li-site occupancy is NOT checked (Li is not a fully occupied site in garnet).

Feature-level residual (used on PDP virtual inputs, where only site totals exist):
    r = Li_comp + 3*La_comp + 4*Zr_comp
        + qbar_Li*Li_site_sto + qbar_La*La_site_sto + qbar_Zr*Zr_site_sto - 24
qbar_<site> = mean oxidation state of the dopants on that site in the *background*
record (dopant identity and co-dopant proportions are held fixed when a PDP changes
a site total). This is a declared modelling choice, not a fact about the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Declared oxidation states (nominal). Changing any value is a protocol change.
OX_STATE = {
    "Ga": 3, "Al": 3, "Fe": 3, "Zn": 2,             # Li-site dopants in this dataset
    "Ta": 5, "Nb": 5, "Sb": 5, "Bi": 5, "W": 6, "Te": 6,
    "Mg": 2, "Sc": 3, "Ti": 4, "Gd": 3, "Sm": 3,     # Zr-site dopants
    "Y": 3, "Ba": 2, "Sr": 2,                        # La-site (Y also appears on Zr site)
}
TOL = 0.05  # |residual| below this counts as consistent (formula units)

SITE_COLS = {"Li": "Li_dop", "La": "La_dop", "Zr": "Zr_dop"}
STO_COLS = {"Li": "Li_site_sto", "La": "La_site_sto", "Zr": "Zr_site_sto"}
COMP_COLS = {"Li": "Li Composition", "La": "La_comp", "Zr": "Zr_comp"}
HOST_Q = {"Li": 1, "La": 3, "Zr": 4}


def _site_members(v) -> set[str]:
    if not isinstance(v, str):
        return set()
    return {p.strip() for p in v.split("/") if p.strip() and p.strip() != "0"}


def dopant_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (record, dopant) with its assigned site and charge."""
    rows = []
    for rid, r in df.iterrows():
        members = {s: _site_members(r.get(c)) for s, c in SITE_COLS.items()}
        for k in (1, 2, 3):
            el, x = r.get(f"dopant_{k}"), r.get(f"x_{k}")
            if not isinstance(el, str) or el in ("Nil", "") or pd.isna(x):
                continue
            sites = [s for s, m in members.items() if el in m]
            site = sites[0] if len(sites) == 1 else ("ambiguous" if sites else "unassigned")
            rows.append({"record_id": rid, "element": el, "x": float(x), "site": site,
                         "q": OX_STATE.get(el, np.nan)})
    return pd.DataFrame(rows, columns=["record_id", "element", "x", "site", "q"])


def site_charge_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per record: Q_<site> = Σ q*x on that site, qbar_<site> = Q/Σx (NaN if undoped)."""
    long = dopant_long(df)
    out = pd.DataFrame(index=df.index)
    for s in ("Li", "La", "Zr"):
        sub = long[long.site == s]
        Q = (sub.q * sub.x).groupby(sub.record_id).sum()
        X = sub.x.groupby(sub.record_id).sum()
        out[f"Q_{s}"] = Q.reindex(df.index).fillna(0.0)
        out[f"X_{s}"] = X.reindex(df.index).fillna(0.0)
        out[f"qbar_{s}"] = (Q / X).reindex(df.index)
    out["n_unassigned"] = long[long.site.isin(["unassigned", "ambiguous"])].groupby("record_id").size().reindex(df.index).fillna(0).astype(int)
    return out


def charge_residual_raw(df: pd.DataFrame) -> pd.Series:
    """Li + 3La + 4Zr + Σ q x - 24 using every listed dopant (site-independent)."""
    long = dopant_long(df)
    dq = (long.q * long.x).groupby(long.record_id).sum().reindex(df.index).fillna(0.0)
    return df["Li Composition"] + 3 * df["La_comp"] + 4 * df["Zr_comp"] + dq - 24


def charge_residual_features(X: pd.DataFrame, qbar: pd.DataFrame) -> pd.Series:
    """Residual on feature-level rows. X uses model feature names
    (Li_comp, La_comp, Zr_comp, *_site_sto); qbar has qbar_Li/La/Zr aligned to X.index."""
    r = X["Li_comp"] + 3 * X["La_comp"] + 4 * X["Zr_comp"] - 24
    for s in ("Li", "La", "Zr"):
        r = r + qbar[f"qbar_{s}"].fillna(0.0) * X[f"{s}_site_sto"]
    return r


def site_residuals(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "r_site_Zr": X["Zr_comp"] + X["Zr_site_sto"] - 2,
        "r_site_La": X["La_comp"] + X["La_site_sto"] - 3,
    }, index=X.index)


def linked_update(row: pd.Series, site: str, x_new: float, qbar: float) -> pd.Series:
    """Change the total dopant amount on `site` to x_new while keeping charge balance.

    Li site (dopant charge q): ΔLi = -q Δx
    Zr site:                    ΔZr = -Δx,  ΔLi = -(q-4) Δx
    La site:                    ΔLa = -Δx,  ΔLi = -(q-3) Δx
    Only site totals / host compositions are changed here; derived descriptor
    features (elnv, irad, A, one-hots, # of Dopants) must be recomputed by the
    caller from the (unchanged) dopant identities.
    """
    z = row.copy()
    dx = x_new - z[f"{site}_site_sto"]
    z[f"{site}_site_sto"] = x_new
    if site == "Li":
        z["Li_comp"] = z["Li_comp"] - qbar * dx
    else:
        host = "Zr_comp" if site == "Zr" else "La_comp"
        z[host] = z[host] - dx
        z["Li_comp"] = z["Li_comp"] - (qbar - HOST_Q[site]) * dx
    return z


def li_coefficient(site: str, qbar: float) -> float:
    """dLi/dx along the linked path for a dopant of mean charge qbar on `site`."""
    return {"Li": -qbar, "Zr": -(qbar - 4), "La": -(qbar - 3)}[site]
