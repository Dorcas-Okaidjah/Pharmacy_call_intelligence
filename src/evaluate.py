"""Score extracted outcomes against gold labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTCOME_FIELDS
from src.data import _norm


def long_table(golds: dict, preds: dict, method: str) -> pd.DataFrame:
    """One row per (call, field) with gold, prediction and correctness."""
    rows = []
    for cid, g in golds.items():
        p = preds.get(cid, {}) or {}
        for f in OUTCOME_FIELDS:
            gv, pv = g.get(f), _norm(p.get(f))
            rows.append({"method": method, "call_id": cid, "field": f, "gold": gv, "pred": pv,
                         "applicable": gv is not None,
                         "correct": (gv == pv) if gv is not None else None,
                         "hallucinated": gv is None and pv is not None})
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0):
    """95% CI resampling whole calls, so correlated fields in one call aren't treated as independent."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_call = {c: values[groups == c] for c in uniq}
    stats = []
    for _ in range(n):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        stats.append(np.concatenate([by_call[c] for c in sample]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-field accuracy table, overall table) for every method in df."""
    app = df[df["applicable"]].copy()
    app["correct"] = app["correct"].astype(float)
    per_field = (app.groupby(["field", "method"])["correct"].agg(["mean", "count"])
                 .rename(columns={"mean": "accuracy", "count": "n"}).reset_index())

    overall = []
    for m, g in df.groupby("method"):
        a = g[g["applicable"]]
        acc = a["correct"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(acc, a["call_id"].to_numpy())
        call_ok = a.groupby("call_id")["correct"].all().mean()
        overall.append({"method": m, "field_accuracy": acc.mean(), "ci95_low": lo, "ci95_high": hi,
                        "calls_fully_correct": call_ok,
                        "hallucination_rate": g.loc[~g["applicable"], "hallucinated"].mean(),
                        "n_calls": g["call_id"].nunique()})
    return per_field, pd.DataFrame(overall)
