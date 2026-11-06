"""Load INSURE-Dial calls and turn their annotations into evaluation labels.

Usage:
    python -m src.data --download      # pull from Hugging Face into data/real and data/synthetic
    python -m src.data --inspect       # print one call's structure to sanity-check the schema
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, HF_DATASET, PHASES, TURN_KEY


# --------------------------------------------------------------------------- download
def download_from_hf() -> None:
    """Save each call as one JSON file, the layout the official INSURE-Dial scripts use."""
    from datasets import load_dataset  # imported here so the rest works without it

    ds = load_dataset(HF_DATASET)
    for split in ("real", "synthetic"):
        out = DATA_DIR / split
        out.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(ds[split]):
            with open(out / f"{split}_{i:04d}.json", "w") as f:
                json.dump({"script": rec["script"], "annotation": rec["annotation"]}, f)
        print(f"saved {len(ds[split])} calls to {out}")


# --------------------------------------------------------------------------- load
def load_calls(split: str, limit: int | None = None) -> list[dict]:
    files = sorted((DATA_DIR / split).glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files in {DATA_DIR / split}. Run `python -m src.data --download` first."
        )
    calls = []
    for fp in files[:limit] if limit else files:
        rec = json.loads(fp.read_text())
        ann = rec["annotation"]
        if isinstance(ann, str):  # some exports store annotation as a JSON string
            ann = json.loads(ann)
        calls.append({"call_id": fp.stem, "script": rec["script"], "annotation": ann})
    return calls


def turn_number(turn: dict, position: int) -> int:
    return turn.get(TURN_KEY, position) if TURN_KEY else position


# --------------------------------------------------------------------------- gold outcomes
def _get(d, *path):
    for key in path:
        try:
            d = d[key]
        except (KeyError, IndexError, TypeError):
            return None
    return d


def _norm(v):
    """Normalise yes/no/status values so 'Yes', 'YES', True all compare equal."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().upper().replace(" ", "_")
    if s in ("", "NA", "N/A", "NONE", "NULL"):
        return None
    return {"TRUE": True, "FALSE": False}.get(s, s)


def gold_outcomes(ann: dict) -> dict:
    """Ground-truth outcome fields, applying the dataset's own business rules:
    coverage only if the record was found; restrictions only if formulary = YES."""
    record_found = _norm(_get(ann, "patient_info", "record_found"))
    status = None
    if record_found:
        ps = _norm(_get(ann, "coverage_check", "plan_status"))
        status = ps if ps in ("ACTIVE", "INACTIVE") else "NOT_ANSWERED"
    out = {"record_found": record_found, "coverage_status": status}
    for i in (0, 1):
        form = _norm(_get(ann, "drugs", i, "formulary", "value"))
        rest = _norm(_get(ann, "drugs", i, "restrictions", "value"))
        out[f"drug{i+1}_formulary"] = form
        out[f"drug{i+1}_restrictions"] = rest if form == "YES" else None
    return out


# --------------------------------------------------------------------------- phase labels
def _span(ann, *path):
    s = _get(ann, *path, "turn_range")
    if not (isinstance(s, list) and len(s) == 2) or s == [0, 0]:
        return None
    return s


def phase_spans(ann: dict) -> list[tuple[str, list[int]]]:
    """(phase, [start, end]) pairs in call order. Drug phases are pooled across drugs."""
    found = _norm(_get(ann, "patient_info", "record_found")) is True
    spans = [(p, _span(ann, p)) for p in ("ivr", "greeting", "patient_info")]
    spans.append(("coverage_check", _span(ann, "coverage_check") if found else None))
    for i in (0, 1):
        form = _norm(_get(ann, "drugs", i, "formulary", "value"))
        rest = _norm(_get(ann, "drugs", i, "restrictions", "value"))
        spans.append(("drug_formulary", _span(ann, "drugs", i, "formulary")))
        spans.append(("drug_restrictions", _span(ann, "drugs", i, "restrictions") if form == "YES" else None))
        spans.append(("drug_copay", _span(ann, "drugs", i, "copay") if form == "YES" and rest == "NO" else None))
    spans.append(("agent_interaction", _span(ann, "agent_interaction")))
    return [(p, s) for p, s in spans if s is not None]


def turn_frame(calls: list[dict]) -> pd.DataFrame:
    """One row per turn with its gold phase. Where phases overlap by a turn
    (allowed by the guidelines), the later phase wins."""
    rows = []
    for c in calls:
        spans = phase_spans(c["annotation"])
        n = len(c["script"])
        for pos, t in enumerate(c["script"]):
            tn = turn_number(t, pos)
            label = "other"
            for phase, (a, b) in spans:
                if a <= tn <= b:
                    label = phase
            prev = c["script"][pos - 1]["content"] if pos > 0 else ""
            rows.append({
                "call_id": c["call_id"], "pos": pos, "rel_pos": pos / max(n - 1, 1),
                "role": t.get("role", ""), "content": t.get("content", ""),
                "prev_content": prev, "phase": label,
            })
    df = pd.DataFrame(rows)
    assert set(df["phase"]).issubset(PHASES)
    return df


# --------------------------------------------------------------------------- CLI
def _inspect():
    for split in ("real", "synthetic"):
        try:
            c = load_calls(split, limit=1)[0]
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"\n=== {split}: {c['call_id']} ({len(c['script'])} turns)")
        print("first turns:", json.dumps(c["script"][:3], indent=1)[:600])
        print("annotation keys:", list(c["annotation"].keys()))
        print("annotation:", json.dumps(c["annotation"], indent=1)[:1500])
        print("gold outcomes:", gold_outcomes(c["annotation"]))
        print("phase spans:", phase_spans(c["annotation"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    if a.download:
        download_from_hf()
    if a.inspect:
        _inspect()
