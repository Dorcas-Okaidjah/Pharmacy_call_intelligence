"""Turn-level call-phase classifier (classical baseline).

Question it answers: can a cheap, fast model tag each turn of a call with its phase
(IVR, patient ID, coverage check, formulary, restrictions, copay ...)? Train on the 1,000
synthetic calls, test on the 50 real calls, and compare with grouped cross-validation
inside the synthetic set. The gap between the two is the synthetic-to-real domain shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from src.config import PHASES


def build_model() -> Pipeline:
    features = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "content"),
        ("prev", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=5000), "prev_content"),
        ("role", OneHotEncoder(handle_unknown="ignore"), ["role"]),
        ("pos", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="uniform"), ["rel_pos"]),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    return Pipeline([("features", features), ("clf", clf)])


def grouped_cv_f1(df: pd.DataFrame, n_splits: int = 5) -> tuple[float, float]:
    """Macro-F1 with calls kept whole within a fold (no turn-level leakage)."""
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(df, df["phase"], df["call_id"]):
        m = build_model().fit(df.iloc[tr], df.iloc[tr]["phase"])
        scores.append(f1_score(df.iloc[te]["phase"], m.predict(df.iloc[te]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def train_and_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    model = build_model().fit(train_df, train_df["phase"])
    pred = model.predict(test_df)
    labels = [p for p in PHASES if p in set(test_df["phase"]) | set(pred)]
    return {
        "macro_f1": f1_score(test_df["phase"], pred, average="macro"),
        "report": classification_report(test_df["phase"], pred, labels=labels,
                                        output_dict=True, zero_division=0),
        "confusion": pd.DataFrame(confusion_matrix(test_df["phase"], pred, labels=labels),
                                  index=labels, columns=labels),
        "predictions": test_df.assign(pred=pred),
    }
