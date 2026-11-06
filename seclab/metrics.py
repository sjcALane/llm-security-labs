"""Evaluation metrics for a phishing detector.

Positive class = phishing (label 1). A detector outputs p, the
probability that an email is phishing, and flags it when p >= threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate(y_true, p, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, false positive rate and ROC AUC.

    recall = TP / (TP + FN): share of phishing emails we catch.
    FPR    = FP / (FP + TN): share of safe emails we wrongly block.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(y_true, p) if len(set(y_true.tolist())) == 2 else float("nan")
    return {
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn) if fp + tn else 0.0),
        "auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def attack_success_rate(y_true, p_clean, p_attacked, threshold: float = 0.5) -> float:
    """Share of phishing emails the detector caught that the attack lets through.

    We only count phishing emails that were flagged before the attack.
    An attack cannot "succeed" on an email the detector already missed.
    """
    y_true = np.asarray(y_true)
    caught = (y_true == 1) & (np.asarray(p_clean) >= threshold)
    if caught.sum() == 0:
        return float("nan")
    evaded = caught & (np.asarray(p_attacked) < threshold)
    return float(evaded.sum() / caught.sum())


def table(rows: dict, keys=("accuracy", "precision", "recall", "f1", "fpr", "auc")):
    """Format {name: metrics} as a small pandas table.

    Entries that are not full metric dictionaries (for example a single
    attack success rate) are skipped.
    """
    import pandas as pd

    rows = {n: m for n, m in rows.items() if isinstance(m, dict) and all(k in m for k in keys)}
    return pd.DataFrame({name: {k: m[k] for k in keys} for name, m in rows.items()}).T.round(3)
