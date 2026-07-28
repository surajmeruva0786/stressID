"""Stage 6 metrics: accuracy / F1 / per-class recall / calibration (ECE) / bootstrap CI."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, roc_auc_score)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> tuple[float, list]:
    """ECE for binary predictions, plus reliability-diagram bins (E13)."""
    conf = np.where(probs >= 0.5, probs, 1.0 - probs)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "acc": None, "conf": None})
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        ece += (m.sum() / len(probs)) * abs(acc - cf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": acc, "conf": cf})
    return float(ece), bins


def binary_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    pred = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(labels, pred, zero_division=0)),
        "recall_0": float(recall_score(labels, pred, pos_label=0, zero_division=0)),
        "recall_1": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        "n": int(len(labels)),
    }
    out["auc"] = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
    out["ece"], out["reliability"] = expected_calibration_error(probs, labels, n_bins)
    return out


def affect3_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    rec = recall_score(labels, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "recall_class0": float(rec[0]), "recall_class1": float(rec[1]),
        "recall_class2": float(rec[2]),
        "confusion": confusion_matrix(labels, pred, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean + percentile CI over per-fold / per-seed scores."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                            seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value for mean(a) - mean(b) != 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    centred = boot - d.mean()
    p = float((np.abs(centred) >= abs(d.mean())).mean())
    return min(max(p, 1.0 / n_boot), 1.0)
