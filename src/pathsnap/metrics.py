"""Evaluation metrics for candidate classification and route reconstruction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Binary metrics over the (observation, candidate) rows."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    try:
        out["log_loss"] = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def top1_accuracy(df: pd.DataFrame, score_col: str = "score") -> float:
    """Per observation, does the highest-scoring candidate match the reference arc?"""
    idx = df.groupby(["trip_id", "obs_idx"])[score_col].idxmax()
    return float(df.loc[idx, "label"].mean())


def top1_predictions(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    idx = df.groupby(["trip_id", "obs_idx"])[score_col].idxmax()
    return df.loc[idx, ["trip_id", "obs_idx", "arc_id", score_col]].sort_values(
        ["trip_id", "obs_idx"]).reset_index(drop=True)


def route_continuity(arc_sequence, network) -> float:
    """Fraction of consecutive arc pairs that are actually connected.

    Identical consecutive arcs count as continuous (the vehicle stayed put);
    otherwise the head node of one arc must be the tail node of the next.
    """
    arcs = np.asarray(arc_sequence)
    arcs = arcs[arcs >= 0]
    if len(arcs) < 2:
        return 1.0
    ok = 0
    for a, b in zip(arcs[:-1], arcs[1:]):
        if a == b or network.arcs[a, 1] == network.arcs[b, 0]:
            ok += 1
    return ok / (len(arcs) - 1)


def sequence_metrics(pred_arcs, gt_arcs) -> dict:
    """Sequence-level agreement between predicted and reference arc sequences."""
    pred, gt = np.asarray(pred_arcs), np.asarray(gt_arcs)
    valid = (pred >= 0) & (gt >= 0)
    pointwise = float((pred[valid] == gt[valid]).mean()) if valid.any() else float("nan")

    def collapse(a):
        a = a[a >= 0]
        return a[np.concatenate([[True], a[1:] != a[:-1]])] if len(a) else a

    p_set, g_set = set(collapse(pred).tolist()), set(collapse(gt).tolist())
    inter = len(p_set & g_set)
    return {
        "pointwise_accuracy": pointwise,
        "arc_set_precision": inter / len(p_set) if p_set else float("nan"),
        "arc_set_recall": inter / len(g_set) if g_set else float("nan"),
        "arc_set_jaccard": inter / len(p_set | g_set) if (p_set | g_set) else float("nan"),
    }
