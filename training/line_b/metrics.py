"""Unified multilabel metrics for Line B.

All metrics are computed from frozen probability arrays.  AP/AUROC are
threshold-free; Precision/Recall/F1 use the caller-provided threshold
(0.5 by default).  Single-class labels yield `None` for AP/AUROC and are
excluded from the Macro AP/AUROC means, with counts reported.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_2d(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true and y_prob must be 2-D")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_prob.shape}")
    return y_true.astype(np.int64), y_prob.astype(np.float64)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Iterable[str],
    threshold: float = 0.5,
) -> dict:
    y_true, y_prob = _as_2d(y_true, y_prob)
    names = list(label_names)
    if len(names) != y_true.shape[1]:
        raise ValueError(f"{len(names)} names for {y_true.shape[1]} columns")

    y_pred = (y_prob >= threshold).astype(np.int64)

    per_label: list[dict] = []
    ap_values: list[float] = []
    auroc_values: list[float] = []

    for index, name in enumerate(names):
        truth = y_true[:, index]
        prob = y_prob[:, index]
        pred = y_pred[:, index]

        support = int(truth.sum())
        tp = int(np.sum((truth == 1) & (pred == 1)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        tn = int(np.sum((truth == 0) & (pred == 0)))

        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, pred, average="binary", zero_division=0
        )

        unique = np.unique(truth)
        if unique.size == 2:
            ap = float(average_precision_score(truth, prob))
            auroc = float(roc_auc_score(truth, prob))
            ap_values.append(ap)
            auroc_values.append(auroc)
        else:
            ap = None
            auroc = None

        per_label.append(
            {
                "label_index": index,
                "label": name,
                "support": support,
                "prevalence": float(support / len(truth)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "average_precision": ap,
                "auroc": auroc,
                "ap_defined": ap is not None,
                "auroc_defined": auroc is not None,
            }
        )

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p = float(np.mean([row["precision"] for row in per_label]))
    macro_r = float(np.mean([row["recall"] for row in per_label]))
    macro_f1 = float(np.mean([row["f1"] for row in per_label]))

    undefined_ap = [row["label"] for row in per_label if row["average_precision"] is None]
    undefined_auroc = [row["label"] for row in per_label if row["auroc"] is None]

    return {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "threshold": float(threshold),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "n_defined_ap": len(ap_values),
        "n_defined_auroc": len(auroc_values),
        "undefined_ap_labels": undefined_ap,
        "undefined_auroc_labels": undefined_auroc,
        "per_label": per_label,
    }
