"""Validation-only per-label F1 threshold calibration for Line B.

Implements the B0 v0.2 secondary threshold policy:

* Candidate thresholds are the observed-probability thresholds returned by
  ``sklearn.metrics.precision_recall_curve`` (no fixed 0.01 grid).
* The threshold maximizing F1 is selected; ``np.nanargmax`` selects the first
  candidate on an exact tie.  Since PR-curve thresholds are ascending, this is
  the smallest threshold on a tie.
* Single-class labels (all positive or all negative) are undefined for
  threshold optimization: fallback to 0.5 and record status ``undefined``.
* Validation-F1 thresholds are fitted on the same Validation split used for the
  fixed 0.5 metrics; their F1 is in-sample optimistic, not an independent Test
  result.
* No Test indices, Test probabilities, Test metrics, BN recalibration, or model
  inference are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    atomic_output_dir,
    expected_seed_from_run_name,
    load_label_names,
    relative_to_root,
    resolve_label_set,
    resolve_task_runs,
    sha256_array,
    sha256_file,
    validate_manifest_checks,
    validate_prediction_arrays,
    write_json,
)

DEFAULT_THRESHOLD = 0.5
METRIC_NAMES = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_ap",
    "macro_auroc",
)
THRESHOLD_RECORD_FIELDS = (
    "run_name",
    "task",
    "label_set",
    "modality",
    "seed",
    "label_index",
    "label",
    "support",
    "default_threshold",
    "calibrated_threshold",
    "validation_best_f1",
    "threshold_status",
    "tied_max_count",
)
PER_LABEL_FIELDS = (
    "run_name",
    "task",
    "label_set",
    "modality",
    "seed",
    "label_index",
    "label",
    "support",
    "prevalence",
    "average_precision",
    "auroc",
    "ap_defined",
    "auroc_defined",
    "default_threshold",
    "calibrated_threshold",
    "threshold_status",
    "tied_max_count",
    "default_precision",
    "default_recall",
    "default_f1",
    "calibrated_precision",
    "calibrated_recall",
    "calibrated_f1",
    "f1_change",
)


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_sd(values: list[float]):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def optimize_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Fit one label's F1 threshold using the historical PR-curve rule."""
    truth = np.asarray(y_true).astype(np.int64)
    probability = np.asarray(probs, dtype=np.float64)
    n_pos = int(truth.sum())
    if n_pos == 0 or n_pos == len(truth):
        return {
            "threshold": float(DEFAULT_THRESHOLD),
            "validation_best_f1": None,
            "status": "undefined_single_class",
            "tied_max_count": 0,
        }

    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if len(thresholds) == 0:
        return {
            "threshold": float(DEFAULT_THRESHOLD),
            "validation_best_f1": None,
            "status": "undefined_no_candidate",
            "tied_max_count": 0,
        }

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    if not np.isfinite(f1).any():
        return {
            "threshold": float(DEFAULT_THRESHOLD),
            "validation_best_f1": None,
            "status": "undefined_nonfinite_f1",
            "tied_max_count": 0,
        }
    best_f1 = float(np.nanmax(f1))
    # np.nanargmax returns the first maximum; thresholds are ascending.
    best_index = int(np.nanargmax(f1))
    tied = np.flatnonzero(f1 == best_f1)
    return {
        "threshold": float(thresholds[best_index]),
        "validation_best_f1": best_f1,
        "status": "defined",
        "tied_max_count": int(len(tied)),
    }


def metrics_at_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> dict:
    truth = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    if threshold_array.ndim == 0:
        threshold_array = np.full(truth.shape[1], float(threshold_array))
    if threshold_array.shape != (truth.shape[1],):
        raise ValueError(f"Threshold shape {threshold_array.shape} != labels {truth.shape[1]}")
    predictions = (probs >= threshold_array[None, :]).astype(np.int64)

    per_label = []
    ap_values = []
    auroc_values = []
    for index, name in enumerate(label_names):
        label_truth = truth[:, index]
        label_pred = predictions[:, index]
        label_prob = probs[:, index]
        support = int(label_truth.sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_truth, label_pred, average="binary", zero_division=0
        )
        if np.unique(label_truth).size == 2:
            ap = float(average_precision_score(label_truth, label_prob))
            auroc = float(roc_auc_score(label_truth, label_prob))
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
                "prevalence": float(support / len(label_truth)),
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
        truth, predictions, average="micro", zero_division=0
    )
    return {
        "n_samples": int(len(truth)),
        "n_labels": int(truth.shape[1]),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean([row["precision"] for row in per_label])),
        "macro_recall": float(np.mean([row["recall"] for row in per_label])),
        "macro_f1": float(np.mean([row["f1"] for row in per_label])),
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "per_label": per_label,
    }


def _manifest_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_model_record(
    manifest_path: Path,
    label_set: str,
    expected_names: list[str],
) -> dict:
    from training.line_b.common import read_json

    manifest = read_json(manifest_path)
    label_config = resolve_label_set(label_set)
    if manifest.get("task") != label_config["task"]:
        raise ValueError(f"Unexpected task in {manifest_path}")
    if manifest.get("label_set") != label_set:
        raise ValueError(f"Unexpected label_set in {manifest_path}")
    if manifest.get("split") != "validation":
        raise ValueError(f"Unexpected split in {manifest_path}")
    validate_manifest_checks(manifest, str(manifest_path))

    run_name = manifest.get("run_name")
    pred_dir = manifest_path.parent
    if run_name != pred_dir.parent.name:
        raise ValueError(f"run_name does not match directory: {manifest_path}")
    modality = manifest.get("modality")
    expected_seed = expected_seed_from_run_name(run_name)
    checkpoint_config = manifest.get("checkpoint", {}).get("config") or {}
    seed_config = checkpoint_config.get("seed")
    if seed_config is None or int(seed_config) != expected_seed:
        raise ValueError(
            f"checkpoint seed {seed_config} != expected {expected_seed} in {manifest_path}"
        )

    probabilities, row_index, labels = validate_prediction_arrays(
        pred_dir, manifest, modality, label_set, expected_names
    )

    checkpoint = manifest.get("checkpoint") or {}
    checkpoint_path = _manifest_path(checkpoint.get("path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_actual_sha = sha256_file(checkpoint_path)
    expected_checkpoint_sha = checkpoint.get("sha256_after") or checkpoint.get("sha256")
    if expected_checkpoint_sha != checkpoint_actual_sha:
        raise ValueError(f"Checkpoint SHA256 mismatch for {run_name}")

    split_file = manifest.get("split_file") or {}
    split_path = _manifest_path(split_file.get("path", ""))
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if split_file.get("sha256") != sha256_file(split_path):
        raise ValueError(f"Split SHA256 mismatch for {run_name}")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "pred_dir": pred_dir,
        "run_name": run_name,
        "modality": modality,
        "seed": expected_seed,
        "seed_config": int(seed_config),
        "probabilities": probabilities,
        "row_index": row_index,
        "labels": labels,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_actual_sha,
        "split_path": split_path,
        "label_names": expected_names,
    }


def _compute_task(
    label_set: str,
    predictions_root: Path,
    allow_partial: bool,
) -> dict:
    label_config = resolve_label_set(label_set)
    task = label_config["task"]
    expected_runs = resolve_task_runs(label_set)
    task_root = Path(predictions_root) / task
    manifests = sorted(task_root.glob("*/*/validation/prediction_manifest.json"))
    expected_count = sum(len(runs) for runs in expected_runs.values())
    if not allow_partial and len(manifests) != expected_count:
        raise ValueError(f"{task}: expected {expected_count} manifests, found {len(manifests)}")
    if not manifests:
        raise FileNotFoundError(f"{task}: no prediction manifests under {task_root}")

    expected_names = load_label_names(label_config["names"])
    records = []
    seen = set()
    for manifest_path in manifests:
        record = _load_model_record(manifest_path, label_set, expected_names)
        if record["run_name"] in seen:
            raise ValueError(f"Duplicate run record: {record['run_name']}")
        seen.add(record["run_name"])
        if record["modality"] not in expected_runs:
            raise ValueError(f"Unknown modality: {record['modality']}")
        if record["run_name"] not in expected_runs[record["modality"]]:
            raise ValueError(f"Run not registered: {record['run_name']}")
        records.append(record)

    if not allow_partial:
        for modality, run_names in expected_runs.items():
            actual = {r["run_name"] for r in records if r["modality"] == modality}
            if actual != set(run_names):
                raise ValueError(f"{task}/{modality}: registered run mismatch")
        if len(records) != expected_count:
            raise ValueError(f"{task}: expected {expected_count} unique records")
    return {
        "task": task,
        "label_set": label_set,
        "label_config": label_config,
        "records": records,
        "partial": bool(allow_partial),
        "expected_count": expected_count,
        "task_root": task_root,
    }


def _calibrate_task(task_info: dict, out_root: Path) -> dict:
    task = task_info["task"]
    label_set = task_info["label_set"]
    label_config = task_info["label_config"]
    threshold_rows = []
    comparison_rows = []
    model_rows = []
    model_summaries = []
    undefined_labels = 0
    undefined_records = []

    task_out = Path(out_root) / task
    with atomic_output_dir(task_out) as temporary:
        for record in task_info["records"]:
            run_name = record["run_name"]
            modality = record["modality"]
            seed = record["seed"]
            probabilities = record["probabilities"]
            truth = record["labels"]
            names = record["label_names"]
            thresholds = []
            threshold_fits = []
            for index, name in enumerate(names):
                fit = optimize_threshold(truth[:, index], probabilities[:, index])
                thresholds.append(fit["threshold"])
                threshold_fits.append({**fit, "label": name, "label_index": index})
                threshold_rows.append(
                    {
                        "run_name": run_name,
                        "task": task,
                        "label_set": label_set,
                        "modality": modality,
                        "seed": seed,
                        "label_index": index,
                        "label": name,
                        "support": int(truth[:, index].sum()),
                        "default_threshold": DEFAULT_THRESHOLD,
                        "calibrated_threshold": fit["threshold"],
                        "validation_best_f1": fit["validation_best_f1"],
                        "threshold_status": fit["status"],
                        "tied_max_count": fit["tied_max_count"],
                    }
                )
                if fit["status"] != "defined":
                    undefined_labels += 1
                    undefined_records.append(
                        {
                            "run_name": run_name,
                            "modality": modality,
                            "seed": seed,
                            "label": name,
                            "status": fit["status"],
                        }
                    )

            thresholds = np.asarray(thresholds, dtype=np.float64)
            default_metrics = metrics_at_thresholds(truth, probabilities, DEFAULT_THRESHOLD, names)
            calibrated_metrics = metrics_at_thresholds(truth, probabilities, thresholds, names)

            for default_row, calibrated_row, fit in zip(
                default_metrics["per_label"], calibrated_metrics["per_label"], threshold_fits
            ):
                comparison_rows.append(
                    {
                        "run_name": run_name,
                        "task": task,
                        "label_set": label_set,
                        "modality": modality,
                        "seed": seed,
                        "label_index": default_row["label_index"],
                        "label": default_row["label"],
                        "support": default_row["support"],
                        "prevalence": default_row["prevalence"],
                        "average_precision": default_row["average_precision"],
                        "auroc": default_row["auroc"],
                        "ap_defined": default_row["ap_defined"],
                        "auroc_defined": default_row["auroc_defined"],
                        "default_threshold": DEFAULT_THRESHOLD,
                        "calibrated_threshold": fit["threshold"],
                        "threshold_status": fit["status"],
                        "tied_max_count": fit["tied_max_count"],
                        "default_precision": default_row["precision"],
                        "default_recall": default_row["recall"],
                        "default_f1": default_row["f1"],
                        "calibrated_precision": calibrated_row["precision"],
                        "calibrated_recall": calibrated_row["recall"],
                        "calibrated_f1": calibrated_row["f1"],
                        "f1_change": calibrated_row["f1"] - default_row["f1"],
                    }
                )

            thresholds_payload = {
                "phase": "B3",
                "task": task,
                "label_set": label_set,
                "run_name": run_name,
                "modality": modality,
                "seed": seed,
                "split": "validation",
                "default_threshold": DEFAULT_THRESHOLD,
                "primary_threshold": DEFAULT_THRESHOLD,
                "secondary_threshold": "validation_f1_per_label",
                "calibration_method": (
                    "per-label F1 maximization on validation; "
                    "precision_recall_curve observed-probability candidates; "
                    "first/smallest threshold on exact ties"
                ),
                "validation_only": True,
                "in_sample_validation_f1": True,
                "test_split_accessed": False,
                "frozen_before_test": True,
                "thresholds": {name: float(thresholds[i]) for i, name in enumerate(names)},
                "validation_best_f1": {
                    fit["label"]: fit["validation_best_f1"] for fit in threshold_fits
                },
                "threshold_status": {
                    fit["label"]: fit["status"] for fit in threshold_fits
                },
                "tied_max_count": {
                    fit["label"]: fit["tied_max_count"] for fit in threshold_fits
                },
                "label_names": names,
                "probabilities_sha256": sha256_array(probabilities),
                "row_index_sha256": sha256_array(record["row_index"]),
                "labels_sha256": sha256_array(truth),
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
            overall_payload = {
                "phase": "B3",
                "task": task,
                "label_set": label_set,
                "run_name": run_name,
                "modality": modality,
                "seed": seed,
                "split": "validation",
                "n_samples": int(len(truth)),
                "n_labels": int(len(names)),
                "default_threshold": DEFAULT_THRESHOLD,
                "primary_threshold": DEFAULT_THRESHOLD,
                "secondary_threshold": "validation_f1_per_label",
                "default": default_metrics,
                "calibrated": calibrated_metrics,
                "validation_f1_note": (
                    "Validation-F1 thresholds are fitted on the same Validation split; "
                    "their F1 is in-sample optimistic and is not a Test estimate."
                ),
                "test_split_accessed": False,
            }

            model_dir = temporary / run_name
            model_dir.mkdir()
            write_json(model_dir / "optimal_thresholds.json", thresholds_payload)
            write_json(model_dir / "validation_overall.json", overall_payload)
            _write_csv(
                model_dir / "validation_per_label_comparison.csv",
                [row for row in comparison_rows if row["run_name"] == run_name],
                PER_LABEL_FIELDS,
            )
            calibration_manifest = {
                "task": task,
                "label_set": label_set,
                "run_name": run_name,
                "modality": modality,
                "seed": seed,
                "split": "validation",
                "test_split_accessed": False,
                "calibration_split": "validation",
                "in_sample_validation_f1": True,
                "inputs": {
                    "prediction_manifest": {
                        "path": relative_to_root(record["manifest_path"]),
                        "sha256": sha256_file(record["manifest_path"]),
                    },
                    "probabilities": {
                        "path": relative_to_root(record["pred_dir"] / "probabilities.npy"),
                        "sha256": sha256_file(record["pred_dir"] / "probabilities.npy"),
                    },
                    "row_index": {
                        "path": relative_to_root(record["pred_dir"] / "row_index.npy"),
                        "sha256": sha256_file(record["pred_dir"] / "row_index.npy"),
                    },
                    "labels": {
                        "path": relative_to_root(record["pred_dir"] / "labels.npy"),
                        "sha256": sha256_file(record["pred_dir"] / "labels.npy"),
                    },
                    "checkpoint": {
                        "path": relative_to_root(record["checkpoint_path"]),
                        "sha256": record["checkpoint_sha256"],
                    },
                    "split": {
                        "path": relative_to_root(record["split_path"]),
                        "sha256": sha256_file(record["split_path"]),
                    },
                    "label_names": {
                        "path": relative_to_root(label_config["names"]),
                        "sha256": sha256_file(label_config["names"]),
                    },
                    "label_matrix": {
                        "path": relative_to_root(label_config["labels"]),
                        "sha256": sha256_file(label_config["labels"]),
                    },
                },
                "outputs": {
                    "optimal_thresholds": sha256_file(model_dir / "optimal_thresholds.json"),
                    "validation_overall": sha256_file(model_dir / "validation_overall.json"),
                    "validation_per_label_comparison": sha256_file(
                        model_dir / "validation_per_label_comparison.csv"
                    ),
                },
                "all_checks_pass": True,
            }
            write_json(model_dir / "calibration_manifest.json", calibration_manifest)

            model_row = {
                "run_name": run_name,
                "task": task,
                "label_set": label_set,
                "modality": modality,
                "seed": seed,
                "n_samples": len(truth),
                "n_labels": len(names),
                "default_threshold": DEFAULT_THRESHOLD,
                "calibrated_thresholds_sha256": sha256_file(model_dir / "optimal_thresholds.json"),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "probabilities_sha256": sha256_array(probabilities),
                "row_index_sha256": sha256_array(record["row_index"]),
                "labels_sha256": sha256_array(truth),
                "default_micro_precision": default_metrics["micro_precision"],
                "default_micro_recall": default_metrics["micro_recall"],
                "default_micro_f1": default_metrics["micro_f1"],
                "default_macro_precision": default_metrics["macro_precision"],
                "default_macro_recall": default_metrics["macro_recall"],
                "default_macro_f1": default_metrics["macro_f1"],
                "default_macro_ap": default_metrics["macro_ap"],
                "default_macro_auroc": default_metrics["macro_auroc"],
                "calibrated_micro_precision": calibrated_metrics["micro_precision"],
                "calibrated_micro_recall": calibrated_metrics["micro_recall"],
                "calibrated_micro_f1": calibrated_metrics["micro_f1"],
                "calibrated_macro_precision": calibrated_metrics["macro_precision"],
                "calibrated_macro_recall": calibrated_metrics["macro_recall"],
                "calibrated_macro_f1": calibrated_metrics["macro_f1"],
                "calibrated_macro_ap": calibrated_metrics["macro_ap"],
                "calibrated_macro_auroc": calibrated_metrics["macro_auroc"],
                "undefined_labels": sum(
                    1 for fit in threshold_fits if fit["status"] != "defined"
                ),
                "test_split_accessed": False,
            }
            model_rows.append(model_row)
            model_summaries.append({
                "run_name": run_name,
                "modality": modality,
                "seed": seed,
                "checkpoint_sha256": model_row["checkpoint_sha256"],
                "probabilities_sha256": model_row["probabilities_sha256"],
                "thresholds_sha256": model_row["calibrated_thresholds_sha256"],
                "default_macro_f1": model_row["default_macro_f1"],
                "calibrated_macro_f1": model_row["calibrated_macro_f1"],
            })

        _write_csv(temporary / "threshold_records.csv", threshold_rows, THRESHOLD_RECORD_FIELDS)
        _write_csv(temporary / "per_label_calibration.csv", comparison_rows, PER_LABEL_FIELDS)

        model_fields = [
            "run_name", "task", "label_set", "modality", "seed", "n_samples", "n_labels",
            "default_threshold", "calibrated_thresholds_sha256", "checkpoint_sha256",
            "probabilities_sha256", "row_index_sha256", "labels_sha256",
        ]
        for prefix in ("default", "calibrated"):
            model_fields.extend(f"{prefix}_{name}" for name in METRIC_NAMES)
        model_fields.extend(["undefined_labels", "test_split_accessed"])
        _write_csv(temporary / "model_metrics.csv", model_rows, model_fields)

        by_modality = defaultdict(list)
        for row in model_rows:
            by_modality[row["modality"]].append(row)
        modality_rows = []
        for modality, rows in sorted(by_modality.items()):
            for policy in ("default", "calibrated"):
                for metric in METRIC_NAMES:
                    values = [row[f"{policy}_{metric}"] for row in rows if row[f"{policy}_{metric}"] is not None]
                    mean, sd = _mean_sd(values)
                    modality_rows.append({
                        "modality": modality,
                        "threshold_policy": policy,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "n_seeds": len(values),
                    })
        _write_csv(
            temporary / "modality_mean_sd.csv",
            modality_rows,
            ["modality", "threshold_policy", "metric", "mean", "sd", "n_seeds"],
        )

        label_groups = defaultdict(list)
        for row in comparison_rows:
            label_groups[(row["modality"], row["label_index"], row["label"])].append(row)
        per_label_summary = []
        for (modality, label_index, label), rows in sorted(label_groups.items()):
            for policy in ("default", "calibrated"):
                for metric in ("precision", "recall", "f1", "average_precision", "auroc"):
                    if metric in ("average_precision", "auroc"):
                        values = [row[metric] for row in rows]
                    else:
                        values = [row[f"{policy}_{metric}"] for row in rows]
                    mean, sd = _mean_sd(values)
                    per_label_summary.append({
                        "modality": modality,
                        "label_index": label_index,
                        "label": label,
                        "threshold_policy": policy,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                        "n_seeds": len(values),
                    })
        _write_csv(
            temporary / "per_label_mean_sd.csv",
            per_label_summary,
            ["modality", "label_index", "label", "threshold_policy", "metric", "mean", "sd", "n_seeds"],
        )

        summary = {
            "phase": "B3",
            "task": task,
            "label_set": label_set,
            "split": "validation",
            "n_models": len(model_rows),
            "n_labels": int(label_config["n_labels"]),
            "n_threshold_records": len(threshold_rows),
            "expected_threshold_records": int(label_config["n_labels"]) * len(model_rows),
            "n_undefined_threshold_records": undefined_labels,
            "undefined_records": undefined_records,
            "default_threshold": DEFAULT_THRESHOLD,
            "primary_threshold": DEFAULT_THRESHOLD,
            "secondary_threshold": "validation_f1_per_label",
            "calibration_method": (
                "precision_recall_curve observed-probability candidates; "
                "np.nanargmax first maximum; ascending thresholds -> smallest threshold on tie; "
                "single-class labels fallback to 0.5 with undefined status"
            ),
            "partial": bool(task_info["partial"]),
            "expected_models": task_info["expected_count"] if not task_info["partial"] else None,
            "no_test_access": True,
            "all_checks_pass": True,
            "validation_f1_note": (
                "Validation-F1 thresholds are fitted on the same Validation split; "
                "their F1 is in-sample optimistic and is not a Test estimate."
            ),
            "output_hashes": {
                name: sha256_file(temporary / name)
                for name in (
                    "threshold_records.csv",
                    "per_label_calibration.csv",
                    "model_metrics.csv",
                    "modality_mean_sd.csv",
                    "per_label_mean_sd.csv",
                )
            },
            "models": model_summaries,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(temporary / "summary.json", summary)
        return summary


def calibrate(
    label_sets: list[str],
    predictions_root: Path,
    out_root: Path,
    allow_partial: bool = False,
) -> list[dict]:
    results = []
    for label_set in label_sets:
        task_info = _compute_task(label_set, Path(predictions_root), allow_partial)
        results.append(_calibrate_task(task_info, Path(out_root)))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", choices=["all", *sorted(LABEL_SETS)], default="all")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=ROOT / "runs" / "LINE-B-PREDICTIONS",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "LINE-B-THRESHOLDS",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a partial model set for temporary sanity checks only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_sets = sorted(LABEL_SETS) if args.label_set == "all" else [args.label_set]
    results = calibrate(
        label_sets,
        args.predictions_root,
        args.out_root,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
