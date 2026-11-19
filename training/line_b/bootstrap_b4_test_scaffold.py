"""B4.3 formal Test scaffold-clustered paired bootstrap (B0 v0.2 §7.2).

Reads only the frozen B4.1 Test probability arrays, the frozen Validation-F1
thresholds and the authoritative scaffold manifest.  It never trains, never
runs Test inference, never re-fits thresholds, never modifies frozen assets and
never runs B4.4+ analyses.

Design
------
* Test ``row_index`` -> ``murcko_scaffold`` from
  ``processed/qm9s/prepared/qm9s_manifest_prepared.csv``.
* Only scaffolds actually present in Test form the resampling clusters.
* Each replicate draws exactly ``n_test_scaffolds`` clusters with replacement;
  molecule weight = scaffold draw count.
* One fixed draw matrix is shared by both tasks, all 18 models, all 3 seeds,
  all labels and all metrics (paired).
* 2,000 replicates, RNG seed 20261117.
* AP/AUROC weighted by draw counts; undefined when a weighted class is empty.
  F1 uses zero_division=0.
* Percentile 95% CI; seed SD is the sample SD (ddof=1) of full-Test point
  estimates and is reported separately from the bootstrap CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    read_json,
    relative_to_root,
    sha256_array,
    sha256_file,
    write_json,
)

DEFAULT_REPS = 2000
DEFAULT_RNG = 20261117
DEV_MAX_REPS = 50
CI_PERCENTILES = (2.5, 97.5)
CHUNK = 250
TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
SEEDS = (42, 123, 2026)
COMPARISONS = (("ir", "raman"), ("uvvis", "ir"), ("uvvis", "raman"))
COMPARISON_LABELS = {
    ("ir", "raman"): "IR-Raman",
    ("uvvis", "ir"): "UV-IR",
    ("uvvis", "raman"): "UV-Raman",
}
POLICIES = ("fixed_0.5", "validation_f1")
METRIC_NAMES = {"ap": "AP", "auroc": "AUROC", "f1": "F1"}
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
DEFAULT_PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
DEFAULT_SCAFFOLD = ROOT / "processed" / "qm9s" / "prepared" / "qm9s_manifest_prepared.csv"
DEFAULT_OUT = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
TEST_N = 12981


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cwd": str(ROOT),
    }


def _load_scaffold_map(path: Path) -> tuple[dict[int, str], dict]:
    mapping: dict[int, str] = {}
    blank = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "row_index" not in reader.fieldnames or "murcko_scaffold" not in reader.fieldnames:
            raise ValueError("scaffold manifest lacks row_index or murcko_scaffold")
        for row in reader:
            index = int(row["row_index"])
            if index in mapping:
                raise ValueError(f"duplicate row_index in scaffold manifest: {index}")
            scaffold = (row.get("murcko_scaffold") or "").strip()
            if not scaffold:
                scaffold = "__MISSING__"
                blank += 1
            mapping[index] = scaffold
    if set(mapping) != set(range(129817)):
        raise ValueError("scaffold manifest row_index is not 0..129816")
    info = {
        "path": relative_to_root(path),
        "sha256": _sha(path),
        "n_rows": len(mapping),
        "n_unique_scaffolds": len(set(mapping.values())),
        "blank_scaffold_rows": blank,
    }
    return mapping, info


def _build_clusters(row_index: np.ndarray, mapping: dict[int, str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    row_index = np.asarray(row_index, dtype=np.int64)
    if len(row_index) != TEST_N or len(np.unique(row_index)) != TEST_N:
        raise ValueError("Test row_index is not 12,981 unique rows")
    missing = [int(i) for i in row_index if int(i) not in mapping]
    if missing:
        raise ValueError(f"Test row_index missing from scaffold manifest: {missing[:10]}")
    names = sorted({mapping[int(i)] for i in row_index})
    index = {name: position for position, name in enumerate(names)}
    cluster_id = np.asarray([index[mapping[int(i)]] for i in row_index], dtype=np.int64)
    sizes = np.bincount(cluster_id, minlength=len(names)).astype(np.int64)
    return cluster_id, names, sizes


def _load_preflight(freeze_path: Path, pin: str, predictions_root: Path) -> dict:
    protocol = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
    protocol_sha = _sha(protocol)
    if _sha(freeze_path) != pin:
        raise SystemExit("FREEZE SHA256 mismatch")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise SystemExit("FREEZE is not a frozen B3 manifest")
    if freeze.get("protocol_sha256") != protocol_sha:
        raise SystemExit("Protocol SHA256 differs from FREEZE record")
    if len(freeze.get("models") or []) != 18:
        raise SystemExit("FREEZE does not register 18 models")

    models = []
    row_index_ref = None
    labels_by_task: dict[str, np.ndarray] = {}
    label_names_by_task: dict[str, list[str]] = {}
    for record in freeze["models"]:
        task = record["task"]
        label_set = record["label_set"]
        modality = record["modality"]
        run_name = record["run_name"]
        out_dir = predictions_root / task / MODALITY_DIRS[modality] / run_name / "test"
        manifest_path = out_dir / "prediction_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"Missing Test manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        probabilities = np.load(out_dir / "probabilities.npy", allow_pickle=False)
        row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
        labels = np.load(out_dir / "labels.npy", allow_pickle=False)
        arrays = manifest["arrays"]
        if sha256_array(probabilities) != arrays["probabilities"]["sha256"]:
            raise SystemExit(f"Probability SHA256 mismatch: {out_dir}")
        if sha256_array(row_index) != arrays["row_index"]["sha256"]:
            raise SystemExit(f"row_index SHA256 mismatch: {out_dir}")
        if sha256_array(labels) != arrays["labels"]["sha256"]:
            raise SystemExit(f"labels SHA256 mismatch: {out_dir}")
        if (manifest.get("freeze") or {}).get("sha256") != pin:
            raise SystemExit(f"FREEZE pin mismatch: {out_dir}")
        if manifest["checkpoint"]["sha256"] != record["checkpoint"]["sha256"]:
            raise SystemExit(f"Checkpoint mismatch: {out_dir}")
        if manifest.get("n_samples") != TEST_N or manifest.get("split") != "test":
            raise SystemExit(f"Unexpected Test manifest identity: {out_dir}")
        names = list(freeze["labels_and_data"]["label_sets"][label_set]["label_names"]["names"])
        if manifest.get("label_names") != names:
            raise SystemExit(f"Label order mismatch: {out_dir}")
        if row_index_ref is None:
            row_index_ref = row_index
        elif not np.array_equal(row_index_ref, row_index):
            raise SystemExit("Test row_index differs across models")
        if task not in labels_by_task:
            labels_by_task[task] = labels
            label_names_by_task[task] = names
        elif not np.array_equal(labels_by_task[task], labels):
            raise SystemExit(f"Test labels differ across {task} models")

        threshold_path = ROOT / record["threshold"]["path"]
        if _sha(threshold_path) != record["threshold"]["sha256"]:
            raise SystemExit(f"Threshold mismatch: {threshold_path}")
        threshold_payload = read_json(threshold_path)
        if threshold_payload.get("label_names") != names:
            raise SystemExit(f"Threshold label order mismatch: {threshold_path}")
        thresholds = np.asarray([threshold_payload["thresholds"][name] for name in names], dtype=np.float64)
        validation_labels = np.load(ROOT / record["labels"]["path"], allow_pickle=False)
        models.append({
            "task": task,
            "label_set": label_set,
            "modality": modality,
            "run_name": run_name,
            "seed": int(record["seed"]),
            "probabilities": probabilities,
            "labels": labels,
            "validation_labels": validation_labels,
            "thresholds": thresholds,
            "manifest": manifest,
            "prediction_manifest_sha256": _sha(manifest_path),
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "threshold_path": relative_to_root(threshold_path),
            "threshold_sha256": _sha(threshold_path),
        })

    # Test access log.
    log_path = ROOT / "runs" / "LINE-B-MANIFESTS" / "test_access_log.json"
    log = read_json(log_path)
    if (log.get("freeze") or {}).get("sha256") != pin:
        raise SystemExit("test_access_log.json does not belong to this FREEZE")
    if len(log.get("entries") or []) != 18:
        raise SystemExit("test_access_log.json does not contain 18 entries")
    if (log.get("historical_exposure") or {}).get("blind_test_claim") is not False:
        raise SystemExit("test_access_log.json must not claim a blind test")

    # B4.2 must be intact and verified (point-estimate reference).
    if not B42.is_dir():
        raise SystemExit(f"B4.2 directory missing: {B42}")
    b42_manifest = read_json(B42 / "analysis_manifest.json")
    for item in b42_manifest["outputs"]:
        path = B42 / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise SystemExit(f"B4.2 output hash mismatch: {item['path']}")
    b42_verification = read_json(B42 / "verification.json")
    if b42_verification.get("status") != "PASS":
        raise SystemExit("B4.2 verification is not PASS")
    if b42_verification["analysis_manifest"]["sha256"] != _sha(B42 / "analysis_manifest.json"):
        raise SystemExit("B4.2 verification does not reference the current analysis_manifest")

    return {
        "freeze": freeze,
        "freeze_path": freeze_path,
        "freeze_sha256": pin,
        "protocol_path": relative_to_root(protocol),
        "protocol_sha256": protocol_sha,
        "models": models,
        "row_index": row_index_ref,
        "labels_by_task": labels_by_task,
        "label_names_by_task": label_names_by_task,
        "test_access_log": {"path": relative_to_root(log_path), "sha256": _sha(log_path), "n_entries": len(log["entries"])},
        "b42": {
            "analysis_manifest": {"path": relative_to_root(B42 / "analysis_manifest.json"), "sha256": _sha(B42 / "analysis_manifest.json")},
            "verification": {"path": relative_to_root(B42 / "verification.json"), "sha256": _sha(B42 / "verification.json")},
            "overall_metrics": {"path": relative_to_root(B42 / "overall_metrics.csv"), "sha256": _sha(B42 / "overall_metrics.csv")},
            "per_label_metrics": {"path": relative_to_root(B42 / "per_label_metrics.csv"), "sha256": _sha(B42 / "per_label_metrics.csv")},
            "seed_comparison": {"path": relative_to_root(B42 / "seed_comparison.csv"), "sha256": _sha(B42 / "seed_comparison.csv")},
            "overall_mean_sd": {"path": relative_to_root(B42 / "overall_mean_sd.csv"), "sha256": _sha(B42 / "overall_mean_sd.csv")},
            "report": {"path": relative_to_root(B42 / "report.md"), "sha256": _sha(B42 / "report.md")},
        },
    }


def _f1_unweighted(truth: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = (np.asarray(probability, dtype=np.float64) >= threshold).astype(np.int64)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _point_metrics(model: dict, names: list[str]) -> dict:
    truth = model["labels"].astype(np.int64)
    probabilities = model["probabilities"]
    label_rows = []
    for index, name in enumerate(names):
        column = truth[:, index]
        unique = np.unique(column)
        ap = float(average_precision_score(column, probabilities[:, index])) if unique.size == 2 else None
        auroc = float(roc_auc_score(column, probabilities[:, index])) if unique.size == 2 else None
        label_rows.append({
            "label": name,
            "index": index,
            "ap": ap,
            "auroc": auroc,
            "f1_fixed_0.5": _f1_unweighted(column, probabilities[:, index], 0.5),
            "f1_validation_f1": _f1_unweighted(column, probabilities[:, index], float(model["thresholds"][index])),
            "support": int(column.sum()),
        })
    ap_values = [row["ap"] for row in label_rows if row["ap"] is not None]
    auroc_values = [row["auroc"] for row in label_rows if row["auroc"] is not None]
    return {
        "labels": label_rows,
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "macro_f1_fixed_0.5": float(np.mean([row["f1_fixed_0.5"] for row in label_rows])),
        "macro_f1_validation_f1": float(np.mean([row["f1_validation_f1"] for row in label_rows])),
        "n_defined_ap": len(ap_values),
        "n_defined_auroc": len(auroc_values),
    }


def _prepare_label(probability: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_truth = truth[order]
    ends = np.flatnonzero(np.r_[sorted_probability[1:] != sorted_probability[:-1], True])
    return order, sorted_truth, ends


def _metrics_batch(draws: np.ndarray, cluster_sorted: np.ndarray, sorted_truth: np.ndarray,
                   ends: np.ndarray, prediction_fixed: np.ndarray, prediction_frozen: np.ndarray) -> dict:
    weights = draws[:, cluster_sorted].astype(np.float64)
    weighted_positive = weights * sorted_truth
    weighted_negative = weights - weighted_positive
    cumulative_positive = np.cumsum(weighted_positive, axis=1)
    cumulative_negative = np.cumsum(weighted_negative, axis=1)
    tp = cumulative_positive[:, ends]
    fp = cumulative_negative[:, ends]
    positive_total = tp[:, -1]
    negative_total = fp[:, -1]
    valid = (positive_total > 0) & (negative_total > 0)

    denominator = tp + fp
    precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    recall = np.divide(tp, positive_total[:, None], out=np.zeros_like(tp), where=positive_total[:, None] > 0)
    delta_recall = np.diff(np.concatenate([np.zeros((tp.shape[0], 1)), recall], axis=1), axis=1)
    ap = (delta_recall * precision).sum(axis=1)

    tpr = np.divide(tp, positive_total[:, None], out=np.zeros_like(tp), where=positive_total[:, None] > 0)
    fpr = np.divide(fp, negative_total[:, None], out=np.zeros_like(fp), where=negative_total[:, None] > 0)
    delta_fpr = np.diff(np.concatenate([np.zeros((fpr.shape[0], 1)), fpr], axis=1), axis=1)
    previous_tpr = np.concatenate([np.zeros((tpr.shape[0], 1)), tpr[:, :-1]], axis=1)
    auroc = (delta_fpr * (tpr + previous_tpr) / 2.0).sum(axis=1)

    ap[~valid] = np.nan
    auroc[~valid] = np.nan

    def f1_for(prediction: np.ndarray) -> np.ndarray:
        tp_w = (weighted_positive * prediction).sum(axis=1)
        fp_w = (weighted_negative * prediction).sum(axis=1)
        fn_w = (weighted_positive * (1.0 - prediction)).sum(axis=1)
        denom = 2.0 * tp_w + fp_w + fn_w
        return np.divide(2.0 * tp_w, denom, out=np.zeros_like(tp_w), where=denom > 0)

    return {
        "ap": ap,
        "auroc": auroc,
        "f1_fixed_0.5": f1_for(prediction_fixed),
        "f1_validation_f1": f1_for(prediction_frozen),
    }


def _bootstrap_metrics(by_seed_modality: dict, names: list[str], cluster_id: np.ndarray,
                       draws: np.ndarray, reps: int) -> dict:
    output = {}
    chunk = CHUNK
    for (seed, modality), model in by_seed_modality.items():
        probabilities = model["probabilities"]
        truth = model["labels"]
        n_labels = len(names)
        values = {key: np.empty((n_labels, reps), dtype=np.float64)
                  for key in ("ap", "auroc", "f1_fixed_0.5", "f1_validation_f1")}
        for index in range(n_labels):
            order, sorted_truth, ends = _prepare_label(probabilities[:, index], truth[:, index])
            cluster_sorted = cluster_id[order]
            prediction_fixed = (probabilities[order, index] >= 0.5).astype(np.float64)
            prediction_frozen = (probabilities[order, index] >= float(model["thresholds"][index])).astype(np.float64)
            for start in range(0, reps, chunk):
                stop = min(start + chunk, reps)
                batch = _metrics_batch(
                    draws[start:stop], cluster_sorted, sorted_truth, ends,
                    prediction_fixed, prediction_frozen,
                )
                for key in values:
                    values[key][index, start:stop] = batch[key]
        ap_defined = np.isfinite(values["ap"])
        auroc_defined = np.isfinite(values["auroc"])
        counts_ap = ap_defined.sum(axis=0)
        counts_auroc = auroc_defined.sum(axis=0)
        macro_ap = np.divide(
            np.nansum(values["ap"], axis=0), counts_ap,
            out=np.full(reps, np.nan), where=counts_ap > 0,
        )
        macro_auroc = np.divide(
            np.nansum(values["auroc"], axis=0), counts_auroc,
            out=np.full(reps, np.nan), where=counts_auroc > 0,
        )
        output[(seed, modality)] = {
            "labels": values,
            "macro_ap": macro_ap,
            "macro_auroc": macro_auroc,
            "macro_f1_fixed_0.5": values["f1_fixed_0.5"].mean(axis=0),
            "macro_f1_validation_f1": values["f1_validation_f1"].mean(axis=0),
            "n_defined_ap": counts_ap,
            "n_defined_auroc": counts_auroc,
        }
    return output


def _support_tier(val_positives: int, test_positives: int) -> str:
    smallest = min(int(val_positives), int(test_positives))
    if smallest < 30:
        return "very_low"
    if smallest < 150:
        return "low"
    return "normal"


def _percentile_ci(values: np.ndarray) -> tuple[float | None, float | None, int]:
    valid = values[np.isfinite(values)]
    if len(valid) >= 2:
        low, high = np.percentile(valid, CI_PERCENTILES)
        return float(low), float(high), int(len(valid))
    return None, None, int(len(valid))


def _bootstrap_task(task: str, label_set: str, models: list[dict], names: list[str],
                    cluster_id: np.ndarray, draws: np.ndarray, reps: int,
                    val_positives: np.ndarray, test_positives: np.ndarray) -> dict:
    by_seed_modality = {(model["seed"], model["modality"]): model for model in models}
    if len(by_seed_modality) != 9:
        raise SystemExit(f"{task}: expected 9 models, got {len(by_seed_modality)}")
    for seed in SEEDS:
        for modality in MODALITIES:
            if (seed, modality) not in by_seed_modality:
                raise SystemExit(f"{task}: missing model for seed={seed}, modality={modality}")

    point = {}
    for (seed, modality), model in by_seed_modality.items():
        point[(seed, modality)] = _point_metrics(model, names)

    metrics = _bootstrap_metrics(by_seed_modality, names, cluster_id, draws, reps)
    tier = {name: _support_tier(val_positives[index], test_positives[index]) for index, name in enumerate(names)}

    # Per-seed comparison arrays.
    label_arrays: dict[tuple, np.ndarray] = {}
    macro_arrays: dict[tuple, np.ndarray] = {}
    macro_label_counts: dict[tuple, np.ndarray] = {}
    for base, other in COMPARISONS:
        comparison = COMPARISON_LABELS[(base, other)]
        for seed in SEEDS:
            a = metrics[(seed, base)]
            b = metrics[(seed, other)]
            # AP / AUROC per label, paired finite only.
            for metric in ("ap", "auroc"):
                for index, name in enumerate(names):
                    av = a["labels"][metric][index]
                    bv = b["labels"][metric][index]
                    diff = np.where(np.isfinite(av) & np.isfinite(bv), av - bv, np.nan)
                    label_arrays[("threshold_free", comparison, metric, "label", seed, name)] = diff
                a_macro = a[f"macro_{metric}"]
                b_macro = b[f"macro_{metric}"]
                diff = np.where(np.isfinite(a_macro) & np.isfinite(b_macro), a_macro - b_macro, np.nan)
                macro_arrays[("threshold_free", comparison, metric, "macro", seed)] = diff
            # F1 both policies.
            for policy in POLICIES:
                key = f"f1_{policy}"
                for index, name in enumerate(names):
                    label_arrays[(policy, comparison, "f1", "label", seed, name)] = a["labels"][key][index] - b["labels"][key][index]
                macro_arrays[(policy, comparison, "f1", "macro", seed)] = a[f"macro_{key}"] - b[f"macro_{key}"]

    point_rows = []
    ci_per_seed = []
    ci_three_seed = []
    replicate_store: dict[str, np.ndarray] = {}

    def _point_value(metric: str, policy: str, base: str, other: str, scope: str,
                     label: str | None, seed: int):
        key = metric.lower() if metric != "F1" else f"f1_{policy}"
        record_a = point[(seed, base)]
        record_b = point[(seed, other)]
        if scope == "label":
            row_a = next(row for row in record_a["labels"] if row["label"] == label)
            row_b = next(row for row in record_b["labels"] if row["label"] == label)
            av, bv = row_a[key], row_b[key]
        else:
            av, bv = record_a[f"macro_{key}"], record_b[f"macro_{key}"]
        if av is None or bv is None:
            return None
        return float(av) - float(bv)

    def _point_label(metric: str, policy: str, comparison: str, scope: str, label: str | None, seed=None):
        base, other = next(pair for pair in COMPARISONS if COMPARISON_LABELS[pair] == comparison)
        if seed is None:
            diffs = [_point_value(metric, policy, base, other, scope, label, current) for current in SEEDS]
            if any(value is None for value in diffs):
                return None, None, None
            return float(np.mean(diffs)), float(np.mean(diffs)), float(statistics.stdev(diffs))
        value = _point_value(metric, policy, base, other, scope, label, seed)
        if value is None:
            return None, None, None
        return value, None, None

    # Point CSV and per-seed CI rows.
    for base, other in COMPARISONS:
        comparison = COMPARISON_LABELS[(base, other)]
        for policy, metric in (("threshold_free", "AP"), ("threshold_free", "AUROC"),
                               ("fixed_0.5", "F1"), ("validation_f1", "F1")):
            metric_key = metric.lower() if metric != "F1" else "f1"
            for scope in ("label", "macro"):
                labels = names if scope == "label" else [None]
                for label in labels:
                    index = names.index(label) if label is not None else ""
                    tier_value = tier[label] if label is not None else ""
                    point_three, _, seed_sd = _point_label(metric, policy, comparison, scope, label, seed=None)
                    for seed in SEEDS:
                        point_value, _, _ = _point_label(metric, policy, comparison, scope, label, seed=seed)
                        point_rows.append({
                            "task": task, "label_set": label_set, "comparison": comparison,
                            "base_modality": base, "other_modality": other,
                            "metric": metric, "threshold_policy": policy, "scope": scope,
                            "label_index": index, "label": label if label is not None else "__macro__",
                            "support_tier": tier_value, "seed": seed,
                            "point_estimate": point_value, "seed_mean": "", "seed_sd": "",
                        })
                        if scope == "label":
                            arr = label_arrays[(policy, comparison, metric_key, scope, seed, label)]
                        else:
                            arr = macro_arrays[(policy, comparison, metric_key, scope, seed)]
                        low, high, n_valid = _percentile_ci(arr)
                        n_undefined = reps - n_valid
                        ci_per_seed.append({
                            "task": task, "label_set": label_set, "comparison": comparison,
                            "base_modality": base, "other_modality": other,
                            "metric": metric, "threshold_policy": policy, "scope": scope,
                            "label_index": index, "label": label if label is not None else "__macro__",
                            "support_tier": tier_value, "seed": seed,
                            "point_estimate": point_value,
                            "ci_low": low, "ci_high": high,
                            "n_valid_reps": n_valid, "n_total_reps": reps,
                            "n_undefined_ap": n_undefined if metric == "AP" else "",
                            "n_undefined_auroc": n_undefined if metric == "AUROC" else "",
                            "n_labels_defined": (
                                1 if scope == "label" else (
                                    len(names) if metric == "F1" else sum(
                                        1 for name in names
                                        if next(row for row in point[(seed, base)]["labels"] if row["label"] == name)[metric_key] is not None
                                        and next(row for row in point[(seed, other)]["labels"] if row["label"] == name)[metric_key] is not None
                                    )
                                )
                            ),                            "n_labels_total": len(names) if scope == "macro" else 1,
                        })
                    # Three-seed bootstrap arrays.
                    per_seed_values = []
                    for seed in SEEDS:
                        if scope == "label":
                            per_seed_values.append(label_arrays[(policy, comparison, metric_key, scope, seed, label)])
                        else:
                            per_seed_values.append(macro_arrays[(policy, comparison, metric_key, scope, seed)])
                    matrix = np.vstack(per_seed_values)
                    if scope == "macro" and metric in ("AP", "AUROC"):
                        label_matrix = np.stack([
                            np.vstack([label_arrays[(policy, comparison, metric_key, "label", seed, name)] for name in names])
                            for seed in SEEDS
                        ], axis=0)  # seeds x labels x reps
                        three = np.full(reps, np.nan)
                        common_counts = np.zeros(reps, dtype=np.int64)
                        for rep in range(reps):
                            diffs = label_matrix[:, :, rep]
                            valid_labels = np.all(np.isfinite(diffs), axis=0)
                            common_counts[rep] = int(valid_labels.sum())
                            if valid_labels.any():
                                three[rep] = float(np.mean(diffs[:, valid_labels].mean(axis=0)))
                    else:
                        valid_reps = np.all(np.isfinite(matrix), axis=0)
                        three = np.full(reps, np.nan)
                        if valid_reps.any():
                            three[valid_reps] = matrix[:, valid_reps].mean(axis=0)
                        common_counts = None
                    low, high, n_valid = _percentile_ci(three)
                    n_undefined = reps - n_valid
                    # Three-seed point and seed SD.
                    if scope == "label":
                        seed_points = []
                        for seed in SEEDS:
                            value, _, _ = _point_label(metric, policy, comparison, scope, label, seed=seed)
                            seed_points.append(value)
                        if any(value is None for value in seed_points):
                            point_mean = seed_sd_value = None
                        else:
                            point_mean = float(np.mean(seed_points))
                            seed_sd_value = float(statistics.stdev(seed_points))
                        n_labels_defined = 1
                    else:
                        if metric in ("AP", "AUROC"):
                            label_points = {}
                            for name in names:
                                values = [_point_label(metric, policy, comparison, "label", name, seed=seed)[0] for seed in SEEDS]
                                if all(value is not None for value in values):
                                    label_points[name] = values
                            if label_points:
                                seed_macros = [np.mean([values[j] for values in label_points.values()]) for j in range(3)]
                                point_mean = float(np.mean(seed_macros))
                                seed_sd_value = float(statistics.stdev(seed_macros))
                            else:
                                point_mean = seed_sd_value = None
                            n_labels_defined = len(label_points)
                        else:
                            seed_macros = []
                            for seed in SEEDS:
                                value, _, _ = _point_label(metric, policy, comparison, "macro", None, seed=seed)
                                seed_macros.append(value)
                            point_mean = None if any(v is None for v in seed_macros) else float(np.mean(seed_macros))
                            seed_sd_value = None if any(v is None for v in seed_macros) else float(statistics.stdev(seed_macros))
                            n_labels_defined = len(names)
                    ci_three_seed.append({
                        "task": task, "label_set": label_set, "comparison": comparison,
                        "base_modality": base, "other_modality": other,
                        "metric": metric, "threshold_policy": policy, "scope": scope,
                        "label_index": index, "label": label if label is not None else "__macro__",
                        "support_tier": tier_value,
                        "point_estimate": point_mean, "seed_mean": point_mean, "seed_sd": seed_sd_value,
                        "ci_low": low, "ci_high": high,
                        "n_valid_reps": n_valid, "n_total_reps": reps,
                        "n_undefined_ap": n_undefined if metric == "AP" else "",
                        "n_undefined_auroc": n_undefined if metric == "AUROC" else "",
                        "n_labels_defined": n_labels_defined,
                        "n_labels_total": len(names) if scope == "macro" else 1,
                    })
                    point_rows.append({
                        "task": task, "label_set": label_set, "comparison": comparison,
                        "base_modality": base, "other_modality": other,
                        "metric": metric, "threshold_policy": policy, "scope": scope,
                        "label_index": index, "label": label if label is not None else "__macro__",
                        "support_tier": tier_value, "seed": "three_seed_mean",
                        "point_estimate": point_mean, "seed_mean": point_mean, "seed_sd": seed_sd_value,
                    })
                    key = "|".join([
                        task, comparison, metric, policy, scope,
                        label if label is not None else "__macro__", "three_seed",
                    ])
                    replicate_store[key] = three
                    for seed in SEEDS:
                        per_seed_key = "|".join([
                            task, comparison, metric, policy, scope,
                            label if label is not None else "__macro__", str(seed),
                        ])
                        if scope == "label":
                            replicate_store[per_seed_key] = label_arrays[(policy, comparison, metric_key, scope, seed, label)]
                        else:
                            replicate_store[per_seed_key] = macro_arrays[(policy, comparison, metric_key, scope, seed)]

    # Undefined counts summary per metric group.
    undefined = Counter()
    for row in ci_three_seed:
        if row["n_undefined_ap"] != "" and int(row["n_undefined_ap"]) > 0:
            undefined[f'{row["task"]}|{row["metric"]}|three_seed'] += int(row["n_undefined_ap"])
        if row["n_undefined_auroc"] != "" and int(row["n_undefined_auroc"]) > 0:
            undefined[f'{row["task"]}|{row["metric"]}|three_seed'] += int(row["n_undefined_auroc"])

    return {
        "task": task,
        "label_set": label_set,
        "names": names,
        "point": point,
        "point_rows": point_rows,
        "ci_per_seed": ci_per_seed,
        "ci_three_seed": ci_three_seed,
        "replicates": replicate_store,
        "tier": tier,
        "macro_label_counts": macro_label_counts,
        "undefined": dict(undefined),
    }


def _point_consistency(task_info: dict, b42_dir: Path, label_names: dict[str, list[str]]) -> dict:
    import csv as csv_module
    overall_path = b42_dir / "overall_metrics.csv"
    per_label_path = b42_dir / "per_label_metrics.csv"
    with overall_path.open(encoding="utf-8-sig", newline="") as handle:
        overall = list(csv_module.DictReader(handle))
    with per_label_path.open(encoding="utf-8-sig", newline="") as handle:
        per_label = list(csv_module.DictReader(handle))
    max_diff = 0.0
    for (seed, modality), metrics in task_info["point"].items():
        rows = [row for row in overall if row["task"] == task_info["task"] and row["modality"] == modality and int(row["seed"]) == seed]
        for policy in POLICIES:
            row = next(r for r in rows if r["policy"] == policy)
            pairs = [
                (metrics["macro_ap"], float(row["macro_ap"])),
                (metrics["macro_auroc"], float(row["macro_auroc"])),
                (metrics[f"macro_f1_{policy}"], float(row["macro_f1"])),
            ]
            for computed, reference in pairs:
                if computed is None:
                    raise SystemExit(f"Undefined point metric while checking B4.2 ({task_info['task']})")
                max_diff = max(max_diff, abs(float(computed) - reference))
        for label_row in metrics["labels"]:
            refs = [row for row in per_label if row["task"] == task_info["task"] and row["modality"] == modality and int(row["seed"]) == seed and row["label"] == label_row["label"]]
            for row in refs:
                if row["policy"] == "fixed_0.5":
                    max_diff = max(max_diff, abs(label_row["f1_fixed_0.5"] - float(row["f1"])))
                elif row["policy"] == "validation_f1":
                    max_diff = max(max_diff, abs(label_row["f1_validation_f1"] - float(row["f1"])))
                if row["policy"] == "fixed_0.5":
                    if label_row["ap"] is not None:
                        max_diff = max(max_diff, abs(label_row["ap"] - float(row["average_precision"])))
                    if label_row["auroc"] is not None:
                        max_diff = max(max_diff, abs(label_row["auroc"] - float(row["auroc"])))
    if max_diff > 1e-6:
        raise SystemExit(f"Point estimates disagree with B4.2 by {max_diff}")
    return {"max_abs_diff_vs_b4_2": max_diff, "tolerance": 1e-6}


def _build_report(task_results: list[dict], scaffold_rows: list[dict], reps: int, rng_seed: int,
                  undefined_summary: dict) -> str:
    lines = []
    lines.append("# Line B B4.3 — Formal Test Scaffold-Clustered Paired Bootstrap")
    lines.append("")
    lines.append("> **Classification: exploratory / post-hoc.** The Test split was opened before B4. "
                 "Confidence intervals describe scaffold-resampling uncertainty under the fixed three-seed models and fixed Test set; "
                 "they are not training-randomness estimates and are not confirmatory significance tests.")
    lines.append("")
    lines.append(f"- Replicates: **{reps}** (RNG seed `{rng_seed}`), scaffold clusters = Test scaffolds.")
    lines.append("- One shared scaffold draw matrix for both tasks, all 18 models, all 3 seeds, all labels and all comparisons.")
    lines.append("- AP/AUROC weighted by scaffold draw counts; undefined when a weighted class is empty (never imputed). "
                 "F1 uses zero_division=0; both `fixed_0.5` and `validation_f1` policies are reported.")
    lines.append("")
    lines.append("## 1. Scaffold clusters")
    lines.append("")
    lines.append("| Task | Test rows | Scaffolds | Min size | Median | Max | Singletons | Resampled molecules (mean) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in scaffold_rows:
        lines.append(f'| {row["task"]} | {row["n_test_rows"]} | {row["n_scaffolds"]} | {row["scaffold_size_min"]} | '
                     f'{row["scaffold_size_median"]} | {row["scaffold_size_max"]} | {row["singleton_scaffolds"]} | '
                     f'{row["resampled_molecules_mean"]:.1f} |')
    lines.append("")
    lines.append("## 2. Full-Test point estimates (3-seed mean ± sample SD)")
    lines.append("")
    lines.append("| Task | Comparison | Metric | Point (3-seed mean) | Seed SD (ddof=1) |")
    lines.append("|---|---|---|---|---|")
    for result in task_results:
        for row in result["ci_three_seed"]:
            if row["scope"] != "macro":
                continue
            metric_label = row["metric"] if row["metric"] != "F1" else f'F1 ({row["threshold_policy"]})'
            point = "" if row["point_estimate"] is None else f'{row["point_estimate"]:+.4f}'
            sd = "" if row["seed_sd"] is None else f'{row["seed_sd"]:.4f}'
            lines.append(f'| {row["task"]} | {row["comparison"]} | {metric_label} | {point} | {sd} |')
    lines.append("")
    lines.append("## 3. Three-seed bootstrap 95% percentile CI (macro)")
    lines.append("")
    lines.append("| Task | Comparison | Metric | Point | CI low | CI high | Valid reps | Undefined | Labels defined |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for result in task_results:
        for row in result["ci_three_seed"]:
            if row["scope"] != "macro":
                continue
            metric_label = row["metric"] if row["metric"] != "F1" else f'F1 ({row["threshold_policy"]})'
            point = "" if row["point_estimate"] is None else f'{row["point_estimate"]:+.4f}'
            low = "" if row["ci_low"] is None else f'{row["ci_low"]:+.4f}'
            high = "" if row["ci_high"] is None else f'{row["ci_high"]:+.4f}'
            lines.append(f'| {row["task"]} | {row["comparison"]} | {metric_label} | {point} | {low} | {high} | '
                         f'{row["n_valid_reps"]} | {row["n_total_reps"] - row["n_valid_reps"]} | {row["n_labels_defined"]} |')
    lines.append("")
    lines.append("## 4. Undefined handling summary")
    lines.append("")
    if undefined_summary:
        lines.append("| Group | Undefined replicates |")
        lines.append("|---|---|")
        for key, value in sorted(undefined_summary.items()):
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("No undefined AP/AUROC replicates in the three-seed macro summaries.")
    lines.append("")
    lines.append("Per-seed CIs, per-label CIs and all undefined counts are in "
                 "`bootstrap_ci_per_seed.csv` / `bootstrap_ci_three_seed.csv`.")
    lines.append("")
    lines.append("## 5. Interpretation boundaries")
    lines.append("")
    lines.append("- Bootstrap CI covers scaffold-sampling uncertainty only, with the models and Test set held fixed.")
    lines.append("- Seed SD (ddof=1, n=3) describes seed-to-seed variability and is **not** combined with the bootstrap CI.")
    lines.append("- A CI including 0 is not evidence of directional stability; a CI excluding 0 is not confirmatory significance.")
    lines.append("- Very-low-support labels are unstable under scaffold resampling and must not carry modality conclusions alone.")
    lines.append("- No Test inference, threshold fitting, model selection or Bootstrap-based tuning was performed in B4.3.")
    lines.append("")
    lines.append("## 6. Provenance")
    lines.append("")
    lines.append("- `analysis_manifest.json` records all input/output SHA256; `verification.json` is the independent check.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--scaffold-manifest", type=Path, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.out_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.out_dir}")
    if args.reps < 2:
        raise SystemExit("At least 2 replicates are required")

    preflight = _load_preflight(args.freeze, args.freeze_sha256, args.predictions_root)
    mapping, scaffold_info = _load_scaffold_map(args.scaffold_manifest)
    row_index = preflight["row_index"]
    cluster_id, cluster_names, cluster_sizes = _build_clusters(row_index, mapping)
    n_scaffolds = len(cluster_names)

    rng = np.random.default_rng(args.rng_seed)
    draws = np.empty((args.reps, n_scaffolds), dtype=np.float32)
    for rep in range(args.reps):
        draws[rep] = np.bincount(rng.integers(0, n_scaffolds, size=n_scaffolds), minlength=n_scaffolds).astype(np.float32)
    draws_sha = _sha_bytes(draws)
    resampled = draws.astype(np.float64) @ cluster_sizes.astype(np.float64)

    models_by_task = defaultdict(list)
    for model in preflight["models"]:
        models_by_task[model["task"]].append(model)

    task_results = []
    scaffold_rows = []
    for task, label_set in TASKS:
        models = models_by_task[task]
        names = preflight["label_names_by_task"][task]
        test_positives = preflight["labels_by_task"][task].sum(axis=0).astype(int)
        validation_positives = models[0]["validation_labels"].sum(axis=0).astype(int)
        result = _bootstrap_task(task, label_set, models, names, cluster_id, draws, args.reps,
                                 validation_positives, test_positives)
        consistency = _point_consistency(result, B42, preflight["label_names_by_task"])
        result["point_consistency"] = consistency
        task_results.append(result)
        scaffold_rows.append({
            "task": task,
            "label_set": label_set,
            "n_test_rows": int(len(row_index)),
            "n_scaffolds": int(n_scaffolds),
            "scaffold_size_min": int(cluster_sizes.min()),
            "scaffold_size_median": float(np.median(cluster_sizes)),
            "scaffold_size_max": int(cluster_sizes.max()),
            "scaffold_size_mean": float(cluster_sizes.mean()),
            "singleton_scaffolds": int((cluster_sizes == 1).sum()),
            "missing_scaffold_rows": 0,
            "blank_scaffold_molecules": int(sum(1 for i in row_index if mapping[int(i)] == "__MISSING__")),
            "reps": int(args.reps),
            "rng_seed": int(args.rng_seed),
            "resampled_molecules_min": float(resampled.min()),
            "resampled_molecules_mean": float(resampled.mean()),
            "resampled_molecules_max": float(resampled.max()),
            "draw_count_min": int(draws.min()),
            "draw_count_mean": float(draws.mean()),
            "draw_count_max": int(draws.max()),
        })

    point_rows = [row for result in task_results for row in result["point_rows"]]
    ci_per_seed = [row for result in task_results for row in result["ci_per_seed"]]
    ci_three_seed = [row for result in task_results for row in result["ci_three_seed"]]
    undefined_summary = {}
    for result in task_results:
        for key, value in result["undefined"].items():
            undefined_summary[key] = undefined_summary.get(key, 0) + value

    outputs = {}
    with _atomic_output_dir(args.out_dir) as temporary:
        point_fields = ["task", "label_set", "comparison", "base_modality", "other_modality", "metric",
                        "threshold_policy", "scope", "label_index", "label", "support_tier", "seed",
                        "point_estimate", "seed_mean", "seed_sd"]
        _write_csv(temporary / "point_comparisons.csv", point_rows, point_fields)
        ci_fields = ["task", "label_set", "comparison", "base_modality", "other_modality", "metric",
                     "threshold_policy", "scope", "label_index", "label", "support_tier", "seed",
                     "point_estimate", "ci_low", "ci_high", "n_valid_reps", "n_total_reps",
                     "n_undefined_ap", "n_undefined_auroc", "n_labels_defined", "n_labels_total"]
        _write_csv(temporary / "bootstrap_ci_per_seed.csv", ci_per_seed, ci_fields)
        ci3_fields = ["task", "label_set", "comparison", "base_modality", "other_modality", "metric",
                      "threshold_policy", "scope", "label_index", "label", "support_tier",
                      "point_estimate", "seed_mean", "seed_sd", "ci_low", "ci_high",
                      "n_valid_reps", "n_total_reps", "n_undefined_ap", "n_undefined_auroc",
                      "n_labels_defined", "n_labels_total"]
        _write_csv(temporary / "bootstrap_ci_three_seed.csv", ci_three_seed, ci3_fields)
        scaffold_fields = list(scaffold_rows[0].keys())
        _write_csv(temporary / "scaffold_summary.csv", scaffold_rows, scaffold_fields)

        replicate_store = {}
        for result in task_results:
            replicate_store.update(result["replicates"])
        np.savez_compressed(temporary / "bootstrap_replicates.npz", **replicate_store)

        report_text = _build_report(task_results, scaffold_rows, args.reps, args.rng_seed, undefined_summary)
        (temporary / "report.md").write_text(report_text, encoding="utf-8")

        elapsed = time.perf_counter() - started
        summary = {
            "phase": "B4.3",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "classification": {
                "status": "exploratory/post-hoc",
                "blind_test_claim": False,
                "confirmatory_significance": False,
            },
            "config": {
                "reps": int(args.reps),
                "rng_seed": int(args.rng_seed),
                "rng": "numpy.random.default_rng; per replicate: bincount(rng.integers(0, n_scaffolds, n_scaffolds))",
                "ci_percentiles": list(CI_PERCENTILES),
                "seed_sd_ddof": 1,
                "comparisons": [COMPARISON_LABELS[pair] for pair in COMPARISONS],
                "policies": list(POLICIES),
                "draws_sha256": draws_sha,
            },
            "scaffolds": scaffold_rows,
            "undefined_summary": undefined_summary,
            "point_consistency": {result["task"]: result["point_consistency"] for result in task_results},
            "counts": {
                "tasks": len(task_results),
                "models": len(preflight["models"]),
                "seeds": len(SEEDS),
                "labels_shared14": len(preflight["label_names_by_task"]["Shared14"]),
                "labels_uvspec15": len(preflight["label_names_by_task"]["UVSPEC15"]),
                "point_rows": len(point_rows),
                "ci_per_seed_rows": len(ci_per_seed),
                "ci_three_seed_rows": len(ci_three_seed),
            },
            "environment": _environment(),
            "elapsed_seconds": round(elapsed, 3),
        }
        write_json(temporary / "summary.json", summary)

        analysis_outputs = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "analysis_manifest.json":
                analysis_outputs.append({
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": _sha(path),
                    "size_bytes": int(path.stat().st_size),
                })
        manifest = {
            "phase": "B4.3",
            "generated_at": summary["generated_at"],
            "classification": summary["classification"],
            "freeze": {
                "path": relative_to_root(args.freeze),
                "sha256": args.freeze_sha256,
                "protocol_path": preflight["protocol_path"],
                "protocol_sha256": preflight["protocol_sha256"],
            },
            "inputs": {
                "scaffold_manifest": scaffold_info,
                "test_access_log": preflight["test_access_log"],
                "b4_2": preflight["b42"],
                "test_predictions": [
                    {
                        "task": model["task"],
                        "label_set": model["label_set"],
                        "modality": model["modality"],
                        "run_name": model["run_name"],
                        "seed": model["seed"],
                        "prediction_manifest_sha256": model["prediction_manifest_sha256"],
                        "probabilities_sha256": model["manifest"]["arrays"]["probabilities"]["sha256"],
                        "row_index_sha256": model["manifest"]["arrays"]["row_index"]["sha256"],
                        "labels_sha256": model["manifest"]["arrays"]["labels"]["sha256"],
                        "checkpoint_sha256": model["checkpoint_sha256"],
                        "threshold_path": model["threshold_path"],
                        "threshold_sha256": model["threshold_sha256"],
                    }
                    for model in preflight["models"]
                ],
            },
            "config": summary["config"],
            "counts": summary["counts"],
            "outputs": analysis_outputs,
            "code": {
                "bootstrap_script": relative_to_root(Path(__file__)),
                "bootstrap_script_sha256": _sha(Path(__file__)),
                "verifier": "training/line_b/verify_b4_3_bootstrap.py",
                "verifier_sha256": _sha(ROOT / "training/line_b/verify_b4_3_bootstrap.py")
                if (ROOT / "training/line_b/verify_b4_3_bootstrap.py").is_file() else None,
                "metrics_sha256": _sha(ROOT / "training/line_b/metrics.py"),
                "common_sha256": _sha(ROOT / "training/line_b/common.py"),
            },
            "environment": summary["environment"],
        }
        write_json(temporary / "analysis_manifest.json", manifest)

    print(json.dumps({
        "out_dir": relative_to_root(args.out_dir),
        "reps": args.reps,
        "n_scaffolds": n_scaffolds,
        "point_rows": len(point_rows),
        "ci_per_seed_rows": len(ci_per_seed),
        "ci_three_seed_rows": len(ci_three_seed),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "draws_sha256": draws_sha,
        "point_consistency": {result["task"]: result["point_consistency"] for result in task_results},
        "undefined_summary": undefined_summary,
    }, ensure_ascii=False, indent=1))


def _sha_bytes(array: np.ndarray) -> str:
    import hashlib
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_output_dir(out_dir: Path):
    from training.line_b.common import atomic_output_dir
    return atomic_output_dir(out_dir)


if __name__ == "__main__":
    main()
