"""Validation-only conditional complementarity and Oracle analysis for Line B.

Definitions follow B0 v0.2 §7.1B/§7.3.  This is a read-only analysis of the
already exported FP32 Validation probabilities and the already fitted
Validation-F1 thresholds.  No model is loaded, no threshold is refitted, and
no Test index or Test prediction is read.

The two tasks are processed independently.  Shared14 and UVSPEC15 arrays are
never concatenated.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    atomic_output_dir,
    expected_seed_from_run_name,
    load_label_names,
    read_json,
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
MODALITIES = ("ir", "raman", "uvvis")
POLICIES = ("fixed_0.5", "validation_f1")
PAIRWISE_MODEL_METRICS = (
    "base_only_correct", "other_only_correct", "both_correct", "both_wrong",
    "fn_rescue", "fp_correction", "new_fp", "new_fn",
    "error_intersection", "error_union", "error_jaccard",
    "conditional_correct_given_base_wrong", "p_other_correct", "conditional_delta",
    "fn_rescue_rate", "fp_correction_rate", "new_fp_rate", "new_fn_rate",
    "or_f1", "or_gain_vs_max_single",
)
THREE_METRICS = (
    "any_correct", "exactly_one_correct", "all_wrong", "all_correct",
    "only_ir_correct", "only_raman_correct", "only_uv_correct",
    "ir_raman_common_fn", "ir_raman_common_fp",
    "uv_common_fn_rescue", "uv_common_fn_rescue_rate",
    "uv_common_fp_correction", "uv_common_fp_correction_rate",
    "ir_raman_both_correct", "ir_raman_both_correct_uv_wrong",
    "ir_raman_both_correct_uv_wrong_positive",
    "ir_raman_both_correct_uv_wrong_negative",
    "ir_raman_both_correct_uv_wrong_rate",
    "ir_raman_any_correct", "ir_raman_any_correct_rate",
    "three_any_correct", "three_any_correct_rate",
    "uv_extra_oracle_correct", "uv_extra_oracle_correct_rate",
    "ir_raman_both_wrong_uv_correct",
    "ir_raman_both_wrong_uv_correct_positive",
    "ir_raman_both_wrong_uv_correct_negative",
    "ir_raman_or_tp", "ir_raman_or_fp", "ir_raman_or_fn", "ir_raman_or_tn",
    "ir_raman_or_precision", "ir_raman_or_recall", "ir_raman_or_f1",
    "or_tp", "or_fp", "or_fn", "or_tn", "or_precision", "or_recall", "or_f1",
    "three_or_minus_ir_raman_or_f1",
    "best_single_f1", "or_minus_best_single_f1",
)


def _write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_sd(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return float(clean[0]), None
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth).astype(np.int64)
    prediction = np.asarray(prediction).astype(np.int64)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = (2.0 * tp / f1_denominator) if f1_denominator > 0 else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_task_inputs(
    label_set: str,
    predictions_root: Path,
    thresholds_root: Path,
    metrics_root: Path,
    allow_partial: bool,
) -> dict:
    label_config = resolve_label_set(label_set)
    task = label_config["task"]
    expected_runs = resolve_task_runs(label_set)
    task_predictions = Path(predictions_root) / task
    task_thresholds = Path(thresholds_root) / task
    metrics_path = Path(metrics_root) / task / "model_metrics.csv"
    expected_names = load_label_names(label_config["names"])
    manifests = sorted(task_predictions.glob("*/*/validation/prediction_manifest.json"))
    expected_count = sum(len(v) for v in expected_runs.values())
    if not manifests:
        raise FileNotFoundError(f"No prediction manifests under {task_predictions}")
    if not allow_partial and len(manifests) != expected_count:
        raise ValueError(f"{task}: expected {expected_count} manifests, found {len(manifests)}")
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    models = []
    seen = set()
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        if manifest.get("task") != task or manifest.get("label_set") != label_set:
            raise ValueError(f"Task/label-set mismatch in {manifest_path}")
        validate_manifest_checks(manifest, str(manifest_path))
        run_name = manifest.get("run_name")
        pred_dir = manifest_path.parent
        if run_name != pred_dir.parent.name:
            raise ValueError(f"Run name/directory mismatch: {manifest_path}")
        modality = manifest.get("modality")
        if modality not in expected_runs or run_name not in expected_runs[modality]:
            raise ValueError(f"Unregistered run: {task}/{modality}/{run_name}")
        expected_seed = expected_seed_from_run_name(run_name)
        seed_config = (manifest.get("checkpoint", {}).get("config") or {}).get("seed")
        if seed_config is None or int(seed_config) != expected_seed:
            raise ValueError(f"Seed mismatch for {run_name}")
        if run_name in seen:
            raise ValueError(f"Duplicate run: {run_name}")
        seen.add(run_name)

        probabilities, row_index, labels = validate_prediction_arrays(
            pred_dir, manifest, modality, label_set, expected_names
        )
        if manifest.get("split") != "validation":
            raise ValueError(f"Prediction manifest is not validation: {manifest_path}")
        split_file = manifest.get("split_file") or {}
        split_path = Path(split_file.get("path", ""))
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        split_sha = sha256_file(split_path)
        if split_file.get("sha256") != split_sha:
            raise ValueError(f"Split SHA mismatch: {run_name}")
        checkpoint = manifest.get("checkpoint") or {}
        checkpoint_path = Path(checkpoint.get("path", ""))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != (checkpoint.get("sha256_after") or checkpoint.get("sha256")):
            raise ValueError(f"Checkpoint SHA mismatch: {run_name}")

        threshold_path = task_thresholds / run_name / "optimal_thresholds.json"
        if not threshold_path.is_file():
            raise FileNotFoundError(threshold_path)
        threshold_sha = sha256_file(threshold_path)
        calibration_manifest_path = task_thresholds / run_name / "calibration_manifest.json"
        if not calibration_manifest_path.is_file():
            raise FileNotFoundError(calibration_manifest_path)
        calibration_manifest = read_json(calibration_manifest_path)
        if calibration_manifest.get("test_split_accessed") is not False:
            raise ValueError(f"Calibration manifest does not assert no Test access: {calibration_manifest_path}")
        if (calibration_manifest.get("outputs") or {}).get("optimal_thresholds") != threshold_sha:
            raise ValueError(f"Threshold file SHA mismatch vs calibration manifest: {run_name}")

        task_summary_path = task_thresholds / "summary.json"
        if task_summary_path.is_file():
            task_summary = read_json(task_summary_path)
            summary_models = {
                row.get("run_name"): row for row in task_summary.get("models", [])
            }
            if run_name in summary_models:
                if summary_models[run_name].get("thresholds_sha256") != threshold_sha:
                    raise ValueError(f"Threshold file SHA mismatch vs threshold summary: {run_name}")
            elif not allow_partial:
                raise ValueError(f"Missing threshold summary entry: {run_name}")
        elif not allow_partial:
            raise FileNotFoundError(task_summary_path)

        thresholds_payload = read_json(threshold_path)
        if thresholds_payload.get("task") != task or thresholds_payload.get("label_set") != label_set:
            raise ValueError(f"Threshold task mismatch: {threshold_path}")
        if thresholds_payload.get("run_name") != run_name:
            raise ValueError(f"Threshold run mismatch: {threshold_path}")
        if thresholds_payload.get("modality") != modality:
            raise ValueError(f"Threshold modality mismatch: {threshold_path}")
        if int(thresholds_payload.get("seed")) != expected_seed:
            raise ValueError(f"Threshold seed mismatch: {threshold_path}")
        if thresholds_payload.get("split") != "validation":
            raise ValueError(f"Threshold split mismatch: {threshold_path}")
        if thresholds_payload.get("test_split_accessed") is not False:
            raise ValueError(f"Threshold does not assert no Test access: {threshold_path}")
        if thresholds_payload.get("label_names") != expected_names:
            raise ValueError(f"Threshold label order mismatch: {threshold_path}")
        threshold_map = thresholds_payload.get("thresholds") or {}
        if list(threshold_map.keys()) != expected_names:
            raise ValueError(f"Threshold label keys mismatch: {threshold_path}")
        threshold_values = np.asarray([threshold_map[name] for name in expected_names], dtype=np.float64)
        if not np.isfinite(threshold_values).all() or threshold_values.min() < 0.0 or threshold_values.max() > 1.0:
            raise ValueError(f"Invalid threshold values: {threshold_path}")
        if thresholds_payload.get("probabilities_sha256") != sha256_array(probabilities):
            raise ValueError(f"Threshold probabilities SHA mismatch: {threshold_path}")
        if thresholds_payload.get("row_index_sha256") != sha256_array(row_index):
            raise ValueError(f"Threshold row_index SHA mismatch: {threshold_path}")
        if thresholds_payload.get("labels_sha256") != sha256_array(labels):
            raise ValueError(f"Threshold labels SHA mismatch: {threshold_path}")
        if thresholds_payload.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"Threshold checkpoint SHA mismatch: {threshold_path}")

        models.append({
            "run_name": run_name,
            "modality": modality,
            "seed": expected_seed,
            "pred_dir": pred_dir,
            "manifest_path": manifest_path,
            "manifest_sha256": sha256_file(manifest_path),
            "probabilities": probabilities,
            "row_index": row_index,
            "labels": labels,
            "probabilities_sha256": sha256_array(probabilities),
            "row_index_sha256": sha256_array(row_index),
            "labels_sha256": sha256_array(labels),
            "split_path": split_path,
            "split_sha256": split_sha,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "threshold_path": threshold_path,
            "thresholds_sha256": threshold_sha,
            "thresholds": threshold_values,
            "thresholds_payload": thresholds_payload,
            "calibration_manifest_path": calibration_manifest_path,
        })

    if not models:
        raise FileNotFoundError(f"{task}: no validated models")
    reference = models[0]
    for model in models[1:]:
        if not np.array_equal(model["row_index"], reference["row_index"]):
            raise ValueError(f"{task}: cross-model row_index mismatch: {model['run_name']}")
        if not np.array_equal(model["labels"], reference["labels"]):
            raise ValueError(f"{task}: cross-model labels mismatch: {model['run_name']}")
        if model["thresholds_payload"].get("label_names") != reference["thresholds_payload"].get("label_names"):
            raise ValueError(f"{task}: cross-model label_names mismatch: {model['run_name']}")

    if not allow_partial:
        for modality, runs in expected_runs.items():
            actual = {m["run_name"] for m in models if m["modality"] == modality}
            if actual != set(runs):
                raise ValueError(f"{task}/{modality}: registered run mismatch")
        if len(models) != expected_count:
            raise ValueError(f"{task}: expected {expected_count} unique models")

    # Pre-specify the best single modality from the already frozen 0.5 metrics
    # (mean Macro AP across the three seeds), not from analysis outcomes.
    metric_rows = list(csv.DictReader(metrics_path.open(encoding="utf-8-sig")))
    if len(metric_rows) != 9:
        raise ValueError(f"{metrics_path}: expected exactly 9 metric rows, found {len(metric_rows)}")
    metric_run_names = [row["run_name"] for row in metric_rows]
    if len(set(metric_run_names)) != 9:
        raise ValueError(f"{metrics_path}: duplicate run rows")
    expected_run_names = {run for runs in expected_runs.values() for run in runs}
    if set(metric_run_names) != expected_run_names:
        raise ValueError(f"{metrics_path}: run matrix does not match registered runs")
    by_modality = defaultdict(list)
    for row in metric_rows:
        run_name = row["run_name"]
        modality = row["modality"]
        if run_name not in expected_runs.get(modality, ()):
            raise ValueError(f"{metrics_path}: run/modality mismatch for {run_name}")
        if int(row["seed"]) != expected_seed_from_run_name(run_name):
            raise ValueError(f"{metrics_path}: seed mismatch for {run_name}")
        if float(row["threshold"]) != DEFAULT_THRESHOLD:
            raise ValueError(f"{metrics_path}: non-0.5 threshold row for {run_name}")
        if int(row["n_samples"]) != 12981:
            raise ValueError(f"{metrics_path}: unexpected n_samples for {run_name}")
        macro_ap = float(row["macro_ap"])
        if not np.isfinite(macro_ap):
            raise ValueError(f"{metrics_path}: non-finite macro_ap for {run_name}")
        by_modality[modality].append(macro_ap)
    if set(by_modality) != set(MODALITIES) or any(len(v) != 3 for v in by_modality.values()):
        raise ValueError(f"{metrics_path}: expected 3 rows per modality")
    modality_ap = {
        modality: float(statistics.mean(values))
        for modality, values in by_modality.items()
    }
    best_single = max(MODALITIES, key=lambda modality: (modality_ap.get(modality, -1.0), -MODALITIES.index(modality)))

    return {
        "task": task,
        "label_set": label_set,
        "label_config": label_config,
        "expected_names": expected_names,
        "models": models,
        "partial": bool(allow_partial),
        "expected_count": expected_count,
        "task_predictions": task_predictions,
        "task_thresholds": task_thresholds,
        "metrics_path": metrics_path,
        "best_single": best_single,
        "modality_macro_ap": modality_ap,
    }


def _threshold_for(model: dict, index: int, policy: str) -> float:
    return DEFAULT_THRESHOLD if policy == "fixed_0.5" else float(model["thresholds"][index])


def _model_rows(task_info: dict) -> list[dict]:
    rows = []
    for model in task_info["models"]:
        for policy in POLICIES:
            for index, label in enumerate(task_info["expected_names"]):
                truth = model["labels"][:, index]
                threshold = _threshold_for(model, index, policy)
                prediction = (model["probabilities"][:, index] >= threshold).astype(np.int64)
                confusion = _confusion(truth, prediction)
                rows.append({
                    "task": task_info["task"],
                    "label_set": task_info["label_set"],
                    "threshold_policy": policy,
                    "seed": model["seed"],
                    "run_name": model["run_name"],
                    "modality": model["modality"],
                    "label_index": index,
                    "label": label,
                    "support": int(truth.sum()),
                    "n_samples": int(len(truth)),
                    "threshold": threshold,
                    "thresholds_sha256": model["thresholds_sha256"],
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "probabilities_sha256": model["probabilities_sha256"],
                    "row_index_sha256": model["row_index_sha256"],
                    "labels_sha256": model["labels_sha256"],
                    **confusion,
                })
    return rows


def _pair_rows(task_info: dict) -> list[dict]:
    rows = []
    seeds = sorted({model["seed"] for model in task_info["models"]})
    for seed in seeds:
        models = {
            modality: next((m for m in task_info["models"] if m["seed"] == seed and m["modality"] == modality), None)
            for modality in MODALITIES
        }
        if any(m is None for m in models.values()):
            raise ValueError(f"Missing modality for seed {seed}")
        for base_modality in MODALITIES:
            for other_modality in MODALITIES:
                if base_modality == other_modality:
                    continue
                base = models[base_modality]
                other = models[other_modality]
                for policy in POLICIES:
                    for index, label in enumerate(task_info["expected_names"]):
                        truth = base["labels"][:, index]
                        base_threshold = _threshold_for(base, index, policy)
                        other_threshold = _threshold_for(other, index, policy)
                        base_pred = (base["probabilities"][:, index] >= base_threshold).astype(np.int64)
                        other_pred = (other["probabilities"][:, index] >= other_threshold).astype(np.int64)
                        base_tp = (truth == 1) & (base_pred == 1)
                        base_tn = (truth == 0) & (base_pred == 0)
                        base_fp = (truth == 0) & (base_pred == 1)
                        base_fn = (truth == 1) & (base_pred == 0)
                        other_correct = other_pred == truth
                        base_correct = base_pred == truth
                        base_wrong = ~base_correct
                        both_wrong = (~base_correct) & (~other_correct)
                        base_only = base_correct & (~other_correct)
                        other_only = (~base_correct) & other_correct
                        both_correct = base_correct & other_correct
                        fn_rescue = base_fn & (other_pred == 1)
                        fp_correction = base_fp & (other_pred == 0)
                        new_fp = base_tn & (other_pred == 1)
                        new_fn = base_tp & (other_pred == 0)
                        error_intersection = (~base_correct) & (~other_correct)
                        error_union = (~base_correct) | (~other_correct)
                        union_count = int(error_union.sum())
                        jaccard = 1.0 if union_count == 0 else float(error_intersection.sum()) / union_count
                        base_wrong_count = int((~base_correct).sum())
                        other_correct_count = int(other_correct.sum())
                        conditional = _ratio(int((base_wrong & other_correct).sum()), base_wrong_count)
                        baseline = _ratio(other_correct_count, len(truth))
                        base_conf = _confusion(truth, base_pred)
                        other_conf = _confusion(truth, other_pred)
                        or_pred = ((base_pred == 1) | (other_pred == 1)).astype(np.int64)
                        or_conf = _confusion(truth, or_pred)
                        max_single = max(
                            base_conf["f1"] if base_conf["f1"] is not None else -1.0,
                            other_conf["f1"] if other_conf["f1"] is not None else -1.0,
                        )
                        rows.append({
                            "task": task_info["task"],
                            "label_set": task_info["label_set"],
                            "threshold_policy": policy,
                            "seed": seed,
                            "base_modality": base_modality,
                            "other_modality": other_modality,
                            "base_run": base["run_name"],
                            "other_run": other["run_name"],
                            "label_index": index,
                            "label": label,
                            "support": int(truth.sum()),
                            "n_samples": int(len(truth)),
                            "n_positive": int((truth == 1).sum()),
                            "n_negative": int((truth == 0).sum()),
                            "base_wrong": base_wrong_count,
                            "base_threshold": base_threshold,
                            "other_threshold": other_threshold,
                            "base_tp": base_conf["tp"],
                            "base_tn": base_conf["tn"],
                            "base_fp": base_conf["fp"],
                            "base_fn": base_conf["fn"],
                            "other_tp": other_conf["tp"],
                            "other_tn": other_conf["tn"],
                            "other_fp": other_conf["fp"],
                            "other_fn": other_conf["fn"],
                            "base_only_correct": int(base_only.sum()),
                            "other_only_correct": int(other_only.sum()),
                            "both_correct": int(both_correct.sum()),
                            "both_wrong": int(both_wrong.sum()),
                            "fn_rescue": int(fn_rescue.sum()),
                            "fp_correction": int(fp_correction.sum()),
                            "new_fp": int(new_fp.sum()),
                            "new_fn": int(new_fn.sum()),
                            "error_intersection": int(error_intersection.sum()),
                            "error_union": int(error_union.sum()),
                            "error_jaccard": jaccard,
                            "error_jaccard_defined": union_count > 0,
                            "conditional_correct_given_base_wrong": conditional,
                            "p_other_correct": baseline,
                            "conditional_delta": None if conditional is None or baseline is None else conditional - baseline,
                            "fn_rescue_rate": _ratio(int(fn_rescue.sum()), base_conf["fn"]),
                            "fp_correction_rate": _ratio(int(fp_correction.sum()), base_conf["fp"]),
                            "new_fp_rate": _ratio(int(new_fp.sum()), int((truth == 0).sum())),
                            "new_fn_rate": _ratio(int(new_fn.sum()), int((truth == 1).sum())),
                            "or_tp": or_conf["tp"],
                            "or_fp": or_conf["fp"],
                            "or_fn": or_conf["fn"],
                            "or_tn": or_conf["tn"],
                            "or_precision": or_conf["precision"],
                            "or_recall": or_conf["recall"],
                            "or_f1": or_conf["f1"],
                            "or_gain_vs_max_single": (
                                None if or_conf["f1"] is None or max_single < 0
                                else or_conf["f1"] - max_single
                            ),
                        })
    return rows


def _three_rows(task_info: dict) -> list[dict]:
    rows = []
    best_single = task_info["best_single"]
    for seed in sorted({m["seed"] for m in task_info["models"]}):
        models = {
            modality: next((m for m in task_info["models"] if m["seed"] == seed and m["modality"] == modality), None)
            for modality in MODALITIES
        }
        if any(m is None for m in models.values()):
            raise ValueError(f"Missing modality for seed {seed}")
        for policy in POLICIES:
            for index, label in enumerate(task_info["expected_names"]):
                truth = models["ir"]["labels"][:, index]
                preds = {}
                corrects = {}
                for modality in MODALITIES:
                    model = models[modality]
                    threshold = _threshold_for(model, index, policy)
                    pred = (model["probabilities"][:, index] >= threshold).astype(np.int64)
                    preds[modality] = pred
                    corrects[modality] = pred == truth
                correct_sum = corrects["ir"].astype(int) + corrects["raman"].astype(int) + corrects["uvvis"].astype(int)
                any_correct = int((correct_sum >= 1).sum())
                exactly_one = int((correct_sum == 1).sum())
                all_wrong = int((correct_sum == 0).sum())
                all_correct = int((correct_sum == 3).sum())
                only_ir = int((corrects["ir"] & (~corrects["raman"]) & (~corrects["uvvis"])).sum())
                only_raman = int((corrects["raman"] & (~corrects["ir"]) & (~corrects["uvvis"])).sum())
                only_uv = int((corrects["uvvis"] & (~corrects["ir"]) & (~corrects["raman"])).sum())
                vib_both_wrong = (~corrects["ir"]) & (~corrects["raman"])
                vib_both_wrong_uv = vib_both_wrong & corrects["uvvis"]
                ir_fn = (truth == 1) & (preds["ir"] == 0)
                raman_fn = (truth == 1) & (preds["raman"] == 0)
                ir_fp = (truth == 0) & (preds["ir"] == 1)
                raman_fp = (truth == 0) & (preds["raman"] == 1)
                common_fn = ir_fn & raman_fn
                common_fp = ir_fp & raman_fp
                uv_common_fn_rescue = common_fn & corrects["uvvis"]
                uv_common_fp_correction = common_fp & corrects["uvvis"]
                both_correct = corrects["ir"] & corrects["raman"]
                both_correct_uv_wrong = both_correct & (~corrects["uvvis"])
                ir_raman_any = corrects["ir"] | corrects["raman"]
                three_any = correct_sum >= 1
                uv_extra_oracle = int(three_any.sum()) - int(ir_raman_any.sum())
                ir_raman_or_pred = ((preds["ir"] == 1) | (preds["raman"] == 1)).astype(np.int64)
                or_pred = ((preds["ir"] == 1) | (preds["raman"] == 1) | (preds["uvvis"] == 1)).astype(np.int64)
                and_pred = ((preds["ir"] == 1) & (preds["raman"] == 1) & (preds["uvvis"] == 1)).astype(np.int64)
                ir_raman_or_conf = _confusion(truth, ir_raman_or_pred)
                or_conf = _confusion(truth, or_pred)
                and_conf = _confusion(truth, and_pred)
                best_pred = preds[best_single]
                best_conf = _confusion(truth, best_pred)
                both_correct_count = int(both_correct.sum())
                common_fn_count = int(common_fn.sum())
                common_fp_count = int(common_fp.sum())
                rows.append({
                    "task": task_info["task"],
                    "label_set": task_info["label_set"],
                    "threshold_policy": policy,
                    "seed": seed,
                    "label_index": index,
                    "label": label,
                    "support": int(truth.sum()),
                    "n_samples": int(len(truth)),
                    "n_positive": int((truth == 1).sum()),
                    "n_negative": int((truth == 0).sum()),
                    "any_correct": any_correct,
                    "exactly_one_correct": exactly_one,
                    "all_wrong": all_wrong,
                    "all_correct": all_correct,
                    "only_ir_correct": only_ir,
                    "only_raman_correct": only_raman,
                    "only_uv_correct": only_uv,
                    "ir_raman_common_fn": common_fn_count,
                    "ir_raman_common_fp": common_fp_count,
                    "uv_common_fn_rescue": int(uv_common_fn_rescue.sum()),
                    "uv_common_fn_rescue_rate": _ratio(int(uv_common_fn_rescue.sum()), common_fn_count),
                    "uv_common_fp_correction": int(uv_common_fp_correction.sum()),
                    "uv_common_fp_correction_rate": _ratio(int(uv_common_fp_correction.sum()), common_fp_count),
                    "ir_raman_both_correct": both_correct_count,
                    "ir_raman_both_correct_uv_wrong": int(both_correct_uv_wrong.sum()),
                    "ir_raman_both_correct_uv_wrong_positive": int((both_correct_uv_wrong & (truth == 1)).sum()),
                    "ir_raman_both_correct_uv_wrong_negative": int((both_correct_uv_wrong & (truth == 0)).sum()),
                    "ir_raman_both_correct_uv_wrong_rate": _ratio(int(both_correct_uv_wrong.sum()), both_correct_count),
                    "ir_raman_any_correct": int(ir_raman_any.sum()),
                    "ir_raman_any_correct_rate": float(ir_raman_any.mean()),
                    "three_any_correct": int(three_any.sum()),
                    "three_any_correct_rate": float(three_any.mean()),
                    "uv_extra_oracle_correct": uv_extra_oracle,
                    "uv_extra_oracle_correct_rate": float(uv_extra_oracle / len(truth)),
                    "ir_raman_both_wrong_uv_correct": int(vib_both_wrong_uv.sum()),
                    "ir_raman_both_wrong_uv_correct_positive": int((vib_both_wrong_uv & (truth == 1)).sum()),
                    "ir_raman_both_wrong_uv_correct_negative": int((vib_both_wrong_uv & (truth == 0)).sum()),
                    "ir_raman_or_tp": ir_raman_or_conf["tp"],
                    "ir_raman_or_fp": ir_raman_or_conf["fp"],
                    "ir_raman_or_fn": ir_raman_or_conf["fn"],
                    "ir_raman_or_tn": ir_raman_or_conf["tn"],
                    "ir_raman_or_precision": ir_raman_or_conf["precision"],
                    "ir_raman_or_recall": ir_raman_or_conf["recall"],
                    "ir_raman_or_f1": ir_raman_or_conf["f1"],
                    "or_tp": or_conf["tp"],
                    "or_fp": or_conf["fp"],
                    "or_fn": or_conf["fn"],
                    "or_tn": or_conf["tn"],
                    "or_precision": or_conf["precision"],
                    "or_recall": or_conf["recall"],
                    "or_f1": or_conf["f1"],
                    "three_or_minus_ir_raman_or_f1": or_conf["f1"] - ir_raman_or_conf["f1"],
                    "and_tp": and_conf["tp"],
                    "and_fp": and_conf["fp"],
                    "and_fn": and_conf["fn"],
                    "and_tn": and_conf["tn"],
                    "and_f1": and_conf["f1"],
                    "best_single_modality": best_single,
                    "best_single_f1": best_conf["f1"],
                    "or_minus_best_single_f1": or_conf["f1"] - best_conf["f1"],
                })
    return rows


def _aggregate(rows: list[dict], group_fields, metrics, value_prefix_map=None) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        base = dict(zip(group_fields, key))
        for metric in metrics:
            values = []
            for row in group:
                field = metric
                if value_prefix_map:
                    field = value_prefix_map.get(metric, metric)
                value = row.get(field)
                if value is not None and value != "":
                    values.append(float(value))
            mean, sd = _mean_sd(values)
            output.append({
                **base,
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "n_seeds": len(values),
                "n_undefined_seeds": len(group) - len(values),
            })
    return output


def _analyze_task(task_info: dict, out_root: Path) -> dict:
    task = task_info["task"]
    label_set = task_info["label_set"]
    model_rows = _model_rows(task_info)
    pairwise_rows = _pair_rows(task_info)
    three_rows = _three_rows(task_info)

    model_metrics = (
        "tp", "tn", "fp", "fn", "precision", "recall", "f1",
    )
    model_mean = _aggregate(
        model_rows,
        ("task", "label_set", "modality", "label", "threshold_policy"),
        model_metrics,
    )
    pairwise_mean = _aggregate(
        pairwise_rows,
        ("task", "label_set", "base_modality", "other_modality", "label", "threshold_policy"),
        PAIRWISE_MODEL_METRICS,
    )
    three_mean = _aggregate(
        three_rows,
        ("task", "label_set", "label", "threshold_policy"),
        THREE_METRICS,
    )

    task_out = Path(out_root) / task / "complementarity"
    with atomic_output_dir(task_out) as temporary:
        _write_csv(
            temporary / "model_metrics.csv",
            model_rows,
            [
                "task", "label_set", "threshold_policy", "seed", "run_name", "modality",
                "label_index", "label", "support", "n_samples", "threshold",
                "thresholds_sha256", "checkpoint_sha256", "probabilities_sha256",
                "row_index_sha256", "labels_sha256", "tp", "tn", "fp", "fn",
                "precision", "recall", "f1",
            ],
        )
        pair_fields = [
            "task", "label_set", "threshold_policy", "seed", "base_modality", "other_modality",
            "base_run", "other_run", "label_index", "label", "support", "n_samples",
            "n_positive", "n_negative", "base_wrong",
            "base_threshold", "other_threshold", "base_tp", "base_tn", "base_fp", "base_fn",
            "other_tp", "other_tn", "other_fp", "other_fn", "base_only_correct",
            "other_only_correct", "both_correct", "both_wrong", "fn_rescue", "fp_correction",
            "new_fp", "new_fn", "error_intersection", "error_union", "error_jaccard",
            "error_jaccard_defined", "conditional_correct_given_base_wrong", "p_other_correct",
            "conditional_delta", "fn_rescue_rate", "fp_correction_rate", "new_fp_rate",
            "new_fn_rate", "or_tp", "or_fp", "or_fn", "or_tn", "or_precision", "or_recall",
            "or_f1", "or_gain_vs_max_single",
        ]
        _write_csv(temporary / "pairwise_metrics.csv", pairwise_rows, pair_fields)
        _write_csv(
            temporary / "three_modality_metrics.csv",
            three_rows,
            [
                "task", "label_set", "threshold_policy", "seed", "label_index", "label",
                "support", "n_samples", "n_positive", "n_negative",
                "any_correct", "exactly_one_correct", "all_wrong",
                "all_correct", "only_ir_correct", "only_raman_correct", "only_uv_correct",
                "ir_raman_common_fn", "ir_raman_common_fp", "uv_common_fn_rescue",
                "uv_common_fn_rescue_rate", "uv_common_fp_correction",
                "uv_common_fp_correction_rate", "ir_raman_both_correct",
                "ir_raman_both_correct_uv_wrong", "ir_raman_both_correct_uv_wrong_positive",
                "ir_raman_both_correct_uv_wrong_negative", "ir_raman_both_correct_uv_wrong_rate",
                "ir_raman_any_correct", "ir_raman_any_correct_rate", "three_any_correct",
                "three_any_correct_rate", "uv_extra_oracle_correct", "uv_extra_oracle_correct_rate",
                "ir_raman_both_wrong_uv_correct", "ir_raman_both_wrong_uv_correct_positive",
                "ir_raman_both_wrong_uv_correct_negative", "ir_raman_or_tp", "ir_raman_or_fp",
                "ir_raman_or_fn", "ir_raman_or_tn", "ir_raman_or_precision", "ir_raman_or_recall",
                "ir_raman_or_f1", "or_tp", "or_fp", "or_fn", "or_tn", "or_precision", "or_recall",
                "or_f1", "three_or_minus_ir_raman_or_f1", "and_tp", "and_fp", "and_fn", "and_tn",
                "and_f1", "best_single_modality", "best_single_f1", "or_minus_best_single_f1",
            ],
        )
        _write_csv(
            temporary / "model_mean_sd.csv",
            model_mean,
            ["task", "label_set", "modality", "label", "threshold_policy", "metric", "mean", "sd", "n_seeds", "n_undefined_seeds"],
        )
        _write_csv(
            temporary / "pairwise_mean_sd.csv",
            pairwise_mean,
            ["task", "label_set", "base_modality", "other_modality", "label", "threshold_policy", "metric", "mean", "sd", "n_seeds", "n_undefined_seeds"],
        )
        _write_csv(
            temporary / "three_modality_mean_sd.csv",
            three_mean,
            ["task", "label_set", "label", "threshold_policy", "metric", "mean", "sd", "n_seeds", "n_undefined_seeds"],
        )

        model_provenance = [
            {
                "run_name": model["run_name"],
                "modality": model["modality"],
                "seed": model["seed"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "probabilities_sha256": model["probabilities_sha256"],
                "row_index_sha256": model["row_index_sha256"],
                "labels_sha256": model["labels_sha256"],
                "thresholds_sha256": model["thresholds_sha256"],
                "prediction_manifest": {
                    "path": relative_to_root(model["manifest_path"]),
                    "sha256": model["manifest_sha256"],
                },
                "thresholds": {
                    "path": relative_to_root(model["threshold_path"]),
                    "sha256": model["thresholds_sha256"],
                },
            }
            for model in task_info["models"]
        ]
        summary = {
            "phase": "B3-A",
            "task": task,
            "label_set": label_set,
            "split": "validation",
            "n_models": len(task_info["models"]),
            "n_labels": len(task_info["expected_names"]),
            "seeds": sorted({m["seed"] for m in task_info["models"]}),
            "modalities": list(MODALITIES),
            "threshold_policies": list(POLICIES),
            "primary_threshold": DEFAULT_THRESHOLD,
            "secondary_threshold": "validation_f1_per_label",
            "best_single_modality": task_info["best_single"],
            "best_single_selection": "maximum mean Validation Macro AP at 0.5 from frozen B3 model metrics",
            "modality_macro_ap": task_info["modality_macro_ap"],
            "n_model_rows": len(model_rows),
            "n_pairwise_rows": len(pairwise_rows),
            "n_three_modality_rows": len(three_rows),
            "n_error_jaccard_undefined": sum(
                1 for row in pairwise_rows if not row["error_jaccard_defined"]
            ),
            "partial": bool(task_info["partial"]),
            "no_test_access": True,
            "all_checks_pass": True,
            "definitions": {
                "fn_rescue": "A FN and B predicts positive for the same sample-label pair",
                "fp_correction": "A FP and B predicts negative for the same sample-label pair",
                "new_fp": "A true-negative and B false-positive",
                "new_fn": "A true-positive and B false-negative",
                "conditional_correct": "P(other correct | base wrong)",
                "common_fn": "IR and Raman are both FN on the same sample-label pair",
                "common_fp": "IR and Raman are both FP on the same sample-label pair",
                "uv_common_fn_rescue": "UV is correct on an IR/Raman common-FN pair",
                "uv_common_fp_correction": "UV is correct on an IR/Raman common-FP pair",
                "uv_extra_oracle_correct": "three-modality any-correct minus IR/Raman any-correct, in sample-label pairs",
                "error_jaccard": "|(FP_A union FN_A) intersect (FP_B union FN_B)| / |union|; 1 when both error sets empty, flagged",
                "double_or_vs_triple_or": "IR/Raman positive OR versus three-modality positive OR; fixed rules, not Oracle",
                "positive_or_fusion": "OR of individual positive predictions; not Oracle",
                "oracle": "uses the true label to determine whether each modality is correct; diagnostic only",
            },
            "validation_f1_note": (
                "Validation-F1 thresholds are in-sample fits on the same Validation split; "
                "complementarity at that policy is secondary/exploratory."
            ),
            "interpretation_boundary": (
                "positive_or_fusion is a fixed prediction rule, not Oracle; "
                "Oracle uses y_true and is diagnostic only; conditional complementarity "
                "is not a direct estimate of deployable fusion gain."
            ),
            "provenance": {
                "models": model_provenance,
                "frozen_b3_metrics": {
                    "path": relative_to_root(task_info["metrics_path"]),
                    "sha256": sha256_file(task_info["metrics_path"]),
                },
            },
            "outputs": {
                name: sha256_file(temporary / name)
                for name in (
                    "model_metrics.csv",
                    "pairwise_metrics.csv",
                    "three_modality_metrics.csv",
                    "model_mean_sd.csv",
                    "pairwise_mean_sd.csv",
                    "three_modality_mean_sd.csv",
                )
            },
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(temporary / "summary.json", summary)
        return summary


def analyze(
    label_sets: list[str],
    predictions_root: Path,
    thresholds_root: Path,
    metrics_root: Path,
    out_root: Path,
    allow_partial: bool = False,
) -> list[dict]:
    results = []
    for label_set in label_sets:
        info = _load_task_inputs(
            label_set, predictions_root, thresholds_root, metrics_root, allow_partial
        )
        results.append(_analyze_task(info, out_root))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", choices=["all", *sorted(LABEL_SETS)], default="all")
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "runs" / "LINE-B-PREDICTIONS")
    parser.add_argument("--thresholds-root", type=Path, default=ROOT / "runs" / "LINE-B-THRESHOLDS")
    parser.add_argument("--metrics-root", type=Path, default=ROOT / "runs" / "LINE-B-METRICS")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "LINE-B-ANALYSIS")
    parser.add_argument("--allow-partial", action="store_true", help="Temporary sanity checks only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_sets = sorted(LABEL_SETS) if args.label_set == "all" else [args.label_set]
    results = analyze(
        label_sets,
        args.predictions_root,
        args.thresholds_root,
        args.metrics_root,
        args.out_root,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
