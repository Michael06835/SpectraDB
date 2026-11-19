"""Paired scaffold-cluster bootstrap for Line B Validation metrics.

This module implements the B0 v0.2 exploratory bootstrap only.  It reads the
frozen Validation prediction/threshold assets, maps Validation row_index values
to the authoritative ``murcko_scaffold`` identities, and resamples scaffold
clusters with replacement.  All modalities, seeds, labels and metrics use the
same draw in each replicate.

No model inference, threshold fitting, Test access or formal Freeze is
performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    MODALITIES,
    PREPARED,
    ROOT as PROJECT_ROOT,
    TASK_RUNS,
    load_label_names,
    read_json,
    relative_to_root,
    resolve_label_set,
    sha256_file,
    write_json,
)
from training.line_b.analyze_line_b_complementarity import _load_task_inputs

DEFAULT_REPS = 20
DEV_MAX_REPS = 50
DEFAULT_RNG = 20261117
CI_PERCENTILES = (2.5, 97.5)
MANIFEST_PATH = PREPARED / "qm9s_manifest_prepared.csv"
COMPARISONS = (("ir", "raman"), ("uvvis", "ir"), ("uvvis", "raman"))
COMPARISON_LABELS = {
    ("ir", "raman"): "IR-Raman",
    ("uvvis", "ir"): "UV-IR",
    ("uvvis", "raman"): "UV-Raman",
}


def _comparison_label(base: str, other: str) -> str:
    return COMPARISON_LABELS[(base, other)]
POLICIES = ("fixed_0.5", "validation_f1")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cwd": str(PROJECT_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short").splitlines(),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_scaffold_map() -> tuple[dict[int, str], dict[str, Any]]:
    """Read authoritative row_index -> murcko_scaffold mapping."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)
    mapping: dict[int, str] = {}
    missing = 0
    with MANIFEST_PATH.open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "row_index" not in reader.fieldnames or "murcko_scaffold" not in reader.fieldnames:
            raise ValueError("manifest lacks row_index or murcko_scaffold")
        for row in reader:
            idx = int(row["row_index"])
            if idx in mapping:
                raise ValueError(f"duplicate row_index in manifest: {idx}")
            scaffold = (row.get("murcko_scaffold") or "").strip()
            if not scaffold:
                scaffold = "__MISSING__"
                missing += 1
            mapping[idx] = scaffold
    expected = set(range(129817))
    if set(mapping) != expected:
        raise ValueError("manifest row_index is not 0..129816")
    return mapping, {
        "path": relative_to_root(MANIFEST_PATH),
        "sha256": _sha(MANIFEST_PATH),
        "n_rows": len(mapping),
        "n_unique_scaffolds": len(set(mapping.values())),
        "missing_scaffold_rows": missing,
    }


def _build_clusters(row_index: np.ndarray, scaffold_map: dict[int, str]) -> tuple[np.ndarray, list[str], dict[str, list[int]], int]:
    row_index = np.asarray(row_index, dtype=np.int64)
    if len(row_index) != len(np.unique(row_index)):
        raise ValueError("Validation row_index contains duplicates")
    if len(row_index) != 12981:
        raise ValueError(f"Unexpected Validation size: {len(row_index)}")
    if row_index.min() < 0 or row_index.max() >= 129817:
        raise ValueError("Validation row_index out of range")
    missing_rows = [int(i) for i in row_index if int(i) not in scaffold_map]
    if missing_rows:
        # A row_index absent from the authoritative manifest cannot be assigned
        # to any scaffold.  This is a hard input failure, unlike a blank
        # scaffold field which is preserved as the explicit __MISSING__ group.
        raise ValueError(f"Validation row_index missing from scaffold manifest: {missing_rows[:10]}")
    cluster_names = sorted({scaffold_map[int(i)] for i in row_index})
    cluster_index = {name: j for j, name in enumerate(cluster_names)}
    cluster_id = np.asarray([cluster_index[scaffold_map[int(i)]] for i in row_index], dtype=np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for position, name in enumerate(cluster_names):
        positions = np.flatnonzero(cluster_id == position).tolist()
        groups[name] = positions
    return cluster_id, cluster_names, groups, 0


CALL_STATS = {"ranking": 0, "f1": 0}


def _ranking_metrics(truth: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict:
    """Threshold-free AP/AUROC; one call computes both."""
    CALL_STATS["ranking"] += 1
    truth = np.asarray(truth).astype(np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    positive_weight = float(weight[truth == 1].sum())
    negative_weight = float(weight[truth == 0].sum())
    if positive_weight > 0 and negative_weight > 0:
        ap = float(average_precision_score(truth, probability, sample_weight=weight))
        auroc = float(roc_auc_score(truth, probability, sample_weight=weight))
    else:
        ap = None
        auroc = None
    return {"ap": ap, "auroc": auroc, "n_pos_weight": positive_weight, "n_neg_weight": negative_weight}


def _classification_f1(truth: np.ndarray, probability: np.ndarray, weight: np.ndarray, threshold: float) -> dict:
    """Threshold-dependent weighted F1; zero_division=0."""
    CALL_STATS["f1"] += 1
    truth = np.asarray(truth).astype(np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    tp = float(weight[(truth == 1) & (prediction == 1)].sum())
    fp = float(weight[(truth == 0) & (prediction == 1)].sum())
    fn = float(weight[(truth == 1) & (prediction == 0)].sum())
    denom = 2.0 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom > 0 else 0.0
    return {"f1": float(f1), "tp_weight": tp, "fp_weight": fp, "fn_weight": fn}


def _weighted_metrics(truth: np.ndarray, probability: np.ndarray, weight: np.ndarray, threshold: float) -> dict:
    """Convenience wrapper retained for independent equivalence tests."""
    ranking = _ranking_metrics(truth, probability, weight)
    classification = _classification_f1(truth, probability, weight, threshold)
    return {**ranking, **classification}


def _support_tier(positive_count: int) -> str:
    # B0's original tier uses min(val positives, test positives).  This C1
    # development path is Validation-only; the tier is explicitly provisional.
    if positive_count < 30:
        return "very_low"
    if positive_count < 150:
        return "low"
    return "normal"


def _point_metrics(models: list[dict], label_set: str, task: str) -> tuple[list[dict], dict]:
    names = load_label_names(resolve_label_set(label_set)["names"])
    rows = []
    macro = {}
    for model in models:
        seed = int(model["seed"])
        modality = model["modality"]
        probs = model["probabilities"]
        truth = model["labels"]
        base_weight = np.ones(len(truth), dtype=np.float64)
        for index, name in enumerate(names):
            support = int(truth[:, index].sum())
            tier = _support_tier(support)
            ranking = _ranking_metrics(truth[:, index], probs[:, index], base_weight)
            fixed = _classification_f1(truth[:, index], probs[:, index], base_weight, 0.5)
            calibrated = _classification_f1(truth[:, index], probs[:, index], base_weight, float(model["thresholds"][index]))
            rows.append({
                "task": task, "label_set": label_set, "threshold_policy": "threshold_free",
                "seed": seed, "modality": modality, "label_index": index, "label": name,
                "support": support, "support_tier": tier, "ap": ranking["ap"], "auroc": ranking["auroc"],
                "f1": None, "n_defined_ap": int(ranking["ap"] is not None),
                "n_defined_auroc": int(ranking["auroc"] is not None), "n_defined_f1": 0,
                "n_total_labels": len(names), "scope": "label",
            })
            for policy, result in (("fixed_0.5", fixed), ("validation_f1", calibrated)):
                rows.append({
                    "task": task, "label_set": label_set, "threshold_policy": policy,
                    "seed": seed, "modality": modality, "label_index": index, "label": name,
                    "support": support, "support_tier": tier, "ap": None, "auroc": None,
                    "f1": result["f1"], "n_defined_ap": 0, "n_defined_auroc": 0,
                    "n_defined_f1": 1, "n_total_labels": len(names), "scope": "label",
                })
        for policy in ("threshold_free", "fixed_0.5", "validation_f1"):
            label_rows = [r for r in rows if r["seed"] == seed and r["modality"] == modality and r["threshold_policy"] == policy and r["scope"] == "label"]
            if policy == "threshold_free":
                ap_vals = [r["ap"] for r in label_rows if r["ap"] is not None]
                auroc_vals = [r["auroc"] for r in label_rows if r["auroc"] is not None]
                row = {
                    "task": task, "label_set": label_set, "threshold_policy": policy,
                    "seed": seed, "modality": modality, "label_index": "", "label": "__macro__",
                    "support": "", "support_tier": "", "ap": float(np.mean(ap_vals)) if ap_vals else None,
                    "auroc": float(np.mean(auroc_vals)) if auroc_vals else None,
                    "f1": None, "n_defined_ap": len(ap_vals), "n_defined_auroc": len(auroc_vals),
                    "n_defined_f1": 0, "n_total_labels": len(names), "scope": "macro",
                }
            else:
                f1_vals = [r["f1"] for r in label_rows]
                row = {
                    "task": task, "label_set": label_set, "threshold_policy": policy,
                    "seed": seed, "modality": modality, "label_index": "", "label": "__macro__",
                    "support": "", "support_tier": "", "ap": None, "auroc": None,
                    "f1": float(np.mean(f1_vals)) if f1_vals else None,
                    "n_defined_ap": 0, "n_defined_auroc": 0, "n_defined_f1": len(f1_vals),
                    "n_total_labels": len(names), "scope": "macro",
                }
            rows.append(row)
            macro[(policy, seed, modality)] = row
    return rows, macro


def _point_comparison(model_rows: list[dict], label_set: str, task: str) -> list[dict]:
    names = load_label_names(resolve_label_set(label_set)["names"])
    by = {(r["threshold_policy"], r["seed"], r["modality"], r["scope"], r["label"]): r for r in model_rows}
    rows = []
    for base, other in COMPARISONS:
        comparison = _comparison_label(base, other)
        for seed in (42, 123, 2026):
            # AP and AUROC are threshold-free.
            for metric in ("ap", "auroc"):
                for scope, labels in (("label", names), ("macro", ["__macro__"])):
                    if scope == "label":
                        for label in labels:
                            a = by[("threshold_free", seed, base, scope, label)]
                            b = by[("threshold_free", seed, other, scope, label)]
                            av, bv = a[metric], b[metric]
                            if av is not None and bv is not None:
                                rows.append({
                                    "task": task, "label_set": label_set,
                                    "comparison": comparison, "base_modality": base,
                                    "other_modality": other, "metric": metric.upper(),
                                    "threshold_policy": "threshold_free", "seed": seed,
                                    "scope": scope, "label": label,
                                    "point_estimate": float(av) - float(bv), "n_labels": 1,
                                })
                    else:
                        diffs = []
                        for name in names:
                            a = by[("threshold_free", seed, base, "label", name)]
                            b = by[("threshold_free", seed, other, "label", name)]
                            if a[metric] is not None and b[metric] is not None:
                                diffs.append(float(a[metric]) - float(b[metric]))
                        if diffs:
                            rows.append({
                                "task": task, "label_set": label_set,
                                "comparison": comparison, "base_modality": base,
                                "other_modality": other, "metric": metric.upper(),
                                "threshold_policy": "threshold_free", "seed": seed,
                                "scope": "macro", "label": None,
                                "point_estimate": float(np.mean(diffs)), "n_labels": len(diffs),
                            })
            # F1 is policy-specific.
            for policy in POLICIES:
                for scope, labels in (("label", names), ("macro", ["__macro__"])):
                    for label in labels:
                        a = by[(policy, seed, base, scope, label)]
                        b = by[(policy, seed, other, scope, label)]
                        rows.append({
                            "task": task, "label_set": label_set,
                            "comparison": comparison, "base_modality": base,
                            "other_modality": other, "metric": "F1",
                            "threshold_policy": policy, "seed": seed,
                            "scope": scope, "label": label if scope == "label" else None,
                            "point_estimate": float(a["f1"]) - float(b["f1"]),
                            "n_labels": 1 if scope == "label" else len(names),
                        })
    return rows


def _bootstrap(
    task_info: dict,
    label_set: str,
    reps: int,
    rng_seed: int,
) -> dict:
    names = load_label_names(resolve_label_set(label_set)["names"])
    models = task_info["models"]
    if len(models) != 9:
        raise ValueError(f"{task_info['task']}: expected 9 models, found {len(models)}")
    by_seed_modality = {}
    for model in models:
        by_seed_modality[(int(model["seed"]), model["modality"])] = model
    for seed in (42, 123, 2026):
        for modality in MODALITIES:
            if (seed, modality) not in by_seed_modality:
                raise ValueError(f"Missing model for seed={seed}, modality={modality}")
    row_index = np.asarray(models[0]["row_index"], dtype=np.int64)
    for model in models[1:]:
        if not np.array_equal(row_index, np.asarray(model["row_index"], dtype=np.int64)):
            raise ValueError("Model row_index arrays are not aligned")
        if not np.array_equal(np.asarray(models[0]["labels"]), np.asarray(model["labels"])):
            raise ValueError("Model label matrices are not aligned")
    scaffold_map, scaffold_info = _load_scaffold_map()
    cluster_id, cluster_names, groups, missing = _build_clusters(row_index, scaffold_map)
    n_clusters = len(cluster_names)
    sizes = np.asarray([len(groups[name]) for name in cluster_names], dtype=np.int64)
    blank_validation = int(sum(1 for i in row_index if scaffold_map[int(i)] == "__MISSING__"))
    if n_clusters == 0:
        raise ValueError("No scaffold clusters in Validation")

    rng = np.random.default_rng(rng_seed)
    values = defaultdict(lambda: np.full(reps, np.nan, dtype=np.float64))
    label_counts = defaultdict(lambda: np.zeros(reps, dtype=np.int64))
    resampled_molecules = np.zeros(reps, dtype=np.int64)

    point_rows, point_macro = _point_metrics(models, label_set, task_info["task"])
    point_diffs = _point_comparison(point_rows, label_set, task_info["task"])
    point_lookup = {}
    point_n_lookup = {}
    for row in point_diffs:
        key = (row["threshold_policy"], row["comparison"], row["metric"], row["scope"], row["seed"], row["label"])
        point_lookup[key] = row["point_estimate"]
        point_n_lookup[key] = row["n_labels"]
    tier_by_label = {r["label"]: r["support_tier"] for r in point_rows if r["scope"] == "label"}

    for rep in range(reps):
        draw_counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters).astype(np.float64)
        weight = draw_counts[cluster_id].astype(np.float64)
        resampled_molecules[rep] = int(weight.sum())
        per_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for seed in (42, 123, 2026):
            per_seed[seed] = {}
            for modality in MODALITIES:
                model = by_seed_modality[(seed, modality)]
                probs = model["probabilities"]
                truth = model["labels"]
                n_labels = len(names)
                arrays = {
                    "ap": np.full(n_labels, np.nan, dtype=np.float64),
                    "auroc": np.full(n_labels, np.nan, dtype=np.float64),
                    "fixed_0.5": np.full(n_labels, np.nan, dtype=np.float64),
                    "validation_f1": np.full(n_labels, np.nan, dtype=np.float64),
                }
                for index in range(n_labels):
                    ranking = _ranking_metrics(truth[:, index], probs[:, index], weight)
                    fixed = _classification_f1(truth[:, index], probs[:, index], weight, 0.5)
                    calibrated = _classification_f1(truth[:, index], probs[:, index], weight, float(model["thresholds"][index]))
                    if ranking["ap"] is not None:
                        arrays["ap"][index] = ranking["ap"]
                        arrays["auroc"][index] = ranking["auroc"]
                    arrays["fixed_0.5"][index] = fixed["f1"]
                    arrays["validation_f1"][index] = calibrated["f1"]
                per_seed[seed][modality] = arrays

        for base, other in COMPARISONS:
            comparison = _comparison_label(base, other)
            for scope in ("label", "macro"):
                label_indexes = list(range(len(names))) if scope == "label" else None
                # AP and AUROC: threshold-free, paired valid labels only.
                for metric in ("ap", "auroc"):
                    for seed in (42, 123, 2026):
                        a = per_seed[seed][base][metric]
                        b = per_seed[seed][other][metric]
                        if scope == "label":
                            for index in range(len(names)):
                                if np.isfinite(a[index]) and np.isfinite(b[index]):
                                    values[("threshold_free", comparison, metric, "label", seed, names[index])][rep] = float(a[index] - b[index])
                        else:
                            pair = a - b
                            pair = pair[np.isfinite(pair)]
                            label_counts[("threshold_free", comparison, metric, "macro", seed)][rep] = int(len(pair))
                            if len(pair):
                                values[("threshold_free", comparison, metric, "macro", seed, "__macro__")][rep] = float(np.mean(pair))
                # F1: two threshold policies, zero_division=0 ensures finite.
                for policy in POLICIES:
                    for seed in (42, 123, 2026):
                        a = per_seed[seed][base][policy]
                        b = per_seed[seed][other][policy]
                        if scope == "label":
                            for index in range(len(names)):
                                values[(policy, comparison, "f1", "label", seed, names[index])][rep] = float(a[index] - b[index])
                        else:
                            values[(policy, comparison, "f1", "macro", seed, "__macro__")][rep] = float(np.mean(a - b))

    ci_rows = []
    metric_names = {"ap": "AP", "auroc": "AUROC", "f1": "F1"}
    policy_metrics = {
        "threshold_free": ("ap", "auroc"),
        "fixed_0.5": ("f1",),
        "validation_f1": ("f1",),
    }
    for policy, metrics in policy_metrics.items():
        for base, other in COMPARISONS:
            comparison = _comparison_label(base, other)
            for metric in metrics:
                for scope in ("label", "macro"):
                    labels = names if scope == "label" else ["__macro__"]
                    for label in labels:
                        index = names.index(label) if scope == "label" else ""
                        per_seed_values = {}
                        per_seed_points = {}
                        for seed in (42, 123, 2026):
                            arr = values.get((policy, comparison, metric, scope, seed, label), np.full(reps, np.nan))
                            per_seed_values[seed] = arr
                            per_seed_points[seed] = point_lookup.get((policy, comparison, metric_names[metric], scope, seed, label if scope == "label" else None))
                        # Per-seed CI.
                        for seed in (42, 123, 2026):
                            arr = per_seed_values[seed]
                            valid = arr[np.isfinite(arr)]
                            if len(valid) >= 2:
                                ci = np.percentile(valid, CI_PERCENTILES)
                                low, high = float(ci[0]), float(ci[1])
                            else:
                                low, high = None, None
                            ci_rows.append({
                                "task": task_info["task"], "label_set": label_set, "threshold_policy": policy,
                                "comparison": comparison, "metric": metric_names[metric], "scope": scope,
                                "seed": seed, "label_index": index, "label": label,
                                "point_estimate": per_seed_points[seed], "ci_low": low, "ci_high": high,
                                "seed_sd": None, "n_valid": int(len(valid)), "n_undefined": int(reps - len(valid)),
                                "n_undefined_ap": int(reps - len(valid)) if metric == "ap" else None,
                                "n_undefined_auroc": int(reps - len(valid)) if metric == "auroc" else None,
                                "n_labels_total": len(names) if scope == "macro" else 1,
                                "n_labels_defined": point_n_lookup.get((policy, comparison, metric_names[metric], scope, seed, label if scope == "label" else None)),
                                "support_tier": tier_by_label.get(label, "") if scope == "label" else "",
                                "direction": comparison,
                            })
                        # 3-seed mean uses the same replicate index for all 3 seeds.
                        matrix = np.vstack([per_seed_values[s] for s in (42, 123, 2026)])
                        valid_rows = np.all(np.isfinite(matrix), axis=0)
                        three = matrix[:, valid_rows].mean(axis=0) if valid_rows.any() else np.asarray([])
                        seed_points = [per_seed_points[s] for s in (42, 123, 2026)]
                        point_defined_labels = point_n_lookup.get((policy, comparison, metric_names[metric], scope, "three_seed_mean", label if scope == "label" else None))
                        if scope == "label":
                            point_defined_labels = 1
                        elif metric == "f1":
                            point_defined_labels = len(names)
                        # Macro AP/AUROC for the 3-seed mean must use one
                        # common valid-label set across all three seeds.
                        if scope == "macro" and metric in ("ap", "auroc"):
                            seed_label_arrays = {}
                            for seed in (42, 123, 2026):
                                seed_label_arrays[seed] = np.vstack([
                                    values.get((policy, comparison, metric, "label", seed, name), np.full(reps, np.nan))
                                    for name in names
                                ])
                            label_matrix = np.stack([seed_label_arrays[s] for s in (42, 123, 2026)], axis=0)  # seeds x labels x reps
                            three_values = np.full(reps, np.nan)
                            common_counts = np.zeros(reps, dtype=np.int64)
                            for rep_index in range(reps):
                                label_diffs = label_matrix[:, :, rep_index]
                                valid_labels = np.all(np.isfinite(label_diffs), axis=0)
                                common_counts[rep_index] = int(valid_labels.sum())
                                if valid_labels.any():
                                    three_values[rep_index] = float(np.mean(label_diffs[:, valid_labels].mean(axis=0)))
                            three = three_values[np.isfinite(three_values)]
                            label_counts[(policy, comparison, metric, "macro", "three_seed_mean")] = common_counts
                            label_points = {}
                            for name in names:
                                pts = [point_lookup.get((policy, comparison, metric_names[metric], "label", s, name)) for s in (42, 123, 2026)]
                                if all(p is not None for p in pts):
                                    label_points[name] = pts
                            if label_points:
                                point_mean = float(np.mean([np.mean(v) for v in label_points.values()]))
                                seed_sd = float(statistics.stdev([np.mean([v[j] for v in label_points.values()]) for j in range(3)]))
                                point_defined_labels = len(label_points)
                            else:
                                point_mean = None
                                seed_sd = None
                                point_defined_labels = 0
                        if len(three) >= 2:
                            ci = np.percentile(three, CI_PERCENTILES)
                            low3, high3 = float(ci[0]), float(ci[1])
                        else:
                            low3, high3 = None, None
                        if not (scope == "macro" and metric in ("ap", "auroc")):
                            point_mean = None if any(v is None for v in seed_points) else float(np.mean(seed_points))
                            seed_sd = None if any(v is None for v in seed_points) else float(statistics.stdev(seed_points))
                        ci_rows.append({
                            "task": task_info["task"], "label_set": label_set, "threshold_policy": policy,
                            "comparison": comparison, "metric": metric_names[metric], "scope": scope,
                            "seed": "three_seed_mean", "label_index": index, "label": label,
                            "point_estimate": point_mean, "ci_low": low3, "ci_high": high3,
                            "seed_sd": seed_sd, "n_valid": int(len(three)), "n_undefined": int(reps - len(three)),
                            "n_undefined_ap": int(reps - len(three)) if metric == "ap" else None,
                            "n_undefined_auroc": int(reps - len(three)) if metric == "auroc" else None,
                            "n_labels_total": len(names) if scope == "macro" else 1,
                            "n_labels_defined": point_defined_labels,
                            "support_tier": tier_by_label.get(label, "") if scope == "label" else "",
                            "direction": comparison,
                        })

    return {
        "task": task_info["task"],
        "label_set": label_set,
        "point_rows": point_rows,
        "point_diffs": point_diffs,
        "ci_rows": ci_rows,
        "label_counts": label_counts,
        "scaffold_info": scaffold_info,
        "cluster_sizes": sizes.tolist(),
        "cluster_names": cluster_names,
        "groups": groups,
        "missing_scaffold_rows": missing,
        "blank_scaffold_rows_validation": blank_validation,
        "resampled_molecules": resampled_molecules,
        "n_clusters": n_clusters,
        "n_samples": int(len(row_index)),
        "reps": reps,
        "rng_seed": rng_seed,
    }


def _check_accepted_inputs(task_info: dict, label_set: str) -> dict:
    verification_path = PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B3" / "b3_5b_manifest_verification_v2.json"
    provenance_path = PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B3" / "code_provenance.json"
    if not verification_path.is_file():
        raise FileNotFoundError(verification_path)
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    verification = read_json(verification_path)
    if verification.get("no_test_access") is not True or verification.get("overall_status") != "PASS":
        raise ValueError("B3.5-B v2 verification is not a passing Validation-only record")
    raw_assets = {Path(r["path"]).name: r for r in verification.get("raw_assets") or []}
    manifest_record = raw_assets.get("qm9s_manifest_prepared.csv")
    structure_record = raw_assets.get("qm9s_structure_library.csv")
    if not manifest_record or not structure_record:
        raise ValueError("B3.5-B v2 does not register manifest/structure library")
    manifest_actual = _sha(MANIFEST_PATH)
    if manifest_actual != manifest_record.get("sha256"):
        raise ValueError("qm9s_manifest_prepared.csv changed since B3.5-B v2")
    structure_actual = _sha(PROJECT_ROOT / "processed/qm9s/structures/qm9s_structure_library.csv")
    if structure_actual != structure_record.get("sha256"):
        raise ValueError("qm9s_structure_library.csv changed since B3.5-B v2")
    metrics_record = (verification.get("metrics") or {}).get(task_info["task"])
    if not metrics_record or not task_info.get("metrics_path"):
        raise ValueError("Missing accepted metrics record")
    metrics_actual = _sha(task_info["metrics_path"])
    if metrics_actual != metrics_record.get("model_metrics_sha256"):
        raise ValueError("Validation metrics changed since B3.5-B v2")
    b3a_record = (verification.get("b3a") or {}).get(task_info["task"])
    if not b3a_record:
        raise ValueError("Missing B3-A summary binding")
    b3a_path = Path(b3a_record["summary_path"])
    if _sha(b3a_path) != b3a_record.get("summary_sha256"):
        raise ValueError("B3-A summary changed since B3.5-B v2")
    b3a_summary = read_json(b3a_path)
    if task_info.get("best_single") != b3a_summary.get("best_single_modality"):
        raise ValueError("Best Single differs from accepted B3-A record")
    provenance = read_json(provenance_path)
    arrays = {(r["task"], r["run_name"]): r for r in (provenance.get("formal_assets") or {}).get("validation_arrays") or []}
    manifests = {(r["task"], r["run_name"]): r for r in (provenance.get("formal_assets") or {}).get("prediction_manifests") or []}
    thresholds = {}
    for r in (provenance.get("formal_assets") or {}).get("threshold_files") or []:
        run_name = Path(r["path"]).parent.name
        thresholds[(r["task"], run_name)] = r
    if len(arrays) != 18 or len(manifests) != 18 or len(thresholds) != 18:
        raise ValueError("Accepted provenance does not contain 18 model records")
    checks = []
    for model in task_info["models"]:
        identity = (task_info["task"], model["run_name"])
        a = arrays.get(identity); mf = manifests.get(identity); th = thresholds.get(identity)
        if a is None or mf is None or th is None:
            raise ValueError(f"Missing accepted provenance for {identity}")
        expected = {
            "probabilities_sha256": a["probabilities"],
            "row_index_sha256": a["row_index"],
            "labels_sha256": a["labels"],
            "manifest_sha256": mf["sha256"],
            "thresholds_sha256": th["sha256"],
        }
        for field, value in expected.items():
            if model[field] != value:
                raise ValueError(f"Input hash mismatch for {identity}/{field}")
        checks.append({"task": task_info["task"], "run_name": model["run_name"], **expected})
    return {
        "verification_path": relative_to_root(verification_path),
        "verification_sha256": _sha(verification_path),
        "manifest": {"path": relative_to_root(MANIFEST_PATH), "sha256": manifest_actual, "used_in_computation": True},
        "structure_library": {"path": "processed/qm9s/structures/qm9s_structure_library.csv", "sha256": structure_actual, "used_in_computation": False},
        "metrics": {"path": relative_to_root(task_info["metrics_path"]), "sha256": metrics_actual, "used_in_computation": False},
        "b3a": {"path": b3a_path.name, "sha256": _sha(b3a_path), "best_single": b3a_summary.get("best_single_modality")},
        "code_provenance": {"path": relative_to_root(provenance_path), "sha256": _sha(provenance_path), "n_models": len(checks), "records": checks},
    }


def _run_task(label_set: str, task: str, reps: int, out_root: Path, rng_seed: int) -> dict:
    from training.line_b.common import atomic_output_dir

    task_info = _load_task_inputs(
        label_set,
        PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS",
        PROJECT_ROOT / "runs" / "LINE-B-THRESHOLDS",
        PROJECT_ROOT / "runs" / "LINE-B-METRICS",
        False,
    )
    accepted_inputs = _check_accepted_inputs(task_info, label_set)
    data = _bootstrap(task_info, label_set, reps, rng_seed)
    task_out = Path(out_root) / task / "bootstrap"
    with atomic_output_dir(task_out) as temporary:
        _write_csv(
            temporary / "scaffold_clusters.csv",
            [{"scaffold": name, "size": len(data["groups"][name])} for name in data["cluster_names"]],
            ["scaffold", "size"],
        )
        _write_csv(
            temporary / "point_estimates.csv",
            data["point_rows"],
            ["task", "label_set", "threshold_policy", "seed", "modality", "label_index", "label", "support", "support_tier", "ap", "auroc", "f1", "n_defined_ap", "n_defined_auroc", "n_defined_f1", "n_total_labels", "scope"],
        )
        _write_csv(
            temporary / "bootstrap_ci.csv",
            data["ci_rows"],
            ["task", "label_set", "threshold_policy", "comparison", "metric", "scope", "seed", "label_index", "label", "point_estimate", "ci_low", "ci_high", "seed_sd", "n_valid", "n_undefined", "n_undefined_ap", "n_undefined_auroc", "n_labels_total", "n_labels_defined", "support_tier", "direction"],
        )
        # Point comparison rows are retained in the machine summary as well.
        summary = {
            "phase": "B3.5-C1",
            "task": task,
            "label_set": label_set,
            "split": "validation",
            "n_models": 9,
            "n_labels": len(load_label_names(resolve_label_set(label_set)["names"])),
            "n_scaffolds": data["n_clusters"],
            "n_samples": data["n_samples"],
            "scaffold_size_min": int(min(data["cluster_sizes"])),
            "scaffold_size_max": int(max(data["cluster_sizes"])),
            "scaffold_size_median": float(statistics.median(data["cluster_sizes"])),
            "scaffold_size_mean": float(np.mean(data["cluster_sizes"])),
            "scaffold_singletons": int(sum(1 for size in data["cluster_sizes"] if size == 1)),
            "scaffold_size_histogram": {
                "1": int(sum(1 for size in data["cluster_sizes"] if size == 1)),
                "2-5": int(sum(1 for size in data["cluster_sizes"] if 2 <= size <= 5)),
                "6-10": int(sum(1 for size in data["cluster_sizes"] if 6 <= size <= 10)),
                ">10": int(sum(1 for size in data["cluster_sizes"] if size > 10)),
            },
            "missing_scaffold_rows": data["missing_scaffold_rows"],
            "blank_scaffold_rows_full": data["scaffold_info"].get("missing_scaffold_rows", 0),
            "blank_scaffold_rows_validation": data["blank_scaffold_rows_validation"],
            "resampled_molecules": {
                "min": int(np.min(data["resampled_molecules"])),
                "max": int(np.max(data["resampled_molecules"])),
                "mean": float(np.mean(data["resampled_molecules"])),
                "median": float(np.median(data["resampled_molecules"])),
                "distribution": {str(k): int((data["resampled_molecules"] == k).sum()) for k in range(int(np.min(data["resampled_molecules"])), int(np.max(data["resampled_molecules"])) + 1)},
            },
            "best_single": task_info.get("best_single"),
            "threshold_policies": {"primary": "fixed_0.5", "secondary": "validation_f1", "ap_auroc": "threshold_free"},
            "support_rule": "B0 tier uses min(val,test) positives; C1 is Validation-only, so support_tier is provisional validation-only and no Test positives are read",
            "very_low_labels": sorted({r["label"] for r in data["point_rows"] if r["scope"] == "label" and r["support_tier"] == "very_low"}),
            "comparisons": [_comparison_label(a, b) for a, b in COMPARISONS],
            "reps": data["reps"],
            "rng_seed": data["rng_seed"],
            "rng": "numpy.random.default_rng; scaffold draw multinomial-equivalent via rng.integers + bincount",
            "resampling": "scaffold clusters with replacement; same draw across all modalities, seeds, labels and metrics; sample weights are cluster draw counts",
            "ci": "95% percentile; CI undefined when n_valid < 2",
            "undefined_policy": "AP/AUROC undefined when either weighted class is empty; paired valid only; F1 zero_division=0",
            "macro_rule": "AP/AUROC macro uses the paired labels defined in the same replicate; 3-seed macro uses labels defined in all 3 seeds; F1 macro uses all labels; n_undefined is reported per group",
            "seed_sd_rule": "sample SD of the three seed point estimates; separate from the paired bootstrap CI",
            "direction_rule": "IR-Raman = IR minus Raman; UV-IR = UV minus IR; UV-Raman = UV minus Raman; never reversed across outputs",
            "no_test_access": True,
            "point_estimates": data["point_rows"],
            "point_comparisons": data["point_diffs"],
            "ci_rows": data["ci_rows"],
            "undefined_counts": [
                {k: r[k] for k in ("comparison", "metric", "scope", "seed", "label", "n_valid", "n_undefined", "n_undefined_ap", "n_undefined_auroc")}
                for r in data["ci_rows"] if r["n_undefined"] > 0
            ],
            "macro_label_count_distributions": {
                "|".join([str(x) for x in key]): {str(k): int((arr == k).sum()) for k in range(0, int(arr.max()) + 1)}
                for key, arr in data["label_counts"].items()
            },
            "scaffold": {
                **data["scaffold_info"],
                "n_scaffolds": data["n_clusters"],
                "n_samples": data["n_samples"],
                "size_min": int(min(data["cluster_sizes"])),
                "size_max": int(max(data["cluster_sizes"])),
                "size_median": float(statistics.median(data["cluster_sizes"])),
                "size_mean": float(np.mean(data["cluster_sizes"])),
                "singletons": int(sum(1 for size in data["cluster_sizes"] if size == 1)),
                "missing_scaffold_rows": data["missing_scaffold_rows"],
                "blank_scaffold_rows_full": data["scaffold_info"].get("missing_scaffold_rows", 0),
                "blank_scaffold_rows_validation": data["blank_scaffold_rows_validation"],
                "missing_group": "__MISSING__" if data["scaffold_info"].get("missing_scaffold_rows", 0) else None,
                "row_index_file_sha256": _sha(PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS" / task / "IR" / str(next(m["run_name"] for m in task_info["models"] if m["modality"] == "ir")) / "validation" / "row_index.npy"),
                "row_index_array_sha256": next(m["row_index_sha256"] for m in task_info["models"] if m["modality"] == "ir"),
            },
            "inputs": {
                "manifest": {"path": "processed/qm9s/prepared/qm9s_manifest_prepared.csv", "sha256": _sha(MANIFEST_PATH)},
                "structure_library": {"path": "processed/qm9s/structures/qm9s_structure_library.csv", "sha256": _sha(PROJECT_ROOT / "processed/qm9s/structures/qm9s_structure_library.csv")},
                "accepted_verification": accepted_inputs,
                "accepted_provenance": accepted_inputs["code_provenance"],
                "predictions_root": "runs/LINE-B-PREDICTIONS",
                "thresholds_root": "runs/LINE-B-THRESHOLDS",
                "models": [
                    {
                        "run_name": m["run_name"], "modality": m["modality"], "seed": int(m["seed"]),
                        "manifest_path": relative_to_root(m["manifest_path"]), "manifest_sha256": m["manifest_sha256"],
                        "probabilities_sha256": m["probabilities_sha256"], "row_index_sha256": m["row_index_sha256"],
                        "labels_sha256": m["labels_sha256"], "split_path": relative_to_root(m["split_path"]),
                        "split_sha256": m["split_sha256"], "checkpoint_sha256": m["checkpoint_sha256"],
                        "threshold_path": relative_to_root(m["threshold_path"]), "thresholds_sha256": m["thresholds_sha256"],
                        "calibration_manifest_path": relative_to_root(m["calibration_manifest_path"]),
                    }
                    for m in task_info["models"]
                ],
            },
            "code_sha256": {
                rel: _sha(PROJECT_ROOT / rel)
                for rel in (
                    "training/line_b/bootstrap_line_b_scaffold.py",
                    "training/line_b/common.py",
                    "training/line_b/metrics.py",
                    "training/line_b/analyze_line_b_complementarity.py",
                )
            },
            "feature_flags": {"validation_development": True, "formal_2000_reps": False},
            "outputs": {
                name: _sha(temporary / name)
                for name in ("scaffold_clusters.csv", "point_estimates.csv", "bootstrap_ci.csv")
            },
            "environment": _environment(),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(temporary / "summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "Shared14", "UVSPEC15"], default="all")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--out-root", type=Path, required=True, help="Required temporary output directory; formal runs are not part of B3.5-C1.")
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG)
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test:
        raise SystemExit("Test is not authorized in B3.5-C1")
    if args.reps < 1 or args.reps > DEV_MAX_REPS:
        raise SystemExit(f"B3.5-C1 is Validation development only; --reps must be 1..{DEV_MAX_REPS}. Formal 2,000 belongs to B4 Test.")
    out_root = args.out_root.resolve()
    if out_root.is_relative_to(PROJECT_ROOT.resolve()):
        raise SystemExit("B3.5-C1 outputs must use a temporary directory outside the project; formal LINE-B-ANALYSIS/LINE-B-FINAL writes are not authorized.")
    label_sets = [("Shared14", "shared14"), ("UVSPEC15", "uvspec15")]
    if args.task != "all":
        label_sets = [pair for pair in label_sets if pair[0] == args.task]
    summaries = []
    for task, label_set in label_sets:
        summaries.append(_run_task(label_set, task, args.reps, args.out_root, args.rng_seed))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
