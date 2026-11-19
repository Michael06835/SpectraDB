"""Extended Validation-only error-complementarity analysis for Line B.

This is the B0 v0.2 section 7.3 companion to the already accepted B3-A
``analyze_line_b_complementarity.py``.  It reuses the accepted loaders and
pair/three-modality calculations, and adds:

* exact McNemar tests with Holm correction;
* per-molecule error distributions;
* Spearman correlations between modalities;
* deterministic top-10% hard-sample sets and Jaccard;
* molecule-level exact-match patterns;
* high-confidence diagnostic FP/FN rates.

No model is loaded, no threshold is refitted, no Test array is read, and no
formal B3-A asset is modified.  Outputs are temporary-development only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    atomic_output_dir,
    load_label_names,
    relative_to_root,
    resolve_label_set,
    sha256_file,
    write_json,
)
from training.line_b.analyze_line_b_complementarity import (
    DEFAULT_THRESHOLD,
    MODALITIES,
    POLICIES,
    _aggregate,
    _load_task_inputs,
    _mean_sd,
    _model_rows,
    _pair_rows,
    _three_rows,
    _write_csv,
)
from training.line_b.bootstrap_line_b_scaffold import _check_accepted_inputs

PHASE = "B3.5-C2"
UNORDERED_PAIRS = (("ir", "raman"), ("uvvis", "ir"), ("uvvis", "raman"))
ORDERED_PAIRS = tuple(
    (a, b) for a in MODALITIES for b in MODALITIES if a != b
)
PAIR_LABELS = {
    ("ir", "raman"): "IR-Raman",
    ("uvvis", "ir"): "UV-IR",
    ("uvvis", "raman"): "UV-Raman",
}
DIRECTION_LABELS = {
    ("ir", "raman"): "IR-Raman",
    ("raman", "ir"): "Raman-IR",
    ("uvvis", "ir"): "UV-IR",
    ("ir", "uvvis"): "IR-UV",
    ("uvvis", "raman"): "UV-Raman",
    ("raman", "uvvis"): "Raman-UV",
}
HC_FP_PROB = 0.90
HC_FN_PROB = 0.10
HARD_FRACTION = 0.10
DEV_MAX_LABELS = 15


def _pair_label(base: str, other: str) -> str:
    return PAIR_LABELS[(base, other)]


def _direction_label(base: str, other: str) -> str:
    return DIRECTION_LABELS[(base, other)]


def _exact_mcnemar(base_only: int, other_only: int) -> tuple[float, int]:
    """Two-sided exact McNemar/equal-probability binomial p-value."""
    b = int(base_only)
    c = int(other_only)
    n = b + c
    if n == 0:
        return 1.0, 0
    k = min(b, c)
    # Log-space tail avoids converting enormous 2**n integers/combinations.
    log_tail = np.logaddexp.reduce([
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        for i in range(k + 1)
    ])
    log_p = math.log(2.0) + float(log_tail) - n * math.log(2.0)
    p = min(1.0, math.exp(log_p)) if log_p < 0 else 1.0
    return p, n


def _holm(rows: list[dict]) -> list[dict]:
    """Holm step-down adjustment inside one explicit family."""
    ordered = sorted(rows, key=lambda row: (row["p_raw"], row["label_index"], row["comparison"]))
    m = len(ordered)
    adjusted = 1.0
    for i, row in enumerate(ordered):
        candidate = min(1.0, (m - i) * row["p_raw"])
        adjusted = max(adjusted, candidate) if i else candidate
        row["p_holm"] = adjusted
        row["family_size"] = m
    return rows


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float | None, str | None]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None, "too_few_samples"
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None, "constant_vector"
    ra = _rank_average(a)
    rb = _rank_average(b)
    if np.all(ra == ra[0]) or np.all(rb == rb[0]):
        return None, "constant_rank_vector"
    ca = ra - ra.mean()
    cb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ca * ca) * np.sum(cb * cb)))
    if denom == 0.0:
        return None, "zero_rank_variance"
    rho = float(np.sum(ca * cb) / denom)
    if not np.isfinite(rho):
        return None, "non_finite_correlation"
    return rho, None


def _hard_set(n_errors: np.ndarray, row_index: np.ndarray) -> tuple[np.ndarray, dict]:
    n_errors = np.asarray(n_errors, dtype=np.int64)
    row_index = np.asarray(row_index, dtype=np.int64)
    n = len(n_errors)
    k = int(math.ceil(HARD_FRACTION * n))
    order = np.lexsort((row_index, -n_errors,))  # primary: -n_errors; tie: row_index asc
    selected = order[:k]
    boundary = int(n_errors[selected[-1]]) if k else 0
    tie_count = int((n_errors == boundary).sum())
    return selected, {
        "n_hard": k,
        "boundary_error": boundary,
        "boundary_tie_count": tie_count,
        "tie_rule": "n_errors descending, then row_index ascending, deterministic ceil(0.10*N)",
    }


def _jaccard(a: set, b: set) -> tuple[int, int, float]:
    union = a | b
    inter = a & b
    return len(inter), len(union), (1.0 if len(union) == 0 else len(inter) / len(union))


def _three_seed_summary(analysis: str, rows: list[dict], group_fields: tuple, metrics: tuple) -> list[dict]:
    """Reuse B3-A _aggregate; add explicit n_valid/n_undefined aliases."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups[key].append(int(row["seed"]))
    bad = {key: seeds for key, seeds in groups.items() if Counter(seeds) != Counter({42: 1, 123: 1, 2026: 1})}
    if bad:
        raise ValueError(f"Three-seed family must contain exactly seeds 42/123/2026 once: {bad}")
    output = _aggregate(rows, group_fields, metrics)
    for row in output:
        row["analysis"] = analysis
        row["n_valid"] = row.get("n_seeds")
        row["n_undefined"] = row.get("n_undefined_seeds")
        row["scope"] = "three_seed_descriptive"
    return output


def _high_conf_seed_overall(rows: list[dict]) -> list[dict]:
    """Collapse labels within each seed before any cross-seed aggregation."""
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["task"], row["label_set"], row["modality"], row["threshold_policy"], int(row["seed"]),
        )].append(row)
    out = []
    for key, group in sorted(groups.items()):
        task, label_set, modality, policy, seed = key
        all_fp = int(sum(r["all_fp"] for r in group))
        all_fn = int(sum(r["all_fn"] for r in group))
        all_errors = all_fp + all_fn
        hc_fp = int(sum(r["high_conf_fp"] for r in group))
        hc_fn = int(sum(r["high_conf_fn"] for r in group))
        hc_errors = hc_fp + hc_fn
        n_pairs = int(sum(r["n_pairs"] for r in group))
        out.append({
            "task": task, "label_set": label_set, "modality": modality,
            "threshold_policy": policy, "seed": seed, "n_labels": len(group),
            "all_fp": all_fp, "all_fn": all_fn, "all_errors": all_errors,
            "high_conf_fp": hc_fp, "high_conf_fn": hc_fn, "high_conf_errors": hc_errors,
            "n_pairs": n_pairs,
            "high_conf_fp_fraction_of_fp": (hc_fp / all_fp) if all_fp else None,
            "high_conf_fn_fraction_of_fn": (hc_fn / all_fn) if all_fn else None,
            "high_conf_fraction_of_errors": (hc_errors / all_errors) if all_errors else None,
            "high_conf_fraction_of_pairs": hc_errors / n_pairs if n_pairs else None,
        })
    return out


def _mcnemar_rows(task_info: dict, pair_rows: list[dict]) -> list[dict]:
    wanted = {(_pair_label(a, b),): (a, b) for a, b in UNORDERED_PAIRS}
    rows = []
    for seed in (42, 123, 2026):
        for policy in POLICIES:
            family = []
            for base, other in UNORDERED_PAIRS:
                label = _pair_label(base, other)
                for index, name in enumerate(task_info["expected_names"]):
                    row = next(
                        (r for r in pair_rows if r["seed"] == seed and r["threshold_policy"] == policy
                         and r["base_modality"] == base and r["other_modality"] == other and r["label_index"] == index),
                        None,
                    )
                    if row is None:
                        raise ValueError(f"Missing pair row {label}/{seed}/{policy}/{name}")
                    p, discordant = _exact_mcnemar(row["base_only_correct"], row["other_only_correct"])
                    n = int(row["n_samples"])
                    base_correct = int(row["base_only_correct"] + row["both_correct"])
                    other_correct = int(row["other_only_correct"] + row["both_correct"])
                    family.append({
                        "task": task_info["task"], "label_set": task_info["label_set"],
                        "seed": seed, "threshold_policy": policy, "label_index": index, "label": name,
                        "comparison": label, "base_modality": base, "other_modality": other,
                        "n_samples": n, "both_correct": int(row["both_correct"]),
                        "base_only_correct": int(row["base_only_correct"]),
                        "other_only_correct": int(row["other_only_correct"]),
                        "both_wrong": int(row["both_wrong"]),
                        "discordant_count": discordant,
                        "base_correct_rate": base_correct / n,
                        "other_correct_rate": other_correct / n,
                        "correct_rate_difference": (base_correct - other_correct) / n,
                        "p_raw": p,
                        "family_id": f"{task_info['task']}|seed={seed}|policy={policy}",
                        "family_size": None, "p_holm": None,
                        "exploratory_only": True,
                    })
            _holm(family)
            rows.extend(family)
    return rows


def _sample_errors(task_info: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    sample_rows = []
    summary_rows = []
    seed_vectors: dict[tuple, dict[int, np.ndarray]] = defaultdict(dict)
    row_index = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    n_labels = len(task_info["expected_names"])
    high_conf_summary = []
    for model in task_info["models"]:
        seed = int(model["seed"])
        modality = model["modality"]
        truth = model["labels"]
        for policy in POLICIES:
            thresholds = np.asarray(
                [0.5 if policy == "fixed_0.5" else model["thresholds"][i] for i in range(n_labels)],
                dtype=np.float64,
            )
            probs = model["probabilities"]
            pred = probs >= thresholds.reshape(1, -1)
            errors = pred != (truth == 1)
            n_errors = errors.sum(axis=1).astype(np.int64)
            hc_fp = (truth == 0) & pred & (probs >= HC_FP_PROB)
            hc_fn = (truth == 1) & (~pred) & (probs <= HC_FN_PROB)
            hc = hc_fp | hc_fn
            n_hc = hc.sum(axis=1).astype(np.int64)
            for i, row_idx in enumerate(row_index):
                sample_rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "modality": modality, "threshold_policy": policy,
                    "row_index": int(row_idx), "n_labels": n_labels,
                    "n_errors": int(n_errors[i]),
                    "hamming_error": float(n_errors[i] / n_labels),
                    "exact_match": bool(n_errors[i] == 0),
                    "n_high_confidence_errors": int(n_hc[i]),
                    "has_high_confidence_error": bool(n_hc[i] > 0),
                })
            freq = {int(k): int(v) for k, v in zip(*np.unique(n_errors, return_counts=True))}
            summary_rows.append({
                "task": task_info["task"], "label_set": task_info["label_set"],
                "seed": seed, "modality": modality, "threshold_policy": policy,
                "n_samples": len(n_errors), "n_labels": n_labels,
                "mean_n_errors": float(n_errors.mean()),
                "median_n_errors": float(np.median(n_errors)),
                "q1_n_errors": float(np.percentile(n_errors, 25, method="linear")),
                "q3_n_errors": float(np.percentile(n_errors, 75, method="linear")),
                "iqr_n_errors": float(np.percentile(n_errors, 75, method="linear") - np.percentile(n_errors, 25, method="linear")),
                "exact_match_count": int((n_errors == 0).sum()),
                "exact_match_rate": float((n_errors == 0).mean()),
                "error_count_distribution": json.dumps(freq, sort_keys=True),
                "n_high_confidence_errors": int(n_hc.sum()),
                "n_samples_with_high_confidence_error": int((n_hc > 0).sum()),
            })
            seed_vectors[(modality, policy)][seed] = n_errors
            # High-confidence diagnostic rates by label.
            for index, name in enumerate(task_info["expected_names"]):
                fp = (truth[:, index] == 0) & pred[:, index]
                fn = (truth[:, index] == 1) & (~pred[:, index])
                all_fp = int(fp.sum()); all_fn = int(fn.sum())
                hc_fp_count = int((fp & (probs[:, index] >= HC_FP_PROB)).sum())
                hc_fn_count = int((fn & (probs[:, index] <= HC_FN_PROB)).sum())
                n_pairs = len(truth)
                high_conf_summary.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "modality": modality, "threshold_policy": policy,
                    "label_index": index, "label": name, "support": int((truth[:, index] == 1).sum()),
                    "n_pairs": n_pairs, "all_fp": all_fp, "all_fn": all_fn,
                    "all_errors": all_fp + all_fn, "high_conf_fp": hc_fp_count,
                    "high_conf_fn": hc_fn_count, "high_conf_errors": hc_fp_count + hc_fn_count,
                    "high_conf_fp_fraction_of_fp": (hc_fp_count / all_fp) if all_fp else None,
                    "high_conf_fn_fraction_of_fn": (hc_fn_count / all_fn) if all_fn else None,
                    "high_conf_fraction_of_errors": ((hc_fp_count + hc_fn_count) / (all_fp + all_fn)) if (all_fp + all_fn) else None,
                    "high_conf_fraction_of_pairs": (hc_fp_count + hc_fn_count) / n_pairs,
                })
    three_rows = []
    for (modality, policy), by_seed in seed_vectors.items():
        stack = np.vstack([by_seed[s] for s in (42, 123, 2026)])
        mean = stack.mean(axis=0)
        sd = stack.std(axis=0, ddof=1)
        for i, row_idx in enumerate(row_index):
            three_rows.append({
                "task": task_info["task"], "label_set": task_info["label_set"],
                "modality": modality, "threshold_policy": policy, "row_index": int(row_idx),
                "mean_n_errors": float(mean[i]), "sd_n_errors": float(sd[i]),
                "exact_match_all_seeds": bool(np.all(stack[:, i] == 0)),
            })
    return sample_rows, summary_rows, three_rows, {"high_conf_summary": high_conf_summary}


def _spearman_rows(task_info: dict, sample_rows: list[dict]) -> list[dict]:
    by = {}
    for row in sample_rows:
        by[(row["seed"], row["threshold_policy"], row["modality"], row["row_index"])] = row["n_errors"]
    rows = []
    for seed in (42, 123, 2026):
        for policy in POLICIES:
            vectors = {}
            for modality in MODALITIES:
                vectors[modality] = np.asarray([
                    by[(seed, policy, modality, int(row_idx))]
                    for row_idx in task_info["models"][0]["row_index"]
                ], dtype=np.float64)
            for a, b in UNORDERED_PAIRS:
                rho, reason = _spearman(vectors[a], vectors[b])
                rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy,
                    "comparison": _pair_label(a, b), "modality_a": a, "modality_b": b,
                    "n": len(vectors[a]), "spearman_rho": rho,
                    "status": "defined" if rho is not None else "undefined",
                    "undefined_reason": reason,
                })
    return rows


def _hard_rows(task_info: dict, sample_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    by = defaultdict(dict)
    for row in sample_rows:
        by[(row["seed"], row["threshold_policy"], row["modality"])][row["row_index"]] = row["n_errors"]
    row_index = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    hard_rows = []
    pair_rows = []
    membership_rows = []
    selected_by = {}
    for seed in (42, 123, 2026):
        for policy in POLICIES:
            selected_sets = {}
            for modality in MODALITIES:
                vec = np.asarray([by[(seed, policy, modality)][int(r)] for r in row_index], dtype=np.int64)
                selected, info = _hard_set(vec, row_index)
                selected_sets[modality] = set(row_index[selected].tolist())
                selected_by[(seed, policy, modality)] = selected_sets[modality]
                hard_rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "modality": modality,
                    **info,
                })
            for a, b in UNORDERED_PAIRS:
                inter, union, jac = _jaccard(selected_sets[a], selected_sets[b])
                pair_rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "comparison": _pair_label(a, b),
                    "modality_a": a, "modality_b": b, "intersection": inter, "union": union,
                    "jaccard": jac, "boundary_tie_note": "Jaccard may be unstable when large n_errors ties define the boundary",
                })
            for row_idx in row_index:
                membership_rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "row_index": int(row_idx),
                    "ir_hard": bool(int(row_idx) in selected_sets["ir"]),
                    "raman_hard": bool(int(row_idx) in selected_sets["raman"]),
                    "uvvis_hard": bool(int(row_idx) in selected_sets["uvvis"]),
                })
    return hard_rows, pair_rows, membership_rows, {"selected_by": selected_by}


def _molecule_exact_rows(task_info: dict) -> tuple[list[dict], list[dict]]:
    row_index = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    membership = []
    summary = []
    for seed in (42, 123, 2026):
        models = {m["modality"]: m for m in task_info["models"] if int(m["seed"]) == seed}
        for policy in POLICIES:
            exact = {}
            for modality in MODALITIES:
                model = models[modality]
                thresholds = np.asarray(
                    [0.5 if policy == "fixed_0.5" else model["thresholds"][i] for i in range(len(task_info["expected_names"]))],
                    dtype=np.float64,
                )
                pred = model["probabilities"] >= thresholds.reshape(1, -1)
                exact[modality] = np.all(pred == (model["labels"] == 1), axis=1)
            all_correct = exact["ir"] & exact["raman"] & exact["uvvis"]
            all_wrong = (~exact["ir"]) & (~exact["raman"]) & (~exact["uvvis"])
            partial = (~all_correct) & (~all_wrong)
            only_ir = exact["ir"] & (~exact["raman"]) & (~exact["uvvis"])
            only_raman = exact["raman"] & (~exact["ir"]) & (~exact["uvvis"])
            only_uv = exact["uvvis"] & (~exact["ir"]) & (~exact["raman"])
            for i, row_idx in enumerate(row_index):
                membership.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "row_index": int(row_idx),
                    "ir_exact_match": bool(exact["ir"][i]), "raman_exact_match": bool(exact["raman"][i]),
                    "uvvis_exact_match": bool(exact["uvvis"][i]),
                })
            n = len(row_index)
            summary.append({
                "task": task_info["task"], "label_set": task_info["label_set"],
                "seed": seed, "threshold_policy": policy, "n_molecules": n,
                "all_three_exact": int(all_correct.sum()), "all_three_exact_rate": float(all_correct.mean()),
                "none_exact": int(all_wrong.sum()), "none_exact_rate": float(all_wrong.mean()),
                "partial_exact": int(partial.sum()), "partial_exact_rate": float(partial.mean()),
                "only_ir_exact": int(only_ir.sum()), "only_ir_exact_rate": float(only_ir.mean()),
                "only_raman_exact": int(only_raman.sum()), "only_raman_exact_rate": float(only_raman.mean()),
                "only_uv_exact": int(only_uv.sum()), "only_uv_exact_rate": float(only_uv.mean()),
            })
    return membership, summary


def _build_three_seed_summary(
    sample_summary: list[dict],
    exact_summary: list[dict],
    spearman: list[dict],
    hard_pairs: list[dict],
    oracle_summary: list[dict],
    pairwise_summary: list[dict],
    or_gains: list[dict],
    high_conf: list[dict],
) -> list[dict]:
    out = []
    out += _three_seed_summary(
        "sample_level_overall_errors", sample_summary,
        ("task", "label_set", "modality", "threshold_policy"),
        ("mean_n_errors", "median_n_errors", "q1_n_errors", "q3_n_errors", "iqr_n_errors",
         "exact_match_count", "exact_match_rate", "n_high_confidence_errors",
         "n_samples_with_high_confidence_error"),
    )
    out += _three_seed_summary(
        "molecule_exact_pattern", exact_summary,
        ("task", "label_set", "threshold_policy"),
        ("all_three_exact", "all_three_exact_rate", "none_exact", "none_exact_rate",
         "partial_exact", "partial_exact_rate", "only_ir_exact", "only_ir_exact_rate",
         "only_raman_exact", "only_raman_exact_rate", "only_uv_exact", "only_uv_exact_rate"),
    )
    out += _three_seed_summary(
        "spearman", spearman,
        ("task", "label_set", "comparison", "threshold_policy"),
        ("spearman_rho",),
    )
    out += _three_seed_summary(
        "hard_sample_jaccard", hard_pairs,
        ("task", "label_set", "comparison", "threshold_policy"),
        ("jaccard",),
    )
    out += _three_seed_summary(
        "sample_label_oracle", oracle_summary,
        ("task", "label_set", "threshold_policy"),
        ("any_correct_rate", "exactly_one_correct_rate", "all_wrong_rate", "all_correct_rate",
         "only_ir_correct_rate", "only_raman_correct_rate", "only_uv_correct_rate",
         "ir_raman_any_correct_rate", "three_any_correct_rate",
         "uv_extra_oracle_correct_rate", "ir_raman_both_wrong_uv_correct_rate"),
    )
    out += _three_seed_summary(
        "conditional_complementarity", pairwise_summary,
        ("task", "label_set", "comparison", "threshold_policy"),
        ("conditional_correct_given_base_wrong", "p_other_correct", "conditional_delta",
         "fn_rescue_rate", "fp_correction_rate", "new_fp_rate", "new_fn_rate", "error_jaccard"),
    )
    out += _three_seed_summary(
        "or_gain", or_gains,
        ("task", "label_set", "comparison", "threshold_policy"),
        ("positive_labels", "zero_labels", "negative_labels", "mean_or_gain"),
    )
    out += _three_seed_summary(
        "high_confidence", _high_conf_seed_overall(high_conf),
        ("task", "label_set", "modality", "threshold_policy"),
        ("high_conf_fp", "high_conf_fn", "high_conf_errors",
         "high_conf_fp_fraction_of_fp", "high_conf_fn_fraction_of_fn",
         "high_conf_fraction_of_errors", "high_conf_fraction_of_pairs"),
    )
    return out


def _oracle_rows(task_info: dict, three_rows: list[dict], pair_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    per_label = []
    summary = []
    gains = []
    pairwise_summary = []
    pairwise_label = []
    for seed in (42, 123, 2026):
        for policy in POLICIES:
            subset = [r for r in three_rows if r["seed"] == seed and r["threshold_policy"] == policy]
            n_total = sum(r["n_samples"] for r in subset)
            stats = {}
            for key in (
                "any_correct", "exactly_one_correct", "all_wrong", "all_correct",
                "only_ir_correct", "only_raman_correct", "only_uv_correct",
                "ir_raman_any_correct", "three_any_correct",
                "uv_extra_oracle_correct", "ir_raman_both_wrong_uv_correct",
            ):
                stats[key] = int(sum(r[key] for r in subset))
                stats[key + "_rate"] = stats[key] / n_total
            summary.append({
                "task": task_info["task"], "label_set": task_info["label_set"],
                "seed": seed, "threshold_policy": policy, "level": "sample_label",
                "n_pairs": n_total, "n_labels": len(task_info["expected_names"]),
                **stats,
                "note": "sample-label pair correctness; not molecule exact_match",
            })
            for r in subset:
                per_label.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "label_index": r["label_index"], "label": r["label"],
                    "n_pairs": r["n_samples"], "any_correct": r["any_correct"],
                    "exactly_one_correct": r["exactly_one_correct"], "all_wrong": r["all_wrong"],
                    "all_correct": r["all_correct"], "only_ir_correct": r["only_ir_correct"],
                    "only_raman_correct": r["only_raman_correct"], "only_uv_correct": r["only_uv_correct"],
                    "ir_raman_any_correct": r["ir_raman_any_correct"],
                    "three_any_correct": r["three_any_correct"],
                    "uv_extra_oracle_correct": r["uv_extra_oracle_correct"],
                    "ir_raman_both_wrong_uv_correct": r["ir_raman_both_wrong_uv_correct"],
                    "or_minus_best_single_f1": r["or_minus_best_single_f1"],
                })
            vals = [r["or_minus_best_single_f1"] for r in subset]
            gains.append({
                "task": task_info["task"], "label_set": task_info["label_set"],
                "seed": seed, "threshold_policy": policy, "comparison": "three_modality_vs_best_single",
                "n_labels": len(vals),
                "positive_labels": int(sum(v > 0 for v in vals)),
                "zero_labels": int(sum(v == 0 for v in vals)),
                "negative_labels": int(sum(v < 0 for v in vals)),
                "mean_or_gain": float(np.mean(vals)),
                "best_single": task_info["best_single"],
                "note": "positive_or_fusion vs protocol-selected best single; OR is not Oracle",
            })
            # Three unordered pairwise OR comparisons, each against max of its two members.
            for a, b in UNORDERED_PAIRS:
                label = _pair_label(a, b)
                subset_pair = [r for r in pair_rows if r["seed"] == seed and r["threshold_policy"] == policy and r["base_modality"] == a and r["other_modality"] == b]
                vals = [r["or_gain_vs_max_single"] for r in subset_pair]
                gains.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "comparison": label,
                    "n_labels": len(vals),
                    "positive_labels": int(sum(v > 0 for v in vals)),
                    "zero_labels": int(sum(v == 0 for v in vals)),
                    "negative_labels": int(sum(v < 0 for v in vals)),
                    "mean_or_gain": float(np.mean(vals)),
                    "best_single": None,
                    "note": "positive_or_fusion vs max of the two member modalities; OR is not Oracle",
                })
            # All six directed conditional-complementarity rows.
            for a, b in ORDERED_PAIRS:
                label = _direction_label(a, b)
                subset_pair = [r for r in pair_rows if r["seed"] == seed and r["threshold_policy"] == policy and r["base_modality"] == a and r["other_modality"] == b]
                fields = ("base_only_correct", "other_only_correct", "both_correct", "both_wrong", "fn_rescue", "fp_correction", "new_fp", "new_fn", "error_intersection", "error_union")
                sums = {f: int(sum(r[f] for r in subset_pair)) for f in fields}
                sums["base_wrong"] = int(sum(r["base_wrong"] for r in subset_pair))
                sums["base_fn"] = int(sum(r["base_fn"] for r in subset_pair))
                sums["base_fp"] = int(sum(r["base_fp"] for r in subset_pair))
                sums["n_positive"] = int(sum(r["n_positive"] for r in subset_pair))
                sums["n_negative"] = int(sum(r["n_negative"] for r in subset_pair))
                sums["n_samples"] = int(sum(r["n_samples"] for r in subset_pair))
                conditional_num = sums["other_only_correct"]
                conditional = (conditional_num / sums["base_wrong"]) if sums["base_wrong"] else None
                p_other = (sums["both_correct"] + sums["other_only_correct"]) / sums["n_samples"]
                jaccard = (1.0 if sums["error_union"] == 0 else sums["error_intersection"] / sums["error_union"])
                pairwise_summary.append({
                    "task": task_info["task"], "label_set": task_info["label_set"],
                    "seed": seed, "threshold_policy": policy, "comparison": label,
                    "modality_a": a, "modality_b": b, "n_labels": len(subset_pair),
                    **sums,
                    "conditional_correct_given_base_wrong": conditional,
                    "p_other_correct": p_other,
                    "conditional_delta": None if conditional is None else conditional - p_other,
                    "error_jaccard": jaccard,
                    "error_jaccard_dual_empty": bool(sums["error_union"] == 0),
                    "fn_rescue_rate": (sums["fn_rescue"] / sums["base_fn"]) if sums["base_fn"] else None,
                    "fp_correction_rate": (sums["fp_correction"] / sums["base_fp"]) if sums["base_fp"] else None,
                    "new_fp_rate": (sums["new_fp"] / sums["n_negative"]) if sums["n_negative"] else None,
                    "new_fn_rate": (sums["new_fn"] / sums["n_positive"]) if sums["n_positive"] else None,
                    "or_gain_vs_max_single_mean": float(np.mean([r["or_gain_vs_max_single"] for r in subset_pair])),
                })
                for r in subset_pair:
                    pairwise_label.append({
                        "task": task_info["task"], "label_set": task_info["label_set"],
                        "seed": seed, "threshold_policy": policy, "comparison": label,
                        "modality_a": a, "modality_b": b,
                        "label_index": r["label_index"], "label": r["label"], "support": r["support"],
                        "base_wrong": r["base_wrong"], "base_fn": r["base_fn"], "base_fp": r["base_fp"],
                        "n_positive": r["n_positive"], "n_negative": r["n_negative"],
                        **{f: r[f] for f in fields},
                        "error_jaccard": r["error_jaccard"],
                        "error_jaccard_defined": r["error_jaccard_defined"],
                        "conditional_correct_given_base_wrong": r["conditional_correct_given_base_wrong"],
                        "p_other_correct": r["p_other_correct"],
                        "conditional_delta": r["conditional_delta"],
                        "fn_rescue_rate": r["fn_rescue_rate"],
                        "fp_correction_rate": r["fp_correction_rate"],
                        "new_fp_rate": r["new_fp_rate"],
                        "new_fn_rate": r["new_fn_rate"],
                        "or_f1": r["or_f1"], "or_gain_vs_max_single": r["or_gain_vs_max_single"],
                    })
    return summary, per_label, gains, pairwise_summary, pairwise_label


def _analyze_task(label_set: str, task: str, out_root: Path) -> dict:
    task_info = _load_task_inputs(
        label_set,
        ROOT / "runs" / "LINE-B-PREDICTIONS",
        ROOT / "runs" / "LINE-B-THRESHOLDS",
        ROOT / "runs" / "LINE-B-METRICS",
        False,
    )
    accepted = _check_accepted_inputs(task_info, label_set)
    model_rows = _model_rows(task_info)
    pair_rows = _pair_rows(task_info)
    three_rows = _three_rows(task_info)
    mcnemar = _mcnemar_rows(task_info, pair_rows)
    sample_errors, sample_summary, three_seed, hc_extra = _sample_errors(task_info)
    spearman = _spearman_rows(task_info, sample_errors)
    hard, hard_pairs, hard_membership, hard_extra = _hard_rows(task_info, sample_errors)
    exact_membership, exact_summary = _molecule_exact_rows(task_info)
    oracle_summary, oracle_label, or_gains, pairwise_summary, pairwise_label = _oracle_rows(task_info, three_rows, pair_rows)
    three_seed_summary = _build_three_seed_summary(
        sample_summary, exact_summary, spearman, hard_pairs, oracle_summary,
        pairwise_summary, or_gains, hc_extra["high_conf_summary"],
    )

    task_out = Path(out_root) / task / "complementarity_extended"
    with atomic_output_dir(task_out) as temporary:
        _write_csv(temporary / "mcnemar_holm.csv", mcnemar, list(mcnemar[0].keys()))
        _write_csv(temporary / "sample_errors.csv", sample_errors, list(sample_errors[0].keys()))
        _write_csv(temporary / "sample_error_summary.csv", sample_summary, list(sample_summary[0].keys()))
        _write_csv(temporary / "sample_error_three_seed.csv", three_seed, list(three_seed[0].keys()))
        _write_csv(temporary / "spearman.csv", spearman, list(spearman[0].keys()))
        _write_csv(temporary / "hard_samples.csv", hard, list(hard[0].keys()))
        _write_csv(temporary / "hard_sample_pairs.csv", hard_pairs, list(hard_pairs[0].keys()))
        _write_csv(temporary / "hard_sample_membership.csv", hard_membership, list(hard_membership[0].keys()))
        _write_csv(temporary / "molecule_exact_match.csv", exact_membership, list(exact_membership[0].keys()))
        _write_csv(temporary / "molecule_exact_summary.csv", exact_summary, list(exact_summary[0].keys()))
        _write_csv(temporary / "oracle_sample_label_summary.csv", oracle_summary, list(oracle_summary[0].keys()))
        _write_csv(temporary / "oracle_sample_label.csv", oracle_label, list(oracle_label[0].keys()))
        _write_csv(temporary / "or_gain_summary.csv", or_gains, list(or_gains[0].keys()))
        _write_csv(temporary / "pairwise_summary.csv", pairwise_summary, list(pairwise_summary[0].keys()))
        _write_csv(temporary / "pairwise_by_label.csv", pairwise_label, list(pairwise_label[0].keys()))
        three_seed_fields = []
        for row in three_seed_summary:
            for name in row:
                if name not in three_seed_fields:
                    three_seed_fields.append(name)
        _write_csv(temporary / "three_seed_summary.csv", three_seed_summary, three_seed_fields)
        _write_csv(temporary / "high_confidence_errors.csv", hc_extra["high_conf_summary"], list(hc_extra["high_conf_summary"][0].keys()))
        output_names = [
            "mcnemar_holm.csv", "sample_errors.csv", "sample_error_summary.csv",
            "sample_error_three_seed.csv", "spearman.csv", "hard_samples.csv",
            "hard_sample_pairs.csv", "hard_sample_membership.csv",
            "molecule_exact_match.csv", "molecule_exact_summary.csv",
            "oracle_sample_label_summary.csv", "oracle_sample_label.csv",
            "or_gain_summary.csv", "pairwise_summary.csv", "pairwise_by_label.csv",
            "three_seed_summary.csv", "high_confidence_errors.csv",
        ]
        family_sizes = sorted({r["family_size"] for r in mcnemar})
        summary = {
            "phase": PHASE,
            "task": task,
            "label_set": label_set,
            "split": "validation",
            "validation_development": True,
            "no_test_access": True,
            "n_models": len(task_info["models"]),
            "n_labels": len(task_info["expected_names"]),
            "seeds": [42, 123, 2026],
            "threshold_policies": {"primary": "fixed_0.5", "secondary": "validation_f1"},
            "best_single": task_info["best_single"],
            "inputs": accepted,
            "code_sha256": {
                rel: sha256_file(ROOT / rel)
                for rel in (
                    "training/line_b/analyze_line_b_complementarity_extended.py",
                    "training/line_b/analyze_line_b_complementarity.py",
                    "training/line_b/common.py",
                    "training/line_b/metrics.py",
                    "training/line_b/bootstrap_line_b_scaffold.py",
                )
            },
            "mcnemar": {
                "test": "two-sided exact McNemar/equal-probability binomial, p=0.5",
                "directions": [PAIR_LABELS[p] for p in UNORDERED_PAIRS],
                "family": "task x seed x threshold_policy; 3 unordered modality pairs x all labels",
                "family_sizes": family_sizes,
                "exploratory_only": True,
                "seed_pvalues_not_averaged": True,
            },
            "pairwise": {
                "directions": [_direction_label(a, b) for a, b in ORDERED_PAIRS],
                "n_directions": 6,
                "n_summary_rows": len(pairwise_summary),
                "n_by_label_rows": len(pairwise_label),
                "conditional_rate_rule": "aggregate other_only_correct / base_wrong counts; never average per-label ratios",
                "error_jaccard_rule": "aggregate error intersection / error union; dual-empty sets map to Jaccard 1 with explicit flag",
            },
            "or_gain": {
                "comparisons": ["three_modality_vs_best_single"] + [_pair_label(a, b) for a, b in UNORDERED_PAIRS],
                "n_rows": len(or_gains),
                "best_single": task_info["best_single"],
            },
            "three_seed_summary": {
                "path": "three_seed_summary.csv",
                "n_rows": len(three_seed_summary),
                "rule": "mean and sample SD across 3 seeds; n_valid/n_undefined; McNemar/Holm p-values excluded",
                "cross_seed_molecule_file": "sample_error_three_seed.csv",
            },
            "output_file_count": len(output_names) + 1,
            "hard_sample": {
                "fraction": HARD_FRACTION,
                "n_per_seed_policy": hard[0]["n_hard"],
                "tie_rule": "n_errors descending, then authoritative row_index ascending; ceil(0.10*N)",
                "boundary_tie_note": "Jaccard can be unstable when a large tie set defines the top-10% boundary",
            },
            "high_confidence": {
                "fp": "FP and p >= 0.90",
                "fn": "FN and p <= 0.10",
                "unit": "sample-label pair",
                "caveat": "raw sigmoid outputs are not calibrated probabilities; diagnostic only",
            },
            "undefined": {
                "spearman_constant": "None with explicit reason",
                "rates": "None when denominator is zero; never replaced with 0",
            },
            "outputs": {
                name: sha256_file(temporary / name) for name in output_names
            },
            "environment": {
                "python": sys.version,
                "python_version": sys.version.split()[0],
                "numpy": np.__version__,
                "cwd": str(ROOT),
            },
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "caveats": [
                "McNemar/Holm are exploratory; scaffold correlation precludes confirmatory claims.",
                "Molecular exact_match and sample-label Oracle are distinct and both reported.",
                "positive_or_fusion is not the true Oracle and its F1 gain is not fusion evidence.",
                "High-confidence outputs are uncalibrated sigmoid extremes, not calibrated posterior reliability.",
            ],
        }
        write_json(temporary / "summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "Shared14", "UVSPEC15"], default="all")
    parser.add_argument("--out-root", type=Path, required=True, help="Required temporary output directory outside the project.")
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test:
        raise SystemExit("Test is not authorized in B3.5-C2")
    out_root = args.out_root.resolve()
    if out_root.is_relative_to(ROOT.resolve()):
        raise SystemExit("B3.5-C2 outputs must use a temporary directory outside the project; formal LINE-B-ANALYSIS/B4 writes are not authorized.")
    tasks = [("Shared14", "shared14"), ("UVSPEC15", "uvspec15")]
    if args.task != "all":
        tasks = [pair for pair in tasks if pair[0] == args.task]
    summaries = [_analyze_task(label_set, task, out_root) for task, label_set in tasks]
    print(json.dumps([{"task": s["task"], "outputs": len(s["outputs"]), "fails": 0} for s in summaries], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
