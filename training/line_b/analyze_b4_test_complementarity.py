"""B4.4 Test error complementarity, OR/AND, Oracle, sample-level and structural subgroups.

Read-only with respect to every frozen asset.  Uses B4.1 Test predictions,
FREEZE thresholds, B4.3 draw matrix (reconstructed and SHA-checked), the
authoritative structure library and the D1 Training quartiles.

Exploratory / post-hoc only.  No Test inference, threshold fitting, model
selection or label selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
SEEDS = (42, 123, 2026)
POLICIES = ("fixed_0.5", "validation_f1")
PAIRS = (("ir", "raman"), ("uvvis", "ir"), ("uvvis", "raman"))
PAIR_LABELS = {("ir", "raman"): "IR-Raman", ("uvvis", "ir"): "UV-IR", ("uvvis", "raman"): "UV-Raman"}
DIRECTIONS = (("ir", "raman"), ("raman", "ir"), ("uvvis", "ir"), ("ir", "uvvis"), ("uvvis", "raman"), ("raman", "uvvis"))
DIRECTION_LABELS = {
    ("ir", "raman"): "IR|wrong->Raman", ("raman", "ir"): "Raman|wrong->IR",
    ("uvvis", "ir"): "UV|wrong->IR", ("ir", "uvvis"): "IR|wrong->UV",
    ("uvvis", "raman"): "UV|wrong->Raman", ("raman", "uvvis"): "Raman|wrong->UV",
}
HC_FP_PROB = 0.90
HC_FN_PROB = 0.10
HARD_FRACTION = 0.10
SUPPORT_MIN_N = 100
SUPPORT_MIN_POS = 30
CI = (2.5, 97.5)
TEST_N = 12981
DEFAULT_REPS = 2000
DEFAULT_RNG = 20261117
CHUNK = 200
FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
B43 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
STRUCTURE = ROOT / "processed" / "qm9s" / "structures" / "qm9s_structure_library.csv"
SCAFFOLD_MANIFEST = ROOT / "processed" / "qm9s" / "prepared" / "qm9s_manifest_prepared.csv"
D1 = ROOT / "runs" / "LINE-B-ANALYSIS" / "validation_development" / "B3.5-D1" / "final_22subgroups_20reps"
DEFAULT_OUT = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4"


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


def _load_preflight(freeze_path: Path, pin: str) -> dict:
    protocol = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
    if _sha(freeze_path) != pin:
        raise SystemExit("FREEZE SHA256 mismatch")
    freeze = read_json(freeze_path)
    if freeze["status"] != "FROZEN" or freeze["phase"] != "B3":
        raise SystemExit("FREEZE is not frozen B3")
    if freeze["protocol_sha256"] != _sha(protocol):
        raise SystemExit("Protocol SHA differs from FREEZE")
    models = []
    row_index_ref = None
    labels_by_task = {}
    names_by_task = {}
    for record in freeze["models"]:
        d = PREDICTIONS / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        manifest = read_json(d / "prediction_manifest.json")
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        row_index = np.load(d / "row_index.npy", allow_pickle=False)
        labels = np.load(d / "labels.npy", allow_pickle=False)
        if sha256_array(probs) != manifest["arrays"]["probabilities"]["sha256"]:
            raise SystemExit(f"probability hash mismatch: {d}")
        if sha256_array(row_index) != manifest["arrays"]["row_index"]["sha256"]:
            raise SystemExit(f"row_index hash mismatch: {d}")
        if sha256_array(labels) != manifest["arrays"]["labels"]["sha256"]:
            raise SystemExit(f"labels hash mismatch: {d}")
        if (manifest.get("freeze") or {}).get("sha256") != pin:
            raise SystemExit(f"FREEZE pin mismatch: {d}")
        if manifest["checkpoint"]["sha256"] != record["checkpoint"]["sha256"]:
            raise SystemExit(f"checkpoint mismatch: {d}")
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        if manifest.get("label_names") != names:
            raise SystemExit(f"label order mismatch: {d}")
        threshold_payload = read_json(ROOT / record["threshold"]["path"])
        if _sha(ROOT / record["threshold"]["path"]) != record["threshold"]["sha256"]:
            raise SystemExit("threshold hash mismatch")
        thresholds = np.asarray([threshold_payload["thresholds"][n] for n in names], dtype=np.float64)
        if row_index_ref is None:
            row_index_ref = row_index
        elif not np.array_equal(row_index_ref, row_index):
            raise SystemExit("row_index differs across models")
        if record["task"] not in labels_by_task:
            labels_by_task[record["task"]] = labels
            names_by_task[record["task"]] = names
        elif not np.array_equal(labels_by_task[record["task"]], labels):
            raise SystemExit("labels differ across models")
        models.append({
            "task": record["task"], "label_set": record["label_set"], "modality": record["modality"],
            "run_name": record["run_name"], "seed": int(record["seed"]),
            "probabilities": probs, "labels": labels, "row_index": row_index,
            "thresholds": thresholds, "names": names,
            "manifest": manifest, "prediction_manifest_sha256": _sha(d / "prediction_manifest.json"),
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "threshold_path": relative_to_root(ROOT / record["threshold"]["path"]),
            "threshold_sha256": _sha(ROOT / record["threshold"]["path"]),
        })
    return {
        "freeze": freeze, "freeze_path": freeze_path, "freeze_sha256": pin,
        "protocol_path": relative_to_root(protocol), "protocol_sha256": _sha(protocol),
        "models": models, "row_index": row_index_ref,
        "labels_by_task": labels_by_task, "names_by_task": names_by_task,
    }


def _load_structure(row_index: np.ndarray) -> dict[int, dict]:
    wanted = set(int(i) for i in row_index)
    rows = {}
    with STRUCTURE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["row_index"])
            if index in wanted:
                rows[index] = row
    if len(rows) != len(wanted):
        raise SystemExit(f"Structure library missing Test rows: {len(wanted)-len(rows)}")
    return rows


def _load_scaffold_map() -> dict[int, str]:
    mapping = {}
    with SCAFFOLD_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["row_index"])] = (row.get("murcko_scaffold") or "").strip() or "__MISSING__"
    return mapping


def _clusters(row_index: np.ndarray, mapping: dict[int, str]):
    names = sorted({mapping[int(i)] for i in row_index})
    index = {name: p for p, name in enumerate(names)}
    cluster_id = np.asarray([index[mapping[int(i)]] for i in row_index], dtype=np.int64)
    sizes = np.bincount(cluster_id, minlength=len(names)).astype(np.int64)
    return names, cluster_id, sizes


def _draws(reps: int, n_scaffolds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, n_scaffolds), dtype=np.float32)
    for rep in range(reps):
        draws[rep] = np.bincount(rng.integers(0, n_scaffolds, size=n_scaffolds), minlength=n_scaffolds).astype(np.float32)
    return draws


def _draws_sha(draws: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(draws.dtype).encode()); h.update(b"|")
    h.update(str(draws.shape).encode()); h.update(b"|")
    h.update(np.ascontiguousarray(draws).tobytes())
    return h.hexdigest()


def _pred(model: dict, policy: str) -> np.ndarray:
    if policy == "fixed_0.5":
        thresholds = np.full(len(model["names"]), 0.5)
    else:
        thresholds = np.asarray(model["thresholds"], dtype=np.float64)
    return model["probabilities"] >= thresholds.reshape(1, -1)


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _confusion(truth: np.ndarray, pred: np.ndarray) -> dict:
    truth = truth.astype(bool); pred = pred.astype(bool)
    tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred)); tn = int(np.sum(~truth & ~pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": _f1_from_counts(tp, fp, fn)}


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    y = y.astype(np.int64)
    if np.unique(y).size != 2:
        return None, None
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def _exact_mcnemar(b: int, c: int) -> tuple[float, int]:
    n = b + c
    if n == 0:
        return 1.0, 0
    k = min(b, c)
    log_tail = np.logaddexp.reduce([
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)
    ])
    log_p = math.log(2.0) + float(log_tail) - n * math.log(2.0)
    return (min(1.0, math.exp(log_p)) if log_p < 0 else 1.0), n


def _holm(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda r: (r["p_raw"], r["label_index"], r["comparison"]))
    m = len(ordered)
    adjusted = 1.0
    for i, row in enumerate(ordered):
        candidate = min(1.0, (m - i) * row["p_raw"])
        adjusted = max(adjusted, candidate) if i else candidate
        row["p_holm"] = adjusted
        row["family_size"] = m


def _spearman(a: np.ndarray, b: np.ndarray):
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), dtype=np.float64)
        i = 0
        while i < len(v):
            j = i + 1
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
            i = j
        return out
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ca, cb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float(np.sum(ca * ca) * np.sum(cb * cb)))
    return None if denom == 0 else float(np.sum(ca * cb) / denom)


def _hard_indices(n_errors: np.ndarray) -> tuple[np.ndarray, int]:
    k = int(math.ceil(HARD_FRACTION * len(n_errors)))
    order = np.lexsort((np.arange(len(n_errors)), -n_errors))
    selected = order[:k]
    boundary = int(n_errors[selected[-1]]) if k else 0
    return selected, boundary


def _point_pairwise(task_info: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    jaccard_rows = []
    mcnemar_rows = []
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    for seed in SEEDS:
        for policy in POLICIES:
            preds = {mod: _pred(models[(seed, mod)], policy) for mod in MODALITIES}
            family = []
            for base, other in PAIRS:
                comparison = PAIR_LABELS[(base, other)]
                pa, pb = preds[base], preds[other]
                for index, name in enumerate(names):
                    y = task_info["labels"][:, index]
                    ca = _confusion(y, pa[:, index]); cb = _confusion(y, pb[:, index])
                    base_correct = pa[:, index] == y
                    other_correct = pb[:, index] == y
                    both_correct = int(np.sum(base_correct & other_correct))
                    base_only = int(np.sum(base_correct & ~other_correct))
                    other_only = int(np.sum(~base_correct & other_correct))
                    both_wrong = int(np.sum(~base_correct & ~other_correct))
                    fn_rescue = int(np.sum((~base_correct & (y == 1)) & other_correct))
                    fp_correction = int(np.sum((~base_correct & (y == 0)) & other_correct))
                    new_fp = int(np.sum(base_correct & (y == 0) & (pb[:, index] == 1)))
                    new_fn = int(np.sum(base_correct & (y == 1) & (pb[:, index] == 0)))
                    error_intersection = both_wrong
                    error_union = len(y) - both_correct
                    jaccard = 1.0 if error_union == 0 else error_intersection / error_union
                    rows.append({
                        "task": task_info["task"], "label_set": task_info["label_set"], "seed": seed,
                        "threshold_policy": policy, "comparison": comparison,
                        "base_modality": base, "other_modality": other, "scope": "label",
                        "label_index": index, "label": name, "n_samples": len(y),
                        "base_tp": ca["tp"], "base_fp": ca["fp"], "base_fn": ca["fn"], "base_tn": ca["tn"],
                        "other_tp": cb["tp"], "other_fp": cb["fp"], "other_fn": cb["fn"], "other_tn": cb["tn"],
                        "base_f1": ca["f1"], "other_f1": cb["f1"],
                        "both_correct": both_correct, "base_only_correct": base_only,
                        "other_only_correct": other_only, "both_wrong": both_wrong,
                        "fn_rescue": fn_rescue, "fp_correction": fp_correction,
                        "new_fp": new_fp, "new_fn": new_fn,
                        "error_intersection": error_intersection, "error_union": error_union,
                        "error_jaccard": jaccard,
                    })
                    jaccard_rows.append({
                        "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                        "comparison": comparison, "scope": "label", "label_index": index, "label": name,
                        "error_intersection": error_intersection, "error_union": error_union,
                        "error_jaccard": jaccard, "dual_empty": bool(error_union == 0),
                    })
                    p, discordant = _exact_mcnemar(base_only, other_only)
                    family.append({
                        "task": task_info["task"], "label_set": task_info["label_set"], "seed": seed,
                        "threshold_policy": policy, "label_index": index, "label": name,
                        "comparison": comparison, "base_modality": base, "other_modality": other,
                        "n_samples": len(y), "both_correct": both_correct,
                        "base_only_correct": base_only, "other_only_correct": other_only,
                        "both_wrong": both_wrong, "discordant_count": discordant,
                        "base_correct_rate": (base_only + both_correct) / len(y),
                        "other_correct_rate": (other_only + both_correct) / len(y),
                        "correct_rate_difference": (base_only - other_only) / len(y),
                        "p_raw": p, "p_holm": None, "family_size": None,
                        "family_id": f'{task_info["task"]}|seed={seed}|policy={policy}',
                    })
                # Macro row.
                totals = {
                    "base_tp": sum(r["base_tp"] for r in rows[-len(names):]),
                    "base_fp": sum(r["base_fp"] for r in rows[-len(names):]),
                    "base_fn": sum(r["base_fn"] for r in rows[-len(names):]),
                    "other_tp": sum(r["other_tp"] for r in rows[-len(names):]),
                    "other_fp": sum(r["other_fp"] for r in rows[-len(names):]),
                    "other_fn": sum(r["other_fn"] for r in rows[-len(names):]),
                }
                rows.append({
                    "task": task_info["task"], "label_set": task_info["label_set"], "seed": seed,
                    "threshold_policy": policy, "comparison": comparison,
                    "base_modality": base, "other_modality": other, "scope": "macro",
                    "label_index": "", "label": "__macro__", "n_samples": len(names) * len(task_info["labels"]),
                    "base_tp": totals["base_tp"], "base_fp": totals["base_fp"], "base_fn": totals["base_fn"], "base_tn": "",
                    "other_tp": totals["other_tp"], "other_fp": totals["other_fp"], "other_fn": totals["other_fn"], "other_tn": "",
                    "base_f1": float(np.mean([r["base_f1"] for r in rows[-len(names)-1:-1]])),
                    "other_f1": float(np.mean([r["other_f1"] for r in rows[-len(names)-1:-1]])),
                    "both_correct": sum(r["both_correct"] for r in rows[-len(names)-1:-1]),
                    "base_only_correct": sum(r["base_only_correct"] for r in rows[-len(names)-1:-1]),
                    "other_only_correct": sum(r["other_only_correct"] for r in rows[-len(names)-1:-1]),
                    "both_wrong": sum(r["both_wrong"] for r in rows[-len(names)-1:-1]),
                    "fn_rescue": sum(r["fn_rescue"] for r in rows[-len(names)-1:-1]),
                    "fp_correction": sum(r["fp_correction"] for r in rows[-len(names)-1:-1]),
                    "new_fp": sum(r["new_fp"] for r in rows[-len(names)-1:-1]),
                    "new_fn": sum(r["new_fn"] for r in rows[-len(names)-1:-1]),
                    "error_intersection": sum(r["error_intersection"] for r in rows[-len(names)-1:-1]),
                    "error_union": sum(r["error_union"] for r in rows[-len(names)-1:-1]),
                    "error_jaccard": None,
                })
                last = rows[-1]
                last["error_jaccard"] = 1.0 if last["error_union"] == 0 else last["error_intersection"] / last["error_union"]
                jaccard_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "comparison": comparison, "scope": "macro", "label_index": "", "label": "__macro__",
                    "error_intersection": last["error_intersection"], "error_union": last["error_union"],
                    "error_jaccard": last["error_jaccard"], "dual_empty": bool(last["error_union"] == 0),
                })
            _holm(family)
            mcnemar_rows.extend(family)
    return rows, jaccard_rows, mcnemar_rows


def _point_or(task_info: dict) -> tuple[list[dict], list[dict]]:
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    pair_rows = []
    three_rows = []
    best_single = task_info["best_single"]
    for seed in SEEDS:
        for policy in POLICIES:
            preds = {mod: _pred(models[(seed, mod)], policy) for mod in MODALITIES}
            f1 = {}
            for mod in MODALITIES:
                f1[mod] = np.asarray([_confusion(task_info["labels"][:, i], preds[mod][:, i])["f1"] for i in range(len(names))])
            for base, other in PAIRS:
                comparison = PAIR_LABELS[(base, other)]
                or_f1, and_f1, gains = [], [], []
                for index in range(len(names)):
                    y = task_info["labels"][:, index]
                    pa, pb = preds[base][:, index], preds[other][:, index]
                    f_or = _confusion(y, pa | pb)["f1"]
                    f_and = _confusion(y, pa & pb)["f1"]
                    or_f1.append(f_or); and_f1.append(f_and)
                    gains.append(f_or - max(f1[base][index], f1[other][index]))
                pair_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "comparison": comparison, "base_modality": base, "other_modality": other, "scope": "macro",
                    "base_f1": float(np.mean(f1[base])), "other_f1": float(np.mean(f1[other])),
                    "or_f1": float(np.mean(or_f1)), "and_f1": float(np.mean(and_f1)),
                    "or_gain_vs_max": float(np.mean(gains)),
                    "n_positive_or_gain_labels": int(sum(1 for g in gains if g > 0)),
                    "n_negative_or_gain_labels": int(sum(1 for g in gains if g < 0)),
                })
            three_or = []
            for index in range(len(names)):
                y = task_info["labels"][:, index]
                union = preds["ir"][:, index] | preds["raman"][:, index] | preds["uvvis"][:, index]
                three_or.append(_confusion(y, union)["f1"])
            three_rows.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                "best_single": best_single,
                "best_single_f1": float(np.mean(f1[best_single])),
                "three_or_f1": float(np.mean(three_or)),
                "or_gain_vs_best_single": float(np.mean(three_or) - np.mean(f1[best_single])),
                "n_positive_or_gain_labels": int(sum(1 for a, b in zip(three_or, f1[best_single]) if a > b)),
                "n_negative_or_gain_labels": int(sum(1 for a, b in zip(three_or, f1[best_single]) if a < b)),
            })
    return pair_rows, three_rows


def _point_oracle(task_info: dict) -> tuple[list[dict], list[dict]]:
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    summary = []
    per_label = []
    for seed in SEEDS:
        for policy in POLICIES:
            correct = {mod: (_pred(models[(seed, mod)], policy) == task_info["labels"]) for mod in MODALITIES}
            n_pairs = task_info["labels"].size
            count = correct["ir"].astype(int) + correct["raman"].astype(int) + correct["uvvis"].astype(int)
            any_correct = int(np.sum(count >= 1))
            exactly_one = int(np.sum(count == 1))
            all_wrong = int(np.sum(count == 0))
            all_correct = int(np.sum(count == 3))
            only = {mod: int(np.sum((count == 1) & correct[mod])) for mod in MODALITIES}
            summary.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy, "scope": "overall",
                "label_index": "", "label": "__overall__", "n_pairs": n_pairs,
                "any_correct": any_correct, "exactly_one_correct": exactly_one,
                "all_wrong": all_wrong, "all_correct": all_correct,
                "only_ir_correct": only["ir"], "only_raman_correct": only["raman"],
                "only_uv_correct": only["uvvis"],
                "any_correct_rate": any_correct / n_pairs,
                "exactly_one_correct_rate": exactly_one / n_pairs,
                "all_wrong_rate": all_wrong / n_pairs,
            })
            for index, name in enumerate(names):
                c = correct["ir"][:, index].astype(int) + correct["raman"][:, index].astype(int) + correct["uvvis"][:, index].astype(int)
                n = len(c)
                per_label.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "label_index": index, "label": name, "n_pairs": n,
                    "support": int(task_info["labels"][:, index].sum()),
                    "any_correct": int(np.sum(c >= 1)), "exactly_one_correct": int(np.sum(c == 1)),
                    "all_wrong": int(np.sum(c == 0)), "all_correct": int(np.sum(c == 3)),
                    "only_ir_correct": int(np.sum((c == 1) & correct["ir"][:, index])),
                    "only_raman_correct": int(np.sum((c == 1) & correct["raman"][:, index])),
                    "only_uv_correct": int(np.sum((c == 1) & correct["uvvis"][:, index])),
                })
    return summary, per_label


def _conditional_indicators(task_info: dict, policy: str) -> dict[int, np.ndarray]:
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    names = task_info["names"]
    indicators = {}
    n_labels = len(names)
    for seed in SEEDS:
        correct = {mod: (_pred(models[(seed, mod)], policy) == task_info["labels"]) for mod in MODALITIES}
        C = np.stack([correct[mod].reshape(-1) for mod in MODALITIES], axis=1).astype(np.float64)  # pairs x 3
        W = 1.0 - C
        columns = [C]
        for base, other in DIRECTIONS:
            columns.append((W[:, MODALITIES.index(base)] * C[:, MODALITIES.index(other)]).reshape(-1, 1))
        both_wrong_ir_raman = W[:, 0] * W[:, 1]
        columns.append(both_wrong_ir_raman.reshape(-1, 1))
        columns.append((both_wrong_ir_raman * C[:, 2]).reshape(-1, 1))
        indicators[seed] = np.concatenate(columns, axis=1)
    return indicators


def _point_conditional(task_info: dict) -> list[dict]:
    names = task_info["names"]
    rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            ind = _conditional_indicators(task_info, policy)[seed]
            total = float(ind.shape[0])
            counts = ind.sum(axis=0)
            c = {mod: counts[MODALITIES.index(mod)] for mod in MODALITIES}
            for d_index, (base, other) in enumerate(DIRECTIONS):
                num = counts[3 + d_index]
                den = total - c[base]
                rate = num / den if den else None
                p_other = c[other] / total if total else None
                rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "metric": "conditional_correct_given_base_wrong",
                    "base_modality": base, "other_modality": other,
                    "n_pairs": int(total), "count_num": float(num), "count_den": float(den),
                    "rate": rate, "p_other_marginal": p_other,
                    "delta": None if rate is None or p_other is None else rate - p_other,
                })
            both_wrong = counts[9]
            both_wrong_uv = counts[10]
            uv_marginal = c["uvvis"] / total if total else None
            rows.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                "metric": "uv_marginal_correct_rate", "base_modality": "", "other_modality": "uvvis",
                "count_num": c["uvvis"], "count_den": total, "rate": uv_marginal,
                "p_other_marginal": uv_marginal, "delta": 0.0,
            })
            rows.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                "metric": "ir_raman_both_wrong_uv_correct_count", "base_modality": "ir", "other_modality": "raman",
                "count_num": both_wrong_uv, "count_den": total, "rate": both_wrong_uv / total,
                "p_other_marginal": "", "delta": "",
            })
            rows.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                "metric": "unique_uv_correct_count", "base_modality": "ir", "other_modality": "raman",
                "count_num": both_wrong_uv, "count_den": total, "rate": both_wrong_uv / total,
                "p_other_marginal": "", "delta": "",
            })
            rescue = both_wrong_uv / both_wrong if both_wrong else None
            rows.append({
                "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                "metric": "uv_rescue_rate", "base_modality": "ir+raman", "other_modality": "uvvis",
                "count_num": both_wrong_uv, "count_den": both_wrong, "rate": rescue,
                "p_other_marginal": uv_marginal,
                "delta": None if rescue is None or uv_marginal is None else rescue - uv_marginal,
            })
    return rows


def _conditional_bootstrap(task_info: dict, draws: np.ndarray, cluster_id: np.ndarray, reps: int) -> list[dict]:
    rows = []
    for policy in POLICIES:
        indicators = _conditional_indicators(task_info, policy)
        arrays = {}
        for seed in SEEDS:
            I = indicators[seed]
            total = np.ones(I.shape[0])
            seed_values = defaultdict(lambda: np.full(reps, np.nan))
            for rep in range(reps):
                w = np.repeat(draws[rep][cluster_id].astype(np.float64), len(task_info["names"]))
                counts = I.T @ w
                c = {mod: counts[MODALITIES.index(mod)] for mod in MODALITIES}
                total_w = float(w.sum())
                for d_index, (base, other) in enumerate(DIRECTIONS):
                    num = counts[3 + d_index]; den = total_w - c[base]
                    rate = num / den if den else np.nan
                    p_other = c[other] / total_w if total_w else np.nan
                    seed_values[("conditional", base, other)][rep] = rate
                    seed_values[("delta", base, other)][rep] = rate - p_other if np.isfinite(rate) else np.nan
                both_wrong = counts[9]; both_wrong_uv = counts[10]
                uv_marginal = c["uvvis"] / total_w if total_w else np.nan
                seed_values[("uv_marginal", "", "")][rep] = uv_marginal
                seed_values[("both_wrong_uv", "", "")][rep] = both_wrong_uv
                seed_values[("rescue", "", "")][rep] = both_wrong_uv / both_wrong if both_wrong else np.nan
                seed_values[("rescue_delta", "", "")][rep] = (both_wrong_uv / both_wrong - uv_marginal) if both_wrong else np.nan
            arrays[seed] = seed_values
        for key in arrays[SEEDS[0]].keys():
            matrix = np.vstack([arrays[seed][key] for seed in SEEDS])
            valid = np.all(np.isfinite(matrix), axis=0)
            three = np.full(reps, np.nan)
            if valid.any():
                three[valid] = matrix[:, valid].mean(axis=0)
            metric, base, other = key
            rows.append({
                "task": task_info["task"], "threshold_policy": policy, "metric": metric,
                "base_modality": base, "other_modality": other, "three_seed": three,
            })
    return rows


def _sample_level(task_info: dict, output_rows: list, spearman_rows: list, hard_rows: list) -> None:
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    for seed in SEEDS:
        for policy in POLICIES:
            vectors = {}
            sets = {}
            for mod in MODALITIES:
                model = models[(seed, mod)]
                pred = _pred(model, policy)
                truth = task_info["labels"].astype(bool)
                errors = pred != truth
                n_errors = errors.sum(axis=1).astype(np.int64)
                hc_fp = (~truth) & pred & (model["probabilities"] >= HC_FP_PROB)
                hc_fn = truth & (~pred) & (model["probabilities"] <= HC_FN_PROB)
                n_hc = (hc_fp | hc_fn).sum(axis=1)
                vectors[mod] = n_errors
                selected, boundary = _hard_indices(n_errors)
                sets[mod] = set(int(i) for i in selected)
                output_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy, "modality": mod,
                    "n_samples": len(n_errors), "n_labels": len(names),
                    "mean_n_errors": float(n_errors.mean()),
                    "median_n_errors": float(np.median(n_errors)),
                    "q1_n_errors": float(np.percentile(n_errors, 25, method="linear")),
                    "q3_n_errors": float(np.percentile(n_errors, 75, method="linear")),
                    "iqr_n_errors": float(np.percentile(n_errors, 75, method="linear") - np.percentile(n_errors, 25, method="linear")),
                    "exact_match_count": int((n_errors == 0).sum()),
                    "exact_match_rate": float((n_errors == 0).mean()),
                    "n_high_confidence_errors": int(n_hc.sum()),
                    "n_samples_with_high_confidence_error": int((n_hc > 0).sum()),
                })
                hard_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy, "modality": mod,
                    "n_hard": len(selected), "boundary_error": boundary, "jaccard": "",
                })
            for a, b in PAIRS:
                rho = _spearman(vectors[a], vectors[b])
                spearman_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "modality_a": a, "modality_b": b, "spearman_rho": rho,
                })
                inter = len(sets[a] & sets[b]); union = len(sets[a] | sets[b])
                hard_rows.append({
                    "task": task_info["task"], "seed": seed, "threshold_policy": policy,
                    "modality": f"{a}-{b}", "n_hard": "", "boundary_error": "",
                    "jaccard": 1.0 if union == 0 else inter / union,
                })


def _structure_definitions(row_index: np.ndarray, structure: dict[int, dict], quartiles: dict):
    q1, q2, q3 = quartiles["q1"], quartiles["q2"], quartiles["q3"]
    def heavy_bin(v):
        if v <= q1: return "Q1"
        if v <= q2: return "Q2"
        if v <= q3: return "Q3"
        return "Q4"
    def tokens(value):
        return {t.strip() for t in (value or "").split(";") if t.strip()}
    def int_field(row, field): return int(float(row[field]))
    definitions = []
    getters = [
        ("ring_count", ("0", "1", "2", ">=3"), lambda r: "0" if int_field(r, "ring_count") == 0 else "1" if int_field(r, "ring_count") == 1 else "2" if int_field(r, "ring_count") == 2 else ">=3"),
        ("aromatic_ring_count", ("0", "1", ">=2"), lambda r: "0" if int_field(r, "aromatic_ring_count") == 0 else "1" if int_field(r, "aromatic_ring_count") == 1 else ">=2"),
        ("n_o_f", ("0", "1", "2", ">=3"), lambda r: str(v) if (v := int_field(r, "nitrogen_count") + int_field(r, "oxygen_count") + int_field(r, "fluorine_count")) < 3 else ">=3"),
        ("heavy_atom_count", ("Q1", "Q2", "Q3", "Q4"), lambda r: heavy_bin(int_field(r, "heavy_atom_count"))),
        ("rotatable_bond_count", ("0", "1", ">=2"), lambda r: "0" if int_field(r, "rotatable_bond_count") == 0 else "1" if int_field(r, "rotatable_bond_count") == 1 else ">=2"),
        ("fraction_csp3", ("<0.5", ">=0.5"), lambda r: "<0.5" if float(r["fraction_csp3"]) < 0.5 else ">=0.5"),
    ]
    rows = [structure[int(i)] for i in row_index]
    for dimension, bins, getter in getters:
        for label in bins:
            mask = np.asarray([getter(r) == label for r in rows], dtype=bool)
            definitions.append({"subgroup": f"{dimension}={label}", "dimension": dimension, "bin": label, "mask": mask, "n_samples": int(mask.sum())})
    for group in ("carbonyl", "aromatic", "nitrile", "alkyne", "amine", "hydroxyl"):
        if group == "carbonyl":
            mask = np.asarray(["carbonyl" in tokens(r.get("functional_group_types", "")) for r in rows], dtype=bool)
        elif group == "aromatic":
            mask = np.asarray([any("aromatic" in t or "aryl" in t for t in tokens(r.get("functional_group_types", ""))) for r in rows], dtype=bool)
        else:
            mask = np.asarray([group in tokens(r.get("functional_group_types", "")) for r in rows], dtype=bool)
        definitions.append({"subgroup": f"functional_group={group}", "dimension": "functional_group_types", "bin": group, "mask": mask, "n_samples": int(mask.sum())})
    return definitions


def _point_subgroups(task_info: dict, definitions: list[dict]) -> tuple[list[dict], list[dict]]:
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    rows = []
    for definition in definitions:
        mask = definition["mask"]
        n = int(mask.sum())
        if n == 0:
            continue
        for index, name in enumerate(names):
            positive = int(task_info["labels"][mask, index].sum())
            sufficient = n >= SUPPORT_MIN_N and positive >= SUPPORT_MIN_POS
            reason = "ok" if sufficient else ("n<100" if n < SUPPORT_MIN_N else "positive<30")
            for seed in SEEDS:
                for mod in MODALITIES:
                    model = models[(seed, mod)]
                    y = task_info["labels"][mask, index]
                    p = model["probabilities"][mask, index]
                    ap, auroc = _metrics(y, p)
                    for policy in POLICIES:
                        pred = _pred(model, policy)[mask, index]
                        rows.append({
                            "task": task_info["task"], "subgroup": definition["subgroup"],
                            "dimension": definition["dimension"], "bin": definition["bin"],
                            "n_samples": n, "label_index": index, "label": name,
                            "positive_count": positive, "seed": seed, "modality": mod,
                            "threshold_policy": policy, "ap": ap, "auroc": auroc,
                            "f1": _confusion(y, pred)["f1"],
                            "support_sufficient": sufficient, "support_reason": reason,
                        })
    return rows


def _subgroup_complementarity(task_info: dict, definitions: list[dict]) -> list[dict]:
    names = task_info["names"]
    models = {(m["seed"], m["modality"]): m for m in task_info["models"]}
    rows = []
    for definition in definitions:
        mask = definition["mask"]
        n = int(mask.sum())
        if n == 0:
            continue
        positive_max = max(int(task_info["labels"][mask, i].sum()) for i in range(len(names)))
        sufficient = n >= SUPPORT_MIN_N and positive_max >= SUPPORT_MIN_POS
        for seed in SEEDS:
            for policy in POLICIES:
                correct = {mod: (_pred(models[(seed, mod)], policy)[mask] == task_info["labels"][mask]) for mod in MODALITIES}
                C = np.stack([correct[mod] for mod in MODALITIES], axis=1).astype(np.int64)
                W = 1 - C
                n_pairs = C.shape[0]
                for base, other in DIRECTIONS:
                    den = int(W[:, MODALITIES.index(base)].sum())
                    num = int((W[:, MODALITIES.index(base)] * C[:, MODALITIES.index(other)]).sum())
                    rate = num / den if den else None
                    p_other = float(C[:, MODALITIES.index(other)].mean())
                    rows.append({
                        "task": task_info["task"], "subgroup": definition["subgroup"],
                        "dimension": definition["dimension"], "bin": definition["bin"],
                        "n_samples": n, "support_sufficient": sufficient,
                        "seed": seed, "threshold_policy": policy,
                        "metric": "conditional_rescue", "base_modality": base, "other_modality": other,
                        "count_num": num, "count_den": den, "rate": rate,
                        "p_other_marginal": p_other,
                        "delta": None if rate is None else rate - p_other,
                    })
                both_wrong = int((W[:, 0] * W[:, 1]).sum())
                both_wrong_uv = int((W[:, 0] * W[:, 1] * C[:, 2]).sum())
                rows.append({
                    "task": task_info["task"], "subgroup": definition["subgroup"],
                    "dimension": definition["dimension"], "bin": definition["bin"],
                    "n_samples": n, "support_sufficient": sufficient,
                    "seed": seed, "threshold_policy": policy,
                    "metric": "ir_raman_both_wrong_uv_correct", "base_modality": "ir", "other_modality": "raman",
                    "count_num": both_wrong_uv, "count_den": n_pairs,
                    "rate": both_wrong_uv / n_pairs, "p_other_marginal": "", "delta": "",
                })
                any_correct = int((C.sum(axis=1) >= 1).sum())
                all_wrong = int((C.sum(axis=1) == 0).sum())
                rows.append({
                    "task": task_info["task"], "subgroup": definition["subgroup"],
                    "dimension": definition["dimension"], "bin": definition["bin"],
                    "n_samples": n, "support_sufficient": sufficient,
                    "seed": seed, "threshold_policy": policy,
                    "metric": "oracle_any_correct", "base_modality": "", "other_modality": "",
                    "count_num": any_correct, "count_den": n_pairs,
                    "rate": any_correct / n_pairs, "p_other_marginal": "", "delta": "",
                })
    return rows


def _prepare_label(probability: np.ndarray, truth: np.ndarray):
    probability = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_truth = truth[order]
    ends = np.flatnonzero(np.r_[sorted_probability[1:] != sorted_probability[:-1], True])
    return order, sorted_truth, ends


def _metrics_batch(draws, cluster_sorted, sorted_truth, ends, prediction_fixed, prediction_frozen):
    weights = draws[:, cluster_sorted].astype(np.float64)
    positive = weights * sorted_truth
    negative = weights - positive
    cum_positive = np.cumsum(positive, axis=1)
    cum_negative = np.cumsum(negative, axis=1)
    tp = cum_positive[:, ends]
    fp = cum_negative[:, ends]
    P = tp[:, -1]
    N = fp[:, -1]
    valid = (P > 0) & (N > 0)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, P[:, None], out=np.zeros_like(tp), where=P[:, None] > 0)
    drecall = np.diff(np.concatenate([np.zeros((tp.shape[0], 1)), recall], axis=1), axis=1)
    ap = (drecall * precision).sum(axis=1)
    tpr = np.divide(tp, P[:, None], out=np.zeros_like(tp), where=P[:, None] > 0)
    fpr = np.divide(fp, N[:, None], out=np.zeros_like(fp), where=N[:, None] > 0)
    dfpr = np.diff(np.concatenate([np.zeros((fpr.shape[0], 1)), fpr], axis=1), axis=1)
    prev_tpr = np.concatenate([np.zeros((tpr.shape[0], 1)), tpr[:, :-1]], axis=1)
    auroc = (dfpr * (tpr + prev_tpr) / 2.0).sum(axis=1)
    ap[~valid] = np.nan; auroc[~valid] = np.nan

    def f1_for(pred):
        tp_w = (positive * pred).sum(axis=1)
        fp_w = (negative * pred).sum(axis=1)
        fn_w = (positive * (1.0 - pred)).sum(axis=1)
        denom = 2 * tp_w + fp_w + fn_w
        return np.divide(2 * tp_w, denom, out=np.zeros_like(tp_w), where=denom > 0)

    return ap, auroc, f1_for(prediction_fixed), f1_for(prediction_frozen)


def _bootstrap_subgroup(mask: np.ndarray, info: dict, cluster_id: np.ndarray, reps: int, rng_seed: int) -> tuple[list[dict], dict]:
    positions = np.flatnonzero(mask)
    local_cluster, remap = np.unique(cluster_id[positions], return_inverse=True)
    sizes = np.bincount(remap, minlength=len(local_cluster)).astype(np.int64)
    draws = _draws(reps, len(local_cluster), rng_seed)
    names = info["names"]
    models = {(m["seed"], m["modality"]): m for m in info["models"]}
    values = {}
    point = {}
    for seed in SEEDS:
        for modality in MODALITIES:
            model = models[(seed, modality)]
            probs = model["probabilities"][positions]
            truth = model["labels"][positions]
            thresholds = model["thresholds"]
            arrays = {key: np.empty((len(names), reps), dtype=np.float64) for key in ("ap", "auroc", "f1_fixed_0.5", "f1_validation_f1")}
            point_label = {key: np.empty(len(names)) for key in arrays}
            for index in range(len(names)):
                order, sorted_truth, ends = _prepare_label(probs[:, index], truth[:, index])
                cluster_sorted = remap[order]
                pred_fixed = (probs[order, index] >= 0.5).astype(np.float64)
                pred_frozen = (probs[order, index] >= float(thresholds[index])).astype(np.float64)
                for start in range(0, reps, CHUNK):
                    stop = min(start + CHUNK, reps)
                    ap, auroc, f1f, f1v = _metrics_batch(
                        draws[start:stop], cluster_sorted, sorted_truth, ends, pred_fixed, pred_frozen
                    )
                    arrays["ap"][index, start:stop] = ap
                    arrays["auroc"][index, start:stop] = auroc
                    arrays["f1_fixed_0.5"][index, start:stop] = f1f
                    arrays["f1_validation_f1"][index, start:stop] = f1v
                y = truth[:, index].astype(np.int64)
                p = probs[:, index].astype(np.float64)
                ap, auroc = _metrics(y, p)
                point_label["ap"][index] = ap if ap is not None else np.nan
                point_label["auroc"][index] = auroc if auroc is not None else np.nan
                point_label["f1_fixed_0.5"][index] = _confusion(y, (p >= 0.5))["f1"]
                point_label["f1_validation_f1"][index] = _confusion(y, (p >= float(thresholds[index])))["f1"]
            valid_ap = np.isfinite(arrays["ap"])
            valid_auroc = np.isfinite(arrays["auroc"])
            counts_ap = valid_ap.sum(axis=0)
            counts_auroc = valid_auroc.sum(axis=0)
            macro = {
                "ap": np.divide(np.nansum(arrays["ap"], axis=0), counts_ap, out=np.full(reps, np.nan), where=counts_ap > 0),
                "auroc": np.divide(np.nansum(arrays["auroc"], axis=0), counts_auroc, out=np.full(reps, np.nan), where=counts_auroc > 0),
                "f1_fixed_0.5": arrays["f1_fixed_0.5"].mean(axis=0),
                "f1_validation_f1": arrays["f1_validation_f1"].mean(axis=0),
            }
            point_macro = {
                "ap": float(np.nanmean(point_label["ap"])) if np.isfinite(point_label["ap"]).any() else np.nan,
                "auroc": float(np.nanmean(point_label["auroc"])) if np.isfinite(point_label["auroc"]).any() else np.nan,
                "f1_fixed_0.5": float(np.mean(point_label["f1_fixed_0.5"])),
                "f1_validation_f1": float(np.mean(point_label["f1_validation_f1"])),
            }
            values[(seed, modality)] = macro
            point[(seed, modality)] = point_macro
    rows = []
    replicate_store = {}
    for base, other in PAIRS:
        comparison = PAIR_LABELS[(base, other)]
        for policy, metric in (("threshold_free", "AP"), ("threshold_free", "AUROC"),
                               ("fixed_0.5", "F1"), ("validation_f1", "F1")):
            key = metric.lower() if metric != "F1" else f"f1_{policy}"
            seed_macros = []
            per_seed_diff = {}
            for seed in SEEDS:
                diff = values[(seed, base)][key] - values[(seed, other)][key]
                per_seed_diff[seed] = diff
            matrix = np.vstack([per_seed_diff[seed] for seed in SEEDS])
            valid = np.all(np.isfinite(matrix), axis=0)
            three = np.full(reps, np.nan)
            if valid.any():
                three[valid] = matrix[:, valid].mean(axis=0)
            seed_point = [point[(seed, base)][key] - point[(seed, other)][key] for seed in SEEDS]
            if any(not np.isfinite(v) for v in seed_point):
                point_mean = None; seed_sd = None
            else:
                point_mean = float(np.mean(seed_point)); seed_sd = float(statistics.stdev(seed_point))
            clean = three[np.isfinite(three)]
            low = float(np.percentile(clean, CI[0])) if len(clean) >= 2 else None
            high = float(np.percentile(clean, CI[1])) if len(clean) >= 2 else None
            rows.append({
                "task": info["task"], "subgroup": info["subgroup"], "dimension": info["dimension"], "bin": info["bin"],
                "n_samples": int(mask.sum()), "policy": policy, "comparison": comparison, "metric": metric,
                "point": point_mean, "seed_sd": seed_sd, "ci_low": low, "ci_high": high,
                "n_valid_reps": len(clean), "n_total_reps": reps,
            })
            replicate_store["|".join([info["task"], info["subgroup"], policy, comparison, metric])] = three
    return rows, replicate_store


def _build_report(pairwise_rows, or_rows, three_or_rows, oracle_summary, conditional_point, conditional_ci,
                  sample_rows, subgroup_metrics, subgroup_bootstrap_rows, counts) -> str:
    lines = ["# Line B B4.4 — Test Error Complementarity & Structural Subgroups", ""]
    lines.append("> **Classification: exploratory / post-hoc.** Test was opened before B4. Positive OR/AND are fixed rules, "
                 "not Oracle; Oracle is a diagnostic upper bound, not a deployable model; subgroup and conditional findings "
                 "carry no confirmatory significance.")
    lines.append("")
    lines.append(f"- Models: 18 (2 tasks × 3 modalities × 3 seeds); two F1 policies (fixed 0.5 primary, frozen Validation-F1 secondary).")
    lines.append(f"- Pairwise rows: {counts['pairwise_rows']}; McNemar rows: {counts['mcnemar_rows']}; subgroup metric rows: {counts['subgroup_metric_rows']}.")
    lines.append("- McNemar: two-sided exact test; Holm family = task × seed × policy across the three modality pairs and all labels (C2 rule).")
    lines.append("- Conditional/OR/Oracle use sample-label pairs; hard samples use deterministic top 10%; high-confidence errors use prob ≥ 0.90 FP / ≤ 0.10 FN (C2 rule).")
    lines.append("")
    lines.append("## 1. Macro pairwise error complementarity (fixed 0.5, three-seed mean)")
    lines.append("")
    lines.append("| Task | Comparison | F1 A | F1 B | Error Jaccard | FN rescue | FP correction | Both wrong |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for comparison in ("IR-Raman", "UV-IR", "UV-Raman"):
            rows = [r for r in pairwise_rows if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["comparison"] == comparison and r["scope"] == "macro"]
            if not rows:
                continue
            lines.append(f"| {task} | {comparison} | {np.mean([r['base_f1'] for r in rows]):.4f} | {np.mean([r['other_f1'] for r in rows]):.4f} | "
                         f"{np.mean([r['error_jaccard'] for r in rows]):.4f} | {int(np.mean([r['fn_rescue'] for r in rows]))} | "
                         f"{int(np.mean([r['fp_correction'] for r in rows]))} | {int(np.mean([r['both_wrong'] for r in rows]))} |")
    lines.append("")
    lines.append("## 2. Positive OR / AND (fixed 0.5, three-seed mean)")
    lines.append("")
    lines.append("| Task | Comparison | F1 OR | F1 AND | OR gain vs max | +gain labels | −gain labels |")
    lines.append("|---|---|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for comparison in ("IR-Raman", "UV-IR", "UV-Raman"):
            rows = [r for r in or_rows if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["comparison"] == comparison]
            if not rows:
                continue
            lines.append(f"| {task} | {comparison} | {np.mean([r['or_f1'] for r in rows]):.4f} | {np.mean([r['and_f1'] for r in rows]):.4f} | "
                         f"{np.mean([r['or_gain_vs_max'] for r in rows]):+.4f} | {int(np.mean([r['n_positive_or_gain_labels'] for r in rows]))} | "
                         f"{int(np.mean([r['n_negative_or_gain_labels'] for r in rows]))} |")
    lines.append("")
    lines.append("### Three-modality OR vs Validation-designated best_single")
    lines.append("")
    lines.append("| Task | best_single | F1 best_single | F1 three-OR | OR gain |")
    lines.append("|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in three_or_rows if r["task"] == task and r["threshold_policy"] == "fixed_0.5"]
        lines.append(f"| {task} | {rows[0]['best_single']} | {np.mean([r['best_single_f1'] for r in rows]):.4f} | "
                     f"{np.mean([r['three_or_f1'] for r in rows]):.4f} | {np.mean([r['or_gain_vs_best_single'] for r in rows]):+.4f} |")
    lines.append("")
    lines.append("## 3. True Oracle (sample-label pairs, three-seed mean)")
    lines.append("")
    lines.append("| Task | Policy | any correct | exactly one | all wrong | all correct | only IR | only Raman | only UV |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for policy in ("fixed_0.5", "validation_f1"):
            rows = [r for r in oracle_summary if r["task"] == task and r["threshold_policy"] == policy and r["scope"] == "overall"]
            lines.append(f"| {task} | {policy} | {np.mean([r['any_correct_rate'] for r in rows]):.4f} | "
                         f"{np.mean([r['exactly_one_correct_rate'] for r in rows]):.4f} | {np.mean([r['all_wrong_rate'] for r in rows]):.4f} | "
                         f"{np.mean([r['all_correct'] / r['n_pairs'] for r in rows]):.4f} | "
                         f"{np.mean([r['only_ir_correct'] for r in rows]):.0f} | {np.mean([r['only_raman_correct'] for r in rows]):.0f} | "
                         f"{np.mean([r['only_uv_correct'] for r in rows]):.0f} |")
    lines.append("")
    lines.append("## 4. Conditional complementarity (fixed 0.5; point over three seeds)")
    lines.append("")
    lines.append("| Task | Direction | P(other correct | base wrong) | marginal P(other) | delta |")
    lines.append("|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for base, other in DIRECTIONS:
            rows = [r for r in conditional_point if r["task"] == task and r["threshold_policy"] == "fixed_0.5"
                    and r["metric"] == "conditional_correct_given_base_wrong" and r["base_modality"] == base and r["other_modality"] == other]
            lines.append(f"| {task} | {base.upper()} wrong → {other.upper()} | {np.mean([r['rate'] for r in rows]):.4f} | "
                         f"{np.mean([r['p_other_marginal'] for r in rows]):.4f} | {np.mean([r['delta'] for r in rows]):+.4f} |")
    lines.append("")
    lines.append("### UV rescue vs marginal (fixed 0.5, three-seed mean, 95% bootstrap CI over B4.3 scaffold draws)")
    lines.append("")
    lines.append("| Task | UV rescue rate | UV marginal | delta | CI low | CI high |")
    lines.append("|---|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for metric, label in (("rescue", "UV rescue"), ("rescue_delta", "rescue − marginal")):
            rows = [r for r in conditional_ci if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["metric"] == metric]
            if not rows:
                continue
            arr = rows[0]["three_seed"]
            clean = arr[np.isfinite(arr)]
            lines.append(f"| {task} | {label} | {np.mean(clean):.4f} | | {np.mean(clean):+.4f} | {np.percentile(clean, 2.5):+.4f} | {np.percentile(clean, 97.5):+.4f} |")
    lines.append("")
    lines.append("## 5. Sample-level and high-confidence errors (fixed 0.5, three-seed mean)")
    lines.append("")
    lines.append("| Task | Modality | mean errors | median | IQR | exact-match rate | HC errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        for modality in MODALITIES:
            rows = [r for r in sample_rows if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["modality"] == modality]
            lines.append(f"| {task} | {modality.upper()} | {np.mean([r['mean_n_errors'] for r in rows]):.3f} | "
                         f"{np.mean([r['median_n_errors'] for r in rows]):.1f} | {np.mean([r['iqr_n_errors'] for r in rows]):.2f} | "
                         f"{np.mean([r['exact_match_rate'] for r in rows]):.4f} | {np.mean([r['n_high_confidence_errors'] for r in rows]):.0f} |")
    lines.append("")
    lines.append("## 6. Structural subgroups")
    lines.append("")
    supported = [r for r in subgroup_metrics if r["support_sufficient"]]
    lines.append(f"- Predefined subgroups: {len({r['subgroup'] for r in subgroup_metrics})}; rows meeting per-label support (n≥100 and positive≥30): {len(supported)} metric rows.")
    lines.append(f"- Heavy-atom quartiles are degenerate by design (D1: Q1=Q2=Q3=9.0; all Test molecules fall in Q1).")
    if subgroup_bootstrap_rows:
        lines.append(f"- Paired within-subgroup scaffold bootstrap CIs computed for {len({r['subgroup'] for r in subgroup_bootstrap_rows})} supported subgroups "
                     f"(500 replicates each, shared draw per subgroup; exploratory).")
    lines.append("- Subgroup rows below support thresholds are retained and flagged `support_sufficient=false`.")
    lines.append("")
    lines.append("## 7. Interpretation boundaries")
    lines.append("")
    lines.append("- Positive OR/AND are fixed rules; OR F1 is not Oracle and may gain recall while losing precision.")
    lines.append("- Oracle correctness is a diagnostic ceiling over three fixed models, not a fusion model.")
    lines.append("- Conditional rescue is descriptive; only a positive delta with a bootstrap CI lower bound > 0 is marked as evidence, and still exploratory.")
    lines.append("- No Test-driven threshold, seed, modality or label selection; no confirmatory significance claims.")
    lines.append("")
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--subgroup-bootstrap-reps", type=int, default=500)
    parser.add_argument("--skip-subgroup-bootstrap", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.out_dir.exists():
        raise SystemExit(f"Output directory exists: {args.out_dir}")
    preflight = _load_preflight(args.freeze, args.freeze_sha256)

    b43 = read_json(B43 / "summary.json")
    mapping = _load_scaffold_map()
    cluster_names, cluster_id, cluster_sizes = _clusters(preflight["row_index"], mapping)
    draws = _draws(int(b43["config"]["reps"]), len(cluster_names), int(b43["config"]["rng_seed"]))
    b43_draws_ok = _draws_sha(draws) == b43["config"]["draws_sha256"]
    if not b43_draws_ok:
        raise SystemExit("B4.3 draw matrix reconstruction does not match recorded SHA")

    structure = _load_structure(preflight["row_index"])
    d1_summary = {task: read_json(D1 / task / "structure_analysis" / "summary.json") for task, _ in TASKS}
    best_single = {task: preflight["freeze"]["analysis"]["b3a_complementarity"][task]["best_single_modality"] for task, _ in TASKS}
    task_infos = {}
    for task, label_set in TASKS:
        task_infos[task] = {
            "task": task, "label_set": label_set,
            "models": [m for m in preflight["models"] if m["task"] == task],
            "labels": preflight["labels_by_task"][task],
            "names": preflight["names_by_task"][task],
            "best_single": best_single[task],
        }

    pairwise_rows, jaccard_rows, mcnemar_rows = [], [], []
    or_rows, three_or_rows = [], []
    oracle_summary, oracle_label = [], []
    conditional_point = []
    sample_rows, spearman_rows, hard_rows = [], [], []
    for task, _label_set in TASKS:
        info = task_infos[task]
        p, j, m = _point_pairwise(info)
        pairwise_rows += p; jaccard_rows += j; mcnemar_rows += m
        o, t = _point_or(info)
        or_rows += o; three_or_rows += t
        s, l = _point_oracle(info)
        oracle_summary += s; oracle_label += l
        conditional_point += _point_conditional(info)
        _sample_level(info, sample_rows, spearman_rows, hard_rows)

    # Conditional bootstrap CIs reusing the accepted B4.3 draw matrix.
    conditional_ci = []
    conditional_reps = {}
    for task, _label_set in TASKS:
        boot = _conditional_bootstrap(task_infos[task], draws, cluster_id, args.bootstrap_reps)
        conditional_ci += boot
    for row in conditional_ci:
        conditional_reps["|".join([row["task"], row["threshold_policy"], row["metric"], row["base_modality"], row["other_modality"]])] = row["three_seed"]

    # Structural subgroups and their within-subgroup paired bootstrap.
    subgroup_metrics = []
    subgroup_comp = []
    subgroup_boot_rows = []
    subgroup_boot_reps = {}
    for task, _label_set in TASKS:
        quartiles = d1_summary[task]["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
        definitions = _structure_definitions(preflight["row_index"], structure, quartiles)
        subgroup_metrics += _point_subgroups(task_infos[task], definitions)
        subgroup_comp += _subgroup_complementarity(task_infos[task], definitions)
        if not args.skip_subgroup_bootstrap:
            for definition in definitions:
                mask = definition["mask"]
                n = int(mask.sum())
                if n == 0:
                    continue
                positive_max = max(int(task_infos[task]["labels"][mask, i].sum()) for i in range(len(task_infos[task]["names"])))
                if n < SUPPORT_MIN_N or positive_max < SUPPORT_MIN_POS:
                    continue
                info = dict(task_infos[task])
                info.update({"subgroup": definition["subgroup"], "dimension": definition["dimension"], "bin": definition["bin"]})
                rows, reps = _bootstrap_subgroup(mask, info, cluster_id, args.subgroup_bootstrap_reps, DEFAULT_RNG)
                subgroup_boot_rows += rows
                subgroup_boot_reps.update(reps)

    # Conditional CI table with three-seed full-Test point estimates.
    def _point_value(task, policy, base, other, kind):
        if kind in ("conditional", "delta"):
            rows = [r for r in conditional_point if r["task"] == task and r["threshold_policy"] == policy
                    and r["metric"] == "conditional_correct_given_base_wrong"
                    and r["base_modality"] == base and r["other_modality"] == other]
            return float(np.mean([r["rate"] if kind == "conditional" else r["delta"] for r in rows]))
        mapping = {
            "uv_marginal": ("uv_marginal_correct_rate", "rate"),
            "both_wrong_uv": ("ir_raman_both_wrong_uv_correct_count", "rate"),
            "rescue": ("uv_rescue_rate", "rate"),
            "rescue_delta": ("uv_rescue_rate", "delta"),
        }
        metric, field = mapping[kind]
        rows = [r for r in conditional_point if r["task"] == task and r["threshold_policy"] == policy and r["metric"] == metric]
        return float(np.mean([r[field] for r in rows]))

    conditional_ci_rows = []
    for row in conditional_ci:
        clean = row["three_seed"][np.isfinite(row["three_seed"])]
        if len(clean) < 2:
            continue
        conditional_ci_rows.append({
            "task": row["task"], "threshold_policy": row["threshold_policy"], "metric": row["metric"],
            "base_modality": row["base_modality"], "other_modality": row["other_modality"],
            "point": _point_value(row["task"], row["threshold_policy"], row["base_modality"], row["other_modality"], row["metric"]),
            "ci_low": float(np.percentile(clean, CI[0])), "ci_high": float(np.percentile(clean, CI[1])),
            "n_valid_reps": int(len(clean)), "n_total_reps": args.bootstrap_reps,
        })

    with __import__("training.line_b.common", fromlist=["atomic_output_dir"]).atomic_output_dir(args.out_dir) as temporary:
        _write_csv(temporary / "pairwise_metrics.csv", pairwise_rows, list(pairwise_rows[0].keys()))
        _write_csv(temporary / "pairwise_error_jaccard.csv", jaccard_rows, list(jaccard_rows[0].keys()))
        _write_csv(temporary / "mcnemar_holm.csv", mcnemar_rows, list(mcnemar_rows[0].keys()))
        _write_csv(temporary / "pairwise_or_and.csv", or_rows, list(or_rows[0].keys()))
        _write_csv(temporary / "three_modality_or.csv", three_or_rows, list(three_or_rows[0].keys()))
        _write_csv(temporary / "oracle_summary.csv", oracle_summary, list(oracle_summary[0].keys()))
        _write_csv(temporary / "oracle_per_label.csv", oracle_label, list(oracle_label[0].keys()))
        _write_csv(temporary / "conditional_complementarity.csv", conditional_point, list(conditional_point[0].keys()))
        _write_csv(temporary / "conditional_bootstrap_ci.csv", conditional_ci_rows, list(conditional_ci_rows[0].keys()))
        _write_csv(temporary / "sample_error_summary.csv", sample_rows, list(sample_rows[0].keys()))
        _write_csv(temporary / "spearman.csv", spearman_rows, list(spearman_rows[0].keys()))
        _write_csv(temporary / "hard_sample_summary.csv", hard_rows, list(hard_rows[0].keys()))
        _write_csv(temporary / "subgroup_metrics.csv", subgroup_metrics, list(subgroup_metrics[0].keys()))
        _write_csv(temporary / "subgroup_complementarity.csv", subgroup_comp, list(subgroup_comp[0].keys()))
        if subgroup_boot_rows:
            _write_csv(temporary / "subgroup_bootstrap_ci.csv", subgroup_boot_rows, list(subgroup_boot_rows[0].keys()))
        np.savez_compressed(temporary / "conditional_replicates.npz", **conditional_reps)
        if subgroup_boot_reps:
            np.savez_compressed(temporary / "subgroup_bootstrap_replicates.npz", **subgroup_boot_reps)

        counts = {
            "pairwise_rows": len(pairwise_rows), "mcnemar_rows": len(mcnemar_rows),
            "oracle_rows": len(oracle_label), "conditional_rows": len(conditional_point),
            "sample_rows": len(sample_rows), "subgroup_metric_rows": len(subgroup_metrics),
            "subgroup_complementarity_rows": len(subgroup_comp),
            "subgroup_bootstrap_rows": len(subgroup_boot_rows),
        }
        report_text = _build_report(pairwise_rows, or_rows, three_or_rows, oracle_summary, conditional_point,
                                    conditional_ci, sample_rows, subgroup_metrics, subgroup_boot_rows, counts)
        (temporary / "report.md").write_text(report_text, encoding="utf-8")

        summary = {
            "phase": "B4.4",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "classification": {"status": "exploratory/post-hoc", "blind_test_claim": False,
                               "confirmatory_significance": False, "positive_or_is_oracle": False},
            "config": {
                "bootstrap_reps_conditional": args.bootstrap_reps,
                "bootstrap_reps_subgroup": 0 if args.skip_subgroup_bootstrap else args.subgroup_bootstrap_reps,
                "b4_3_draws_sha256": b43["config"]["draws_sha256"],
                "b4_3_draws_reconstructed_match": b43_draws_ok,
                "rng_seed": int(b43["config"]["rng_seed"]),
                "hc_fp_prob": HC_FP_PROB, "hc_fn_prob": HC_FN_PROB, "hard_fraction": HARD_FRACTION,
                "support_min_n": SUPPORT_MIN_N, "support_min_positive": SUPPORT_MIN_POS,
                "best_single": best_single,
                "conditional_ci_percentiles": list(CI),
            },
            "preflight": {
                "protocol_sha256": preflight["protocol_sha256"], "freeze_sha256": preflight["freeze_sha256"],
                "row_index_sha256": sha256_array(preflight["row_index"]), "n_models": len(preflight["models"]),
            },
            "counts": counts,
            "sources": {
                "b4_1": {"predictions_root": relative_to_root(PREDICTIONS),
                          "prediction_manifests": [m["prediction_manifest_sha256"] for m in preflight["models"]]},
                "b4_2": {"analysis_manifest": _sha(B42 / "analysis_manifest.json"), "verification": _sha(B42 / "verification.json")},
                "b4_3": {"analysis_manifest": _sha(B43 / "analysis_manifest.json"), "verification": _sha(B43 / "verification.json")},
                "test_access_log": {"path": "runs/LINE-B-MANIFESTS/test_access_log.json",
                                    "sha256": _sha(ROOT / "runs/LINE-B-MANIFESTS/test_access_log.json")},
                "structure_library": {"path": relative_to_root(STRUCTURE), "sha256": _sha(STRUCTURE)},
                "scaffold_manifest": {"path": relative_to_root(SCAFFOLD_MANIFEST), "sha256": _sha(SCAFFOLD_MANIFEST)},
                "d1_training_cooccurrence": {
                    task: {name: _sha(D1 / task / "structure_analysis" / name)
                           for name in ("training_cooccurrence_counts.csv", "training_cooccurrence_phi.csv",
                                        "training_cooccurrence_cramers_v.csv", "training_cooccurrence_support.csv")}
                    for task, _ in TASKS
                },
            },
            "environment": _environment(),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        write_json(temporary / "summary.json", summary)
        analysis_outputs = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "analysis_manifest.json":
                analysis_outputs.append({"path": path.relative_to(temporary).as_posix(),
                                         "sha256": _sha(path), "size_bytes": int(path.stat().st_size)})
        manifest = {
            "phase": "B4.4", "generated_at": summary["generated_at"],
            "classification": summary["classification"], "config": summary["config"],
            "inputs": {
                "freeze": {"path": relative_to_root(args.freeze), "sha256": args.freeze_sha256},
                "b4_1": summary["sources"]["b4_1"], "b4_2": summary["sources"]["b4_2"], "b4_3": summary["sources"]["b4_3"],
                "test_access_log": summary["sources"]["test_access_log"],
                "structure_library": summary["sources"]["structure_library"],
                "scaffold_manifest": summary["sources"]["scaffold_manifest"],
                "d1_training_cooccurrence": summary["sources"]["d1_training_cooccurrence"],
            },
            "outputs": analysis_outputs,
            "code": {
                "analysis_script": relative_to_root(Path(__file__)),
                "analysis_script_sha256": _sha(Path(__file__)),
                "verifier": "training/line_b/verify_b4_4_analysis.py",
                "verifier_sha256": _sha(ROOT / "training/line_b/verify_b4_4_analysis.py")
                if (ROOT / "training/line_b/verify_b4_4_analysis.py").is_file() else None,
            },
            "environment": summary["environment"],
        }
        write_json(temporary / "analysis_manifest.json", manifest)

    print(json.dumps({"out_dir": relative_to_root(args.out_dir), "counts": counts,
                      "elapsed_seconds": summary["elapsed_seconds"],
                      "b4_3_draws_match": b43_draws_ok}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
