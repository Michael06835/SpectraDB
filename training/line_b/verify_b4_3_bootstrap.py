"""Independent acceptance verification for the B4.3 Test scaffold bootstrap.

Recomputes, from the frozen B4.1 Test predictions and the authoritative
scaffold manifest:

* the shared scaffold draw matrix (same RNG seed and draw construction);
* full-Test point estimates and comparisons;
* a sample of full bootstrap replicates with an independent weighted-metric
  implementation (scikit-learn weighted AP/AUROC + count-based weighted F1);
* percentile CIs and valid/undefined counts from the stored replicate arrays;
* seed SD with ddof=1;
* frozen-asset immutability.

It does not import or call the B4.3 bootstrap script's metric functions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

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
TEST_N = 12981
CI = (2.5, 97.5)
SAMPLE_REPS = (0, 1, 2, 17, 250, 777, 1500, 1999)
DEFAULT_ANALYSIS = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
DEFAULT_PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
DEFAULT_SCAFFOLD = ROOT / "processed" / "qm9s" / "prepared" / "qm9s_manifest_prepared.csv"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check_id: str, ok: bool, detail: str = "") -> None:
        self.items.append({"id": check_id, "ok": bool(ok), "detail": detail})

    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["ok"]]


def _csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    if value is None or value == "":
        return None
    return float(value)


def _load_freeze(path: Path, pin: str):
    if sha256_file(path) != pin:
        raise SystemExit("FREEZE SHA256 mismatch")
    freeze = read_json(path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise SystemExit("FREEZE is not a frozen B3 manifest")
    protocol = ROOT / freeze["protocol_path"]
    if sha256_file(protocol) != freeze["protocol_sha256"]:
        raise SystemExit("Protocol SHA256 mismatch")
    if len(freeze.get("models") or []) != 18:
        raise SystemExit("FREEZE does not register 18 models")
    return freeze


def _load_models(freeze: dict, pin: str, predictions_root: Path) -> list[dict]:
    models = []
    for record in freeze["models"]:
        out_dir = predictions_root / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        manifest = read_json(out_dir / "prediction_manifest.json")
        probabilities = np.load(out_dir / "probabilities.npy", allow_pickle=False)
        row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
        labels = np.load(out_dir / "labels.npy", allow_pickle=False)
        if sha256_array(probabilities) != manifest["arrays"]["probabilities"]["sha256"]:
            raise SystemExit(f"probability hash mismatch: {out_dir}")
        if sha256_array(row_index) != manifest["arrays"]["row_index"]["sha256"]:
            raise SystemExit(f"row_index hash mismatch: {out_dir}")
        if sha256_array(labels) != manifest["arrays"]["labels"]["sha256"]:
            raise SystemExit(f"labels hash mismatch: {out_dir}")
        if (manifest.get("freeze") or {}).get("sha256") != pin:
            raise SystemExit(f"FREEZE pin mismatch: {out_dir}")
        if manifest["checkpoint"]["sha256"] != record["checkpoint"]["sha256"]:
            raise SystemExit(f"checkpoint mismatch: {out_dir}")
        threshold_payload = read_json(ROOT / record["threshold"]["path"])
        if sha256_file(ROOT / record["threshold"]["path"]) != record["threshold"]["sha256"]:
            raise SystemExit(f"threshold mismatch: {record['run_name']}")
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        thresholds = np.asarray([threshold_payload["thresholds"][name] for name in names], dtype=np.float64)
        models.append({
            "task": record["task"],
            "label_set": record["label_set"],
            "modality": record["modality"],
            "run_name": record["run_name"],
            "seed": int(record["seed"]),
            "probabilities": probabilities,
            "labels": labels,
            "row_index": row_index,
            "thresholds": thresholds,
            "names": names,
            "manifest": manifest,
        })
    return models


def _load_scaffold_map(path: Path) -> dict[int, str]:
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scaffold = (row.get("murcko_scaffold") or "").strip() or "__MISSING__"
            mapping[int(row["row_index"])] = scaffold
    if set(mapping) != set(range(129817)):
        raise SystemExit("scaffold manifest row_index is not 0..129816")
    return mapping


def _clusters(row_index: np.ndarray, mapping: dict[int, str]):
    names = sorted({mapping[int(i)] for i in row_index})
    index = {name: position for position, name in enumerate(names)}
    cluster_id = np.asarray([index[mapping[int(i)]] for i in row_index], dtype=np.int64)
    sizes = np.bincount(cluster_id, minlength=len(names)).astype(np.int64)
    return names, cluster_id, sizes


def _f1(truth, probability, threshold, weight):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = (np.asarray(probability, dtype=np.float64) >= threshold).astype(np.int64)
    w = np.asarray(weight, dtype=np.float64)
    tp = float(w[(truth == 1) & (prediction == 1)].sum())
    fp = float(w[(truth == 0) & (prediction == 1)].sum())
    fn = float(w[(truth == 1) & (prediction == 0)].sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _rep_metrics(models: list[dict], names: list[str], weight: np.ndarray) -> dict:
    output = {}
    for model in models:
        probabilities = model["probabilities"]
        truth = model["labels"]
        arrays = {}
        for metric in ("ap", "auroc"):
            arrays[metric] = np.full(len(names), np.nan)
        arrays["f1_fixed_0.5"] = np.zeros(len(names))
        arrays["f1_validation_f1"] = np.zeros(len(names))
        for index in range(len(names)):
            column = truth[:, index].astype(np.int64)
            positive_weight = float(weight[column == 1].sum())
            negative_weight = float(weight[column == 0].sum())
            if positive_weight > 0 and negative_weight > 0:
                arrays["ap"][index] = float(average_precision_score(column, probabilities[:, index], sample_weight=weight))
                arrays["auroc"][index] = float(roc_auc_score(column, probabilities[:, index], sample_weight=weight))
            arrays["f1_fixed_0.5"][index] = _f1(column, probabilities[:, index], 0.5, weight)
            arrays["f1_validation_f1"][index] = _f1(column, probabilities[:, index], float(model["thresholds"][index]), weight)
        arrays["macro_ap"] = float(np.nanmean(arrays["ap"])) if np.isfinite(arrays["ap"]).any() else np.nan
        arrays["macro_auroc"] = float(np.nanmean(arrays["auroc"])) if np.isfinite(arrays["auroc"]).any() else np.nan
        arrays["macro_f1_fixed_0.5"] = float(np.mean(arrays["f1_fixed_0.5"]))
        arrays["macro_f1_validation_f1"] = float(np.mean(arrays["f1_validation_f1"]))
        output[(model["seed"], model["modality"])] = arrays
    return output


def _point_metrics(model: dict, names: list[str]) -> dict:
    weight = np.ones(TEST_N)
    return _rep_metrics([model], names, weight)[(model["seed"], model["modality"])]


def _diff_key(task: str, comparison: str, metric: str, policy: str, scope: str, label: str, suffix: str) -> str:
    return "|".join([task, comparison, metric, policy, scope, label, suffix])


def _comparison_diffs(task: str, per_model: dict, names: list[str], suffix_map: dict) -> dict:
    output = {}
    for base, other in COMPARISONS:
        comparison = COMPARISON_LABELS[(base, other)]
        for seed in SEEDS:
            a = per_model[(seed, base)]
            b = per_model[(seed, other)]
            for metric in ("ap", "auroc"):
                metric_name = metric.upper()
                for index, name in enumerate(names):
                    av, bv = a[metric][index], b[metric][index]
                    if np.isfinite(av) and np.isfinite(bv):
                        output[_diff_key(task, comparison, metric_name, "threshold_free", "label", name, suffix_map[seed])] = float(av - bv)
                valid = np.isfinite(a[metric]) & np.isfinite(b[metric])
                if valid.any():
                    output[_diff_key(task, comparison, metric_name, "threshold_free", "macro", "__macro__", suffix_map[seed])] = float(np.mean(a[metric][valid] - b[metric][valid]))
            for policy in POLICIES:
                key = f"f1_{policy}"
                for index, name in enumerate(names):
                    output[_diff_key(task, comparison, "F1", policy, "label", name, suffix_map[seed])] = float(a[key][index] - b[key][index])
                output[_diff_key(task, comparison, "F1", policy, "macro", "__macro__", suffix_map[seed])] = float(np.mean(a[key] - b[key]))
    # Three-seed summaries: mean of seed-level differences per replicate.
    for base, other in COMPARISONS:
        comparison = COMPARISON_LABELS[(base, other)]
        for metric in ("ap", "auroc"):
            metric_name = metric.upper()
            label_diffs = {}
            for name in names:
                values = []
                for seed in SEEDS:
                    a = per_model[(seed, base)][metric][names.index(name)]
                    b = per_model[(seed, other)][metric][names.index(name)]
                    values.append(float(a - b) if np.isfinite(a) and np.isfinite(b) else None)
                if all(value is not None for value in values):
                    label_diffs[name] = values
                    output[_diff_key(task, comparison, metric_name, "threshold_free", "label", name, "three_seed")] = float(np.mean(values))
            if label_diffs:
                output[_diff_key(task, comparison, metric_name, "threshold_free", "macro", "__macro__", "three_seed")] = float(
                    np.mean([np.mean(values) for values in label_diffs.values()])
                )
        for policy in POLICIES:
            key = f"f1_{policy}"
            for name in names:
                values = [per_model[(seed, base)][key][names.index(name)] - per_model[(seed, other)][key][names.index(name)] for seed in SEEDS]
                output[_diff_key(task, comparison, "F1", policy, "label", name, "three_seed")] = float(np.mean(values))
            macro_values = []
            for seed in SEEDS:
                a = per_model[(seed, base)][f"macro_{key}"]
                b = per_model[(seed, other)][f"macro_{key}"]
                macro_values.append(float(a - b))
            output[_diff_key(task, comparison, "F1", policy, "macro", "__macro__", "three_seed")] = float(np.mean(macro_values))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--scaffold-manifest", type=Path, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    analysis = args.analysis_dir.resolve()
    if not analysis.is_dir():
        raise SystemExit(f"Analysis directory missing: {analysis}")
    checks = Checks()
    freeze = _load_freeze(args.freeze, args.freeze_sha256)
    models = _load_models(freeze, args.freeze_sha256, args.predictions_root)
    checks.add("models.count_18", len(models) == 18, str(len(models)))

    summary = read_json(analysis / "summary.json")
    checks.add("summary.phase", summary.get("phase") == "B4.3")
    checks.add("summary.reps_2000", int(summary["config"]["reps"]) == 2000, str(summary["config"]["reps"]))
    checks.add("summary.rng_seed", int(summary["config"]["rng_seed"]) == 20261117)
    checks.add("summary.classification", summary["classification"]["status"] == "exploratory/post-hoc"
               and summary["classification"]["blind_test_claim"] is False)

    manifest = read_json(analysis / "analysis_manifest.json")
    checks.add("manifest.phase", manifest.get("phase") == "B4.3")
    checks.add("manifest.freeze_pin", manifest["freeze"]["sha256"] == args.freeze_sha256)
    bad_outputs = []
    for item in manifest["outputs"]:
        path = analysis / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            bad_outputs.append(item["path"])
    checks.add("manifest.outputs", not bad_outputs, ",".join(bad_outputs[:5]))
    checks.add("manifest.outputs_count", len(manifest["outputs"]) == 7, str(len(manifest["outputs"])))

    # Scaffolds and draws.
    mapping = _load_scaffold_map(args.scaffold_manifest)
    row_index = models[0]["row_index"]
    cluster_names, cluster_id, cluster_sizes = _clusters(row_index, mapping)
    checks.add("scaffolds.count", len(cluster_names) == int(summary["scaffolds"][0]["n_scaffolds"]))
    checks.add("scaffolds.singletons", int((cluster_sizes == 1).sum()) == int(summary["scaffolds"][0]["singleton_scaffolds"]))
    checks.add("scaffolds.sizes", int(cluster_sizes.min()) == int(summary["scaffolds"][0]["scaffold_size_min"])
               and int(cluster_sizes.max()) == int(summary["scaffolds"][0]["scaffold_size_max"]))
    reps = int(summary["config"]["reps"])
    sample_reps = sorted({0, reps - 1} | {min(rep, reps - 1) for rep in SAMPLE_REPS})
    rng = np.random.default_rng(int(summary["config"]["rng_seed"]))
    draws = np.empty((reps, len(cluster_names)), dtype=np.float32)
    for rep in range(reps):
        draws[rep] = np.bincount(rng.integers(0, len(cluster_names), size=len(cluster_names)), minlength=len(cluster_names)).astype(np.float32)
    digest = hashlib.sha256()
    digest.update(str(draws.dtype).encode("ascii")); digest.update(b"|")
    digest.update(str(draws.shape).encode("ascii")); digest.update(b"|")
    digest.update(np.ascontiguousarray(draws).tobytes())
    checks.add("draws.sha256", digest.hexdigest() == summary["config"]["draws_sha256"])

    # CSVs and replicate store.
    point_rows = _csv_rows(analysis / "point_comparisons.csv")
    ci_per_seed = _csv_rows(analysis / "bootstrap_ci_per_seed.csv")
    ci_three = _csv_rows(analysis / "bootstrap_ci_three_seed.csv")
    scaffold_rows = _csv_rows(analysis / "scaffold_summary.csv")
    store = np.load(analysis / "bootstrap_replicates.npz")
    checks.add("csv.point_rows", len(point_rows) == 1488, str(len(point_rows)))
    checks.add("csv.ci_per_seed_rows", len(ci_per_seed) == 1116, str(len(ci_per_seed)))
    checks.add("csv.ci_three_rows", len(ci_three) == 372, str(len(ci_three)))
    checks.add("npz.arrays", len(store.files) == 1488, str(len(store.files)))

    models_by_task = {task: [m for m in models if m["task"] == task] for task, _ in TASKS}

    # Independent point estimates and comparisons.
    point_lookup = {(row["task"], row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"], str(row["seed"])): row for row in point_rows}
    point_max_diff = 0.0
    for task, _label_set in TASKS:
        names = models_by_task[task][0]["names"]
        point = {seed: {} for seed in SEEDS}
        per_model = {}
        for model in models_by_task[task]:
            metrics = _point_metrics(model, names)
            per_model[(model["seed"], model["modality"])] = metrics
        diffs = _comparison_diffs(task, per_model, names, {seed: str(seed) for seed in SEEDS})
        for key, value in diffs.items():
            if key.endswith("|three_seed"):
                continue
            parts = key.split("|")
            row = point_lookup.get((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]))
            checks.add(f"point.present.{key}", row is not None)
            if row is not None:
                point_max_diff = max(point_max_diff, abs(float(row["point_estimate"]) - value))
        # Three-seed point rows.
        for row in ci_three:
            if row["task"] != task:
                continue
            suffix = "three_seed"
            key = _diff_key(task, row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"], suffix)
            if key in diffs:
                point_max_diff = max(point_max_diff, abs(float(row["point_estimate"]) - diffs[key]))
    checks.add("point.max_abs_diff", point_max_diff <= 1e-9, f"{point_max_diff:.3e}")

    # B4.2 consistency.
    overall = _csv_rows(B42 / "overall_metrics.csv")
    per_label = _csv_rows(B42 / "per_label_metrics.csv")
    b42_max = 0.0
    for task, _label_set in TASKS:
        names = models_by_task[task][0]["names"]
        for model in models_by_task[task]:
            metrics = _point_metrics(model, names)
            rows = [row for row in overall if row["task"] == task and row["modality"] == model["modality"] and int(row["seed"]) == model["seed"]]
            for policy in POLICIES:
                row = next(r for r in rows if r["policy"] == policy)
                b42_max = max(b42_max,
                              abs(metrics["macro_ap"] - float(row["macro_ap"])),
                              abs(metrics["macro_auroc"] - float(row["macro_auroc"])),
                              abs(metrics[f"macro_f1_{policy}"] - float(row["macro_f1"])))
            for index, name in enumerate(names):
                refs = [row for row in per_label if row["task"] == task and row["modality"] == model["modality"]
                        and int(row["seed"]) == model["seed"] and row["label"] == name]
                fixed = next(r for r in refs if r["policy"] == "fixed_0.5")
                validated = next(r for r in refs if r["policy"] == "validation_f1")
                b42_max = max(b42_max,
                              abs(metrics["ap"][index] - float(fixed["average_precision"])),
                              abs(metrics["auroc"][index] - float(fixed["auroc"])),
                              abs(metrics["f1_fixed_0.5"][index] - float(fixed["f1"])),
                              abs(metrics["f1_validation_f1"][index] - float(validated["f1"])))
    checks.add("b42.point_consistency", b42_max <= 1e-6, f"{b42_max:.3e}")

    # Sample replicate recomputation.
    sample_max_diff = 0.0
    for task, _label_set in TASKS:
        names = models_by_task[task][0]["names"]
        for rep in sample_reps:
            weight = draws[rep][cluster_id].astype(np.float64)
            per_model = _rep_metrics(models_by_task[task], names, weight)
            diffs = _comparison_diffs(task, per_model, names, {seed: str(seed) for seed in SEEDS})
            for key, value in diffs.items():
                if key not in store.files:
                    checks.add(f"rep.key.{key}", False, "missing npz key")
                    continue
                stored = float(store[key][rep])
                if np.isfinite(stored) and np.isfinite(value):
                    sample_max_diff = max(sample_max_diff, abs(stored - value))
                elif np.isnan(stored) != np.isnan(value):
                    checks.add(f"rep.nan.{key}.{rep}", False, f"stored={stored} computed={value}")
            # Three-seed keys are present in diffs; compare too.
            for key, value in diffs.items():
                if key.endswith("|three_seed") and key in store.files:
                    stored = float(store[key][rep])
                    if np.isfinite(stored) and np.isfinite(value):
                        sample_max_diff = max(sample_max_diff, abs(stored - value))
    checks.add("replicates.sample_max_abs_diff", sample_max_diff <= 1e-9, f"{sample_max_diff:.3e}")

    # CI recomputation from stored arrays.
    ci_max_diff = 0.0
    undefined_ok = True
    for row in ci_per_seed:
        suffix = str(int(row["seed"]))
        key = _diff_key(row["task"], row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"], suffix)
        arr = store[key]
        valid = arr[np.isfinite(arr)]
        if len(valid) >= 2:
            low, high = np.percentile(valid, CI)
            ci_max_diff = max(ci_max_diff, abs(float(row["ci_low"]) - float(low)), abs(float(row["ci_high"]) - float(high)))
        else:
            checks.add(f"ci.per_seed.undefined.{key}", row["ci_low"] == "" and row["ci_high"] == "")
        n_valid = int(row["n_valid_reps"])
        if row["metric"] != "F1":
            n_undefined = float(row["n_undefined_ap"] or 0) if row["metric"] == "AP" else float(row["n_undefined_auroc"] or 0)
            undefined_ok = undefined_ok and n_valid == len(valid) and n_undefined + len(valid) == reps
        else:
            undefined_ok = undefined_ok and n_valid == len(valid)
    for row in ci_three:
        key = _diff_key(row["task"], row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"], "three_seed")
        arr = store[key]
        valid = arr[np.isfinite(arr)]
        if len(valid) >= 2:
            low, high = np.percentile(valid, CI)
            ci_max_diff = max(ci_max_diff, abs(float(row["ci_low"]) - float(low)), abs(float(row["ci_high"]) - float(high)))
        n_valid = int(row["n_valid_reps"])
        if row["metric"] != "F1":
            n_undefined = float(row["n_undefined_ap"] or 0) if row["metric"] == "AP" else float(row["n_undefined_auroc"] or 0)
            undefined_ok = undefined_ok and n_valid == len(valid) and n_undefined + len(valid) == reps
        else:
            undefined_ok = undefined_ok and n_valid == len(valid)
    checks.add("ci.max_abs_diff", ci_max_diff <= 1e-9, f"{ci_max_diff:.3e}")
    checks.add("ci.valid_counts", undefined_ok)

    # Seed SD (ddof=1) from three seed point rows.
    sd_max_diff = 0.0
    seed_point = {}
    for row in point_rows:
        if row["seed"] == "three_seed_mean":
            continue
        key = (row["task"], row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"])
        seed_point.setdefault(key, {})[int(row["seed"])] = _float(row["point_estimate"])
    for row in ci_three:
        key = (row["task"], row["comparison"], row["metric"], row["threshold_policy"], row["scope"], row["label"])
        values = seed_point.get(key, {})
        if len(values) == 3 and all(value is not None for value in values.values()):
            expected = statistics.stdev([values[seed] for seed in SEEDS])
            sd_max_diff = max(sd_max_diff, abs(float(row["seed_sd"]) - expected))
    checks.add("seed_sd.ddof1", sd_max_diff <= 1e-9, f"{sd_max_diff:.3e}")

    # Frozen assets unchanged.
    checks.add("frozen.freeze_sha", sha256_file(args.freeze) == args.freeze_sha256)
    predictions_count = sum(1 for _ in args.predictions_root.rglob("*") if _.is_file())
    checks.add("frozen.prediction_files_72", predictions_count == 72, str(predictions_count))
    b42_manifest = read_json(B42 / "analysis_manifest.json")
    b42_bad = []
    for item in b42_manifest["outputs"]:
        path = B42 / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            b42_bad.append(item["path"])
    checks.add("frozen.b42_outputs", not b42_bad, ",".join(b42_bad[:5]))
    checks.add("frozen.b42_verification", read_json(B42 / "verification.json").get("status") == "PASS")
    log_path = ROOT / "runs" / "LINE-B-MANIFESTS" / "test_access_log.json"
    checks.add("frozen.test_access_log", sha256_file(log_path) == manifest["inputs"]["test_access_log"]["sha256"])

    failures = checks.failed()
    status = "PASS" if not failures else "FAILED"
    report = {
        "phase": "B4.3-verification",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "analysis_manifest": {"path": relative_to_root(analysis / "analysis_manifest.json"), "sha256": sha256_file(analysis / "analysis_manifest.json")},
        "freeze": {"path": relative_to_root(args.freeze), "sha256": args.freeze_sha256},
        "summary": {
            "n_checks": len(checks.items),
            "n_failed": len(failures),
            "reps_verified": reps,
            "sample_reps": sample_reps,
            "point_max_abs_diff": point_max_diff,
            "b42_max_abs_diff": b42_max,
            "sample_rep_max_abs_diff": sample_max_diff,
            "ci_max_abs_diff": ci_max_diff,
            "seed_sd_max_abs_diff": sd_max_diff,
            "n_scaffolds": len(cluster_names),
        },
        "checks": checks.items,
        "failures": failures,
    }
    out_path = args.json_out or (analysis / "verification.json")
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing verification: {out_path}")
    write_json(out_path, report)
    print(json.dumps({
        "status": status,
        "n_checks": len(checks.items),
        "n_failed": len(failures),
        "failed_ids": [item["id"] for item in failures],
        "point_max_diff": point_max_diff,
        "b42_max_diff": b42_max,
        "sample_rep_max_diff": sample_max_diff,
        "ci_max_diff": ci_max_diff,
        "seed_sd_max_diff": sd_max_diff,
        "verification": relative_to_root(out_path),
    }, ensure_ascii=False, indent=1))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
