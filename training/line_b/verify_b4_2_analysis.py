"""Independent acceptance verification for B4.2 Test performance analysis.

Recomputes every metric from the frozen Test probability arrays with an
independent implementation (count-based P/R/F1 + scikit-learn AP/AUROC),
compares it with the B4.2 CSVs, checks seed aggregation/threshold provenance,
and confirms that no frozen asset changed.

Writes ``verification.json`` inside the analysis directory.
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
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
POLICIES = ("fixed_0.5", "validation_f1")
OVERALL_METRICS = (
    "micro_precision", "micro_recall", "micro_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "macro_ap", "macro_auroc",
)
TOLERANCE = 1e-7
TEST_N = 12981
DEFAULT_ANALYSIS = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
DEFAULT_PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check_id: str, ok: bool, detail: str = "") -> None:
        self.items.append({"id": check_id, "ok": bool(ok), "detail": detail})

    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["ok"]]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _independent_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
    y_pred = (y_prob >= thresholds).astype(np.int64)
    per_label = []
    ap_values, auroc_values = [], []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        prob = y_prob[:, index]
        pred = y_pred[:, index]
        tp = int(np.sum((truth == 1) & (pred == 1)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        unique = np.unique(truth)
        ap = float(average_precision_score(truth, prob)) if unique.size == 2 else None
        auroc = float(roc_auc_score(truth, prob)) if unique.size == 2 else None
        if ap is not None:
            ap_values.append(ap)
        if auroc is not None:
            auroc_values.append(auroc)
        per_label.append({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "average_precision": ap, "auroc": auroc,
            "support": int(truth.sum()),
        })
    tp_total = sum(row["tp"] for row in per_label)
    fp_total = sum(row["fp"] for row in per_label)
    fn_total = sum(row["fn"] for row in per_label)
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean([row["precision"] for row in per_label])),
        "macro_recall": float(np.mean([row["recall"] for row in per_label])),
        "macro_f1": float(np.mean([row["f1"] for row in per_label])),
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "per_label": per_label,
    }


def _tree_digest(root: Path, names: set[str]) -> str:
    digest = hashlib.sha256()
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(root))
            if relative not in names:
                files[relative] = sha256_file(path)
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relative].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.resolve()
    if not analysis_dir.is_dir():
        raise SystemExit(f"Analysis directory not found: {analysis_dir}")
    checks = Checks()

    if sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit("FREEZE SHA256 pin mismatch")
    freeze = read_json(args.freeze)
    protocol_path = ROOT / freeze["protocol_path"]
    checks.add("freeze.status", freeze.get("status") == "FROZEN" and freeze.get("phase") == "B3")
    checks.add("freeze.protocol_sha", sha256_file(protocol_path) == freeze["protocol_sha256"])

    manifest_path = analysis_dir / "analysis_manifest.json"
    checks.add("manifest.exists", manifest_path.is_file())
    if not manifest_path.is_file():
        raise SystemExit("analysis_manifest.json missing")
    manifest = read_json(manifest_path)
    checks.add("manifest.phase", manifest.get("phase") == "B4.2")
    checks.add("manifest.freeze_pin", (manifest.get("freeze") or {}).get("sha256") == args.freeze_sha256)
    checks.add("manifest.classification",
               (manifest.get("classification") or {}).get("status") == "exploratory/post-hoc"
               and (manifest.get("classification") or {}).get("blind_test_claim") is False)
    outputs = manifest.get("outputs") or []
    checks.add("manifest.outputs_count", len(outputs) == 15, str(len(outputs)))
    missing, mismatched = [], []
    for output in outputs:
        path = analysis_dir / output["path"]
        if not path.is_file():
            missing.append(output["path"])
        elif sha256_file(path) != output["sha256"] or path.stat().st_size != output["size_bytes"]:
            mismatched.append(output["path"])
    checks.add("manifest.outputs_exist", not missing, ",".join(missing[:5]))
    checks.add("manifest.outputs_sha", not mismatched, ",".join(mismatched[:5]))

    # Load frozen Test predictions and thresholds.
    labels_by_task = {
        task: list(freeze["labels_and_data"]["label_sets"][label_set]["label_names"]["names"])
        for task, label_set in TASKS
    }
    models: dict[tuple[str, str], dict] = {}
    row_indexes = []
    for record in freeze["models"]:
        task = record["task"]
        label_set = record["label_set"]
        run_name = record["run_name"]
        out_dir = args.predictions_root / task / MODALITY_DIRS[record["modality"]] / run_name / "test"
        test_manifest = read_json(out_dir / "prediction_manifest.json")
        checks.add(f"prediction.{task}.{run_name}.freeze_pin",
                   (test_manifest.get("freeze") or {}).get("sha256") == args.freeze_sha256)
        probabilities = np.load(out_dir / "probabilities.npy", allow_pickle=False)
        row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
        labels = np.load(out_dir / "labels.npy", allow_pickle=False)
        checks.add(f"prediction.{task}.{run_name}.array_sha",
                   sha256_array(probabilities) == test_manifest["arrays"]["probabilities"]["sha256"]
                   and sha256_array(row_index) == test_manifest["arrays"]["row_index"]["sha256"]
                   and sha256_array(labels) == test_manifest["arrays"]["labels"]["sha256"])
        checks.add(f"prediction.{task}.{run_name}.checkpoint",
                   test_manifest["checkpoint"]["sha256"] == record["checkpoint"]["sha256"])
        with np.load(ROOT / record["split_file"]["path"], allow_pickle=False) as archive:
            test_indices = np.asarray(archive["test"], dtype=np.int64)
        checks.add(f"prediction.{task}.{run_name}.row_index",
                   np.array_equal(row_index, test_indices) and len(np.unique(row_index)) == TEST_N)
        checks.add(f"prediction.{task}.{run_name}.label_order",
                   test_manifest.get("label_names") == labels_by_task[task])
        threshold_payload = read_json(ROOT / record["threshold"]["path"])
        checks.add(f"prediction.{task}.{run_name}.threshold_source",
                   sha256_file(ROOT / record["threshold"]["path"]) == record["threshold"]["sha256"]
                   and threshold_payload.get("label_names") == labels_by_task[task]
                   and threshold_payload.get("test_split_accessed") is False)
        thresholds = np.asarray([threshold_payload["thresholds"][name] for name in labels_by_task[task]], dtype=np.float64)
        validation_labels = np.load(ROOT / record["labels"]["path"], allow_pickle=False)
        models[(task, run_name)] = {
            "record": record,
            "probabilities": probabilities,
            "labels": labels,
            "validation_labels": validation_labels,
            "row_index": row_index,
            "thresholds": thresholds,
        }
        row_indexes.append(row_index)
    checks.add("predictions.models_18", len(models) == 18, str(len(models)))
    checks.add("predictions.row_index_consistent",
               all(np.array_equal(row_indexes[0], other) for other in row_indexes[1:]))

    # Recompute every metric and compare to CSV.
    overall_csv = _read_csv(analysis_dir / "overall_metrics.csv")
    per_label_csv = _read_csv(analysis_dir / "per_label_metrics.csv")
    checks.add("csv.overall_rows_36", len(overall_csv) == 36, str(len(overall_csv)))
    checks.add("csv.per_label_rows_522", len(per_label_csv) == 522, str(len(per_label_csv)))

    max_overall_diff = 0.0
    max_label_diff = 0.0
    overall_by_key = {(row["task"], row["policy"], row["run_name"]): row for row in overall_csv}
    per_label_by_key = {(row["task"], row["policy"], row["run_name"], row["label"]): row for row in per_label_csv}
    for (task, run_name), model in models.items():
        for policy in POLICIES:
            thresholds = np.full(len(labels_by_task[task]), 0.5) if policy == "fixed_0.5" else model["thresholds"]
            expected = _independent_metrics(model["labels"], model["probabilities"], thresholds)
            row = overall_by_key.get((task, policy, run_name))
            checks.add(f"metrics.overall_present.{task}.{run_name}.{policy}", row is not None)
            if row is None:
                continue
            for metric in OVERALL_METRICS:
                actual = float(row[metric])
                difference = abs(actual - expected[metric])
                max_overall_diff = max(max_overall_diff, difference)
                checks.add(f"metrics.overall.{task}.{run_name}.{policy}.{metric}", difference <= TOLERANCE,
                           f"diff={difference:.3e}")
            for index, label in enumerate(labels_by_task[task]):
                label_row = per_label_by_key.get((task, policy, run_name, label))
                checks.add(f"metrics.label_present.{task}.{policy}.{run_name}.{label}", label_row is not None)
                if label_row is None:
                    continue
                expected_label = expected["per_label"][index]
                for metric in ("precision", "recall", "f1"):
                    difference = abs(float(label_row[metric]) - expected_label[metric])
                    max_label_diff = max(max_label_diff, difference)
                    checks.add(f"metrics.label.{task}.{policy}.{run_name}.{label}.{metric}", difference <= TOLERANCE,
                               f"diff={difference:.3e}")
                for metric in ("average_precision", "auroc"):
                    expected_value = expected_label[metric]
                    if expected_value is None:
                        checks.add(f"metrics.label.{task}.{policy}.{run_name}.{label}.{metric}", label_row[metric] == "")
                    else:
                        difference = abs(float(label_row[metric]) - expected_value)
                        max_label_diff = max(max_label_diff, difference)
                        checks.add(f"metrics.label.{task}.{policy}.{run_name}.{label}.{metric}", difference <= TOLERANCE,
                                   f"diff={difference:.3e}")
                for count_metric in ("tp", "fp", "fn", "tn", "support"):
                    checks.add(f"metrics.count.{task}.{policy}.{run_name}.{label}.{count_metric}",
                               int(label_row[count_metric]) == int(expected_label[count_metric]))
                checks.add(f"metrics.test_positives.{task}.{policy}.{run_name}.{label}",
                           int(label_row["test_positives"]) == int(model["labels"][:, index].sum()))
                checks.add(f"metrics.val_positives.{task}.{policy}.{run_name}.{label}",
                           int(label_row["val_positives"]) == int(model["validation_labels"][:, index].sum()))
                expected_threshold = 0.5 if policy == "fixed_0.5" else float(model["thresholds"][index])
                checks.add(f"metrics.threshold.{task}.{policy}.{run_name}.{label}",
                           abs(float(label_row["threshold"]) - expected_threshold) <= TOLERANCE)

    # Seed aggregation.
    seed_csv = _read_csv(analysis_dir / "seed_comparison.csv")
    mean_sd_csv = _read_csv(analysis_dir / "overall_mean_sd.csv")
    modality_csv = _read_csv(analysis_dir / "modality_comparison.csv")
    overall_values = {
        (row["task"], row["policy"], row["run_name"], metric): float(row[metric])
        for row in overall_csv for metric in OVERALL_METRICS
    }
    seed_map: dict[tuple, list[float]] = {}
    for row in seed_csv:
        key = (row["task"], row["policy"], row["modality"], row["metric"])
        seed_map.setdefault(key, []).append(float(row["value"]))
    checks.add("seeds.groups", len(seed_map) == 2 * 2 * 3 * len(OVERALL_METRICS), str(len(seed_map)))
    for key, values in seed_map.items():
        checks.add(f"seeds.three.{key}", len(values) == 3, str(len(values)))
    for row in mean_sd_csv:
        key = (row["task"], row["policy"], row["modality"], row["metric"])
        values = seed_map.get(key, [])
        if len(values) != 3:
            checks.add(f"mean_sd.values.{key}", False, str(len(values)))
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        checks.add(f"mean_sd.{key}", abs(float(row["mean"]) - mean) <= TOLERANCE
                   and abs(float(row["sd"]) - sd) <= TOLERANCE)
    for row in seed_csv:
        key = (row["task"], row["policy"], row["modality"], row["metric"], row["seed"])
        run_rows = [r for r in overall_csv if r["task"] == row["task"] and r["policy"] == row["policy"] and r["modality"] == row["modality"] and r["seed"] == row["seed"]]
        expected = float(run_rows[0][row["metric"]]) if run_rows else None
        checks.add(f"seeds.value.{key}", expected is not None and abs(float(row["value"]) - expected) <= TOLERANCE)

    # Modality ranking.
    for task, _label_set in TASKS:
        for policy in POLICIES:
            for metric in OVERALL_METRICS:
                rows = [r for r in mean_sd_csv if r["task"] == task and r["policy"] == policy and r["metric"] == metric]
                ordered = sorted(rows, key=lambda r: float(r["mean"]), reverse=True)
                for rank, row in enumerate(ordered, start=1):
                    match = [r for r in modality_csv if r["task"] == task and r["policy"] == policy
                             and r["metric"] == metric and r["modality"] == row["modality"]]
                    checks.add(f"modality.rank.{task}.{policy}.{metric}.{row['modality']}",
                               bool(match) and int(match[0]["rank"]) == rank
                               and abs(float(match[0]["delta_vs_best"]) - (float(row["mean"]) - float(ordered[0]["mean"]))) <= TOLERANCE)

    # Per-label comparison completeness.
    comparison_csv = _read_csv(analysis_dir / "per_label_modality_comparison.csv")
    checks.add("label_comparison.rows_29", len(comparison_csv) == 29, str(len(comparison_csv)))
    for row in comparison_csv:
        checks.add(f"label_comparison.{row['task']}.{row['label']}",
                   row["best_ap_modality"] in MODALITIES and row["best_f1_modality"] in MODALITIES)

    # Figures and report.
    figures = sorted((analysis_dir / "figures").glob("*.png"))
    checks.add("figures.count_7", len(figures) == 7, str(len(figures)))
    for figure in figures:
        try:
            with Image.open(figure) as image:
                width, height = image.size
            checks.add(f"figures.valid.{figure.name}", width >= 400 and height >= 300 and figure.stat().st_size > 10000,
                       f"{width}x{height}")
        except Exception as exc:  # pragma: no cover - defensive
            checks.add(f"figures.valid.{figure.name}", False, str(exc))
    report = (analysis_dir / "report.md").read_text(encoding="utf-8")
    checks.add("report.classification",
               "exploratory" in report and "post-hoc" in report and "blind" in report)

    # No Bootstrap products.
    checks.add("no_bootstrap_output",
               not any("bootstrap" in output["path"].lower() for output in outputs))

    # Frozen assets unchanged.
    checks.add("frozen.freeze_sha", sha256_file(args.freeze) == args.freeze_sha256)
    b3_digest = _tree_digest(
        ROOT / "runs/LINE-B-MANIFESTS/B3",
        {"code_provenance.json", "b3_5b_manifest_verification.json", "b3_5b_manifest_verification_v2.json"},
    )
    checks.add("frozen.b3_root",
               b3_digest == "c099ae978071e14886b6f2e66997ee93eb80f6e4ecbfd91f0383df89ecb96a44",
               b3_digest)
    frozen_ok = True
    for record in freeze["models"]:
        validation_manifest = ROOT / record["prediction_manifest"]["path"]
        checkpoint = ROOT / record["checkpoint"]["path"]
        threshold = ROOT / record["threshold"]["path"]
        calibration = ROOT / record["calibration_manifest"]["path"]
        if sha256_file(validation_manifest) != record["prediction_manifest"]["sha256"]:
            frozen_ok = False
        if sha256_file(checkpoint) != record["checkpoint"]["sha256"]:
            frozen_ok = False
        if sha256_file(threshold) != record["threshold"]["sha256"]:
            frozen_ok = False
        if sha256_file(calibration) != record["calibration_manifest"]["sha256"]:
            frozen_ok = False
        for key in ("probabilities", "row_index", "labels"):
            if sha256_array(np.load(validation_manifest.parent / f"{key}.npy", allow_pickle=False)) != record[key]["array_sha256"]:
                frozen_ok = False
    checks.add("frozen.assets", frozen_ok)
    access_log = ROOT / "runs/LINE-B-MANIFESTS/test_access_log.json"
    checks.add("frozen.test_access_log",
               (manifest["inputs"]["test_access_log"] or {}).get("sha256") == sha256_file(access_log))

    failures = checks.failed()
    status = "PASS" if not failures else "FAILED"
    report_payload = {
        "phase": "B4.2-verification",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "analysis_manifest": {
            "path": relative_to_root(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "freeze": {"path": relative_to_root(args.freeze), "sha256": args.freeze_sha256},
        "summary": {
            "n_checks": len(checks.items),
            "n_failed": len(failures),
            "n_models": len(models),
            "max_overall_metric_diff": max_overall_diff,
            "max_per_label_metric_diff": max_label_diff,
            "n_figures": len(figures),
        },
        "checks": checks.items,
        "failures": failures,
    }
    out_path = args.json_out or (analysis_dir / "verification.json")
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing verification: {out_path}")
    write_json(out_path, report_payload)
    print(json.dumps({
        "status": status,
        "n_checks": len(checks.items),
        "n_failed": len(failures),
        "failed_ids": [item["id"] for item in failures],
        "max_overall_diff": max_overall_diff,
        "max_per_label_diff": max_label_diff,
        "verification": relative_to_root(out_path),
    }, ensure_ascii=False, indent=1))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
