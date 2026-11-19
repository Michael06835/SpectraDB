"""Line B Validation metric aggregation for Shared14 and UVSPEC15.

Reads only the exported validation prediction directories.  No model is
loaded, no threshold is fitted, and no Test array is read.
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
    ROOT as PROJECT_ROOT,
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
from training.line_b.metrics import compute_multilabel_metrics


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _round(value, digits: int = 8):
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [float(x) for x in values if x is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def _seed_from_run_name(run_name: str):
    if run_name.endswith("-SEED123"):
        return 123
    if run_name.endswith("-SEED2026"):
        return 2026
    if run_name.endswith("-FULL"):
        return 42
    return None


def discover_predictions(predictions_root: Path, label_set: str = "shared14") -> list[dict]:
    root = Path(predictions_root)
    label_config = resolve_label_set(label_set)
    expected_task = label_config["task"]
    expected_n_labels = int(label_config["n_labels"])
    expected_names = load_label_names(label_config["names"])
    manifests = sorted(root.glob("*/*/validation/prediction_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No validation prediction manifests under {root}")

    records: list[dict] = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        if manifest.get("task") != expected_task:
            raise ValueError(f"Unexpected task in {manifest_path}: {manifest.get('task')}")
        if manifest.get("label_set") != label_set:
            raise ValueError(f"Unexpected label set in {manifest_path}: {manifest.get('label_set')}")
        if manifest.get("split") != "validation":
            raise ValueError(f"Unexpected split in {manifest_path}")
        if len(manifest.get("label_names") or []) != expected_n_labels:
            raise ValueError(f"Unexpected label count in {manifest_path}")
        validate_manifest_checks(manifest, str(manifest_path))

        pred_dir = manifest_path.parent
        run_name = manifest["run_name"]
        if run_name != pred_dir.parent.name:
            raise ValueError(f"Manifest run_name does not match directory: {manifest_path}")
        expected_seed = expected_seed_from_run_name(run_name)
        seed_config = (manifest.get("checkpoint", {}).get("config") or {}).get("seed")
        if seed_config is None:
            raise ValueError(f"Missing checkpoint seed in {manifest_path}")
        if int(seed_config) != expected_seed:
            raise ValueError(
                f"Checkpoint seed {seed_config} != expected {expected_seed} in {manifest_path}"
            )

        probabilities, row_index, labels = validate_prediction_arrays(
            pred_dir, manifest, manifest["modality"], label_set, expected_names
        )
        if row_index.shape != (12981,):
            raise ValueError(f"Unexpected row_index shape in {pred_dir}: {row_index.shape}")

        seed_metadata_mismatch = False
        records.append(
            {
                "manifest": manifest,
                "manifest_path": manifest_path,
                "pred_dir": pred_dir,
                "probabilities": probabilities,
                "row_index": row_index,
                "labels": labels,
                "modality": manifest["modality"],
                "run_name": run_name,
                "seed": expected_seed,
                "seed_config": int(seed_config),
                "seed_metadata_mismatch": seed_metadata_mismatch,
            }
        )

    reference = records[0]
    for record in records[1:]:
        if not np.array_equal(record["row_index"], reference["row_index"]):
            raise ValueError(f"row_index mismatch: {record['pred_dir']}")
        if not np.array_equal(record["labels"], reference["labels"]):
            raise ValueError(f"labels mismatch: {record['pred_dir']}")

    return records


def evaluate(
    predictions_root: Path,
    label_names_path: Path,
    threshold: float,
    require_full_matrix: bool = False,
    label_set: str = "shared14",
) -> dict:
    label_config = resolve_label_set(label_set)
    labels_names = load_label_names(label_names_path)
    if len(labels_names) != label_config["n_labels"]:
        raise ValueError(
            f"Expected {label_config['n_labels']} labels for {label_set}, "
            f"got {len(labels_names)}"
        )
    records = discover_predictions(predictions_root, label_set=label_set)

    if require_full_matrix:
        expected_runs = resolve_task_runs(label_set)
        if len(records) != 9:
            raise ValueError(f"Expected exactly 9 records, found {len(records)}")
        run_names = [record["run_name"] for record in records]
        if len(set(run_names)) != 9:
            raise ValueError(f"Duplicate run records detected: {run_names}")
        expected_by_modality = {
            modality: set(expected_runs[modality])
            for modality in expected_runs
        }
        actual_by_modality = defaultdict(set)
        actual_counts = defaultdict(int)
        for record in records:
            actual_by_modality[record["modality"]].add(record["run_name"])
            actual_counts[record["modality"]] += 1
        if dict(actual_by_modality) != expected_by_modality:
            raise ValueError(
                f"Registered {label_set} run matrix mismatch: "
                f"actual={dict(actual_by_modality)}, expected={expected_by_modality}"
            )
        if dict(actual_counts) != {"ir": 3, "raman": 3, "uvvis": 3}:
            raise ValueError(f"Unexpected per-modality record counts: {dict(actual_counts)}")
        for modality, run_names in expected_runs.items():
            seeds = sorted(
                _seed_from_run_name(run_name)
                for run_name in run_names
            )
            if seeds != [42, 123, 2026]:
                raise ValueError(f"Unexpected seed matrix for {label_set}/{modality}: {seeds}")
            actual_seeds = sorted(
                record["seed_config"]
                for record in records
                if record["modality"] == modality
            )
            if actual_seeds != [42, 123, 2026]:
                raise ValueError(
                    f"Actual seed matrix mismatch for {label_set}/{modality}: {actual_seeds}"
                )

    per_model_rows: list[dict] = []
    per_label_rows: list[dict] = []
    fp16_rows: list[dict] = []

    for record in records:
        manifest = record["manifest"]
        if manifest["label_names"] != labels_names:
            raise ValueError(f"Label order mismatch: {record['pred_dir']}")

        result = compute_multilabel_metrics(
            record["labels"],
            record["probabilities"],
            labels_names,
            threshold=threshold,
        )
        seed = record["seed"]
        base = {
            "run_name": record["run_name"],
            "modality": record["modality"],
            "seed": seed,
            "seed_config": record["seed_config"],
            "seed_metadata_mismatch": record["seed_metadata_mismatch"],
        }
        per_model_rows.append(
            {
                **base,
                "n_samples": result["n_samples"],
                "n_labels": result["n_labels"],
                "threshold": result["threshold"],
                "micro_precision": _round(result["micro_precision"]),
                "micro_recall": _round(result["micro_recall"]),
                "micro_f1": _round(result["micro_f1"]),
                "macro_precision": _round(result["macro_precision"]),
                "macro_recall": _round(result["macro_recall"]),
                "macro_f1": _round(result["macro_f1"]),
                "macro_ap": _round(result["macro_ap"]),
                "macro_auroc": _round(result["macro_auroc"]),
                "n_defined_ap": result["n_defined_ap"],
                "n_defined_auroc": result["n_defined_auroc"],
                "undefined_ap_labels": ";".join(result["undefined_ap_labels"]),
                "undefined_auroc_labels": ";".join(result["undefined_auroc_labels"]),
            }
        )

        for row in result["per_label"]:
            per_label_rows.append(
                {
                    **base,
                    "label_index": row["label_index"],
                    "label": row["label"],
                    "support": row["support"],
                    "prevalence": _round(row["prevalence"]),
                    "precision": _round(row["precision"]),
                    "recall": _round(row["recall"]),
                    "f1": _round(row["f1"]),
                    "average_precision": _round(row["average_precision"]),
                    "auroc": _round(row["auroc"]),
                    "ap_defined": row["ap_defined"],
                    "auroc_defined": row["auroc_defined"],
                }
            )

        historical = manifest.get("checkpoint", {}).get("val_metrics") or {}
        mapping = {
            "micro_f1": result["micro_f1"],
            "macro_f1": result["macro_f1"],
            "mAP": result["macro_ap"],
            "macro_AUROC": result["macro_auroc"],
        }
        for metric, fp32_value in mapping.items():
            if metric not in historical:
                continue
            fp16_value = float(historical[metric])
            fp32_value = float(fp32_value) if fp32_value is not None else None
            fp16_rows.append(
                {
                    **base,
                    "metric": metric,
                    "fp16_value": _round(fp16_value),
                    "fp32_value": _round(fp32_value),
                    "delta": _round(fp32_value - fp16_value) if fp32_value is not None else "",
                    "abs_delta": _round(abs(fp32_value - fp16_value)) if fp32_value is not None else "",
                }
            )

    # Per-modality aggregate of model-level metrics.
    by_modality = defaultdict(list)
    for row in per_model_rows:
        by_modality[row["modality"]].append(row)

    modality_rows: list[dict] = []
    metric_names = [
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_ap",
        "macro_auroc",
    ]
    for modality, rows in sorted(by_modality.items()):
        for metric in metric_names:
            values = [row[metric] for row in rows if row[metric] != ""]
            mean, sd = _mean_sd(values)
            modality_rows.append(
                {
                    "modality": modality,
                    "metric": metric,
                    "mean": _round(mean),
                    "sd": _round(sd),
                    "n_seeds": len(values),
                }
            )

    # Per-label mean/SD across seeds within each modality.
    label_groups = defaultdict(list)
    for row in per_label_rows:
        label_groups[(row["modality"], row["label_index"], row["label"])].append(row)

    per_label_summary: list[dict] = []
    for (modality, label_index, label), rows in sorted(label_groups.items()):
        support_mean, _ = _mean_sd([row["support"] for row in rows])
        prevalence_mean, _ = _mean_sd([row["prevalence"] for row in rows])
        ap_mean, ap_sd = _mean_sd([row["average_precision"] for row in rows])
        auroc_mean, auroc_sd = _mean_sd([row["auroc"] for row in rows])
        f1_mean, f1_sd = _mean_sd([row["f1"] for row in rows])
        precision_mean, precision_sd = _mean_sd([row["precision"] for row in rows])
        recall_mean, recall_sd = _mean_sd([row["recall"] for row in rows])
        per_label_summary.append(
            {
                "modality": modality,
                "label_index": label_index,
                "label": label,
                "support_mean": _round(support_mean),
                "prevalence_mean": _round(prevalence_mean),
                "average_precision_mean": _round(ap_mean),
                "average_precision_sd": _round(ap_sd),
                "auroc_mean": _round(auroc_mean),
                "auroc_sd": _round(auroc_sd),
                "f1_mean": _round(f1_mean),
                "f1_sd": _round(f1_sd),
                "precision_mean": _round(precision_mean),
                "precision_sd": _round(precision_sd),
                "recall_mean": _round(recall_mean),
                "recall_sd": _round(recall_sd),
                "n_seeds": len(rows),
            }
        )

    return {
        "records": records,
        "per_model_rows": per_model_rows,
        "per_label_rows": per_label_rows,
        "modality_rows": modality_rows,
        "per_label_summary": per_label_summary,
        "fp16_rows": fp16_rows,
        "label_names": labels_names,
    }


def _build_summary(
    result: dict,
    predictions_root: Path,
    label_names_path: Path,
    threshold: float,
    label_set: str,
    phase: str,
) -> dict:
    records = result["records"]
    label_config = resolve_label_set(label_set)
    return {
        "task": label_config["task"],
        "label_set": label_set,
        "phase": phase,
        "split": "validation",
        "threshold": float(threshold),
        "n_models": len(records),
        "n_labels": int(label_config["n_labels"]),
        "modalities": sorted({record["modality"] for record in records}),
        "n_per_modality": {
            modality: sum(1 for record in records if record["modality"] == modality)
            for modality in sorted({record["modality"] for record in records})
        },
        "label_names": result["label_names"],
        "predictions_root": relative_to_root(predictions_root),
        "label_names_path": relative_to_root(label_names_path),
        "label_names_sha256": sha256_file(label_names_path),
        "row_index_sha256": sha256_array(records[0]["row_index"]),
        "labels_sha256": sha256_array(records[0]["labels"]),
        "n_samples": int(len(records[0]["row_index"])),
        "no_test_access": all(
            record["manifest"].get("checks", {}).get("test_split_accessed") is False
            for record in records
        ),
        "seed_audit": [
            {
                "run_name": record["run_name"],
                "modality": record["modality"],
                "seed_run_label": record["seed"],
                "seed_config": record["seed_config"],
                "seed_metadata_mismatch": record["seed_metadata_mismatch"],
            }
            for record in records
        ],
        "unique_config_seeds": {
            modality: sorted(
                {
                    int(record["seed_config"])
                    for record in records
                    if record["modality"] == modality and record["seed_config"] is not None
                }
            )
            for modality in sorted({record["modality"] for record in records})
        },
        "per_model": result["per_model_rows"],
        "modality_metrics": result["modality_rows"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _build_readme(result: dict, threshold: float, label_set: str, phase: str) -> str:
    label_config = resolve_label_set(label_set)
    lines = [
        f"Line B {phase} · {label_config['task']} Validation metrics",
        f"Threshold: {threshold}",
        "Split: validation only (no Test access)",
        "",
        "Per-modality mean ± sample SD (n=3 seeds; see seed_audit.csv)",
        "Modality | Micro-F1 | Macro-F1 | Macro-AP | Macro-AUROC",
        "---|---|---:|---:|---:|---:",
    ]
    by_modality = defaultdict(list)
    for row in result["per_model_rows"]:
        by_modality[row["modality"]].append(row)
    for modality, rows in sorted(by_modality.items()):
        values = {}
        for metric in ["micro_f1", "macro_f1", "macro_ap", "macro_auroc"]:
            mean, sd = _mean_sd([row[metric] for row in rows if row[metric] != ""])
            values[metric] = f"{mean:.6f} ± {sd:.6f}" if mean is not None else "NA"
        lines.append(
            f"{modality} | {values['micro_f1']} | {values['macro_f1']} | "
            f"{values['macro_ap']} | {values['macro_auroc']}"
        )
    mismatches = [
        row for row in result["per_model_rows"] if row["seed_metadata_mismatch"]
    ]
    if mismatches:
        lines.extend(["", "Seed metadata mismatch detected:"])
        for row in mismatches:
            lines.append(
                f"  {row['run_name']}: run-label seed={row['seed']} but "
                f"checkpoint.config.seed={row['seed_config']}"
            )
        lines.append(
            "  The modality mean/SD above uses run labels, not the duplicate config seed; "
            "treat the affected modality as having fewer unique trained seeds."
        )
    lines.extend(
        [
            "",
            "Per-model metrics: model_metrics.csv",
            "Per-label metrics: per_label_all_models.csv",
            "Per-label mean/SD: per_label_mean_sd.csv",
            "FP16 reference deltas: fp16_reference_delta.csv",
            "Summary: summary.json",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SETS),
        default="shared14",
        help="Label task to evaluate; Shared14 remains the backward-compatible default.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=None,
        help="Defaults to runs/LINE-B-PREDICTIONS/<Task>.",
    )
    parser.add_argument(
        "--label-names",
        type=Path,
        default=None,
        help="Defaults to the authoritative label-name file for --label-set.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to runs/LINE-B-METRICS/<Task>.",
    )
    parser.add_argument("--phase", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow fewer than 9 models (synthetic/partial checks only).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_config = resolve_label_set(args.label_set)
    phase = args.phase or label_config["phase"]
    predictions_root = args.predictions_root or (
        PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS" / label_config["task"]
    )
    label_names_path = args.label_names or label_config["names"]
    out_dir = args.out_dir or (
        PROJECT_ROOT / "runs" / "LINE-B-METRICS" / label_config["task"]
    )

    result = evaluate(
        predictions_root,
        label_names_path,
        args.threshold,
        require_full_matrix=not args.allow_partial,
        label_set=args.label_set,
    )
    summary = _build_summary(
        result,
        predictions_root,
        label_names_path,
        args.threshold,
        args.label_set,
        phase,
    )

    with atomic_output_dir(out_dir) as temporary:
        _write_csv(
            temporary / "model_metrics.csv",
            result["per_model_rows"],
            [
                "run_name",
                "modality",
                "seed",
                "seed_config",
                "seed_metadata_mismatch",
                "n_samples",
                "n_labels",
                "threshold",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "macro_ap",
                "macro_auroc",
                "n_defined_ap",
                "n_defined_auroc",
                "undefined_ap_labels",
                "undefined_auroc_labels",
            ],
        )
        _write_csv(
            temporary / "per_label_all_models.csv",
            result["per_label_rows"],
            [
                "run_name",
                "modality",
                "seed",
                "seed_config",
                "seed_metadata_mismatch",
                "label_index",
                "label",
                "support",
                "prevalence",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "auroc",
                "ap_defined",
                "auroc_defined",
            ],
        )
        _write_csv(
            temporary / "per_label_mean_sd.csv",
            result["per_label_summary"],
            [
                "modality",
                "label_index",
                "label",
                "support_mean",
                "prevalence_mean",
                "average_precision_mean",
                "average_precision_sd",
                "auroc_mean",
                "auroc_sd",
                "f1_mean",
                "f1_sd",
                "precision_mean",
                "precision_sd",
                "recall_mean",
                "recall_sd",
                "n_seeds",
            ],
        )
        _write_csv(
            temporary / "modality_mean_sd.csv",
            result["modality_rows"],
            ["modality", "metric", "mean", "sd", "n_seeds"],
        )
        _write_csv(
            temporary / "fp16_reference_delta.csv",
            result["fp16_rows"],
            [
                "run_name",
                "modality",
                "seed",
                "seed_config",
                "seed_metadata_mismatch",
                "metric",
                "fp16_value",
                "fp32_value",
                "delta",
                "abs_delta",
            ],
        )
        write_json(temporary / "summary.json", summary)
        (temporary / "README_summary.txt").write_text(
            _build_readme(result, args.threshold, args.label_set, phase),
            encoding="utf-8",
        )

    print((out_dir / "README_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
