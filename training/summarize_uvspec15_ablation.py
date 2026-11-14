#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the 24 UVSPEC15 BCE/WBCE x seed x BN x threshold settings.

Run at SpectraDB root:
    python training/summarize_uvspec15_ablation.py

Reads only recorded JSON/CSV; does NOT load PyTorch, retrain, recalculate
predictions, modify checkpoints, or use Test scores to choose configurations.
Output: runs/UVSPEC15-ABLATION-SUMMARY/ (created only if all checks pass).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path, PureWindowsPath

RUN_IDS = {
    "BCE": {
        42: "EXP-UVVIS-003-UVSPEC15-FULL",
        123: "EXP-UVVIS-003-UVSPEC15-SEED123",
        2026: "EXP-UVVIS-003-UVSPEC15-SEED2026",
    },
    "WBCE": {
        42: "EXP-UVVIS-004-UVSPEC15-WBCE",
        123: "EXP-UVVIS-004-UVSPEC15-WBCE-SEED123",
        2026: "EXP-UVVIS-004-UVSPEC15-WBCE-SEED2026",
    },
}
SEEDS = (42, 123, 2026)
METRICS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "mAP", "macro_AUROC")
F1_METRICS = METRICS[:4]
DISPLAY_METRICS = ("micro_f1", "macro_f1", "mAP", "macro_AUROC")
TOLERANCE = 0.0025  # Evaluation roundoff; FAIL on a substantively different checkpoint.


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_metric(source: dict, key: str, context: str) -> float:
    if key not in source or source[key] is None:
        raise KeyError(f"Missing metric {key} in {context}")
    value = float(source[key])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid metric {key}={value} in {context}")
    return value


def compare_metrics(left: dict, right: dict, keys, left_name: str, right_name: str) -> None:
    for key in keys:
        a = require_metric(left, key, left_name)
        b = require_metric(right, key, right_name)
        if abs(a - b) > TOLERANCE:
            raise ValueError(
                f"Checkpoint/result mismatch for {key}: {left_name}={a:.6f}, "
                f"{right_name}={b:.6f} (tolerance {TOLERANCE}). "
                "Do not use stale threshold files; check original run and seed."
            )


def original_baseline(raw: dict) -> dict:
    keys = {
        "micro_f1": "test_micro_f1", "macro_f1": "test_macro_f1",
        "micro_precision": "test_micro_precision", "micro_recall": "test_micro_recall",
        "mAP": "test_mAP", "macro_AUROC": "test_macro_AUROC",
    }
    return {metric: require_metric(raw, source_key, "test_metrics.json")
            for metric, source_key in keys.items()}


def metric_row(run_name, loss, seed, split, bn, calibrated, summary: dict,
               diag: dict) -> dict:
    branch = "validation" if split == "Validation" else "test"
    suffix = "calibrated" if calibrated else "default"
    payload = summary[f"{branch}_{suffix}"]
    # Original threshold-calibration scripts record test_threshold_free, but
    # generally not validation_threshold_free. For Validation, the BEFORE
    # section of the full BN diagnostic supplies the uncalibrated rankings.
    if bn == "Original":
        free = summary.get(f"{branch}_threshold_free") or diag["before"][branch]
    else:
        free = summary[f"{branch}_threshold_free"]
    output = {
        "run_name": run_name, "loss": loss, "seed": seed, "split": split,
        "bn": bn, "threshold": "Validation-F1" if calibrated else "0.5",
    }
    for metric in METRICS:
        source = free if metric in ("mAP", "macro_AUROC") else payload
        output[metric] = require_metric(source, metric, f"{run_name}/{split}/{bn}/{suffix}")
    return output


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required per-label file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def per_label_records(root: Path, label_names: list[str], run_name: str,
                      loss: str, seed: int, bn: str, folder: Path) -> list[dict]:
    rows = []
    for split, filename in (("Validation", "validation_threshold_comparison.csv"),
                            ("Test", "threshold_comparison.csv")):
        data = read_csv(folder / filename)
        if len(data) != len(label_names):
            raise ValueError(f"Expected {len(label_names)} rows in {folder / filename}, found {len(data)}")
        for j, entry in enumerate(data):
            if int(entry.get("label_index", -1)) != j or entry.get("label") != label_names[j]:
                raise ValueError(f"Label order mismatch at row {j}: {folder / filename}")
            support_key = "positive_val_samples" if split == "Validation" else "positive_test_samples"
            if support_key not in entry:
                raise KeyError(f"Missing {support_key} in {folder / filename}")
            for calibrated in (False, True):
                prefix = "calibrated" if calibrated else "default"
                row = {
                    "run_name": run_name, "loss": loss, "seed": seed,
                    "split": split, "bn": bn,
                    "threshold": "Validation-F1" if calibrated else "0.5",
                    "label_index": j, "label": label_names[j],
                    "positive_samples": int(entry[support_key]),
                    "threshold_value": float(entry["optimal_threshold"]) if calibrated else 0.5,
                }
                for key in ("precision", "recall", "f1"):
                    row[key] = require_metric(entry, f"{prefix}_{key}", str(folder / filename))
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gather(root: Path):
    label_path = root / "processed" / "qm9s" / "prepared" / "uvspec15_label_names.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing UVSPEC15 label list: {label_path}")
    label_names = json.loads(label_path.read_text(encoding="utf-8-sig"))
    if not isinstance(label_names, list) or len(label_names) != 15 or len(set(label_names)) != 15:
        raise ValueError("UVSPEC15 labels must contain 15 unique names")

    records, per_label, qa_log = [], [], []
    for loss, seeds in RUN_IDS.items():
        for seed, run_name in seeds.items():
            original = root / "runs" / run_name
            bn_dir = root / "runs" / "UVSPEC15-BN-RECALIBRATED" / run_name
            orig_summary = load_json(original / "threshold_calibration_summary.json")
            bn_summary = load_json(bn_dir / "threshold_calibration" / "threshold_calibration_summary.json")
            baseline = original_baseline(load_json(original / "test_metrics.json"))
            diag = load_json(root / "runs" / "UVSPEC15-BN-DIAGNOSTICS"
                             / run_name / "batchnorm_full_diagnostic.json")
            manifest = load_json(bn_dir / "bn_recalibration_manifest.json")
            checkpoint_file = bn_dir / "best_model_bn_recalibrated.pt"
            if not checkpoint_file.is_file():
                raise FileNotFoundError(f"Missing recalibrated checkpoint: {checkpoint_file}")
            if int(manifest.get("seed", -1)) != seed or int(manifest.get("train_samples_used", -1)) != 103853:
                raise ValueError(f"Wrong BN manifest seed/train set for {run_name}")
            if bn_summary.get("seed") != seed or bn_summary.get("experiment") != run_name:
                raise ValueError(f"Wrong BN threshold summary provenance: {run_name}")
            # Older single-run Full BN diagnostics do not record requested_seed.
            # Check it when available; verify the recorded source checkpoint in all
            # cases, so a missing optional field never weakens run provenance.
            recorded_seed = diag.get("requested_seed", diag.get("checkpoint_config_seed"))
            if recorded_seed is not None and int(recorded_seed) != seed:
                raise ValueError(
                    f"Wrong BN diagnostic seed for {run_name}: "
                    f"recorded={recorded_seed}, expected={seed}"
                )
            if diag.get("experiment") != run_name:
                raise ValueError(
                    f"Wrong BN diagnostic experiment for {run_name}: "
                    f"recorded={diag.get('experiment')!r}"
                )
            source = diag.get("source_checkpoint")
            # PureWindowsPath also parses the Windows paths when this script is
            # syntax-checked or tested on another operating system.
            source_path = PureWindowsPath(str(source or "").replace("/", "\\"))
            if (
                source_path.name.casefold() != "best_model.pt"
                or source_path.parent.name.casefold() != run_name.casefold()
                or source_path.parent.parent.name.casefold() != "runs"
            ):
                raise ValueError(
                    f"Wrong BN diagnostic source checkpoint for {run_name}: {source!r}"
                )
            if not diag.get("baseline_test_reproduced") or not diag.get("model_weights_unchanged"):
                raise ValueError(f"BN diagnostic not verified: {run_name}")

            # Original threshold calibration MUST be for the latest original model,
            # not a superseded Seed 42 checkpoint.
            compare_metrics(baseline, orig_summary["test_default"], F1_METRICS,
                            f"{run_name}/test_metrics.json", "original threshold summary")
            compare_metrics(baseline, orig_summary["test_threshold_free"],
                            ("mAP", "macro_AUROC"),
                            f"{run_name}/test_metrics.json", "original threshold summary")
            compare_metrics(diag["before"]["test"], baseline, METRICS,
                            "full BN diagnostic before/test", f"{run_name}/test_metrics.json")
            compare_metrics(diag["after"]["test"], bn_summary["test_default"], F1_METRICS,
                            "full BN diagnostic after/test", "BN threshold summary default/test")
            compare_metrics(diag["after"]["test"], bn_summary["test_threshold_free"],
                            ("mAP", "macro_AUROC"),
                            "full BN diagnostic after/test", "BN threshold summary free/test")
            compare_metrics(diag["after"]["validation"], bn_summary["validation_threshold_free"],
                            ("mAP", "macro_AUROC"),
                            "full BN diagnostic after/validation", "BN threshold summary free/validation")

            for split in ("Validation", "Test"):
                for bn, summary in (("Original", orig_summary), ("Recalibrated", bn_summary)):
                    for calibrated in (False, True):
                        records.append(metric_row(run_name, loss, seed, split, bn,
                                                  calibrated, summary, diag))
            for bn, folder in (("Original", original),
                               ("Recalibrated", bn_dir / "threshold_calibration")):
                per_label += per_label_records(root, label_names, run_name, loss, seed, bn, folder)
            qa_log.append({"run_name": run_name, "loss": loss, "seed": seed,
                           "original_baseline_reproduced": True,
                           "bn_saved_diagnostic_reproduced": True,
                           "label_order_verified": True})
    if len(records) != 48 or len(per_label) != 720:
        raise AssertionError(f"Wrong record counts: {len(records)} overall / {len(per_label)} per-label")
    return records, per_label, qa_log


def aggregates(records: list[dict]) -> list[dict]:
    grouped = {}
    for entry in records:
        k = (entry["split"], entry["loss"], entry["bn"], entry["threshold"])
        grouped.setdefault(k, []).append(entry)
    if len(grouped) != 16:
        raise AssertionError(f"Expected 16 groups (8 conditions x 2 splits), found {len(grouped)}")
    output = []
    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda x: x["seed"])
        if tuple(e["seed"] for e in entries) != SEEDS:
            raise ValueError(f"Three seeds not complete: {key}")
        row = dict(zip(("split", "loss", "bn", "threshold"), key))
        row["n_seeds"] = 3
        for metric in METRICS:
            vals = [e[metric] for e in entries]
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_sd"] = statistics.stdev(vals)
        output.append(row)
    return output


def paired_effects(records: list[dict]) -> list[dict]:
    by_key = {(r["split"], r["loss"], r["seed"], r["bn"], r["threshold"]): r
              for r in records}
    out = []
    for split in ("Validation", "Test"):
        for seed in SEEDS:
            for loss in ("BCE", "WBCE"):
                for bn in ("Original", "Recalibrated"):
                    a = by_key[(split, loss, seed, bn, "0.5")]
                    b = by_key[(split, loss, seed, bn, "Validation-F1")]
                    out.append(make_difference(split, seed, loss, bn, "Threshold calibration", a, b))
                for threshold in ("0.5", "Validation-F1"):
                    a = by_key[(split, loss, seed, "Original", threshold)]
                    b = by_key[(split, loss, seed, "Recalibrated", threshold)]
                    out.append(make_difference(split, seed, loss, threshold, "BN recalibration", a, b))
            for bn in ("Original", "Recalibrated"):
                for threshold in ("0.5", "Validation-F1"):
                    a = by_key[(split, "BCE", seed, bn, threshold)]
                    b = by_key[(split, "WBCE", seed, bn, threshold)]
                    out.append(make_difference(split, seed, bn, threshold, "WBCE minus BCE", a, b))
    return out


def make_difference(split, seed, context_a, context_b, change, before, after):
    row = {"split": split, "seed": seed, "comparison": change,
           "context_a": context_a, "context_b": context_b,
           "from": f"{before['loss']}/{before['bn']}/{before['threshold']}",
           "to": f"{after['loss']}/{after['bn']}/{after['threshold']}"}
    row.update({f"delta_{metric}": after[metric] - before[metric] for metric in METRICS})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="SpectraDB project root (default: parent of training/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="New output directory; refuses to overwrite existing directory")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir if args.out_dir else
               root / "runs" / "UVSPEC15-ABLATION-SUMMARY").resolve()
    if out_dir.exists():
        parser.error(f"Output already exists; refusing to overwrite: {out_dir}")
    print("Reading and cross-checking all six runs (read-only)...")
    records, per_label, qa = gather(root)
    summary = aggregates(records)
    paired = paired_effects(records)
    out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(out_dir / "test_24_conditions.csv", [r for r in records if r["split"] == "Test"])
    write_csv(out_dir / "validation_24_conditions.csv", [r for r in records if r["split"] == "Validation"])
    write_csv(out_dir / "mean_sd_8_conditions_each_split.csv", summary)
    write_csv(out_dir / "per_label_720_rows.csv", per_label)
    write_csv(out_dir / "paired_effects.csv", paired)
    write_csv(out_dir / "audit_checks.csv", qa)

    lines = ["UVSPEC15 complete ablation summary", "", "All six original checkpoints and six BN checkpoints",
             "24 conditions per split; 8 conditions x 3 seeds", "",
             "Mean ± sample SD on Test (n=3)", "",
             "Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC"]
    for row in summary:
        if row["split"] != "Test":
            continue
        numbers = [f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}" for metric in DISPLAY_METRICS]
        lines.append(" | ".join((row["loss"], row["bn"], row["threshold"], *numbers)))
    lines += ["", "Selection policy: use Validation for strategy selection; Test scores are evaluation only.",
              "The result table does not establish that within-training epoch oscillations were eliminated.",
              "All source results, checkpoints, and thresholds were read-only."]
    with (out_dir / "README_summary.txt").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nFiles written to:", out_dir)
    print("PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 720 per-label rows")


if __name__ == "__main__":
    main()
