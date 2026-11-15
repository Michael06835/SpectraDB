#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the IR v1 24-condition ablation (2 loss x 3 seed x 2 BN x 2 threshold).

IR port of training/summarize_uvspec15_ablation.py. Reads only recorded
JSON/CSV; it does NOT load PyTorch, run any forward pass, refit any threshold,
or modify checkpoints. All 24 conditions per split are cross-checked against the
approved Full BN Diagnostic records, and the output directory is created only if
every audit check passes.

Per-condition metric sources (exactly what metric_row() reads):

  F1-dependent metrics (Micro-F1, Macro-F1, Micro-Precision, Micro-Recall)
      always come from the matching threshold_calibration_summary.json:
        0.5            -> validation_default / test_default
        Validation-F1  -> validation_calibrated / test_calibrated

  Threshold-free metrics (mAP, Macro-AUROC)
        Test           -> summary.test_threshold_free
        Validation     -> the BN diagnostic block (before for Original,
                          after for Recalibrated), because the calibration
                          summaries do not record validation_threshold_free

  The BN diagnostic blocks are therefore used for (a) threshold-free Validation
  metrics and (b) cross-checks against the calibration summaries. The 0.5 F1
  metrics are NOT read from the diagnostic JSON; the two sources are verified to
  agree within TOLERANCE before any row is emitted.

Inputs (all read-only):
  runs/<run_name>/                        six original training runs
  runs/IR-BN-DIAGNOSTICS/<run_name>/      six Full BN diagnostics
  runs/IR-BN-RECALIBRATED/<run_name>/     six recalibrated checkpoints + manifests
  runs/IR-THRESHOLD-CALIBRATION/<run>/    six original-checkpoint calibrations
  runs/IR-BN-THRESHOLD-CALIBRATION/<run>/ six recalibrated-checkpoint calibrations

Output: runs/IR-ABLATION-SUMMARY/ (only created if all checks pass)

Run at the SpectraDB root:
    python training/summarize_ir_ablation.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]

RUN_IDS = {
    "BCE": {
        42: "EXP-IR-001-FULL",
        123: "EXP-IR-001-SEED123",
        2026: "EXP-IR-001-SEED2026",
    },
    "WBCE": {
        42: "EXP-IR-002-FULL",
        123: "EXP-IR-002-SEED123",
        2026: "EXP-IR-002-SEED2026",
    },
}
SEEDS = (42, 123, 2026)
LOSS_KEYS = {"BCE": "bce", "WBCE": "wbce"}

METRICS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "mAP", "macro_AUROC")
F1_METRICS = METRICS[:4]
FREE_METRICS = ("mAP", "macro_AUROC")
DISPLAY_METRICS = ("micro_f1", "macro_f1", "mAP", "macro_AUROC")

TOLERANCE = 0.0025  # evaluation roundoff; FAIL on a substantively different artifact

EXPECTED_N_LABELS = 14
EXPECTED_TRAIN_SAMPLES = 103855
EXPECTED_CONDITIONS_PER_SPLIT = 24
EXPECTED_RECORDS = 48
EXPECTED_PER_LABEL_ROWS = 672  # 14 labels x 24 conditions x 2 splits
EXPECTED_GROUPS = 16  # 2 loss x 2 bn x 2 threshold x 2 splits

# Support tiers for per-label analysis (documented in README_summary.txt).
SUPPORT_VERY_LOW = 30
SUPPORT_LOW = 150


# ---------------------------------------------------------------- helpers ---

def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                f"Artifact mismatch for {key}: {left_name}={a:.6f}, "
                f"{right_name}={b:.6f} (tolerance {TOLERANCE}). "
                "Do not mix stale calibration summaries with a different checkpoint."
            )


def original_baseline(raw: dict) -> dict:
    keys = {
        "micro_f1": "test_micro_f1",
        "macro_f1": "test_macro_f1",
        "micro_precision": "test_micro_precision",
        "micro_recall": "test_micro_recall",
        "mAP": "test_mAP",
        "macro_AUROC": "test_macro_AUROC",
    }
    return {metric: require_metric(raw, source_key, "test_metrics.json")
            for metric, source_key in keys.items()}


def metric_row(run_name, loss, seed, split, bn, calibrated, summary: dict, diag: dict) -> dict:
    branch = "validation" if split == "Validation" else "test"
    suffix = "calibrated" if calibrated else "default"
    payload = summary[f"{branch}_{suffix}"]
    # The calibration summaries record test_threshold_free only; for Validation,
    # the matching BN diagnostic block supplies the uncalibrated rankings.
    if summary.get(f"{branch}_threshold_free") is not None:
        free = summary[f"{branch}_threshold_free"]
    else:
        free = diag["before" if bn == "Original" else "after"][branch]
    output = {
        "run_name": run_name,
        "loss": loss,
        "seed": seed,
        "split": split,
        "bn": bn,
        "threshold": "Validation-F1" if calibrated else "0.5",
    }
    for metric in METRICS:
        source = free if metric in FREE_METRICS else payload
        output[metric] = require_metric(
            source, metric, f"{run_name}/{split}/{bn}/{suffix}"
        )
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
            raise ValueError(
                f"Expected {len(label_names)} rows in {folder / filename}, found {len(data)}"
            )
        for j, entry in enumerate(data):
            if int(entry.get("label_index", -1)) != j or entry.get("label") != label_names[j]:
                raise ValueError(
                    f"Label order mismatch at row {j}: {folder / filename}"
                )
            support_key = "positive_val_samples" if split == "Validation" else "positive_test_samples"
            if support_key not in entry:
                raise KeyError(f"Missing {support_key} in {folder / filename}")
            for calibrated in (False, True):
                prefix = "calibrated" if calibrated else "default"
                row = {
                    "run_name": run_name,
                    "loss": loss,
                    "seed": seed,
                    "split": split,
                    "bn": bn,
                    "threshold": "Validation-F1" if calibrated else "0.5",
                    "label_index": j,
                    "label": label_names[j],
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


# ------------------------------------------------------------- collection ---

def gather(root: Path):
    label_path = root / "processed" / "qm9s" / "prepared" / "functional_group_label_names.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing IR label list: {label_path}")
    label_names = json.loads(label_path.read_text(encoding="utf-8-sig"))
    if (not isinstance(label_names, list) or len(label_names) != EXPECTED_N_LABELS
            or len(set(label_names)) != EXPECTED_N_LABELS):
        raise ValueError(f"IR labels must contain {EXPECTED_N_LABELS} unique names")

    records, per_label, qa_log, registry = [], [], [], []

    for loss, seeds in RUN_IDS.items():
        for seed, run_name in seeds.items():
            run_dir = root / "runs" / run_name
            diag_dir = root / "runs" / "IR-BN-DIAGNOSTICS" / run_name
            recal_dir = root / "runs" / "IR-BN-RECALIBRATED" / run_name
            orig_cal_dir = root / "runs" / "IR-THRESHOLD-CALIBRATION" / run_name
            recal_cal_dir = root / "runs" / "IR-BN-THRESHOLD-CALIBRATION" / run_name

            orig_summary = load_json(orig_cal_dir / "threshold_calibration_summary.json")
            recal_summary = load_json(recal_cal_dir / "threshold_calibration_summary.json")
            baseline = original_baseline(load_json(run_dir / "test_metrics.json"))
            diag = load_json(diag_dir / "batchnorm_full_diagnostic.json")
            manifest = load_json(recal_dir / "bn_recalibration_manifest.json")
            recal_ckpt = recal_dir / "best_model_bn_recalibrated.pt"
            orig_ckpt = run_dir / "best_model.pt"
            for required in (recal_ckpt, orig_ckpt):
                if not required.is_file():
                    raise FileNotFoundError(f"Missing checkpoint: {required}")

            # --- provenance chain (seed / loss / hashes) -------------------
            if int(manifest.get("seed", -1)) != seed:
                raise ValueError(f"BN manifest seed mismatch for {run_name}")
            if int(manifest.get("train_samples_used", -1)) != EXPECTED_TRAIN_SAMPLES:
                raise ValueError(f"BN manifest train-sample mismatch for {run_name}")
            if manifest.get("run_directory") != run_name:
                raise ValueError(f"BN manifest run_directory mismatch for {run_name}")
            if manifest.get("saved_sha256") != sha256_file(recal_ckpt):
                raise ValueError(f"BN manifest saved_sha256 mismatch for {run_name}")
            if manifest.get("source_sha256") != sha256_file(orig_ckpt):
                raise ValueError(f"BN manifest source_sha256 mismatch for {run_name}")
            if recal_summary.get("bn_state") != "recalibrated":
                raise ValueError(f"Not a recalibrated-BN threshold summary: {run_name}")
            if recal_summary.get("checkpoint_sha256") != manifest["saved_sha256"]:
                raise ValueError(f"Recalibrated calibration does not match the saved checkpoint: {run_name}")
            for summary, label in ((orig_summary, "original"), (recal_summary, "recalibrated")):
                if summary.get("experiment") != run_name:
                    raise ValueError(f"{label} threshold summary experiment mismatch: {run_name}")
                if int(summary.get("seed", -1)) != seed:
                    raise ValueError(f"{label} threshold summary seed mismatch: {run_name}")
                if summary.get("loss") != LOSS_KEYS[loss]:
                    raise ValueError(f"{label} threshold summary loss mismatch: {run_name}")
                if int(summary.get("num_labels", -1)) != EXPECTED_N_LABELS:
                    raise ValueError(f"{label} threshold summary label count mismatch: {run_name}")
            if orig_summary.get("checkpoint_sha256") != manifest["source_sha256"]:
                raise ValueError(f"Original calibration does not match the original checkpoint: {run_name}")
            if diag.get("experiment") != run_name:
                raise ValueError(f"BN diagnostic experiment mismatch for {run_name}")
            recorded_seed = diag.get("requested_seed", diag.get("checkpoint_config_seed"))
            if recorded_seed is not None and int(recorded_seed) != seed:
                raise ValueError(f"BN diagnostic seed mismatch for {run_name}")
            source = PureWindowsPath(str(diag.get("source_checkpoint", "")).replace("/", "\\"))
            if (source.name.casefold() != "best_model.pt"
                    or source.parent.name.casefold() != run_name.casefold()
                    or source.parent.parent.name.casefold() != "runs"):
                raise ValueError(f"BN diagnostic source checkpoint mismatch for {run_name}")
            if not diag.get("baseline_test_reproduced") or not diag.get("model_weights_unchanged"):
                raise ValueError(f"BN diagnostic not verified: {run_name}")

            # --- cross-checks ----------------------------------------------
            compare_metrics(baseline, orig_summary["test_default"], F1_METRICS,
                            f"{run_name}/test_metrics.json", "original threshold summary")
            compare_metrics(baseline, orig_summary["test_threshold_free"], FREE_METRICS,
                            f"{run_name}/test_metrics.json", "original threshold summary")
            compare_metrics(diag["before"]["test"], baseline, METRICS,
                            "full BN diagnostic before/test", f"{run_name}/test_metrics.json")
            compare_metrics(diag["before"]["validation"], orig_summary["validation_default"],
                            F1_METRICS, "full BN diagnostic before/validation",
                            "original threshold summary default/validation")
            compare_metrics(diag["after"]["test"], recal_summary["test_default"], F1_METRICS,
                            "full BN diagnostic after/test", "BN threshold summary default/test")
            compare_metrics(diag["after"]["test"], recal_summary["test_threshold_free"], FREE_METRICS,
                            "full BN diagnostic after/test", "BN threshold summary free/test")
            compare_metrics(diag["after"]["validation"], recal_summary["validation_default"],
                            F1_METRICS, "full BN diagnostic after/validation",
                            "BN threshold summary default/validation")

            # --- 24 conditions per split -----------------------------------
            for split in ("Validation", "Test"):
                for bn, summary in (("Original", orig_summary),
                                    ("Recalibrated", recal_summary)):
                    for calibrated in (False, True):
                        records.append(metric_row(run_name, loss, seed, split, bn,
                                                  calibrated, summary, diag))
                        registry.append({
                            "run_name": run_name,
                            "loss": loss,
                            "seed": seed,
                            "split": split,
                            "bn": bn,
                            "threshold": "Validation-F1" if calibrated else "0.5",
                            "metric_source": (
                                f"{'IR-BN-THRESHOLD-CALIBRATION' if bn == 'Recalibrated' else 'IR-THRESHOLD-CALIBRATION'}"
                                f"/{run_name}/threshold_calibration_summary.json "
                                f"[{'test' if split == 'Test' else 'validation'}_"
                                f"{'calibrated' if calibrated else 'default'}]"
                            ),
                            "threshold_free_source": (
                                f"IR-BN-DIAGNOSTICS/{run_name}/batchnorm_full_diagnostic.json "
                                f"[{'after' if bn == 'Recalibrated' else 'before'}."
                                f"{'test' if split == 'Test' else 'validation'}]"
                            ),
                            "per_label_source": (
                                f"{'IR-BN-THRESHOLD-CALIBRATION' if bn == 'Recalibrated' else 'IR-THRESHOLD-CALIBRATION'}"
                                f"/{run_name}/"
                                f"{'threshold_comparison.csv' if split == 'Test' else 'validation_threshold_comparison.csv'}"
                            ),
                        })

            for bn, folder in (("Original", orig_cal_dir), ("Recalibrated", recal_cal_dir)):
                per_label += per_label_records(root, label_names, run_name, loss, seed, bn, folder)

            qa_log.append({
                "run_name": run_name,
                "loss": loss,
                "seed": seed,
                "original_baseline_reproduced": True,
                "bn_saved_diagnostic_reproduced": True,
                "label_order_verified": True,
                "manifest_hash_chain_verified": True,
            })

    if len(records) != EXPECTED_RECORDS:
        raise AssertionError(f"Expected {EXPECTED_RECORDS} split-level records, found {len(records)}")
    if len(per_label) != EXPECTED_PER_LABEL_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_PER_LABEL_ROWS} per-label rows, found {len(per_label)}"
        )
    unique_conditions = {(r["loss"], r["seed"], r["bn"], r["threshold"], r["split"]) for r in records}
    if len(unique_conditions) != EXPECTED_RECORDS:
        raise AssertionError("Duplicate conditions detected in the 24-condition matrix")

    # Threshold-free metrics must be identical for the two threshold conditions of
    # the same (model, BN) pair; a difference here means the wiring is wrong.
    by_key = {(r["loss"], r["seed"], r["bn"], r["threshold"], r["split"]): r for r in records}
    for loss in ("BCE", "WBCE"):
        for seed in SEEDS:
            for bn in ("Original", "Recalibrated"):
                for split in ("Validation", "Test"):
                    a = by_key[(loss, seed, bn, "0.5", split)]
                    b = by_key[(loss, seed, bn, "Validation-F1", split)]
                    for metric in FREE_METRICS:
                        if abs(a[metric] - b[metric]) > 1e-9:
                            raise AssertionError(
                                f"Threshold-free metric {metric} differs between threshold "
                                f"conditions: {loss}/{seed}/{bn}/{split}"
                            )

    return records, per_label, qa_log, registry, label_names


# ------------------------------------------------------------ statistics ---

def aggregates(records: list[dict]) -> list[dict]:
    grouped = {}
    for entry in records:
        key = (entry["split"], entry["loss"], entry["bn"], entry["threshold"])
        grouped.setdefault(key, []).append(entry)
    if len(grouped) != EXPECTED_GROUPS:
        raise AssertionError(
            f"Expected {EXPECTED_GROUPS} groups (8 conditions x 2 splits), found {len(grouped)}"
        )
    output = []
    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda x: x["seed"])
        if tuple(e["seed"] for e in entries) != SEEDS:
            raise ValueError(f"Three seeds not complete: {key}")
        row = dict(zip(("split", "loss", "bn", "threshold"), key))
        row["n_seeds"] = 3
        for metric in METRICS:
            values = [e[metric] for e in entries]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values)
        output.append(row)
    return output


def make_difference(split, seed, context_a, context_b, comparison, before, after):
    row = {
        "split": split,
        "seed": seed,
        "comparison": comparison,
        "context_a": context_a,
        "context_b": context_b,
        "from": f"{before['loss']}/{before['bn']}/{before['threshold']}",
        "to": f"{after['loss']}/{after['bn']}/{after['threshold']}",
    }
    row.update({f"delta_{metric}": after[metric] - before[metric] for metric in METRICS})
    return row


def paired_effects(records: list[dict]) -> list[dict]:
    by_key = {(r["split"], r["loss"], r["seed"], r["bn"], r["threshold"]): r
              for r in records}
    out = []
    for split in ("Validation", "Test"):
        for seed in SEEDS:
            for loss in ("BCE", "WBCE"):
                # Threshold calibration effect, at fixed BN state.
                for bn in ("Original", "Recalibrated"):
                    out.append(make_difference(
                        split, seed, bn, "0.5", "Threshold calibration",
                        by_key[(split, loss, seed, bn, "0.5")],
                        by_key[(split, loss, seed, bn, "Validation-F1")]))
                # BN recalibration effect, at fixed threshold.
                for threshold in ("0.5", "Validation-F1"):
                    out.append(make_difference(
                        split, seed, threshold, threshold, "BN recalibration",
                        by_key[(split, loss, seed, "Original", threshold)],
                        by_key[(split, loss, seed, "Recalibrated", threshold)]))
                # BN x threshold interaction: (Recal-Orig | ValF1) - (Recal-Orig | 0.5)
                interaction = {"split": split, "seed": seed,
                               "comparison": "BN x threshold interaction",
                               "context_a": "0.5", "context_b": "Validation-F1",
                               "from": f"{loss}/Original->Recalibrated @0.5",
                               "to": f"{loss}/Original->Recalibrated @Validation-F1"}
                for metric in METRICS:
                    interaction[f"delta_{metric}"] = (
                        by_key[(split, loss, seed, "Recalibrated", "Validation-F1")][metric]
                        - by_key[(split, loss, seed, "Original", "Validation-F1")][metric]
                        - (by_key[(split, loss, seed, "Recalibrated", "0.5")][metric]
                           - by_key[(split, loss, seed, "Original", "0.5")][metric])
                    )
                out.append(interaction)
            # WBCE minus BCE at fixed BN and threshold.
            for bn in ("Original", "Recalibrated"):
                for threshold in ("0.5", "Validation-F1"):
                    out.append(make_difference(
                        split, seed, bn, threshold, "WBCE minus BCE",
                        by_key[(split, "BCE", seed, bn, threshold)],
                        by_key[(split, "WBCE", seed, bn, threshold)]))
    return out


def paired_summary(paired: list[dict]) -> list[dict]:
    grouped = {}
    for row in paired:
        key = (row["split"], row["comparison"], row["context_a"], row["context_b"])
        grouped.setdefault(key, []).append(row)
    output = []
    for key in sorted(grouped):
        entries = sorted(grouped[key], key=lambda x: (x["from"], x["to"], x["seed"]))
        # Group by (from, to) so each comparison family is summarised separately.
        families = {}
        for entry in entries:
            families.setdefault((entry["from"], entry["to"]), []).append(entry)
        for (from_label, to_label), members in sorted(families.items()):
            if len(members) != 3:
                continue  # interaction rows are per (seed) only once per loss
            row = {"split": key[0], "comparison": key[1],
                   "from": from_label, "to": to_label, "n_seeds": len(members)}
            for metric in METRICS:
                values = [m[f"delta_{metric}"] for m in members]
                row[f"delta_{metric}_mean"] = statistics.mean(values)
                row[f"delta_{metric}_sd"] = statistics.stdev(values)
            output.append(row)
    return output


def per_label_summary(per_label: list[dict], label_names: list[str]) -> tuple[list[dict], list[dict]]:
    """Per-label F1 levels (mean over seeds) and per-label deltas."""
    conditions = [(loss, bn, threshold)
                  for loss in ("BCE", "WBCE")
                  for bn in ("Original", "Recalibrated")
                  for threshold in ("0.5", "Validation-F1")]

    levels = {}
    deltas = {}
    for row in per_label:
        key = (row["split"], row["label"], row["loss"], row["bn"], row["threshold"])
        levels.setdefault(key, []).append(row["f1"])

    rows = []
    for label in label_names:
        support_val = max(
            r["positive_samples"] for r in per_label
            if r["label"] == label and r["split"] == "Validation"
        )
        support_test = max(
            r["positive_samples"] for r in per_label
            if r["label"] == label and r["split"] == "Test"
        )
        smallest = min(support_val, support_test)
        if smallest < SUPPORT_VERY_LOW:
            tier = "very_low"
        elif smallest < SUPPORT_LOW:
            tier = "low"
        else:
            tier = "normal"
        row = {"label": label, "positive_val": support_val,
               "positive_test": support_test, "support_tier": tier}
        for split in ("Validation", "Test"):
            for loss, bn, threshold in conditions:
                values = levels[(split, label, loss, bn, threshold)]
                row[f"{split.lower()}_{loss}_{bn}_{threshold}_f1_mean"] = statistics.mean(values)
        rows.append(row)

    # Delta rows: threshold calibration at each BN state, and BN effect at each
    # threshold, per label and split.
    families = []
    for split in ("Validation", "Test"):
        for loss in ("BCE", "WBCE"):
            for bn in ("Original", "Recalibrated"):
                families.append((split, loss, f"Threshold calibration @{bn}",
                                 "0.5", "Validation-F1", bn, bn))
            for threshold in ("0.5", "Validation-F1"):
                families.append((split, loss, f"BN recalibration @{threshold}",
                                 threshold, threshold, "Original", "Recalibrated"))

    delta_rows = []
    for split, loss, comparison, thr_a, thr_b, bn_a, bn_b in families:
        for label in label_names:
            by_seed = {}
            for row in per_label:
                if (row["split"], row["label"], row["loss"]) != (split, label, loss):
                    continue
                by_seed.setdefault(row["seed"], {})[(row["bn"], row["threshold"])] = row["f1"]
            paired = [
                by_seed[s][(bn_b, thr_b)] - by_seed[s][(bn_a, thr_a)]
                for s in SEEDS if s in by_seed
            ]
            delta_rows.append({
                "split": split,
                "loss": loss,
                "comparison": comparison,
                "label": label,
                "n_seeds": len(paired),
                "delta_f1_mean": statistics.mean(paired),
                "delta_f1_sd": statistics.stdev(paired) if len(paired) > 1 else None,
            })
    return rows, delta_rows


# ------------------------------------------------------------------ main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="SpectraDB project root (default: parent of training/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="New output directory; refuses to overwrite existing directory")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir if args.out_dir
               else root / "runs" / "IR-ABLATION-SUMMARY").resolve()
    if out_dir.exists():
        parser.error(f"Output already exists; refusing to overwrite: {out_dir}")

    print("Reading and cross-checking the IR v1 ablation artifacts (read-only)...")
    records, per_label, qa, registry, label_names = gather(root)
    summary = aggregates(records)
    paired = paired_effects(records)
    paired_agg = paired_summary(paired)
    label_levels, label_deltas = per_label_summary(per_label, label_names)

    out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(out_dir / "test_24_conditions.csv",
              [r for r in records if r["split"] == "Test"])
    write_csv(out_dir / "validation_24_conditions.csv",
              [r for r in records if r["split"] == "Validation"])
    write_csv(out_dir / "mean_sd_8_conditions_each_split.csv", summary)
    write_csv(out_dir / "paired_effects.csv", paired)
    write_csv(out_dir / "paired_effects_summary.csv", paired_agg)
    write_csv(out_dir / "per_label_672_rows.csv", per_label)
    write_csv(out_dir / "per_label_f1_levels.csv", label_levels)
    write_csv(out_dir / "per_label_deltas.csv", label_deltas)
    write_csv(out_dir / "condition_registry.csv", registry)
    write_csv(out_dir / "audit_checks.csv", qa)

    lines = []
    lines.append("IR v1 complete ablation summary (24 conditions per split)")
    lines.append("")
    lines.append("2 loss x 3 seed x 2 BN x 2 threshold = 24 conditions; Validation 24 + Test 24.")
    lines.append("Per-label rows: 14 labels x 24 conditions x 2 splits = 672.")
    lines.append("All values are read from recorded JSON/CSV; no forward pass, no threshold refit.")
    lines.append("")
    lines.append("Test, mean +/- sample SD over 3 seeds (n=3)")
    lines.append("Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC")
    for row in summary:
        if row["split"] != "Test":
            continue
        numbers = [f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"
                   for metric in DISPLAY_METRICS]
        lines.append(" | ".join((row["loss"], row["bn"], row["threshold"], *numbers)))
    lines.append("")
    lines.append("Validation, mean +/- sample SD over 3 seeds (n=3)")
    lines.append("Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC")
    for row in summary:
        if row["split"] != "Validation":
            continue
        numbers = [f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"
                   for metric in DISPLAY_METRICS]
        lines.append(" | ".join((row["loss"], row["bn"], row["threshold"], *numbers)))
    lines.append("")
    lines.append("Paired effects on Validation (mean +/- sample SD of the per-seed difference)")
    for row in paired_agg:
        if row["split"] != "Validation":
            continue
        lines.append(
            f"{row['comparison']:<26s} {row['from']} -> {row['to']}: "
            f"dMicro-F1={row['delta_micro_f1_mean']:+.4f} +/- {row['delta_micro_f1_sd']:.4f}, "
            f"dMacro-F1={row['delta_macro_f1_mean']:+.4f} +/- {row['delta_macro_f1_sd']:.4f}, "
            f"dmAP={row['delta_mAP_mean']:+.4f} +/- {row['delta_mAP_sd']:.4f}"
        )
    lines.append("")
    lines.append("Selection policy (per docs/experiments/IR_v1_experimental_strategy.md)")
    lines.append("- Checkpoints are frozen; only Validation may be used to compare methods.")
    lines.append("- Test is evaluation only and never participates in selection.")
    lines.append("- No Seed is chosen by Test score.")
    lines.append("- No single primary selection metric is defined in the current documents,")
    lines.append("  so this summary reports multi-metric Validation comparisons and leaves")
    lines.append("  the configuration decision to the reviewer. It does not invent a rule.")
    lines.append("- BN recalibration and threshold calibration are NOT assumed to be beneficial.")
    lines.append("- n=3 seeds: no significance claims; SD is descriptive only.")
    lines.append("")
    lines.append("Support tiers for per-label analysis (see per_label_f1_levels.csv)")
    lines.append(f"- very_low: min(val,test) positives < {SUPPORT_VERY_LOW} (e.g. nitro: 11 val / 15 test)")
    lines.append(f"- low     : min(val,test) positives < {SUPPORT_LOW}")
    lines.append("- Threshold fitting on 11 validation positives is unstable; treat nitro results")
    lines.append("  as indicative only, and do not let a single sparse label drive the conclusion.")
    lines.append("")
    lines.append("Validation-vs-Test consistency")
    lines.append("Per-label thresholds are optimized on Validation, making their in-sample F1")
    lines.append("estimates optimistic. Improvement in Micro-F1 is not guaranteed.")
    lines.append("Compare the Validation and Test columns; the summary deliberately does not")
    lines.append("use Test to re-rank configurations.")
    lines.append("")
    lines.append("Metric sources (archival note)")
    lines.append("- F1-dependent metrics (Micro/Macro-F1, Micro-Precision, Micro-Recall) are read")
    lines.append("  from the matching threshold_calibration_summary.json:")
    lines.append("  0.5 -> validation_default / test_default; Validation-F1 -> *_calibrated.")
    lines.append("- Threshold-free metrics (mAP, Macro-AUROC): Test from summary.test_threshold_free;")
    lines.append("  Validation from the BN diagnostic block (before=Original, after=Recalibrated),")
    lines.append("  because the calibration summaries do not record validation_threshold_free.")
    lines.append("- The 0.5 F1 metrics are NOT read from the BN diagnostic JSON; both sources are")
    lines.append("  verified to agree within 0.0025 before any condition row is emitted.")
    lines.append("")
    lines.append("Output files")
    for name in ("test_24_conditions.csv", "validation_24_conditions.csv",
                 "mean_sd_8_conditions_each_split.csv", "paired_effects.csv",
                 "paired_effects_summary.csv", "per_label_672_rows.csv",
                 "per_label_f1_levels.csv", "per_label_deltas.csv",
                 "condition_registry.csv", "audit_checks.csv"):
        lines.append(f"- {name}")

    with (out_dir / "README_summary.txt").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print("Files written to:", out_dir)
    print("PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows")


if __name__ == "__main__":
    main()
