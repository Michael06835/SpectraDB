"""Rebuild and verify the UVVIS-v1 final summary from source artifacts.

This script is deliberately read-only with respect to ``runs/UVVIS-v1-FINAL``.
It rebuilds the seven text/CSV summaries from the three source run directories
and the Line B B1 metrics, then compares them with the frozen FINAL.

The four PNG assets in FINAL are not regenerated here (they depend on plotting
libraries and, for the Seed-42 figures, on the local training log).  ``--verify``
reports their frozen hashes and marks them as not rebuilt.

Usage:
    python scripts/build_uvvis_v1_corrected_summary.py --verify
    python scripts/build_uvvis_v1_corrected_summary.py --output runs/UVVIS-v1-REBUILD
    python scripts/build_uvvis_v1_corrected_summary.py --output ... --test-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FINAL_DIR = RUNS / "UVVIS-v1-FINAL"
LABEL_NAMES_PATH = ROOT / "processed" / "qm9s" / "prepared" / "functional_group_label_names.json"
B1_METRICS_PATH = RUNS / "LINE-B-METRICS" / "Shared14" / "model_metrics.csv"
B1_SEED_AUDIT_PATH = RUNS / "LINE-B-METRICS" / "Shared14" / "seed_audit.csv"
B1_INVENTORY_PATH = RUNS / "LINE-B-MANIFESTS" / "B1" / "prediction_inventory.csv"

RUN_NAMES = {
    "seed42": "EXP-UVVIS-001-FULL",
    "seed123": "EXP-UVVIS-001-SEED123",
    "seed2026": "EXP-UVVIS-001-SEED2026",
}
EXPECTED_SEEDS = {"seed42": 42, "seed123": 123, "seed2026": 2026}
TEST_METRICS = [
    "test_loss",
    "test_micro_f1",
    "test_macro_f1",
    "test_micro_precision",
    "test_micro_recall",
    "test_mAP",
    "test_macro_AUROC",
]
VALIDATION_METRICS = [
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_ap",
    "macro_auroc",
]
REBUILT_FILES = [
    "uvvis_final_summary.json",
    "UVVIS_FINAL_SUMMARY.md",
    "uvvis_seed_summary.csv",
    "uvvis_seed_summary.json",
    "uvvis_per_label_3seed_summary.csv",
    "uvvis_ablation_summary.csv",
    "README.md",
]
IMAGE_FILES = [
    "training_loss_seed42.png",
    "validation_metrics_seed42.png",
    "per_label_f1_3seed.png",
    "bce_weighting_comparison.png",
]


class ReproError(RuntimeError):
    """Raised when source inputs are missing or inconsistent."""


class MissingInputError(ReproError, FileNotFoundError):
    """Raised when a required source file is absent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise MissingInputError(f"Missing {description}: {path}")


def load_labels(root: Path) -> list[str]:
    require_file(root / LABEL_NAMES_PATH.relative_to(ROOT), "Shared14 label names")
    labels = read_json(root / LABEL_NAMES_PATH.relative_to(ROOT))
    if not isinstance(labels, list) or len(labels) != 14:
        raise ReproError(f"Expected 14 Shared14 labels, got {len(labels) if isinstance(labels, list) else type(labels)}")
    if len(set(labels)) != len(labels):
        raise ReproError("Shared14 label names contain duplicates")
    return labels


def validate_run(root: Path, seed_key: str, expected_seed: int, require_checkpoint: bool = True) -> dict:
    """Validate one seed directory and its config/test/labels/checkpoint."""
    run_name = RUN_NAMES[seed_key]
    run_dir = root / "runs" / run_name
    labels = load_labels(root)
    for name in ("config.json", "test_metrics.json", "test_per_label_metrics.csv"):
        require_file(run_dir / name, f"{run_name}/{name}")

    config = read_json(run_dir / "config.json")
    if config.get("experiment") != run_name:
        raise ReproError(f"{run_name}: config experiment={config.get('experiment')!r}")
    if config.get("seed") != expected_seed:
        raise ReproError(f"{run_name}: config seed={config.get('seed')!r}, expected {expected_seed}")
    if config.get("modality") != "uvvis":
        raise ReproError(f"{run_name}: config modality={config.get('modality')!r}")
    if config.get("label_names") not in (None, labels):
        raise ReproError(f"{run_name}: config label order mismatch")

    test = read_json(run_dir / "test_metrics.json")
    if test.get("experiment") != run_name:
        raise ReproError(f"{run_name}: test_metrics experiment={test.get('experiment')!r}")
    if test.get("modality") not in (None, "uvvis"):
        raise ReproError(f"{run_name}: test_metrics modality mismatch")
    for metric in TEST_METRICS:
        if metric not in test:
            raise ReproError(f"{run_name}: missing test metric {metric}")
        if not math.isfinite(float(test[metric])):
            raise ReproError(f"{run_name}: non-finite test metric {metric}")

    rows = read_csv(run_dir / "test_per_label_metrics.csv")
    if len(rows) != 14:
        raise ReproError(f"{run_name}: expected 14 per-label rows, got {len(rows)}")
    if [row.get("label") for row in rows] != labels:
        raise ReproError(f"{run_name}: per-label order does not match label names")
    supports = [int(row["positive_test_samples"]) for row in rows]
    if supports != [int(row["positive_test_samples"]) for row in read_csv(
        root / "runs" / RUN_NAMES["seed42"] / "test_per_label_metrics.csv"
    )]:
        raise ReproError(f"{run_name}: Test support differs from seed42")

    if require_checkpoint:
        require_file(run_dir / "best_model.pt", f"{run_name}/best_model.pt")
        import torch

        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        ckpt_config = checkpoint.get("config", {})
        if ckpt_config.get("seed") != expected_seed:
            raise ReproError(f"{run_name}: checkpoint seed={ckpt_config.get('seed')!r}, expected {expected_seed}")
        if ckpt_config.get("experiment") not in (None, run_name):
            raise ReproError(f"{run_name}: checkpoint experiment mismatch")
        if checkpoint.get("label_names") != labels:
            raise ReproError(f"{run_name}: checkpoint label order mismatch")

    return {"run_name": run_name, "config": config, "test": test, "rows": rows}


def validate_b1(root: Path, metrics_path: Path, require: bool = True):
    """Validate B1 UV validation provenance; return rows or None."""
    if not metrics_path.is_file():
        if require:
            raise MissingInputError(
                f"Missing Validation B1 source: {metrics_path}. "
                "Use --test-only for an explicitly incomplete Test-only summary."
            )
        return None

    rows = read_csv(metrics_path)
    uv_rows = [row for row in rows if row.get("modality") == "uvvis"]
    if len(uv_rows) != 3:
        raise ReproError(f"Expected 3 UV B1 rows, got {len(uv_rows)}")
    if {int(row["seed"]) for row in uv_rows} != {42, 123, 2026}:
        raise ReproError("B1 UV seeds are not exactly 42/123/2026")
    if {row["run_name"] for row in uv_rows} != set(RUN_NAMES.values()):
        raise ReproError("B1 UV run names do not match the three UVVIS runs")
    for row in uv_rows:
        if int(row["n_samples"]) != 12981 or int(row["n_labels"]) != 14:
            raise ReproError("B1 UV row has unexpected n_samples/n_labels")
        if float(row["threshold"]) != 0.5:
            raise ReproError("B1 UV row threshold is not 0.5")
        if row["seed_config"] != row["seed"] or row["seed_metadata_mismatch"].lower() != "false":
            raise ReproError("B1 UV seed metadata is inconsistent")

    # The aggregate CSV must be backed by the B1 audit inventory and the
    # per-seed prediction manifests, not only by the frozen FINAL.
    audit_path = root / B1_SEED_AUDIT_PATH.relative_to(ROOT)
    inventory_path = root / B1_INVENTORY_PATH.relative_to(ROOT)
    require_file(audit_path, "B1 seed audit")
    require_file(inventory_path, "B1 prediction inventory")
    audit_rows = [row for row in read_csv(audit_path) if row.get("modality") == "uvvis"]
    if len(audit_rows) != 3:
        raise ReproError("B1 seed audit does not contain exactly 3 UV rows")
    inv_rows = [row for row in read_csv(inventory_path) if row.get("modality") == "uvvis"]
    if len(inv_rows) != 3:
        raise ReproError("B1 prediction inventory does not contain exactly 3 UV rows")
    for inv in inv_rows:
        if inv.get("seed_config") != inv.get("seed_run_label"):
            raise ReproError("B1 inventory seed metadata mismatch")
        if int(inv.get("n_samples", -1)) != 12981:
            raise ReproError("B1 inventory n_samples mismatch")
        if inv.get("checks_all_true", "").lower() != "true":
            raise ReproError("B1 inventory checks are not all true")
        if inv.get("repeat_bitwise_equal", "").lower() != "true":
            raise ReproError("B1 inventory repeat check is not bitwise equal")
        pred_dir = root / Path(inv["prediction_dir"])
        manifest_path = pred_dir / "prediction_manifest.json"
        require_file(manifest_path, f"B1 manifest for {inv['run_name']}")
        manifest = read_json(manifest_path)
        if int(manifest["checkpoint"]["config"]["seed"]) != int(inv["seed_config"]):
            raise ReproError("B1 manifest checkpoint seed mismatch")
        if int(manifest.get("n_samples", -1)) != 12981:
            raise ReproError("B1 manifest n_samples mismatch")
        checks = manifest.get("checks", {})
        expected_false = {"test_split_accessed", "checkpoint_mutated", "bn_buffers_mutated"}
        if not checks:
            raise ReproError("B1 manifest has no checks")
        for key, value in checks.items():
            if key in expected_false:
                if value is not False:
                    raise ReproError(f"B1 manifest check {key} must be False")
            elif value is not True:
                raise ReproError(f"B1 manifest check {key} is not True")
        if "test_split_accessed" not in checks:
            raise ReproError("B1 manifest does not assert test_split_accessed")

    return uv_rows


def aggregate_validation(uv_rows) -> dict:
    validation = {}
    for metric in VALIDATION_METRICS:
        values = [float(row[metric]) for row in uv_rows]
        mean, sd = mean_sd(values)
        validation[metric] = {
            "mean": mean,
            "sample_sd": sd,
            "values": {row["seed"]: float(row[metric]) for row in uv_rows},
        }
    return validation


def build_ablation_rows(root: Path) -> list[dict]:
    baseline_path = root / "runs" / "EXP-UVVIS-001-FULL" / "test_metrics.json"
    weighted_path = root / "runs" / "EXP-UVVIS-002-FULL" / "test_metrics.json"
    threshold_path = root / "runs" / "EXP-UVVIS-001-FULL" / "threshold_calibration_summary.json"
    for path in (baseline_path, weighted_path, threshold_path):
        require_file(path, path.name)
    baseline = read_json(baseline_path)
    weighted = read_json(weighted_path)
    threshold = read_json(threshold_path)["test_calibrated"]
    return [
        {
            "experiment": "EXP-UVVIS-001-FULL",
            "description": "Ordinary BCE, threshold=0.5",
            "micro_f1": baseline["test_micro_f1"],
            "macro_f1": baseline["test_macro_f1"],
            "precision": baseline["test_micro_precision"],
            "recall": baseline["test_micro_recall"],
            "mAP": baseline["test_mAP"],
            "macro_AUROC": baseline["test_macro_AUROC"],
        },
        {
            "experiment": "EXP-UVVIS-002-FULL",
            "description": "sqrt(pos_weight), cap=20",
            "micro_f1": weighted["test_micro_f1"],
            "macro_f1": weighted["test_macro_f1"],
            "precision": weighted["test_micro_precision"],
            "recall": weighted["test_micro_recall"],
            "mAP": weighted["test_mAP"],
            "macro_AUROC": weighted["test_macro_AUROC"],
        },
        {
            "experiment": "EXP-UVVIS-001-CALIBRATED",
            "description": "Seed42 BCE + validation-calibrated thresholds",
            "micro_f1": threshold["micro_f1"],
            "macro_f1": threshold["macro_f1"],
            "precision": threshold["micro_precision"],
            "recall": threshold["micro_recall"],
            "mAP": baseline["test_mAP"],
            "macro_AUROC": baseline["test_macro_AUROC"],
        },
    ]


def write_ablation_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["experiment", "description", "micro_f1", "macro_f1", "precision", "recall", "mAP", "macro_AUROC"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {path}")
    else:
        path.mkdir(parents=True)


def build_summary(
    root: Path,
    output: Path,
    *,
    test_only: bool = False,
    b1_metrics_path: Path | None = None,
    require_checkpoint: bool = True,
) -> dict:
    """Build the 7 text/CSV summaries into ``output`` without touching FINAL."""
    labels = load_labels(root)
    runs = {
        seed: validate_run(root, seed, EXPECTED_SEEDS[seed], require_checkpoint=require_checkpoint)
        for seed in RUN_NAMES
    }
    b1_path = b1_metrics_path or (root / B1_METRICS_PATH.relative_to(ROOT))
    uv_rows = validate_b1(root, b1_path, require=not test_only)

    test = {seed: runs[seed]["test"] for seed in RUN_NAMES}
    per_label = {seed: runs[seed]["rows"] for seed in RUN_NAMES}

    summary = {}
    for metric in TEST_METRICS:
        values = [float(test[seed][metric]) for seed in RUN_NAMES]
        mean, sd = mean_sd(values)
        summary[metric] = {
            "mean": mean,
            "sample_sd": sd,
            "values": {seed: float(test[seed][metric]) for seed in RUN_NAMES},
        }

    seed_runs = {}
    for seed, run in RUN_NAMES.items():
        seed_runs[seed] = {
            "experiment": run,
            "modality": "uvvis",
            "best_epoch": test[seed].get("best_epoch"),
            "best_val_mAP": test[seed].get("best_val_mAP"),
            **{metric: test[seed][metric] for metric in TEST_METRICS},
            "checkpoint": str((root / "runs" / run / "best_model.pt").relative_to(root)),
            "checkpoint_sha256": sha256_file(root / "runs" / run / "best_model.pt"),
            "test_metrics_sha256": sha256_file(root / "runs" / run / "test_metrics.json"),
        }

    validation = aggregate_validation(uv_rows) if uv_rows else None
    per_label_rows = []
    for index, label in enumerate(labels):
        rows = {seed: per_label[seed][index] for seed in RUN_NAMES}
        if any(row["label"] != label for row in rows.values()):
            raise ReproError(f"Label order mismatch at {label}")
        out = {
            "label": label,
            "positive_test_samples_seed42": int(rows["seed42"]["positive_test_samples"]),
        }
        for field in ["precision", "recall", "f1", "average_precision", "auroc"]:
            for seed in RUN_NAMES:
                out[f"{field}_{seed}"] = float(rows[seed][field])
            mean, sd = mean_sd([out[f"{field}_{seed}"] for seed in RUN_NAMES])
            out[f"{field}_mean"] = mean
            out[f"{field}_sd"] = sd
        per_label_rows.append(out)

    prepare_output(output)
    payload = {
        "model_version": "UVVIS-v1",
        "status": "CURRENT_FINAL",
        "task": "QM9S UV-Vis -> 14 shared functional-group / structural-feature labels",
        "correction": {
            "reason": "Original EXP-UVVIS-001-SEED2026 was trained with SEED=42",
            "old_summary_snapshot": "archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/superseded_uvvis_v1_final/",
            "new_seed2026_checkpoint_sha256": seed_runs["seed2026"]["checkpoint_sha256"],
            "new_seed2026_test_metrics_sha256": seed_runs["seed2026"]["test_metrics_sha256"],
            "corrected_at": "2026-11-18",
        },
        "architecture": {
            "backbone": "1D-CNN / residual 1D CNN",
            "embedding_dim": 256,
            "input_points": 701,
            "normalization": "max",
            "loss": "BCEWithLogitsLoss",
            "threshold": 0.5,
        },
        "three_seed_test_summary": summary,
        "seed_runs": seed_runs,
        "validation_three_seed_summary_fp32": validation,
        "validation_source": "runs/LINE-B-METRICS/Shared14/model_metrics.csv" if validation else None,
        "test_source": {
            seed: str((root / "runs" / run / "test_metrics.json").relative_to(root))
            for seed, run in RUN_NAMES.items()
        },
        "per_label_source": {
            seed: str((root / "runs" / run / "test_per_label_metrics.csv").relative_to(root))
            for seed, run in RUN_NAMES.items()
        },
    }
    if test_only:
        payload["validation_status"] = "TEST_ONLY_MISSING_B1"

    (output / "uvvis_final_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output / "uvvis_seed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "seed42", "seed123", "seed2026", "mean", "sample_sd"])
        for metric in TEST_METRICS:
            values = summary[metric]["values"]
            w.writerow([
                metric, values["seed42"], values["seed123"], values["seed2026"],
                summary[metric]["mean"], summary[metric]["sample_sd"],
            ])

    (output / "uvvis_seed_summary.json").write_text(
        json.dumps({metric: summary[metric] for metric in TEST_METRICS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = ["label", "positive_test_samples_seed42"]
    for field in ["precision", "recall", "f1", "average_precision", "auroc"]:
        fieldnames.extend([f"{field}_{seed}" for seed in RUN_NAMES])
        fieldnames.extend([f"{field}_mean", f"{field}_sd"])
    with (output / "uvvis_per_label_3seed_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(per_label_rows)

    write_ablation_csv(output / "uvvis_ablation_summary.csv", build_ablation_rows(root))

    validation_section = ""
    if validation:
        validation_section = f"""
## Validation (B1 FP32 predictions, 3 seeds)

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | {validation['micro_f1']['mean']:.4f} ± {validation['micro_f1']['sample_sd']:.4f} |
| Macro-F1 | {validation['macro_f1']['mean']:.4f} ± {validation['macro_f1']['sample_sd']:.4f} |
| Macro-AP | {validation['macro_ap']['mean']:.4f} ± {validation['macro_ap']['sample_sd']:.4f} |
| Macro-AUROC | {validation['macro_auroc']['mean']:.4f} ± {validation['macro_auroc']['sample_sd']:.4f} |
"""
    else:
        validation_section = """
## Validation

Not available: B1 Validation metrics were deliberately omitted in `--test-only`
mode. This output is a Test-only summary and is not a complete FINAL rebuild.
"""

    md = f"""# UVVIS-v1 final 3-seed summary

Status: **CURRENT_FINAL**

The original `EXP-UVVIS-001-SEED2026` was trained with `SEED=42`; the archived
pre-correction snapshot is preserved under
`archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/superseded_uvvis_v1_final/`.
The corrected run uses the same experiment ID, `SEED=2026`, and the original
training/evaluation protocol.

## Test (historical AMP evaluation, 3 seeds)

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | {summary['test_micro_f1']['mean']:.4f} ± {summary['test_micro_f1']['sample_sd']:.4f} |
| Macro-F1 | {summary['test_macro_f1']['mean']:.4f} ± {summary['test_macro_f1']['sample_sd']:.4f} |
| Micro-Precision | {summary['test_micro_precision']['mean']:.4f} ± {summary['test_micro_precision']['sample_sd']:.4f} |
| Micro-Recall | {summary['test_micro_recall']['mean']:.4f} ± {summary['test_micro_recall']['sample_sd']:.4f} |
| mAP | {summary['test_mAP']['mean']:.4f} ± {summary['test_mAP']['sample_sd']:.4f} |
| Macro-AUROC | {summary['test_macro_AUROC']['mean']:.4f} ± {summary['test_macro_AUROC']['sample_sd']:.4f} |
{validation_section}
The historical Test summary is separate from Line B B4's unified FP32 Test
evaluation. No Test result was used to select model, loss, threshold, BN or
hyperparameters.
"""
    (output / "UVVIS_FINAL_SUMMARY.md").write_text(md, encoding="utf-8")
    if test_only:
        readme = (
            "UVVIS-v1 Test-only summary generated from actual test_metrics.json, "
            "test_per_label_metrics.csv and threshold_calibration_summary.json. "
            "B1 Validation metrics were omitted. No training or inference was performed.\n"
        )
    else:
        readme = (
            "UVVIS-v1 final summary generated from actual test_metrics.json and "
            "test_per_label_metrics.csv of seeds 42/123/2026. "
            "No training or inference was performed.\n"
        )
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "output": str(output),
        "files": REBUILT_FILES,
        "test_only": test_only,
        "validation_verified": validation is not None,
        "validation_status": "VERIFIED" if validation else "TEST_ONLY_MISSING_B1",
    }


def _semantic_equal(name: str, a: Path, b: Path) -> bool:
    if a.read_bytes() == b.read_bytes():
        return True
    if name.endswith(".json"):
        return read_json(a) == read_json(b)
    if name.endswith(".csv"):
        return read_csv(a) == read_csv(b) and a.read_text(encoding="utf-8-sig") == b.read_text(encoding="utf-8-sig")
    return False


def verify(root: Path = ROOT) -> dict:
    final = root / "runs" / "UVVIS-v1-FINAL"
    missing = [name for name in REBUILT_FILES + IMAGE_FILES if not (final / name).is_file()]
    if missing:
        return {"status": "FAIL", "reason": "missing FINAL files", "missing": missing, "complete": False}

    with tempfile.TemporaryDirectory(prefix="uvvis-v1-verify-") as tmp:
        report = build_summary(root, Path(tmp))
        rebuilt_dir = Path(tmp)
        comparisons = {}
        for name in REBUILT_FILES:
            built = rebuilt_dir / name
            frozen = final / name
            comparisons[name] = {
                "built_sha256": sha256_file(built),
                "frozen_sha256": sha256_file(frozen),
                "byte_equal": built.read_bytes() == frozen.read_bytes(),
                "size_equal": built.stat().st_size == frozen.stat().st_size,
                "semantic_equal": _semantic_equal(name, built, frozen),
            }
        images = {}
        for name in IMAGE_FILES:
            path = final / name
            images[name] = {
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else None,
                "rebuilt": False,
                "note": "frozen image; not regenerated by this summary builder",
            }

    summary_ok = all(item["byte_equal"] for item in comparisons.values())
    return {
        "status": "PASS" if summary_ok else "FAIL",
        "complete": summary_ok and report["validation_verified"],
        "summary_rebuild_complete": summary_ok and report["validation_verified"],
        "byte_reproduction_complete": False,
        "complete_definition": "7 text/CSV summaries plus Validation provenance; 4 PNGs are frozen-only",
        "images_rebuilt": False,
        "validation_verified": report["validation_verified"],
        "rebuilt_files": comparisons,
        "images_not_rebuilt": images,
        "all_11_final_files_present": True,
        "scope": {
            "byte_rebuilt": REBUILT_FILES,
            "not_rebuilt": IMAGE_FILES,
            "requires_local_checkpoint": True,
            "requires_b1_validation": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true", help="Rebuild in a temporary directory and compare with FINAL")
    group.add_argument("--output", type=Path, help="Write rebuilt summaries into a new/empty directory")
    parser.add_argument("--test-only", action="store_true", help="Explicitly omit B1 Validation when it is unavailable")
    args = parser.parse_args(argv)

    if args.verify:
        if args.test_only:
            parser.error("--verify requires full Validation; use --output --test-only for a Test-only summary")
        report = verify(ROOT)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["complete"] else 1

    try:
        report = build_summary(ROOT, args.output, test_only=args.test_only)
    except (ReproError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
