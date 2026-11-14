"""Threshold calibration of saved BN-recalibrated UVSPEC15 checkpoints.

From SpectraDB root:
    python training/calibrate_uvspec15_bn_thresholds.py --run 003 --seed 42

Optional: --run 003 --seed 123/2026, or --run 004 --seed 42/123/2026.
This is *only* an inference pass: per-label thresholds are fitted on Validation
and then evaluated on Test. Checkpoints and earlier threshold files are READ-ONLY.
The procedure matches calibrate_uvvis_uvspec15_thresholds.py: maximize each
label's validation F1 using precision_recall_curve thresholds; 0.5 baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, precision_recall_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel

PREPARED = ROOT / "processed" / "qm9s" / "prepared"
UVVIS_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "uvspec15_labels.npy"
LABEL_NAMES_PATH = PREPARED / "uvspec15_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"
BN_OUTPUT_ROOT = ROOT / "runs" / "UVSPEC15-BN-RECALIBRATED"

RUN_IDS = {
    "003": {42: "EXP-UVVIS-003-UVSPEC15-FULL",
            123: "EXP-UVVIS-003-UVSPEC15-SEED123",
            2026: "EXP-UVVIS-003-UVSPEC15-SEED2026"},
    "004": {42: "EXP-UVVIS-004-UVSPEC15-WBCE",
             123: "EXP-UVVIS-004-UVSPEC15-WBCE-SEED123",
             2026: "EXP-UVVIS-004-UVSPEC15-WBCE-SEED2026"},
}
NUM_WORKERS = 4
BATCH_SIZE = 512
DEFAULT_THRESHOLD = 0.5
OUTPUT_FILES = (
    "optimal_thresholds.json",
    "validation_threshold_comparison.csv",
    "threshold_comparison.csv",
    "test_per_label_metrics_calibrated.csv",
    "threshold_calibration_summary.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect(model, loader, device, use_amp):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x)["logits"]

            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())

    return (
        np.concatenate(ys).astype(np.int32),
        np.concatenate(ps).astype(np.float32),
    )


def overall_metrics(y_true, probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if thresholds.ndim == 0:
        thresholds = np.full(
            y_true.shape[1],
            float(thresholds),
            dtype=np.float32,
        )

    pred = (probs >= thresholds[None, :]).astype(np.int32)

    return {
        "micro_f1": float(
            f1_score(y_true, pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(y_true, pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, pred, average="micro", zero_division=0)
        ),
    }


def threshold_free_metrics(y_true, probs):
    aps, aucs = [], []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        ps = probs[:, j]

        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)

        if n_pos == 0:
            ap = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ap = average_precision_score(yt, ps)

        if n_pos == 0 or n_neg == 0:
            auc = np.nan
        else:
            auc = roc_auc_score(yt, ps)

        aps.append(ap)
        aucs.append(auc)

    return {
        "mAP": float(np.nanmean(aps)),
        "macro_AUROC": float(np.nanmean(aucs)),
    }


def optimize_threshold(y_true, probs, fallback=0.5):
    n_pos = int(y_true.sum())

    if n_pos == 0 or n_pos == len(y_true):
        return float(fallback), np.nan

    p, r, thresholds = precision_recall_curve(y_true, probs)

    if len(thresholds) == 0:
        return float(fallback), np.nan

    p = p[:-1]
    r = r[:-1]

    denom = p + r
    f1 = np.divide(
        2 * p * r,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )

    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def per_label_metrics(y_true, probs, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if thresholds.ndim == 0:
        thresholds = np.full(
            y_true.shape[1],
            float(thresholds),
            dtype=np.float32,
        )

    rows = []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        pred = (probs[:, j] >= thresholds[j]).astype(np.int32)

        rows.append({
            "positive_samples": int(yt.sum()),
            "negative_samples": int(len(yt) - yt.sum()),
            "threshold": float(thresholds[j]),
            "precision": float(
                precision_score(yt, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(yt, pred, zero_division=0)
            ),
            "f1": float(
                f1_score(yt, pred, zero_division=0)
            ),
        })

    return rows



def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only threshold fitting for UVSPEC15 BN checkpoint")
    parser.add_argument("--run", choices=["003", "004"], default="003")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    args = parser.parse_args()
    if args.seed not in RUN_IDS[args.run]:
        parser.error(f"No configured checkpoint for run={args.run}, seed={args.seed}")

    run_name = RUN_IDS[args.run][args.seed]
    bn_dir = BN_OUTPUT_ROOT / run_name
    checkpoint_path = bn_dir / "best_model_bn_recalibrated.pt"
    manifest_path = bn_dir / "bn_recalibration_manifest.json"
    source_path = ROOT / "runs" / run_name / "best_model.pt"
    out_dir = bn_dir / "threshold_calibration"

    required_files = (UVVIS_PATH, LABEL_PATH, LABEL_NAMES_PATH, SPLIT_PATH,
                      checkpoint_path, manifest_path, source_path)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    if out_dir.exists():
        raise FileExistsError(f"Calibration output already exists; refusing to overwrite: {out_dir}")

    names = json.loads(LABEL_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != 15 or len(set(names)) != 15:
        raise ValueError("Expected 15 unique UVSPEC15 labels")
    with np.load(SPLIT_PATH) as split:
        train_idx, val_idx, test_idx = (
            np.asarray(split[k], dtype=np.int64) for k in ("train", "val", "test")
        )
    if tuple(map(len, (train_idx, val_idx, test_idx))) != (103853, 12981, 12981):
        raise RuntimeError("Unexpected UVSPEC15 scaffold split sizes")
    if np.load(LABEL_PATH, mmap_mode="r").shape != (129817, 15):
        raise RuntimeError("UVSPEC15 label matrix is not (129817, 15)")
    if np.load(UVVIS_PATH, mmap_mode="r").shape != (129817, 701):
        raise RuntimeError("UVVIS matrix is not (129817, 701)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum = sha256_file(checkpoint_path)
    if manifest.get("saved_sha256") != checksum:
        raise RuntimeError("Saved BN checkpoint SHA-256 differs from its manifest")
    if manifest.get("source_sha256") != sha256_file(source_path):
        raise RuntimeError("Source checkpoint differs from the file recorded during BN saving")
    if int(manifest.get("seed", -1)) != args.seed or int(manifest.get("train_samples_used", -1)) != len(train_idx):
        raise RuntimeError("BN manifest seed or train sample count mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    bn_meta = checkpoint.get("bn_recalibration", {})
    if (int(config.get("seed", -1)) != args.seed
            or int(bn_meta.get("seed", -1)) != args.seed
            or bn_meta.get("source_sha256") != manifest.get("source_sha256")):
        raise RuntimeError("Checkpoint / BN metadata / requested seed mismatch")
    if config.get("label_names") != names or checkpoint.get("label_names") != names:
        raise RuntimeError("UVSPEC15 labels or order mismatch")
    if (int(config.get("batch_size", -1)) != BATCH_SIZE
            or int(config.get("num_workers", -1)) != NUM_WORKERS
            or int(config.get("embedding_dim", -1)) != 256
            or config.get("normalization") != "max"
            or float(config.get("threshold", -1)) != DEFAULT_THRESHOLD):
        raise RuntimeError("BN checkpoint differs from established UVSPEC15 training configuration")
    if bool(config.get("use_pos_weight")) != (args.run == "004"):
        raise RuntimeError("Wrong BCE versus Weighted BCE checkpoint")
    if args.run == "004" and (config.get("pos_weight_transform") != "sqrt"
            or float(config.get("pos_weight_cap", -1)) != 20.0):
        raise RuntimeError("Weighted BCE parameters mismatch")

    model = SpectrumStructureModel(num_labels=len(names), embedding_dim=256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    def loader(indices):
        dataset = QM9SSpectrumDataset(
            spectra_path=UVVIS_PATH,
            labels_path=LABEL_PATH,
            indices=indices,
            normalization="max",
        )
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
            persistent_workers=True, prefetch_factor=2,
        )

    print("=" * 85)
    print(f"{run_name} | BN-recalibrated checkpoint -> Threshold Calibration")
    print("=" * 85)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Seed: {args.seed}; Device: {device}; Batch size: {BATCH_SIZE}; Workers: {NUM_WORKERS}")
    print(f"Validation: {len(val_idx)}; Test: {len(test_idx)}")
    print("Optimizing each threshold from Validation ONLY; original checkpoints are read-only")
    print("Collecting validation probabilities...")
    val_y, val_p = collect(model, loader(val_idx), device, device.type == "cuda")
    print("Collecting test probabilities...")
    test_y, test_p = collect(model, loader(test_idx), device, device.type == "cuda")

    thresholds, val_best_f1 = [], []
    print("\n" + "=" * 85)
    print("VALIDATION-OPTIMIZED PER-LABEL THRESHOLDS")
    print("=" * 85)
    print(f"{'label':47s}{'positive':>10s}{'threshold':>12s}{'best_F1':>12s}")
    for j, name in enumerate(names):
        th, f1 = optimize_threshold(val_y[:, j], val_p[:, j], fallback=DEFAULT_THRESHOLD)
        thresholds.append(th)
        val_best_f1.append(f1)
        print(f"{name:47s}{int(val_y[:, j].sum()):10d}{th:12.4f}{f1:12.4f}")
    thresholds = np.asarray(thresholds, dtype=np.float32)

    val_default = overall_metrics(val_y, val_p, DEFAULT_THRESHOLD)
    val_calibrated = overall_metrics(val_y, val_p, thresholds)
    test_default = overall_metrics(test_y, test_p, DEFAULT_THRESHOLD)
    test_calibrated = overall_metrics(test_y, test_p, thresholds)
    val_free = threshold_free_metrics(val_y, val_p)
    test_free = threshold_free_metrics(test_y, test_p)

    # Diagnostic cross-check: no reliance on the ORIGINAL baseline test_metrics.json,
    # which describes an entirely different BN state. Never optimize on Test.
    diagnostic_file = (ROOT / "runs" / "UVSPEC15-BN-DIAGNOSTICS"
                       / run_name / "batchnorm_full_diagnostic.json")
    diagnostic_compare = None
    if diagnostic_file.is_file():
        diagnostic = json.loads(diagnostic_file.read_text(encoding="utf-8"))
        expected = diagnostic.get("after", {}).get("validation", {})
        mAP_expected = expected.get("mAP")
        if mAP_expected is not None:
            delta = float(val_free["mAP"] - float(mAP_expected))
            diagnostic_compare = {"prior_val_mAP": float(mAP_expected),
                                  "saved_checkpoint_val_mAP": val_free["mAP"],
                                  "difference": delta}
            if abs(delta) > 0.003:
                raise RuntimeError(
                    f"Saved BN checkpoint Validation mAP differs from prior BN diagnostic "
                    f"by {delta:+.4f}. Investigate before calibration; no results saved."
                )
            print(f"PASS: saved BN checkpoint reproduces prior Validation mAP (difference {delta:+.6f})")

    for split_name, default, calibrated, free in (
        ("VALIDATION", val_default, val_calibrated, val_free),
        ("TEST", test_default, test_calibrated, test_free),
    ):
        print("\n" + "=" * 85)
        print(f"{split_name} COMPARISON")
        print("=" * 85)
        for title, vals in (("Default 0.5", default), ("Calibrated", calibrated)):
            print(f"{title:13s}: Micro-F1={vals['micro_f1']:.4f} | "
                  f"Macro-F1={vals['macro_f1']:.4f} | "
                  f"Precision={vals['micro_precision']:.4f} | "
                  f"Recall={vals['micro_recall']:.4f}")
        print(f"Threshold-free : mAP={free['mAP']:.4f} | AUROC={free['macro_AUROC']:.4f}")

    val_d = per_label_metrics(val_y, val_p, DEFAULT_THRESHOLD)
    val_c = per_label_metrics(val_y, val_p, thresholds)
    test_d = per_label_metrics(test_y, test_p, DEFAULT_THRESHOLD)
    test_c = per_label_metrics(test_y, test_p, thresholds)

    # Independent subfolder: neither the source run nor existing calibrated outputs are touched.
    out_dir.mkdir(parents=True, exist_ok=False)
    th_output = {
        "experiment": run_name,
        "seed": args.seed,
        "source": "validation split only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checksum,
        "bn_recalibrated": True,
        "default_threshold": DEFAULT_THRESHOLD,
        "thresholds": {n: float(thresholds[i]) for i, n in enumerate(names)},
        "validation_best_f1": {n: (None if np.isnan(val_best_f1[i]) else float(val_best_f1[i]))
                               for i, n in enumerate(names)},
    }
    with (out_dir / OUTPUT_FILES[0]).open("x", encoding="utf-8") as f:
        json.dump(th_output, f, ensure_ascii=False, indent=2)

    def save_comparison(filename, rows_default, rows_calibrated, split_name):
        with (out_dir / filename).open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label_index", "label", f"positive_{split_name}_samples",
                "optimal_threshold", f"{split_name}_best_F1" if split_name == "val" else "val_best_F1",
                "default_precision", "default_recall", "default_f1",
                "calibrated_precision", "calibrated_recall", "calibrated_f1", "f1_change",
            ])
            for j, name in enumerate(names):
                d = rows_default[j]
                c = rows_calibrated[j]
                writer.writerow([
                    j, name, d["positive_samples"], float(thresholds[j]),
                    "" if np.isnan(val_best_f1[j]) else float(val_best_f1[j]),
                    d["precision"], d["recall"], d["f1"],
                    c["precision"], c["recall"], c["f1"], c["f1"] - d["f1"],
                ])

    save_comparison(OUTPUT_FILES[1], val_d, val_c, "val")
    save_comparison(OUTPUT_FILES[2], test_d, test_c, "test")

    with (out_dir / OUTPUT_FILES[3]).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_index", "label", "positive_test_samples", "negative_test_samples",
                         "threshold", "precision", "recall", "f1"])
        for j, name in enumerate(names):
            v = test_c[j]
            writer.writerow([j, name, v["positive_samples"], v["negative_samples"],
                             v["threshold"], v["precision"], v["recall"], v["f1"]])

    summary = {
        "experiment": run_name,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checksum,
        "label_set": "UVSPEC15",
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "model_postprocessing": "TRAIN-only BatchNorm buffer recalibration",
        "calibration": "Per-label F1 optimization on Validation split only",
        "default_threshold": DEFAULT_THRESHOLD,
        "validation_default": val_default,
        "validation_calibrated": val_calibrated,
        "validation_threshold_free": val_free,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "test_threshold_free": test_free,
        "prior_bn_diagnostic_val_check": diagnostic_compare,
        "note": "Test never used to optimize thresholds. mAP/AUROC are threshold-free.",
    }
    with (out_dir / OUTPUT_FILES[4]).open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 85)
    print("Threshold calibration completed. Saved (existing files not overwritten):")
    print("=" * 85)
    for filename in OUTPUT_FILES:
        print("  ", out_dir / filename)
    print("Original and BN-recalibrated checkpoints were NOT modified.")


if __name__ == "__main__":
    main()
