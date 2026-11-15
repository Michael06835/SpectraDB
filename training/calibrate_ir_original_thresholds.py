"""IR original-checkpoint threshold calibration (UVSPEC15 v1 pipeline).

Parameterized IR port of training/calibrate_uvvis_uvspec15_thresholds.py, the
validated UVSPEC15 v1 original-checkpoint calibration. Only the data interface,
label set, paths, experiment mapping and output location differ.

No training, no BN recalibration, no change to model weights, label schema or
dataset split. Class thresholds are optimized ONLY on validation and then
frozen; Test is evaluated afterwards and never used to fit or select anything.

Outputs go to a NEW directory (runs/IR-THRESHOLD-CALIBRATION/<run_name>/) so the
historical BCE Seed 42 artifacts in runs/EXP-IR-001-FULL/ stay untouched.

Usage (from the SpectraDB root in the spectra-db environment), one model per run:
  python training/calibrate_ir_original_thresholds.py --loss bce  --seed 42
  python training/calibrate_ir_original_thresholds.py --loss bce  --seed 123
  python training/calibrate_ir_original_thresholds.py --loss bce  --seed 2026
  python training/calibrate_ir_original_thresholds.py --loss wbce --seed 42
  python training/calibrate_ir_original_thresholds.py --loss wbce --seed 123
  python training/calibrate_ir_original_thresholds.py --loss wbce --seed 2026
"""

from pathlib import Path
import argparse
import hashlib
import sys, json, csv, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel

PREPARED = ROOT / "processed" / "qm9s" / "prepared"
IR_PATH = PREPARED / "ir_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "ir_scaffold_split_valid.npz"

OUTPUT_ROOT = ROOT / "runs" / "IR-THRESHOLD-CALIBRATION"

RUN_IDS = {
    "bce": {
        42: "EXP-IR-001-FULL",
        123: "EXP-IR-001-SEED123",
        2026: "EXP-IR-001-SEED2026",
    },
    "wbce": {
        42: "EXP-IR-002-FULL",
        123: "EXP-IR-002-SEED123",
        2026: "EXP-IR-002-SEED2026",
    },
}

NUM_WORKERS = 4  # Deliberately identical to the original IR training runs.
BATCH_SIZE = 512
EMBEDDING_DIM = 256
NORMALIZATION = "max"
DEFAULT_THRESHOLD = 0.5

EXPECTED_N_LABELS = 14
EXPECTED_SPECTRA_SHAPE = (129817, 3501)
EXPECTED_LABEL_SHAPE = (129817, 14)
EXPECTED_SPLIT_SIZES = (103855, 12981, 12981)

BASELINE_ATOL = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_checkpoint(checkpoint, config, run_name, loss_key, seed, label_names):
    """Refuse to calibrate a checkpoint that does not match the requested run."""
    problems = []

    if len(label_names) != EXPECTED_N_LABELS or len(set(label_names)) != EXPECTED_N_LABELS:
        problems.append(f"expected {EXPECTED_N_LABELS} unique IR labels, got {len(label_names)}")
    if checkpoint.get("label_names") != label_names:
        problems.append("checkpoint label_names do not match the IR label order")
    if config.get("label_names") != label_names:
        problems.append("checkpoint config label_names mismatch")
    if config.get("experiment") != run_name:
        problems.append(
            f"checkpoint experiment {config.get('experiment')!r} != {run_name!r}"
        )
    if int(config.get("seed", -1)) != seed:
        problems.append(f"checkpoint seed {config.get('seed')} != {seed}")
    if bool(config.get("use_pos_weight")) != (loss_key == "wbce"):
        problems.append("bce/wbce ordinary vs weighted BCE setup does not match")
    if loss_key == "wbce":
        if config.get("pos_weight_transform") != "sqrt":
            problems.append("wbce checkpoint is missing pos_weight_transform='sqrt'")
        if float(config.get("pos_weight_cap", -1)) != 20.0:
            problems.append("wbce checkpoint is missing pos_weight_cap=20.0")
    if int(config.get("num_workers", -1)) != NUM_WORKERS:
        problems.append(f"num_workers {config.get('num_workers')} != {NUM_WORKERS}")
    if int(config.get("batch_size", -1)) != BATCH_SIZE:
        problems.append(f"batch_size {config.get('batch_size')} != {BATCH_SIZE}")
    if int(config.get("embedding_dim", -1)) != EMBEDDING_DIM:
        problems.append(f"embedding_dim {config.get('embedding_dim')} != {EMBEDDING_DIM}")
    if config.get("normalization") != NORMALIZATION:
        problems.append(f"normalization {config.get('normalization')!r} != {NORMALIZATION!r}")
    if float(config.get("threshold", -1)) != DEFAULT_THRESHOLD:
        problems.append(f"default threshold {config.get('threshold')} != {DEFAULT_THRESHOLD}")

    if problems:
        raise RuntimeError(
            "Checkpoint verification failed:\n  " + "\n  ".join(problems)
        )


def main():
    parser = argparse.ArgumentParser(
        description="IR original-checkpoint threshold calibration (UVSPEC15 v1 flow)."
    )
    parser.add_argument("--loss", choices=["bce", "wbce"], default="bce")
    parser.add_argument("--seed", type=int, choices=[42, 123, 2026], default=42)
    args = parser.parse_args()

    if args.seed not in RUN_IDS[args.loss]:
        parser.error(
            f"loss={args.loss} has no configured seed {args.seed} "
            f"(available: {sorted(RUN_IDS[args.loss])})"
        )

    run_name = RUN_IDS[args.loss][args.seed]
    run_dir = ROOT / "runs" / run_name
    checkpoint_path = run_dir / "best_model.pt"
    baseline_path = run_dir / "test_metrics.json"
    out_dir = OUTPUT_ROOT / run_name

    required = [IR_PATH, LABEL_PATH, LABEL_NAMES_PATH, SPLIT_PATH, checkpoint_path, baseline_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    # Fail closed: never overwrite an existing calibration directory.
    if out_dir.exists():
        raise FileExistsError(
            f"Calibration output already exists: {out_dir}. "
            "Nothing was overwritten. Inspect or relocate it manually."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print()
    print("=" * 80)
    print(f"{run_name} | original-checkpoint threshold calibration")
    print("=" * 80)
    print()
    print("Device :", device)
    if device.type == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))
        print("CUDA   :", torch.version.cuda)
    print("PyTorch:", torch.__version__)
    print("AMP    :", use_amp)
    print("Workers:", NUM_WORKERS)

    with open(LABEL_NAMES_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    split = np.load(SPLIT_PATH)
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    spectra_check = np.load(IR_PATH, mmap_mode="r")
    label_check = np.load(LABEL_PATH, mmap_mode="r")
    if tuple(spectra_check.shape) != EXPECTED_SPECTRA_SHAPE:
        raise ValueError(f"Unexpected IR matrix shape: {spectra_check.shape}")
    if tuple(label_check.shape) != EXPECTED_LABEL_SHAPE:
        raise ValueError(f"Unexpected IR label matrix shape: {label_check.shape}")
    if (len(train_idx), len(val_idx), len(test_idx)) != EXPECTED_SPLIT_SIZES:
        raise ValueError(
            f"Unexpected IR scaffold split sizes: "
            f"{(len(train_idx), len(val_idx), len(test_idx))}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    verify_checkpoint(checkpoint, config, run_name, args.loss, args.seed, label_names)

    batch_size = int(config["batch_size"])
    embedding_dim = int(config["embedding_dim"])
    normalization = config["normalization"]
    default_threshold = float(config["threshold"])
    source_sha256 = sha256_file(checkpoint_path)

    print()
    print("Validation samples:", len(val_idx))
    print("Test samples      :", len(test_idx))
    print("Checkpoint        :", checkpoint_path)
    print("Checkpoint sha256 :", source_sha256)
    print("Loss / Seed       :", args.loss, "/", args.seed)
    print("Best epoch        :", checkpoint.get("epoch", "unknown"))
    print("Best val mAP      :", checkpoint.get("best_val_mAP", "unknown"))
    print("Batch size        :", batch_size)
    print("Normalization     :", normalization)
    print("Default threshold :", default_threshold)
    print("PASS: checkpoint source, seed, loss config and label order verified")

    val_ds = QM9SSpectrumDataset(
        spectra_path=IR_PATH,
        labels_path=LABEL_PATH,
        indices=val_idx,
        normalization=normalization,
    )

    test_ds = QM9SSpectrumDataset(
        spectra_path=IR_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization=normalization,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
        "prefetch_factor": 2,
    }

    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    model = SpectrumStructureModel(
        num_labels=len(label_names),
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    print()
    print("Collecting validation probabilities...")
    val_y, val_p = collect(model, val_loader, device, use_amp)

    print("Collecting test probabilities...")
    test_y, test_p = collect(model, test_loader, device, use_amp)

    thresholds = []
    val_best_f1 = []

    print()
    print("=" * 80)
    print("VALIDATION-OPTIMIZED PER-LABEL THRESHOLDS")
    print("=" * 80)
    print()
    print(f"{'label':16s}{'positive':>10s}{'threshold':>12s}{'best_F1':>12s}")
    print("-" * 50)

    for j, name in enumerate(label_names):
        threshold, best_f1 = optimize_threshold(
            val_y[:, j],
            val_p[:, j],
            fallback=default_threshold,
        )

        thresholds.append(threshold)
        val_best_f1.append(best_f1)

        best_text = "nan" if np.isnan(best_f1) else f"{best_f1:.4f}"

        print(
            f"{name:16s}"
            f"{int(val_y[:, j].sum()):10d}"
            f"{threshold:12.4f}"
            f"{best_text:>12s}"
        )

    thresholds = np.asarray(thresholds, dtype=np.float32)

    # The thresholds are fitted strictly on Validation and persisted BEFORE any
    # Test metric is computed, so the record proves Test could not have influenced them.
    out_dir.mkdir(parents=True, exist_ok=False)

    thresholds_json = {
        "experiment": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": source_sha256,
        "source": "validation split only",
        "default_threshold": default_threshold,
        "thresholds": {
            name: float(thresholds[j])
            for j, name in enumerate(label_names)
        },
        "validation_best_f1": {
            name: (
                None
                if np.isnan(val_best_f1[j])
                else float(val_best_f1[j])
            )
            for j, name in enumerate(label_names)
        },
        "frozen_before_test_metrics": True,
    }

    with open(out_dir / "optimal_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_json, f, ensure_ascii=False, indent=2)

    print()
    print("Thresholds frozen on Validation and saved before Test evaluation:")
    print(" ", out_dir / "optimal_thresholds.json")

    val_default = overall_metrics(val_y, val_p, default_threshold)
    val_calibrated = overall_metrics(val_y, val_p, thresholds)
    test_default = overall_metrics(test_y, test_p, default_threshold)
    test_calibrated = overall_metrics(test_y, test_p, thresholds)
    test_free = threshold_free_metrics(test_y, test_p)

    # Independent reproducibility check against the frozen baseline.
    # Do NOT use the test scores to select or change calibration thresholds.
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_checks = {
        "test_micro_f1": test_default["micro_f1"],
        "test_macro_f1": test_default["macro_f1"],
        "test_micro_precision": test_default["micro_precision"],
        "test_micro_recall": test_default["micro_recall"],
        "test_mAP": test_free["mAP"],
        "test_macro_AUROC": test_free["macro_AUROC"],
    }
    for key, observed in baseline_checks.items():
        if key not in baseline:
            raise ValueError(f"Frozen baseline is missing {key}")
        if not np.isclose(observed, float(baseline[key]), rtol=0, atol=BASELINE_ATOL):
            raise RuntimeError(
                f"Baseline reproduction check failed for {key}: "
                f"checkpoint={observed:.6f}, recorded={float(baseline[key]):.6f}. "
                "Check labels, indices, preprocessing and inference before proceeding."
            )
    print("PASS: frozen baseline reproduced with threshold=0.5")

    print()
    print("=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)
    print()
    print(
        "Default 0.5 : "
        f"Micro-F1={val_default['micro_f1']:.4f} | "
        f"Macro-F1={val_default['macro_f1']:.4f} | "
        f"Precision={val_default['micro_precision']:.4f} | "
        f"Recall={val_default['micro_recall']:.4f}"
    )
    print(
        "Calibrated  : "
        f"Micro-F1={val_calibrated['micro_f1']:.4f} | "
        f"Macro-F1={val_calibrated['macro_f1']:.4f} | "
        f"Precision={val_calibrated['micro_precision']:.4f} | "
        f"Recall={val_calibrated['micro_recall']:.4f}"
    )

    print()
    print("=" * 80)
    print("TEST COMPARISON")
    print("=" * 80)
    print()
    print(
        "Default 0.5 : "
        f"Micro-F1={test_default['micro_f1']:.4f} | "
        f"Macro-F1={test_default['macro_f1']:.4f} | "
        f"Precision={test_default['micro_precision']:.4f} | "
        f"Recall={test_default['micro_recall']:.4f}"
    )
    print(
        "Calibrated  : "
        f"Micro-F1={test_calibrated['micro_f1']:.4f} | "
        f"Macro-F1={test_calibrated['macro_f1']:.4f} | "
        f"Precision={test_calibrated['micro_precision']:.4f} | "
        f"Recall={test_calibrated['micro_recall']:.4f}"
    )
    print(
        f"Threshold-free: mAP={test_free['mAP']:.4f} | "
        f"Macro-AUROC={test_free['macro_AUROC']:.4f}"
    )

    default_rows = per_label_metrics(test_y, test_p, default_threshold)
    calibrated_rows = per_label_metrics(test_y, test_p, thresholds)
    validation_default_rows = per_label_metrics(val_y, val_p, default_threshold)
    validation_calibrated_rows = per_label_metrics(val_y, val_p, thresholds)

    with open(
        out_dir / "validation_threshold_comparison.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index", "label", "positive_val_samples",
            "optimal_threshold", "validation_best_f1",
            "default_precision", "default_recall", "default_f1",
            "calibrated_precision", "calibrated_recall", "calibrated_f1",
            "f1_change",
        ])
        for j, name in enumerate(label_names):
            d = validation_default_rows[j]
            c = validation_calibrated_rows[j]
            writer.writerow([
                j, name, d["positive_samples"], float(thresholds[j]),
                "" if np.isnan(val_best_f1[j]) else float(val_best_f1[j]),
                d["precision"], d["recall"], d["f1"],
                c["precision"], c["recall"], c["f1"], c["f1"] - d["f1"],
            ])

    with open(
        out_dir / "threshold_comparison.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_val_samples",
            "optimal_threshold",
            "val_best_f1",
            "positive_test_samples",
            "default_precision",
            "default_recall",
            "default_f1",
            "calibrated_precision",
            "calibrated_recall",
            "calibrated_f1",
            "f1_change",
        ])
        for j, name in enumerate(label_names):
            d = default_rows[j]
            c = calibrated_rows[j]
            writer.writerow([
                j,
                name,
                int(val_y[:, j].sum()),
                float(thresholds[j]),
                "" if np.isnan(val_best_f1[j]) else float(val_best_f1[j]),
                d["positive_samples"],
                d["precision"],
                d["recall"],
                d["f1"],
                c["precision"],
                c["recall"],
                c["f1"],
                c["f1"] - d["f1"],
            ])

    with open(
        out_dir / "test_per_label_metrics_calibrated.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_test_samples",
            "negative_test_samples",
            "threshold",
            "precision",
            "recall",
            "f1",
        ])
        for j, name in enumerate(label_names):
            row = calibrated_rows[j]
            writer.writerow([
                j,
                name,
                row["positive_samples"],
                row["negative_samples"],
                row["threshold"],
                row["precision"],
                row["recall"],
                row["f1"],
            ])

    with open(
        out_dir / "test_per_label_metrics_default.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "label_index",
            "label",
            "positive_test_samples",
            "negative_test_samples",
            "threshold",
            "precision",
            "recall",
            "f1",
        ])
        for j, name in enumerate(label_names):
            row = default_rows[j]
            writer.writerow([
                j,
                name,
                row["positive_samples"],
                row["negative_samples"],
                row["threshold"],
                row["precision"],
                row["recall"],
                row["f1"],
            ])

    summary = {
        "experiment": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": source_sha256,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "label_set": "IR_FG14",
        "num_labels": len(label_names),
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "normalization": normalization,
        "calibration": "per-label F1 optimization on validation split",
        "default_threshold": default_threshold,
        "validation_default": val_default,
        "validation_calibrated": val_calibrated,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "test_threshold_free": test_free,
        "note": (
            "Thresholds were selected on validation data only and frozen before any "
            "test metric was computed. mAP and AUROC are threshold-free and therefore "
            "unchanged by the threshold choice."
        ),
    }

    with open(
        out_dir / "threshold_calibration_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("Threshold calibration completed")
    print("=" * 80)
    print()
    print("Saved (new directory, nothing overwritten):")
    for name in (
        "optimal_thresholds.json",
        "validation_threshold_comparison.csv",
        "threshold_comparison.csv",
        "test_per_label_metrics_calibrated.csv",
        "test_per_label_metrics_default.csv",
        "threshold_calibration_summary.json",
    ):
        print(" ", out_dir / name)
    print("Original checkpoint untouched:", checkpoint_path)


if __name__ == "__main__":
    main()
