"""UVSPEC15 post-hoc threshold calibration, aligned with UVVIS-001 protocol.

Run at SpectraDB root:
    python training/calibrate_uvvis_uvspec15_thresholds_seed123.py

No training, no changes to model weights, label schema or dataset split.
Class thresholds are optimized ONLY on validation and evaluated on held-out test.
Training/evaluation num_workers stays 4 for controlled comparisons.
"""

from pathlib import Path
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
UVVIS_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "uvspec15_labels.npy"
LABEL_NAMES_PATH = PREPARED / "uvspec15_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"

RUN_DIR = ROOT / "runs" / "EXP-UVVIS-003-UVSPEC15-SEED123"
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"

NUM_WORKERS = 4  # Deliberately identical to the original UVVIS-001 training run.
DEFAULT_THRESHOLD = 0.5


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


def main():
    required = [
        UVVIS_PATH,
        LABEL_PATH,
        LABEL_NAMES_PATH,
        SPLIT_PATH,
        CHECKPOINT_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file:\n{path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print()
    print("=" * 80)
    print("EXP-UVVIS-003-UVSPEC15-FULL threshold calibration")
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
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})
    batch_size = int(config.get("batch_size", 512))
    embedding_dim = int(config.get("embedding_dim", 256))
    normalization = config.get("normalization", "max")
    default_threshold = float(
        config.get("threshold", DEFAULT_THRESHOLD)
    )

    # Fail before running inference if the checkpoint and labels are mismatched.
    if len(label_names) != 15 or len(set(label_names)) != 15:
        raise ValueError("UVSPEC15 requires exactly 15 unique label names")
    if checkpoint.get("label_names") != label_names:
        raise ValueError("Checkpoint label_names do not match UVSPEC15 label order")
    if config.get("experiment") != "EXP-UVVIS-003-UVSPEC15-FULL":
        raise ValueError("Checkpoint comes from a different experiment")
    if config.get("label_names") != label_names:
        raise ValueError("Checkpoint config label_names mismatch")
    if config.get("use_pos_weight") is not False:
        raise ValueError("Expected the ordinary BCE baseline checkpoint")
    if int(config.get("num_workers", -1)) != NUM_WORKERS:
        raise ValueError("num_workers mismatch: original training must use 4")
    if batch_size != 512 or embedding_dim != 256 or normalization != "max":
        raise ValueError("Training configuration mismatch: investigate before calibration")
    if default_threshold != DEFAULT_THRESHOLD:
        raise ValueError("Expected original default classification threshold 0.5")
    with np.load(SPLIT_PATH) as split_check:
        train_idx = np.asarray(split_check["train"], dtype=np.int64)
    if (len(train_idx), len(val_idx), len(test_idx)) != (103853, 12981, 12981):
        raise ValueError("Unexpected UV scaffold split sizes")
    label_check = np.load(LABEL_PATH, mmap_mode="r")
    if label_check.ndim != 2 or label_check.shape[1] != 15:
        raise ValueError(f"Wrong UVSPEC15 label matrix shape: {label_check.shape}")

    print()
    print("Validation samples:", len(val_idx))
    print("Test samples      :", len(test_idx))
    print("Checkpoint        :", CHECKPOINT_PATH)
    print("Best epoch        :", checkpoint.get("epoch", "unknown"))
    print("Best val mAP      :", checkpoint.get("best_val_mAP", "unknown"))
    print("Batch size        :", batch_size)
    print("Normalization     :", normalization)
    print("Default threshold :", default_threshold)

    val_ds = QM9SSpectrumDataset(
        spectra_path=UVVIS_PATH,
        labels_path=LABEL_PATH,
        indices=val_idx,
        normalization=normalization,
    )

    test_ds = QM9SSpectrumDataset(
        spectra_path=UVVIS_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization=normalization,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    model = SpectrumStructureModel(
        num_labels=len(label_names),
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

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
    print(f"{'label':47s}{'positive':>10s}{'threshold':>12s}{'best_F1':>12s}")
    print("-" * 81)

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
            f"{name:47s}"
            f"{int(val_y[:, j].sum()):10d}"
            f"{threshold:12.4f}"
            f"{best_text:>12s}"
        )

    thresholds = np.asarray(thresholds, dtype=np.float32)

    val_default = overall_metrics(
        val_y,
        val_p,
        default_threshold,
    )
    val_calibrated = overall_metrics(
        val_y,
        val_p,
        thresholds,
    )

    test_default = overall_metrics(
        test_y,
        test_p,
        default_threshold,
    )
    test_calibrated = overall_metrics(
        test_y,
        test_p,
        thresholds,
    )

    test_free = threshold_free_metrics(
        test_y,
        test_p,
    )

    # An independent reproducibility check against the frozen 003 baseline.
    # Do NOT use the test scores to select or change calibration thresholds.
    baseline_path = RUN_DIR / "test_metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing frozen baseline metrics: {baseline_path}")
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
        if not np.isclose(observed, float(baseline[key]), rtol=0, atol=0.001):
            raise RuntimeError(
                f"Baseline reproduction check failed for {key}: "
                f"checkpoint={observed:.6f}, recorded={float(baseline[key]):.6f}. "
                "Check labels, indices, preprocessing and inference before proceeding."
            )
    print("PASS: frozen 003 baseline reproduced with threshold=0.5")

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

    thresholds_json = {
        "experiment": "EXP-UVVIS-003-UVSPEC15-FULL",
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
    }

    with open(
        RUN_DIR / "optimal_thresholds.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds_json,
            f,
            ensure_ascii=False,
            indent=2,
        )

    default_rows = per_label_metrics(
        test_y,
        test_p,
        default_threshold,
    )
    calibrated_rows = per_label_metrics(
        test_y,
        test_p,
        thresholds,
    )

    # Validation diagnostics: all threshold choices were made here.
    validation_default_rows = per_label_metrics(
        val_y,
        val_p,
        default_threshold,
    )
    validation_calibrated_rows = per_label_metrics(
        val_y,
        val_p,
        thresholds,
    )
    with open(
        RUN_DIR / "validation_threshold_comparison.csv",
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
        RUN_DIR / "threshold_comparison.csv",
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
        RUN_DIR / "test_per_label_metrics_calibrated.csv",
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

    summary = {
        "experiment": "EXP-UVVIS-003-UVSPEC15-FULL",
        "checkpoint": "best_model.pt",
        "label_set": "UVSPEC15",
        "num_workers": NUM_WORKERS,
        "calibration": "per-label F1 optimization on validation split",
        "default_threshold": default_threshold,
        "validation_default": val_default,
        "validation_calibrated": val_calibrated,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "test_threshold_free": test_free,
        "note": (
            "Thresholds were selected on validation data only "
            "and then frozen for test. mAP and AUROC are "
            "threshold-free and therefore unchanged."
        ),
    }

    with open(
        RUN_DIR / "threshold_calibration_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Threshold calibration completed")
    print("=" * 80)
    print()
    print("Saved:")
    print(" ", RUN_DIR / "optimal_thresholds.json")
    print(" ", RUN_DIR / "validation_threshold_comparison.csv")
    print(" ", RUN_DIR / "threshold_comparison.csv")
    print(" ", RUN_DIR / "test_per_label_metrics_calibrated.csv")
    print(" ", RUN_DIR / "threshold_calibration_summary.json")


if __name__ == "__main__":
    main()
