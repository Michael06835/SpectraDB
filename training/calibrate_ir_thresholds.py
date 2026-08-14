from pathlib import Path
import sys
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import IRStructureModel


PREPARED = ROOT / "processed" / "qm9s" / "prepared"
RUN_DIR = ROOT / "runs" / "EXP-IR-001-FULL"

IR_PATH = PREPARED / "ir_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "ir_scaffold_split_valid.npz"

CHECKPOINT_PATH = RUN_DIR / "best_model.pt"

OUTPUT_THRESHOLDS = RUN_DIR / "optimal_thresholds.json"
OUTPUT_COMPARISON = RUN_DIR / "threshold_comparison.csv"
OUTPUT_TEST_PER_LABEL = RUN_DIR / "test_per_label_metrics_calibrated.csv"
OUTPUT_SUMMARY = RUN_DIR / "threshold_calibration_summary.json"


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 1024

# Inference only, so we deliberately keep workers at 0.
# This also avoids the RAM expansion we saw during training.
NUM_WORKERS = 0

DEFAULT_THRESHOLD = 0.5


# ============================================================
# Inference
# ============================================================

def collect_predictions(
    model,
    loader,
    device,
):
    model.eval()

    all_targets = []
    all_probabilities = []

    use_amp = (
        device.type == "cuda"
    )

    with torch.no_grad():

        for batch in loader:

            x = batch["spectrum"].to(
                device,
                non_blocking=True,
            )

            y = batch["labels"]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(x)["logits"]

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.append(
                y.numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    targets = np.concatenate(
        all_targets,
        axis=0,
    ).astype(np.int32)

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    ).astype(np.float32)

    return targets, probabilities


# ============================================================
# Find optimal threshold for ONE label
# ============================================================

def find_best_f1_threshold(
    y_true,
    y_score,
):

    positive_count = int(
        y_true.sum()
    )

    negative_count = int(
        len(y_true) - positive_count
    )

    # Threshold optimization is not meaningful
    # if validation contains only one class.
    if positive_count == 0 or negative_count == 0:

        return (
            DEFAULT_THRESHOLD,
            np.nan,
            np.nan,
            np.nan,
        )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    # precision / recall have one more element
    # than thresholds.
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    denominator = (
        precision_t + recall_t
    )

    f1 = np.divide(
        2.0
        * precision_t
        * recall_t,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=np.float64,
        ),
        where=denominator > 0,
    )

    best_index = int(
        np.argmax(f1)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    best_f1 = float(
        f1[best_index]
    )

    best_precision = float(
        precision_t[best_index]
    )

    best_recall = float(
        recall_t[best_index]
    )

    return (
        best_threshold,
        best_f1,
        best_precision,
        best_recall,
    )


# ============================================================
# Multilabel evaluation using independent thresholds
# ============================================================

def evaluate_with_thresholds(
    targets,
    probabilities,
    thresholds,
):

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    predictions = (
        probabilities
        >= thresholds[None, :]
    ).astype(np.int32)

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_precision = precision_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(
            micro_precision
        ),
        "micro_recall": float(
            micro_recall
        ),
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("EXP-IR-001 Threshold Calibration")
    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        label_names = json.load(f)

    num_labels = len(
        label_names
    )

    print()
    print(
        "Number of labels:",
        num_labels,
    )

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    split = np.load(
        SPLIT_PATH
    )

    val_idx = np.asarray(
        split["val"],
        dtype=np.int64,
    )

    test_idx = np.asarray(
        split["test"],
        dtype=np.int64,
    )

    print()
    print(
        "Validation samples:",
        len(val_idx),
    )

    print(
        "Test samples      :",
        len(test_idx),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    val_dataset = QM9SSpectrumDataset(
        spectra_path=IR_PATH,
        labels_path=LABEL_PATH,
        indices=val_idx,
        normalization="max",
    )

    test_dataset = QM9SSpectrumDataset(
        spectra_path=IR_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization="max",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
    )

    embedding_dim = config.get(
        "embedding_dim",
        256,
    )

    model = IRStructureModel(
        num_labels=num_labels,
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print()
    print(
        "Loaded checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation mAP:",
        checkpoint[
            "best_val_mAP"
        ],
    )

    # ========================================================
    # Obtain validation predictions
    # ========================================================

    print()
    print("=" * 80)
    print("Running validation inference")
    print("=" * 80)

    (
        val_targets,
        val_probabilities,
    ) = collect_predictions(
        model,
        val_loader,
        device,
    )

    # ========================================================
    # Find one threshold per label
    # ========================================================

    print()
    print("=" * 80)
    print("Finding optimal thresholds")
    print("=" * 80)

    thresholds = []

    validation_results = []

    for j, name in enumerate(
        label_names
    ):

        y_true = val_targets[:, j]

        y_score = (
            val_probabilities[:, j]
        )

        (
            threshold,
            best_f1,
            best_precision,
            best_recall,
        ) = find_best_f1_threshold(
            y_true,
            y_score,
        )

        thresholds.append(
            threshold
        )

        validation_results.append({
            "label": name,
            "positive_val_samples":
                int(y_true.sum()),
            "threshold":
                threshold,
            "best_val_f1":
                best_f1,
            "best_val_precision":
                best_precision,
            "best_val_recall":
                best_recall,
        })

        print(
            f"{name:15s}"
            f" | positives="
            f"{int(y_true.sum()):5d}"
            f" | threshold="
            f"{threshold:.4f}"
            f" | F1="
            f"{best_f1:.4f}"
            f" | P="
            f"{best_precision:.4f}"
            f" | R="
            f"{best_recall:.4f}"
        )

    thresholds = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    # ========================================================
    # Save thresholds BEFORE looking at test metrics
    # ========================================================

    threshold_dict = {
        name: float(threshold)
        for name, threshold
        in zip(
            label_names,
            thresholds,
        )
    }

    with open(
        OUTPUT_THRESHOLDS,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            threshold_dict,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Thresholds frozen and saved:"
    )

    print(
        OUTPUT_THRESHOLDS
    )

    # ========================================================
    # Compare on validation
    # ========================================================

    default_thresholds = np.full(
        num_labels,
        DEFAULT_THRESHOLD,
        dtype=np.float32,
    )

    val_default = evaluate_with_thresholds(
        val_targets,
        val_probabilities,
        default_thresholds,
    )

    val_calibrated = evaluate_with_thresholds(
        val_targets,
        val_probabilities,
        thresholds,
    )

    print()
    print("=" * 80)
    print("VALIDATION comparison")
    print("=" * 80)

    print()
    print("Threshold = 0.5")

    print(
        "Micro-F1:",
        f"{val_default['micro_f1']:.4f}",
    )

    print(
        "Macro-F1:",
        f"{val_default['macro_f1']:.4f}",
    )

    print()
    print(
        "Per-label calibrated thresholds"
    )

    print(
        "Micro-F1:",
        f"{val_calibrated['micro_f1']:.4f}",
    )

    print(
        "Macro-F1:",
        f"{val_calibrated['macro_f1']:.4f}",
    )

    # ========================================================
    # NOW evaluate frozen thresholds on test
    # ========================================================

    print()
    print("=" * 80)
    print("Running TEST inference")
    print("=" * 80)

    (
        test_targets,
        test_probabilities,
    ) = collect_predictions(
        model,
        test_loader,
        device,
    )

    test_default = evaluate_with_thresholds(
        test_targets,
        test_probabilities,
        default_thresholds,
    )

    test_calibrated = evaluate_with_thresholds(
        test_targets,
        test_probabilities,
        thresholds,
    )

    print()
    print("=" * 80)
    print("TEST comparison")
    print("=" * 80)

    print()
    print("Original threshold = 0.5")

    print(
        "Micro-F1 :",
        f"{test_default['micro_f1']:.4f}",
    )

    print(
        "Macro-F1 :",
        f"{test_default['macro_f1']:.4f}",
    )

    print(
        "Precision:",
        f"{test_default['micro_precision']:.4f}",
    )

    print(
        "Recall   :",
        f"{test_default['micro_recall']:.4f}",
    )

    print()
    print(
        "Validation-calibrated thresholds"
    )

    print(
        "Micro-F1 :",
        f"{test_calibrated['micro_f1']:.4f}",
    )

    print(
        "Macro-F1 :",
        f"{test_calibrated['macro_f1']:.4f}",
    )

    print(
        "Precision:",
        f"{test_calibrated['micro_precision']:.4f}",
    )

    print(
        "Recall   :",
        f"{test_calibrated['micro_recall']:.4f}",
    )

    # ========================================================
    # Per-label test comparison
    # ========================================================

    default_preds = (
        test_default["predictions"]
    )

    calibrated_preds = (
        test_calibrated["predictions"]
    )

    rows = []

    for j, name in enumerate(
        label_names
    ):

        y_true = test_targets[:, j]

        pred_default = (
            default_preds[:, j]
        )

        pred_calibrated = (
            calibrated_preds[:, j]
        )

        default_precision = precision_score(
            y_true,
            pred_default,
            zero_division=0,
        )

        default_recall = recall_score(
            y_true,
            pred_default,
            zero_division=0,
        )

        default_f1 = f1_score(
            y_true,
            pred_default,
            zero_division=0,
        )

        calibrated_precision = (
            precision_score(
                y_true,
                pred_calibrated,
                zero_division=0,
            )
        )

        calibrated_recall = (
            recall_score(
                y_true,
                pred_calibrated,
                zero_division=0,
            )
        )

        calibrated_f1 = (
            f1_score(
                y_true,
                pred_calibrated,
                zero_division=0,
            )
        )

        delta_f1 = (
            calibrated_f1
            - default_f1
        )

        rows.append({
            "label": name,
            "positive_test_samples":
                int(y_true.sum()),
            "threshold":
                float(
                    thresholds[j]
                ),
            "f1_default_0.5":
                float(
                    default_f1
                ),
            "f1_calibrated":
                float(
                    calibrated_f1
                ),
            "delta_f1":
                float(
                    delta_f1
                ),
            "precision_default":
                float(
                    default_precision
                ),
            "precision_calibrated":
                float(
                    calibrated_precision
                ),
            "recall_default":
                float(
                    default_recall
                ),
            "recall_calibrated":
                float(
                    calibrated_recall
                ),
        })

    # --------------------------------------------------------
    # Print per-label comparison
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Per-label TEST comparison")
    print("=" * 80)

    for row in rows:

        print(
            f"{row['label']:15s}"
            f" | t="
            f"{row['threshold']:.4f}"
            f" | F1 "
            f"{row['f1_default_0.5']:.4f}"
            f" -> "
            f"{row['f1_calibrated']:.4f}"
            f" | delta="
            f"{row['delta_f1']:+.4f}"
        )

    # ========================================================
    # Save CSV
    # ========================================================

    with open(
        OUTPUT_COMPARISON,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        fieldnames = [
            "label",
            "positive_test_samples",
            "threshold",
            "f1_default_0.5",
            "f1_calibrated",
            "delta_f1",
            "precision_default",
            "precision_calibrated",
            "recall_default",
            "recall_calibrated",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # Same information under a more explicit filename.
    with open(
        OUTPUT_TEST_PER_LABEL,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        fieldnames = [
            "label",
            "positive_test_samples",
            "threshold",
            "f1_default_0.5",
            "f1_calibrated",
            "delta_f1",
            "precision_default",
            "precision_calibrated",
            "recall_default",
            "recall_calibrated",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Save summary JSON
    # ========================================================

    summary = {
        "checkpoint_epoch":
            int(checkpoint["epoch"]),

        "threshold_selection_set":
            "validation",

        "default_threshold":
            DEFAULT_THRESHOLD,

        "thresholds":
            threshold_dict,

        "validation": {
            "default_micro_f1":
                val_default["micro_f1"],
            "default_macro_f1":
                val_default["macro_f1"],
            "calibrated_micro_f1":
                val_calibrated["micro_f1"],
            "calibrated_macro_f1":
                val_calibrated["macro_f1"],
        },

        "test": {
            "default_micro_f1":
                test_default["micro_f1"],
            "default_macro_f1":
                test_default["macro_f1"],
            "calibrated_micro_f1":
                test_calibrated["micro_f1"],
            "calibrated_macro_f1":
                test_calibrated["macro_f1"],
            "default_precision":
                test_default[
                    "micro_precision"
                ],
            "default_recall":
                test_default[
                    "micro_recall"
                ],
            "calibrated_precision":
                test_calibrated[
                    "micro_precision"
                ],
            "calibrated_recall":
                test_calibrated[
                    "micro_recall"
                ],
        },
    }

    with open(
        OUTPUT_SUMMARY,
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

    print(
        " ",
        OUTPUT_THRESHOLDS,
    )

    print(
        " ",
        OUTPUT_COMPARISON,
    )

    print(
        " ",
        OUTPUT_TEST_PER_LABEL,
    )

    print(
        " ",
        OUTPUT_SUMMARY,
    )


if __name__ == "__main__":
    main()