from pathlib import Path
import sys
import json
import csv
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel


PREPARED = ROOT / "processed" / "qm9s" / "prepared"

UVVIS_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"

RUN_DIR = ROOT / "runs" / "EXP-UVVIS-001-FULL"
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"

DEFAULT_THRESHOLD = 0.5


# ============================================================
# Prediction helper
# ============================================================

def collect_probabilities(
    model,
    loader,
    device,
    use_amp,
):
    model.eval()

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            x = batch["spectrum"].to(
                device,
                non_blocking=True,
            )

            y = batch["labels"].to(
                device,
                non_blocking=True,
            )

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
                y.detach().cpu().numpy()
            )

            all_probabilities.append(
                probabilities.detach().cpu().numpy()
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
# Threshold optimization
# ============================================================

def optimize_threshold_for_label(
    y_true,
    y_score,
):
    """
    Choose the threshold that maximizes F1 on the validation set.

    precision_recall_curve returns:
        precision:  len(thresholds) + 1
        recall:     len(thresholds) + 1
        thresholds: len(thresholds)

    Therefore F1 is evaluated on precision[:-1] / recall[:-1].
    """

    n_pos = int(y_true.sum())

    if n_pos == 0:
        return DEFAULT_THRESHOLD, np.nan

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_score,
        )
    )

    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD, np.nan

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    denom = (
        precision_t
        + recall_t
    )

    f1_values = np.divide(
        2.0
        * precision_t
        * recall_t,
        denom,
        out=np.zeros_like(
            denom,
            dtype=np.float64,
        ),
        where=denom > 0,
    )

    best_index = int(
        np.nanargmax(f1_values)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    best_f1 = float(
        f1_values[best_index]
    )

    return best_threshold, best_f1


# ============================================================
# Metrics
# ============================================================

def global_metrics(
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

    return {
        "micro_f1": float(
            f1_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_precision": float(
            precision_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
    }


def per_label_metrics(
    targets,
    probabilities,
    thresholds,
):
    rows = []

    for j in range(
        targets.shape[1]
    ):

        y_true = (
            targets[:, j]
        )

        y_pred = (
            probabilities[:, j]
            >= thresholds[j]
        ).astype(np.int32)

        rows.append({
            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():

    required_files = [
        UVVIS_PATH,
        LABEL_PATH,
        LABEL_NAMES_PATH,
        SPLIT_PATH,
        CHECKPOINT_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file does not exist:\n{path}"
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_amp = (
        device.type == "cuda"
    )

    print()
    print("=" * 80)
    print("UV-Vis threshold calibration")
    print("=" * 80)

    print()
    print("Device :", device)

    if device.type == "cuda":

        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load checkpoint
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

    batch_size = int(
        config.get(
            "batch_size",
            512,
        )
    )

    embedding_dim = int(
        config.get(
            "embedding_dim",
            256,
        )
    )

    normalization = config.get(
        "normalization",
        "max",
    )

    print(
        "Checkpoint best epoch:",
        checkpoint.get("epoch"),
    )

    print(
        "Checkpoint best val mAP:",
        f"{checkpoint.get('best_val_mAP', float('nan')):.4f}",
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "DataLoader workers: 4"
    )

    # --------------------------------------------------------
    # Labels / split
    # --------------------------------------------------------

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        label_names = json.load(f)

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

    val_dataset = QM9SSpectrumDataset(
        spectra_path=UVVIS_PATH,
        labels_path=LABEL_PATH,
        indices=val_idx,
        normalization=normalization,
    )

    test_dataset = QM9SSpectrumDataset(
        spectra_path=UVVIS_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization=normalization,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SpectrumStructureModel(
        num_labels=len(label_names),
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Collect validation probabilities
    # --------------------------------------------------------

    print()
    print("Collecting validation probabilities...")

    val_targets, val_probabilities = (
        collect_probabilities(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
        )
    )

    # --------------------------------------------------------
    # Optimize one threshold per label on validation only
    # --------------------------------------------------------

    optimal_thresholds = []
    validation_rows = []

    print()
    print("=" * 80)
    print("Validation threshold optimization")
    print("=" * 80)

    for j, name in enumerate(
        label_names
    ):

        y_true = (
            val_targets[:, j]
        )

        y_score = (
            val_probabilities[:, j]
        )

        threshold, best_f1 = (
            optimize_threshold_for_label(
                y_true,
                y_score,
            )
        )

        optimal_thresholds.append(
            threshold
        )

        n_pos = int(
            y_true.sum()
        )

        validation_rows.append({
            "label_index": j,
            "label": name,
            "validation_positive_samples":
                n_pos,
            "optimal_threshold":
                threshold,
            "validation_best_f1":
                best_f1,
        })

        print(
            f"{name:15s}"
            f" positives={n_pos:5d}"
            f" threshold={threshold:.4f}"
            f" val_best_F1={best_f1:.4f}"
        )

    optimal_thresholds = np.asarray(
        optimal_thresholds,
        dtype=np.float32,
    )

    default_thresholds = np.full(
        len(label_names),
        DEFAULT_THRESHOLD,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validation comparison
    # --------------------------------------------------------

    val_default_metrics = (
        global_metrics(
            val_targets,
            val_probabilities,
            default_thresholds,
        )
    )

    val_calibrated_metrics = (
        global_metrics(
            val_targets,
            val_probabilities,
            optimal_thresholds,
        )
    )

    print()
    print("=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)

    print(
        "Default 0.5 :"
        f" micro-F1={val_default_metrics['micro_f1']:.4f}"
        f" macro-F1={val_default_metrics['macro_f1']:.4f}"
        f" precision={val_default_metrics['micro_precision']:.4f}"
        f" recall={val_default_metrics['micro_recall']:.4f}"
    )

    print(
        "Calibrated  :"
        f" micro-F1={val_calibrated_metrics['micro_f1']:.4f}"
        f" macro-F1={val_calibrated_metrics['macro_f1']:.4f}"
        f" precision={val_calibrated_metrics['micro_precision']:.4f}"
        f" recall={val_calibrated_metrics['micro_recall']:.4f}"
    )

    # --------------------------------------------------------
    # Test: apply frozen validation-derived thresholds
    # --------------------------------------------------------

    print()
    print("Collecting test probabilities...")

    test_targets, test_probabilities = (
        collect_probabilities(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
        )
    )

    test_default_metrics = (
        global_metrics(
            test_targets,
            test_probabilities,
            default_thresholds,
        )
    )

    test_calibrated_metrics = (
        global_metrics(
            test_targets,
            test_probabilities,
            optimal_thresholds,
        )
    )

    test_default_per_label = (
        per_label_metrics(
            test_targets,
            test_probabilities,
            default_thresholds,
        )
    )

    test_calibrated_per_label = (
        per_label_metrics(
            test_targets,
            test_probabilities,
            optimal_thresholds,
        )
    )

    print()
    print("=" * 80)
    print("TEST COMPARISON")
    print("=" * 80)

    print(
        "Default 0.5 :"
        f" micro-F1={test_default_metrics['micro_f1']:.4f}"
        f" macro-F1={test_default_metrics['macro_f1']:.4f}"
        f" precision={test_default_metrics['micro_precision']:.4f}"
        f" recall={test_default_metrics['micro_recall']:.4f}"
    )

    print(
        "Calibrated  :"
        f" micro-F1={test_calibrated_metrics['micro_f1']:.4f}"
        f" macro-F1={test_calibrated_metrics['macro_f1']:.4f}"
        f" precision={test_calibrated_metrics['micro_precision']:.4f}"
        f" recall={test_calibrated_metrics['micro_recall']:.4f}"
    )

    # --------------------------------------------------------
    # Save optimal thresholds
    # --------------------------------------------------------

    threshold_json = {
        name: float(
            optimal_thresholds[j]
        )
        for j, name in enumerate(
            label_names
        )
    }

    with open(
        RUN_DIR / "optimal_thresholds.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            threshold_json,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Save threshold comparison CSV
    # --------------------------------------------------------

    comparison_path = (
        RUN_DIR
        / "threshold_comparison.csv"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label_index",
            "label",
            "validation_positive_samples",
            "optimal_threshold",
            "validation_best_f1",
            "test_default_precision",
            "test_default_recall",
            "test_default_f1",
            "test_calibrated_precision",
            "test_calibrated_recall",
            "test_calibrated_f1",
            "test_f1_change",
        ])

        for row in validation_rows:

            j = row[
                "label_index"
            ]

            default_row = (
                test_default_per_label[j]
            )

            calibrated_row = (
                test_calibrated_per_label[j]
            )

            writer.writerow([
                j,
                row["label"],
                row[
                    "validation_positive_samples"
                ],
                row[
                    "optimal_threshold"
                ],
                row[
                    "validation_best_f1"
                ],
                default_row[
                    "precision"
                ],
                default_row[
                    "recall"
                ],
                default_row[
                    "f1"
                ],
                calibrated_row[
                    "precision"
                ],
                calibrated_row[
                    "recall"
                ],
                calibrated_row[
                    "f1"
                ],
                calibrated_row[
                    "f1"
                ]
                - default_row[
                    "f1"
                ],
            ])

    # --------------------------------------------------------
    # Save calibrated per-label test metrics
    # --------------------------------------------------------

    calibrated_path = (
        RUN_DIR
        / "test_per_label_metrics_calibrated.csv"
    )

    with open(
        calibrated_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label_index",
            "label",
            "optimal_threshold",
            "positive_test_samples",
            "precision",
            "recall",
            "f1",
        ])

        for j, name in enumerate(
            label_names
        ):

            y_true = (
                test_targets[:, j]
            )

            row = (
                test_calibrated_per_label[j]
            )

            writer.writerow([
                j,
                name,
                float(
                    optimal_thresholds[j]
                ),
                int(
                    y_true.sum()
                ),
                row[
                    "precision"
                ],
                row[
                    "recall"
                ],
                row[
                    "f1"
                ],
            ])

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {
        "experiment":
            "EXP-UVVIS-001-FULL",

        "modality":
            "uvvis",

        "checkpoint":
            "runs/EXP-UVVIS-001-FULL/best_model.pt",

        "best_epoch":
            int(
                checkpoint["epoch"]
            ),

        "best_val_mAP":
            float(
                checkpoint[
                    "best_val_mAP"
                ]
            ),

        "threshold_source":
            "validation per-label F1 optimization",

        "default_threshold":
            DEFAULT_THRESHOLD,

        "validation_default":
            val_default_metrics,

        "validation_calibrated":
            val_calibrated_metrics,

        "test_default":
            test_default_metrics,

        "test_calibrated":
            test_calibrated_metrics,
    }

    with open(
        RUN_DIR
        / "threshold_calibration_summary.json",
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
    print("Saved files:")

    for filename in [
        "optimal_thresholds.json",
        "threshold_comparison.csv",
        "test_per_label_metrics_calibrated.csv",
        "threshold_calibration_summary.json",
    ]:
        print(
            " ",
            RUN_DIR / filename
        )


if __name__ == "__main__":

    main()