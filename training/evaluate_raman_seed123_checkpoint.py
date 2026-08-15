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
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel


PREPARED = ROOT / "processed" / "qm9s" / "prepared"

RAMAN_PATH = PREPARED / "raman_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "raman_scaffold_split_valid.npz"

RUN_DIR = ROOT / "runs" / "EXP-RAMAN-001-SEED123"
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(
    targets,
    probabilities,
    threshold=0.5,
):
    """
    Same metric definitions as training/train_raman_full.py.
    """

    targets = np.asarray(
        targets,
        dtype=np.int32,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    metrics = {}

    metrics["micro_f1"] = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    metrics["macro_f1"] = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    metrics["micro_precision"] = precision_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    metrics["micro_recall"] = recall_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    ap_per_label = []
    auroc_per_label = []

    for j in range(targets.shape[1]):

        y_true = targets[:, j]
        y_score = probabilities[:, j]

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        if n_pos == 0:
            ap = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ap = average_precision_score(
                    y_true,
                    y_score,
                )

        if n_pos == 0 or n_neg == 0:
            auroc = np.nan
        else:
            auroc = roc_auc_score(
                y_true,
                y_score,
            )

        ap_per_label.append(ap)
        auroc_per_label.append(auroc)

    metrics["mAP"] = float(
        np.nanmean(ap_per_label)
    )

    metrics["macro_AUROC"] = float(
        np.nanmean(auroc_per_label)
    )

    metrics["ap_per_label"] = ap_per_label
    metrics["auroc_per_label"] = auroc_per_label

    return metrics


def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp,
    threshold,
):

    model.eval()

    total_loss = 0.0
    n_samples = 0

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

                output = model(x)
                logits = output["logits"]

                loss = criterion(
                    logits,
                    y,
                )

            batch_size = x.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            n_samples += batch_size

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
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    metrics = calculate_metrics(
        targets,
        probabilities,
        threshold=threshold,
    )

    metrics["loss"] = (
        total_loss / n_samples
    )

    return (
        metrics,
        targets,
        probabilities,
    )


# ============================================================
# Main
# ============================================================

def main():

    required_files = [
        RAMAN_PATH,
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
    print("EXP-RAMAN-001-SEED123 checkpoint evaluation")
    print("=" * 80)

    print()
    print("Device :", device)

    if device.type == "cuda":

        print(
            "GPU    :",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA   :",
            torch.version.cuda,
        )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "AMP    :",
        use_amp,
    )

    # --------------------------------------------------------
    # Load checkpoint and its original experiment config
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

    threshold = float(
        config.get(
            "threshold",
            0.5,
        )
    )

    if config.get(
        "use_pos_weight",
        False,
    ):
        raise RuntimeError(
            "This evaluator is intended for EXP-RAMAN-001-SEED123 "
            "with ordinary BCEWithLogitsLoss (use_pos_weight=False)."
        )

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Best epoch:",
        checkpoint.get("epoch"),
    )

    print(
        "Best validation mAP:",
        f"{checkpoint.get('best_val_mAP', float('nan')):.4f}",
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Normalization:",
        normalization,
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "DataLoader workers: 0"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        label_names = json.load(f)

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    split = np.load(
        SPLIT_PATH
    )

    test_idx = np.asarray(
        split["test"],
        dtype=np.int64,
    )

    print()
    print(
        "Test samples:",
        len(test_idx),
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    #
    # num_workers=0 deliberately avoids Windows WinError 1455
    # caused by spawning additional PyTorch worker processes.
    # This does not change samples, model, threshold, or metrics.
    # --------------------------------------------------------

    test_dataset = QM9SSpectrumDataset(
        spectra_path=RAMAN_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization=normalization,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
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

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TEST")
    print("=" * 80)

    test_metrics, (
        test_targets
    ), (
        test_probabilities
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=threshold,
    )

    print()
    print(
        "Test loss        :",
        f"{test_metrics['loss']:.6f}",
    )

    print(
        "Test Micro-F1    :",
        f"{test_metrics['micro_f1']:.4f}",
    )

    print(
        "Test Macro-F1    :",
        f"{test_metrics['macro_f1']:.4f}",
    )

    print(
        "Test Precision   :",
        f"{test_metrics['micro_precision']:.4f}",
    )

    print(
        "Test Recall      :",
        f"{test_metrics['micro_recall']:.4f}",
    )

    print(
        "Test mAP         :",
        f"{test_metrics['mAP']:.4f}",
    )

    print(
        "Test Macro-AUROC :",
        f"{test_metrics['macro_AUROC']:.4f}",
    )

    # --------------------------------------------------------
    # Per-label test metrics
    # --------------------------------------------------------

    test_predictions = (
        test_probabilities
        >= threshold
    ).astype(
        np.int32
    )

    per_label_path = (
        RUN_DIR
        / "test_per_label_metrics.csv"
    )

    with open(
        per_label_path,
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
            "precision",
            "recall",
            "f1",
            "average_precision",
            "auroc",
        ])

        for j, name in enumerate(
            label_names
        ):

            y_true = (
                test_targets[:, j]
            )

            y_pred = (
                test_predictions[:, j]
            )

            positive_count = int(
                y_true.sum()
            )

            negative_count = int(
                len(y_true)
                - positive_count
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            ap = (
                test_metrics[
                    "ap_per_label"
                ][j]
            )

            auroc = (
                test_metrics[
                    "auroc_per_label"
                ][j]
            )

            writer.writerow([
                j,
                name,
                positive_count,
                negative_count,
                precision,
                recall,
                f1,
                ap,
                auroc,
            ])

    # --------------------------------------------------------
    # Overall test result JSON
    # --------------------------------------------------------

    final_result = {

        "experiment":
            "EXP-RAMAN-001-SEED123",

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

        "test_loss":
            float(
                test_metrics[
                    "loss"
                ]
            ),

        "test_micro_f1":
            float(
                test_metrics[
                    "micro_f1"
                ]
            ),

        "test_macro_f1":
            float(
                test_metrics[
                    "macro_f1"
                ]
            ),

        "test_micro_precision":
            float(
                test_metrics[
                    "micro_precision"
                ]
            ),

        "test_micro_recall":
            float(
                test_metrics[
                    "micro_recall"
                ]
            ),

        "test_mAP":
            float(
                test_metrics[
                    "mAP"
                ]
            ),

        "test_macro_AUROC":
            float(
                test_metrics[
                    "macro_AUROC"
                ]
            ),
    }

    with open(
        RUN_DIR
        / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Checkpoint evaluation completed")
    print("=" * 80)

    print()
    print("Saved files:")

    print(
        " ",
        RUN_DIR / "test_metrics.json"
    )

    print(
        " ",
        RUN_DIR
        / "test_per_label_metrics.csv"
    )


if __name__ == "__main__":

    main()
