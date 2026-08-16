from pathlib import Path
import sys
import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel

PREPARED = ROOT / "processed" / "qm9s" / "prepared"

UVVIS_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"

SOURCE_RUN = ROOT / "runs" / "EXP-UVVIS-001-FULL"
CHECKPOINT_PATH = SOURCE_RUN / "best_model.pt"

RUN_DIR = ROOT / "runs" / "EXP-UVVIS-001-BN-FULL-DIAG"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 512
NUM_WORKERS = 4
THRESHOLD = 0.5


def calculate_metrics(targets, probabilities, threshold=0.5):
    targets = np.asarray(targets, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= threshold).astype(np.int32)

    metrics = {
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
                loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            probabilities = torch.sigmoid(logits)

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
    metrics["loss"] = float(
        total_loss / n_samples
    )

    return metrics


def get_batchnorm_layers(model):
    return [
        module
        for module in model.modules()
        if isinstance(
            module,
            torch.nn.BatchNorm1d,
        )
    ]


def recalibrate_batchnorm(
    model,
    train_loader,
    device,
    use_amp,
):
    """
    Recompute BatchNorm running statistics using TRAIN SET ONLY.

    Model weights are unchanged.
    Only running_mean, running_var, and num_batches_tracked
    are recomputed.
    """

    bn_layers = get_batchnorm_layers(model)

    if len(bn_layers) == 0:
        raise RuntimeError(
            "No BatchNorm1d layers found."
        )

    model.eval()

    original_momenta = {}

    for bn in bn_layers:
        original_momenta[id(bn)] = bn.momentum

        bn.reset_running_stats()
        bn.momentum = None
        bn.train()

    n_batches = 0
    n_samples = 0

    with torch.no_grad():
        for batch in train_loader:
            x = batch["spectrum"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _ = model(x)

            n_batches += 1
            n_samples += x.size(0)

    for bn in bn_layers:
        bn.momentum = original_momenta[id(bn)]
        bn.eval()

    model.eval()

    return {
        "num_batchnorm_layers":
            len(bn_layers),
        "train_batches_seen":
            n_batches,
        "train_samples_seen":
            n_samples,
    }


def print_metrics(title, metrics):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Loss        :",
        f"{metrics['loss']:.6f}",
    )
    print(
        "Micro-F1    :",
        f"{metrics['micro_f1']:.4f}",
    )
    print(
        "Macro-F1    :",
        f"{metrics['macro_f1']:.4f}",
    )
    print(
        "Precision   :",
        f"{metrics['micro_precision']:.4f}",
    )
    print(
        "Recall      :",
        f"{metrics['micro_recall']:.4f}",
    )
    print(
        "mAP         :",
        f"{metrics['mAP']:.4f}",
    )
    print(
        "Macro-AUROC :",
        f"{metrics['macro_AUROC']:.4f}",
    )


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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = device.type == "cuda"

    print()
    print("=" * 80)
    print(
        "EXP-UVVIS-001-FULL BatchNorm diagnostic"
    )
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

    print("PyTorch:", torch.__version__)
    print("AMP    :", use_amp)
    print("Workers:", NUM_WORKERS)

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        label_names = json.load(f)

    split = np.load(SPLIT_PATH)

    train_idx = np.asarray(
        split["train"],
        dtype=np.int64,
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
    print("Train:", len(train_idx))
    print("Val  :", len(val_idx))
    print("Test :", len(test_idx))

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
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
            THRESHOLD,
        )
    )

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )
    print(
        "Best epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Best validation mAP:",
        checkpoint.get(
            "best_val_mAP",
            "unknown",
        ),
    )
    print(
        "Normalization:",
        normalization,
    )
    print(
        "Threshold:",
        threshold,
    )

    train_dataset = QM9SSpectrumDataset(
        spectra_path=UVVIS_PATH,
        labels_path=LABEL_PATH,
        indices=train_idx,
        normalization=normalization,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = SpectrumStructureModel(
        num_labels=len(label_names),
        embedding_dim=embedding_dim,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    before_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=threshold,
    )

    before_test = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=threshold,
    )

    print_metrics(
        "BEFORE BN RECALIBRATION - VALIDATION",
        before_val,
    )

    print_metrics(
        "BEFORE BN RECALIBRATION - TEST",
        before_test,
    )

    print()
    print("=" * 80)
    print(
        "RECALIBRATING BATCHNORM USING TRAIN SET ONLY"
    )
    print("=" * 80)

    bn_info = recalibrate_batchnorm(
        model=model,
        train_loader=train_loader,
        device=device,
        use_amp=use_amp,
    )

    print()
    print(
        "BatchNorm layers:",
        bn_info[
            "num_batchnorm_layers"
        ],
    )
    print(
        "Train batches seen:",
        bn_info[
            "train_batches_seen"
        ],
    )
    print(
        "Train samples seen:",
        bn_info[
            "train_samples_seen"
        ],
    )

    after_val = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=threshold,
    )

    after_test = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=threshold,
    )

    print_metrics(
        "AFTER BN RECALIBRATION - VALIDATION",
        after_val,
    )

    print_metrics(
        "AFTER BN RECALIBRATION - TEST",
        after_test,
    )

    metric_keys = [
        "loss",
        "micro_f1",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "mAP",
        "macro_AUROC",
    ]

    delta = {
        "val": {
            key:
                float(
                    after_val[key]
                    - before_val[key]
                )
            for key in metric_keys
        },
        "test": {
            key:
                float(
                    after_test[key]
                    - before_test[key]
                )
            for key in metric_keys
        },
    }

    print()
    print("=" * 80)
    print("CHANGE AFTER BN RECALIBRATION")
    print("=" * 80)

    print()

    for split_name in [
        "val",
        "test",
    ]:
        print(split_name.upper())

        print(
            "  Micro-F1    :",
            f"{delta[split_name]['micro_f1']:+.4f}",
        )
        print(
            "  Macro-F1    :",
            f"{delta[split_name]['macro_f1']:+.4f}",
        )
        print(
            "  Precision   :",
            f"{delta[split_name]['micro_precision']:+.4f}",
        )
        print(
            "  Recall      :",
            f"{delta[split_name]['micro_recall']:+.4f}",
        )
        print(
            "  mAP         :",
            f"{delta[split_name]['mAP']:+.4f}",
        )
        print(
            "  Macro-AUROC :",
            f"{delta[split_name]['macro_AUROC']:+.4f}",
        )
        print()

    result = {
        "experiment":
            "EXP-UVVIS-001-BN-FULL-DIAG",
        "source_checkpoint":
            str(CHECKPOINT_PATH),
        "recalibration_source":
            "train split only",
        "batch_size":
            BATCH_SIZE,
        "num_workers":
            NUM_WORKERS,
        "normalization":
            normalization,
        "threshold":
            threshold,
        "batchnorm":
            bn_info,
        "before": {
            "validation":
                before_val,
            "test":
                before_test,
        },
        "after": {
            "validation":
                after_val,
            "test":
                after_test,
        },
        "delta":
            delta,
        "note":
            (
                "BatchNorm running statistics were reset and "
                "recomputed using the training split only. "
                "Model weights were not updated."
            ),
    }

    json_path = (
        RUN_DIR
        / "batchnorm_full_diagnostic.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    txt_path = (
        RUN_DIR
        / "batchnorm_full_diagnostic.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "EXP-UVVIS-001-FULL BatchNorm diagnostic\n"
        )
        f.write(
            "=" * 50 + "\n\n"
        )
        f.write(
            f"Source checkpoint: {CHECKPOINT_PATH}\n"
        )
        f.write(
            "Recalibration source: TRAIN split only\n"
        )
        f.write(
            f"BatchNorm layers: "
            f"{bn_info['num_batchnorm_layers']}\n"
        )
        f.write(
            f"Train samples used: "
            f"{bn_info['train_samples_seen']}\n\n"
        )

        f.write("BEFORE TEST\n")
        f.write(
            f"Micro-F1: {before_test['micro_f1']:.4f}\n"
        )
        f.write(
            f"Macro-F1: {before_test['macro_f1']:.4f}\n"
        )
        f.write(
            f"Precision: {before_test['micro_precision']:.4f}\n"
        )
        f.write(
            f"Recall: {before_test['micro_recall']:.4f}\n"
        )
        f.write(
            f"mAP: {before_test['mAP']:.4f}\n"
        )
        f.write(
            f"Macro-AUROC: {before_test['macro_AUROC']:.4f}\n\n"
        )

        f.write("AFTER TEST\n")
        f.write(
            f"Micro-F1: {after_test['micro_f1']:.4f}\n"
        )
        f.write(
            f"Macro-F1: {after_test['macro_f1']:.4f}\n"
        )
        f.write(
            f"Precision: {after_test['micro_precision']:.4f}\n"
        )
        f.write(
            f"Recall: {after_test['micro_recall']:.4f}\n"
        )
        f.write(
            f"mAP: {after_test['mAP']:.4f}\n"
        )
        f.write(
            f"Macro-AUROC: {after_test['macro_AUROC']:.4f}\n\n"
        )

        f.write("DELTA TEST\n")
        f.write(
            f"Micro-F1: {delta['test']['micro_f1']:+.4f}\n"
        )
        f.write(
            f"Macro-F1: {delta['test']['macro_f1']:+.4f}\n"
        )
        f.write(
            f"Precision: {delta['test']['micro_precision']:+.4f}\n"
        )
        f.write(
            f"Recall: {delta['test']['micro_recall']:+.4f}\n"
        )
        f.write(
            f"mAP: {delta['test']['mAP']:+.4f}\n"
        )
        f.write(
            f"Macro-AUROC: {delta['test']['macro_AUROC']:+.4f}\n"
        )

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 80)

    print()
    print("Saved:")
    print(" ", json_path)
    print(" ", txt_path)


if __name__ == "__main__":
    main()
