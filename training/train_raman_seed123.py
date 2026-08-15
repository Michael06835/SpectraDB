from pathlib import Path
import sys
import json
import csv
import time
import random
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


# ============================================================
# EXP-RAMAN-001-SEED123 configuration
# ============================================================

SEED = 123

# ------------------------------------------------------------
# Hardware / DataLoader
# ------------------------------------------------------------

# RTX 5070 Laptop 8 GB:
# first formal run uses 512.
# If CUDA OOM occurs, change this ONLY to 256 and rerun.
BATCH_SIZE = 512

NUM_WORKERS = 4

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

EMBEDDING_DIM = 256

# ------------------------------------------------------------
# Spectrum preprocessing
# ------------------------------------------------------------

# Each spectrum is divided by its own maximum absolute intensity.
NORMALIZATION = "max"

# ------------------------------------------------------------
# Multilabel prediction
# ------------------------------------------------------------

THRESHOLD = 0.5

# First baseline deliberately uses ordinary BCE.
# Class weighting will be investigated in a later experiment.
USE_POS_WEIGHT = False


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    targets,
    probabilities,
    threshold=0.5,
):
    """
    targets:
        [N, C] binary matrix

    probabilities:
        [N, C] sigmoid probabilities
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

    # --------------------------------------------------------
    # Threshold-dependent global metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Per-label threshold-free metrics
    # --------------------------------------------------------

    ap_per_label = []
    auroc_per_label = []

    for j in range(targets.shape[1]):

        y_true = targets[:, j]
        y_score = probabilities[:, j]

        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)

        # Average precision requires positive examples
        # for meaningful interpretation.
        if n_pos == 0:
            ap = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ap = average_precision_score(
                    y_true,
                    y_score,
                )

        # AUROC requires both positive and negative classes.
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


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp,
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
        threshold=THRESHOLD,
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
# Main experiment
# ============================================================

def main():

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(SEED)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        # Fixed spectral input length means cuDNN can benchmark
        # convolution algorithms for better throughput.
        torch.backends.cudnn.benchmark = True

    else:
        device = torch.device("cpu")

    use_amp = (
        device.type == "cuda"
    )

    print()
    print("=" * 80)
    print("EXP-RAMAN-001-SEED123")
    print("QM9S Raman -> structural feature baseline")
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

        props = torch.cuda.get_device_properties(0)

        print(
            "VRAM   :",
            f"{props.total_memory / 1024**3:.2f} GB",
        )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "AMP    :",
        use_amp,
    )

    print(
        "Batch  :",
        BATCH_SIZE,
    )

    print(
        "Workers:",
        NUM_WORKERS,
    )


    # ========================================================
    # Check required files
    # ========================================================

    required_files = [
        RAMAN_PATH,
        LABEL_PATH,
        LABEL_NAMES_PATH,
        SPLIT_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file does not exist:\n{path}"
            )


    # ========================================================
    # Load label names
    # ========================================================

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        label_names = json.load(f)

    num_labels = len(label_names)

    print()
    print("Labels:", num_labels)

    for i, name in enumerate(label_names):

        print(
            f"  {i:2d}: {name}"
        )


    # ========================================================
    # Load scaffold split
    # ========================================================

    split = np.load(
        SPLIT_PATH
    )

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
    print("=" * 80)
    print("Dataset split")
    print("=" * 80)

    print(
        "Train:",
        len(train_idx),
    )

    print(
        "Val  :",
        len(val_idx),
    )

    print(
        "Test :",
        len(test_idx),
    )

    print(
        "Total:",
        len(train_idx)
        + len(val_idx)
        + len(test_idx),
    )


    # ========================================================
    # Datasets
    # ========================================================

    train_dataset = QM9SSpectrumDataset(
        spectra_path=RAMAN_PATH,
        labels_path=LABEL_PATH,
        indices=train_idx,
        normalization=NORMALIZATION,
    )

    val_dataset = QM9SSpectrumDataset(
        spectra_path=RAMAN_PATH,
        labels_path=LABEL_PATH,
        indices=val_idx,
        normalization=NORMALIZATION,
    )

    test_dataset = QM9SSpectrumDataset(
        spectra_path=RAMAN_PATH,
        labels_path=LABEL_PATH,
        indices=test_idx,
        normalization=NORMALIZATION,
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    loader_common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (
            device.type == "cuda"
        ),
    }

    if NUM_WORKERS > 0:

        loader_common[
            "persistent_workers"
        ] = True

        loader_common[
            "prefetch_factor"
        ] = 2


    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )


    # ========================================================
    # Model
    # ========================================================

    model = SpectrumStructureModel(
        num_labels=num_labels,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    n_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("=" * 80)
    print("Model")
    print("=" * 80)

    print(
        "Total parameters    :",
        f"{n_params:,}",
    )

    print(
        "Trainable parameters:",
        f"{n_trainable:,}",
    )

    print(
        "Embedding dimension :",
        EMBEDDING_DIM,
    )


    # ========================================================
    # Loss
    # ========================================================

    if USE_POS_WEIGHT:

        # Not used in EXP-RAMAN-001.
        # Reserved for future class-balanced experiment.

        all_labels = np.load(
            LABEL_PATH,
            mmap_mode="r",
        )

        train_labels = np.asarray(
            all_labels[train_idx],
            dtype=np.float32,
        )

        positives = train_labels.sum(
            axis=0
        )

        negatives = (
            len(train_labels)
            - positives
        )

        pos_weight = (
            negatives
            / np.maximum(
                positives,
                1.0,
            )
        )

        pos_weight_tensor = torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )

        print()
        print(
            "Loss: BCEWithLogitsLoss(pos_weight)"
        )

    else:

        criterion = (
            torch.nn.BCEWithLogitsLoss()
        )

        print()
        print(
            "Loss: BCEWithLogitsLoss"
        )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ========================================================
    # AMP scaler
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )


    # ========================================================
    # Configuration log
    # ========================================================

    config = {

        "experiment":
            "EXP-RAMAN-001-SEED123",
        
        "modality":
            "raman",

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "num_workers":
            NUM_WORKERS,

        "max_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "embedding_dim":
            EMBEDDING_DIM,

        "normalization":
            NORMALIZATION,

        "threshold":
            THRESHOLD,

        "use_pos_weight":
            USE_POS_WEIGHT,

        "loss":
            "BCEWithLogitsLoss",

        "optimizer":
            "AdamW",

        "amp":
            use_amp,

        "train_size":
            len(train_idx),

        "val_size":
            len(val_idx),

        "test_size":
            len(test_idx),

        "label_names":
            label_names,
    }


    with open(
        RUN_DIR / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # CSV training log
    # ========================================================

    log_path = (
        RUN_DIR
        / "training_log.csv"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_micro_f1",
            "val_macro_f1",
            "val_micro_precision",
            "val_micro_recall",
            "val_mAP",
            "val_macro_AUROC",
            "learning_rate",
            "epoch_seconds",
            "peak_cuda_memory_GB",
        ])


    # ========================================================
    # Training state
    # ========================================================

    best_val_map = -np.inf

    best_checkpoint_path = (
        RUN_DIR
        / "best_model.pt"
    )

    epochs_without_improvement = 0


    print()
    print("=" * 80)
    print("Training started")
    print("=" * 80)

    print()
    print(
        f"Maximum epochs : {MAX_EPOCHS}"
    )

    print(
        f"Early stopping : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Learning rate  : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay   : "
        f"{WEIGHT_DECAY}"
    )

    print()


    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats()


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        n_train = 0

        for batch in train_loader:

            x = batch["spectrum"].to(
                device,
                non_blocking=True,
            )

            y = batch["labels"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                output = model(x)

                logits = output[
                    "logits"
                ]

                loss = criterion(
                    logits,
                    y,
                )


            # ------------------------------------------------
            # Backpropagation with AMP
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()


            batch_n = x.size(0)

            running_loss += (
                loss.item()
                * batch_n
            )

            n_train += batch_n


        train_loss = (
            running_loss
            / n_train
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )


        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        if device.type == "cuda":

            peak_cuda_memory_gb = (
                torch.cuda
                .max_memory_allocated()
                / 1024**3
            )

        else:

            peak_cuda_memory_gb = 0.0


        print(
            f"Epoch "
            f"{epoch:03d}/{MAX_EPOCHS}"
            f" | "
            f"train_loss="
            f"{train_loss:.5f}"
            f" | "
            f"val_loss="
            f"{val_metrics['loss']:.5f}"
            f" | "
            f"micro-F1="
            f"{val_metrics['micro_f1']:.4f}"
            f" | "
            f"macro-F1="
            f"{val_metrics['macro_f1']:.4f}"
            f" | "
            f"mAP="
            f"{val_metrics['mAP']:.4f}"
            f" | "
            f"AUROC="
            f"{val_metrics['macro_AUROC']:.4f}"
            f" | "
            f"time="
            f"{epoch_seconds:.1f}s"
            f" | "
            f"VRAM="
            f"{peak_cuda_memory_gb:.2f}GB"
        )


        # ----------------------------------------------------
        # Write training log
        # ----------------------------------------------------

        with open(
            log_path,
            "a",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_metrics[
                    "loss"
                ],
                val_metrics[
                    "micro_f1"
                ],
                val_metrics[
                    "macro_f1"
                ],
                val_metrics[
                    "micro_precision"
                ],
                val_metrics[
                    "micro_recall"
                ],
                val_metrics[
                    "mAP"
                ],
                val_metrics[
                    "macro_AUROC"
                ],
                current_lr,
                epoch_seconds,
                peak_cuda_memory_gb,
            ])


        # ----------------------------------------------------
        # Save best model according to validation mAP
        # ----------------------------------------------------

        current_val_map = (
            val_metrics["mAP"]
        )

        if (
            current_val_map
            > best_val_map
        ):

            best_val_map = (
                current_val_map
            )

            epochs_without_improvement = 0


            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_mAP":
                    best_val_map,

                "val_metrics": {
                    "loss":
                        val_metrics["loss"],

                    "micro_f1":
                        val_metrics["micro_f1"],

                    "macro_f1":
                        val_metrics["macro_f1"],

                    "mAP":
                        val_metrics["mAP"],

                    "macro_AUROC":
                        val_metrics[
                            "macro_AUROC"
                        ],
                },

                "label_names":
                    label_names,

                "config":
                    config,
            }


            torch.save(
                checkpoint,
                best_checkpoint_path,
            )


            print(
                "  -> BEST MODEL SAVED"
                f" | val mAP="
                f"{best_val_map:.4f}"
            )


        else:

            epochs_without_improvement += 1


            print(
                "  -> no improvement"
                f" "
                f"("
                f"{epochs_without_improvement}"
                f"/"
                f"{EARLY_STOPPING_PATIENCE}"
                f")"
            )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break


    # ========================================================
    # Load best checkpoint
    # ========================================================

    print()
    print("=" * 80)
    print("Loading best checkpoint")
    print("=" * 80)


    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    best_epoch = int(
        checkpoint["epoch"]
    )


    print(
        "Best epoch          :",
        best_epoch,
    )

    print(
        "Best validation mAP :",
        f"{checkpoint['best_val_mAP']:.4f}",
    )


    # ========================================================
    # FINAL TEST
    #
    # Test set is evaluated only after model selection.
    # ========================================================

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


    # ========================================================
    # Per-label test metrics
    # ========================================================

    test_predictions = (
        test_probabilities
        >= THRESHOLD
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


    # ========================================================
    # Overall test result JSON
    # ========================================================

    final_result = {

        "experiment":
            "EXP-RAMAN-001-SEED123",
        
        "modality":
            "raman",

        "best_epoch":
            best_epoch,

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


    # ========================================================
    # Completion
    # ========================================================

    print()
    print("=" * 80)
    print("EXP-RAMAN-001-SEED123 completed")
    print("=" * 80)

    print()
    print("Saved files:")

    print(
        " ",
        RUN_DIR / "config.json"
    )

    print(
        " ",
        RUN_DIR / "training_log.csv"
    )

    print(
        " ",
        RUN_DIR / "best_model.pt"
    )

    print(
        " ",
        RUN_DIR / "test_metrics.json"
    )

    print(
        " ",
        RUN_DIR
        / "test_per_label_metrics.csv"
    )


# ============================================================
# IMPORTANT FOR WINDOWS:
# DataLoader with num_workers > 0 must start from here.
# ============================================================

if __name__ == "__main__":

    main()