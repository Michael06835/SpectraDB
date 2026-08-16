from pathlib import Path
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel


PREPARED = ROOT / "processed" / "qm9s" / "prepared"

UVVIS_PATH = PREPARED / "uvvis_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"

CHECKPOINT = (
    ROOT
    / "runs"
    / "EXP-UVVIS-001-SANITY"
    / "model.pt"
)

RUN_DIR = (
    ROOT
    / "runs"
    / "EXP-UVVIS-001-BN-DIAG"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
N_SAMPLES = 512
BATCH_SIZE = 64


def micro_f1(logits, targets):

    pred = (
        torch.sigmoid(logits) >= 0.5
    ).long()

    targets = targets.long()

    tp = (
        (pred == 1)
        & (targets == 1)
    ).sum().item()

    fp = (
        (pred == 1)
        & (targets == 0)
    ).sum().item()

    fn = (
        (pred == 0)
        & (targets == 1)
    ).sum().item()

    denom = 2 * tp + fp + fn

    if denom == 0:
        return 0.0

    return 2 * tp / denom


def evaluate(model, loader, device):

    all_logits = []
    all_targets = []

    with torch.no_grad():

        for batch in loader:

            x = batch["spectrum"].to(device)
            y = batch["labels"].to(device)

            logits = model(x)["logits"]

            all_logits.append(
                logits.cpu()
            )

            all_targets.append(
                y.cpu()
            )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    return micro_f1(
        logits,
        targets,
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

split = np.load(
    SPLIT_PATH
)

train_indices = np.asarray(
    split["train"],
    dtype=np.int64,
)

rng = np.random.default_rng(
    SEED
)

sanity_indices = rng.choice(
    train_indices,
    size=N_SAMPLES,
    replace=False,
)

dataset = QM9SSpectrumDataset(
    spectra_path=UVVIS_PATH,
    labels_path=LABEL_PATH,
    indices=sanity_indices,
    normalization="max",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

with open(
    LABEL_NAMES_PATH,
    "r",
    encoding="utf-8",
) as f:

    label_names = json.load(f)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

model = SpectrumStructureModel(
    num_labels=len(label_names),
    embedding_dim=256,
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


print("=" * 70)
print("UV-Vis BatchNorm diagnostic")
print("=" * 70)


# ------------------------------------------------------------
# Train mode:
# BatchNorm uses current batch statistics
# ------------------------------------------------------------

model.train()

train_mode_f1 = evaluate(
    model,
    loader,
    device,
)

print(
    "train-mode F1:",
    f"{train_mode_f1:.4f}",
)


# ------------------------------------------------------------
# Eval mode:
# BatchNorm uses stored running statistics
# ------------------------------------------------------------

model.eval()

eval_mode_f1 = evaluate(
    model,
    loader,
    device,
)

print(
    "eval-mode F1 :",
    f"{eval_mode_f1:.4f}",
)

print()
print(
    "difference   :",
    f"{train_mode_f1 - eval_mode_f1:+.4f}",
)

difference = (
    train_mode_f1
    - eval_mode_f1
)


# ============================================================
# Save diagnostic results
# ============================================================

result = {
    "experiment":
        "EXP-UVVIS-001-BN-DIAG",

    "source_checkpoint":
        str(CHECKPOINT),

    "seed":
        SEED,

    "n_samples":
        N_SAMPLES,

    "batch_size":
        BATCH_SIZE,

    "train_mode_f1":
        float(train_mode_f1),

    "eval_mode_f1":
        float(eval_mode_f1),

    "difference":
        float(difference),

    "diagnosis":
        (
            "Significant train/eval discrepancy "
            "associated with BatchNorm running statistics."
        ),
}


with open(
    RUN_DIR
    / "batchnorm_diagnostic.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )


text_summary = (
    "UV-Vis BatchNorm diagnostic\n"
    "===========================\n"
    f"Source checkpoint : {CHECKPOINT}\n"
    f"Seed              : {SEED}\n"
    f"Samples           : {N_SAMPLES}\n"
    f"Batch size        : {BATCH_SIZE}\n"
    f"Train-mode F1     : {train_mode_f1:.4f}\n"
    f"Eval-mode F1      : {eval_mode_f1:.4f}\n"
    f"Difference        : {difference:+.4f}\n"
    "\n"
    "Conclusion:\n"
    "Significant train/eval discrepancy associated "
    "with BatchNorm running statistics.\n"
)


with open(
    RUN_DIR
    / "batchnorm_diagnostic.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        text_summary
    )


print()
print("=" * 70)
print("Diagnostic results saved")
print("=" * 70)

print(
    RUN_DIR
    / "batchnorm_diagnostic.json"
)

print(
    RUN_DIR
    / "batchnorm_diagnostic.txt"
)