from pathlib import Path
import sys
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_SAMPLES = 512
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

PREPARED = ROOT / "processed" / "qm9s" / "prepared"

RAMAN_PATH = PREPARED / "raman_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "raman_scaffold_split_valid.npz"

RUN_DIR = ROOT / "runs" / "EXP-RAMAN-001-SANITY"
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("EXP-RAMAN-001-SANITY")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Data
# ============================================================

split = np.load(SPLIT_PATH)

train_indices = np.asarray(
    split["train"],
    dtype=np.int64
)

rng = np.random.default_rng(SEED)

sanity_indices = rng.choice(
    train_indices,
    size=N_SAMPLES,
    replace=False,
)

dataset = QM9SSpectrumDataset(
    spectra_path=RAMAN_PATH,
    labels_path=LABEL_PATH,
    indices=sanity_indices,
    normalization="max",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


with open(
    LABEL_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:
    label_names = json.load(f)


# ============================================================
# Inspect sanity subset
# ============================================================

all_subset_labels = np.load(
    LABEL_PATH,
    mmap_mode="r"
)[sanity_indices]

positive_counts = all_subset_labels.sum(axis=0)

print()
print("Sanity subset positive counts:")

for name, count in zip(
    label_names,
    positive_counts
):
    print(f"{name:15s}: {int(count):4d}")


# ============================================================
# Model
# ============================================================

model = SpectrumStructureModel(
    num_labels=len(label_names),
    embedding_dim=256,
).to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)


# ============================================================
# Metrics
# ============================================================

def micro_f1_from_logits(logits, targets):
    preds = (
        torch.sigmoid(logits) >= 0.5
    ).to(torch.int64)

    targets = targets.to(torch.int64)

    tp = (
        (preds == 1) & (targets == 1)
    ).sum().item()

    fp = (
        (preds == 1) & (targets == 0)
    ).sum().item()

    fn = (
        (preds == 0) & (targets == 1)
    ).sum().item()

    denom = 2 * tp + fp + fn

    if denom == 0:
        return 0.0

    return 2 * tp / denom


# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("Training")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    epoch_logits = []
    epoch_targets = []

    for batch in loader:

        x = batch["spectrum"].to(
            device,
            non_blocking=True
        )

        y = batch["labels"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(x)

        logits = out["logits"]

        loss = criterion(
            logits,
            y
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * x.size(0)
        )

        epoch_logits.append(
            logits.detach().cpu()
        )

        epoch_targets.append(
            y.detach().cpu()
        )

    avg_loss = (
        running_loss / len(dataset)
    )

    logits_all = torch.cat(
        epoch_logits,
        dim=0
    )

    targets_all = torch.cat(
        epoch_targets,
        dim=0
    )

    micro_f1 = micro_f1_from_logits(
        logits_all,
        targets_all
    )

    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"loss={avg_loss:.6f} | "
            f"micro-F1={micro_f1:.4f}"
        )


# ============================================================
# Final evaluation on the SAME 512 samples
# ============================================================

model.eval()

all_logits = []
all_targets = []

eval_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

with torch.no_grad():
    for batch in eval_loader:

        x = batch["spectrum"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)["logits"]

        all_logits.append(
            logits.cpu()
        )

        all_targets.append(
            y.cpu()
        )


all_logits = torch.cat(
    all_logits,
    dim=0
)

all_targets = torch.cat(
    all_targets,
    dim=0
)

final_f1 = micro_f1_from_logits(
    all_logits,
    all_targets
)


print()
print("=" * 70)
print("Sanity result")
print("=" * 70)

print("Final micro-F1:", f"{final_f1:.4f}")


checkpoint = {
    "model_state_dict":
        model.state_dict(),

    "label_names":
        label_names,

    "embedding_dim":
        256,

    "normalization":
        "max",

    "seed":
        SEED,
    
    "modality":
        "raman",
}

torch.save(
    checkpoint,
    RUN_DIR / "model.pt"
)

print()
print(
    "Checkpoint saved to:",
    RUN_DIR / "model.pt"
)
