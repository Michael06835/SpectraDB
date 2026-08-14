from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import (
    QM9SSpectrumDataset,
    load_label_names,
)


PREPARED = ROOT / "processed" / "qm9s" / "prepared"

IR_PATH = PREPARED / "ir_float32.npy"
LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "ir_scaffold_split_valid.npz"


def identify_split_keys(split):
    """
    Support common split-key naming conventions.
    """

    candidates = {
        "train": [
            "train",
            "train_idx",
            "train_indices",
        ],
        "val": [
            "val",
            "valid",
            "validation",
            "val_idx",
            "val_indices",
        ],
        "test": [
            "test",
            "test_idx",
            "test_indices",
        ],
    }

    result = {}

    for role, names in candidates.items():
        found = [
            name for name in names
            if name in split.files
        ]

        if len(found) != 1:
            raise RuntimeError(
                f"Could not uniquely identify '{role}' split. "
                f"Available keys: {split.files}"
            )

        result[role] = found[0]

    return result


print("=" * 70)
print("QM9S IR DataLoader validation")
print("=" * 70)

split = np.load(SPLIT_PATH)

print()
print("Split file:")
print(SPLIT_PATH)

print()
print("Available split keys:")
print(split.files)

keys = identify_split_keys(split)

train_idx = split[keys["train"]]
val_idx = split[keys["val"]]
test_idx = split[keys["test"]]

print()
print("Resolved split keys:")
print("train:", keys["train"])
print("val  :", keys["val"])
print("test :", keys["test"])

print()
print("Split sizes:")
print("train:", len(train_idx))
print("val  :", len(val_idx))
print("test :", len(test_idx))
print("total:", len(train_idx) + len(val_idx) + len(test_idx))

label_names = load_label_names(LABEL_NAMES_PATH)

print()
print("Labels:")
print(label_names)
print("Number of labels:", len(label_names))


train_dataset = QM9SSpectrumDataset(
    spectra_path=IR_PATH,
    labels_path=LABEL_PATH,
    indices=train_idx,

    # For now we only validate raw data loading.
    # Normalization will be fixed before formal training.
    normalization=None,
)

val_dataset = QM9SSpectrumDataset(
    spectra_path=IR_PATH,
    labels_path=LABEL_PATH,
    indices=val_idx,
    normalization=None,
)

test_dataset = QM9SSpectrumDataset(
    spectra_path=IR_PATH,
    labels_path=LABEL_PATH,
    indices=test_idx,
    normalization=None,
)


loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


batch = next(iter(loader))

x = batch["spectrum"]
y = batch["labels"]
idx = batch["row_index"]


print()
print("=" * 70)
print("First training batch")
print("=" * 70)

print("spectrum shape:", x.shape)
print("spectrum dtype:", x.dtype)

print("labels shape  :", y.shape)
print("labels dtype  :", y.dtype)

print("row_index shape:", idx.shape)

print()
print("Spectrum statistics:")
print("min :", x.min().item())
print("max :", x.max().item())
print("mean:", x.mean().item())
print("std :", x.std().item())

print()
print("First 10 row indices:")
print(idx[:10].tolist())

print()
print("First 5 samples' active labels:")

for i in range(min(5, len(idx))):
    active = [
        label_names[j]
        for j in torch.where(y[i] > 0.5)[0].tolist()
    ]

    print(
        f"row_index={idx[i].item():6d} -> {active}"
    )


print()
print("=" * 70)
print("CUDA information")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version    :", torch.version.cuda)
    print("GPU             :", torch.cuda.get_device_name(0))


print()
print("DataLoader validation completed successfully.")
