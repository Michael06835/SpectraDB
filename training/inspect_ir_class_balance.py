from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PREPARED = ROOT / "processed" / "qm9s" / "prepared"

LABEL_PATH = PREPARED / "functional_group_labels.npy"
LABEL_NAMES_PATH = PREPARED / "functional_group_label_names.json"
SPLIT_PATH = PREPARED / "ir_scaffold_split_valid.npz"


def main():

    labels = np.load(
        LABEL_PATH,
        mmap_mode="r"
    )

    split = np.load(
        SPLIT_PATH
    )

    train_idx = np.asarray(
        split["train"],
        dtype=np.int64
    )

    with open(
        LABEL_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        label_names = json.load(f)

    train_labels = np.asarray(
        labels[train_idx],
        dtype=np.float32
    )

    n_train = len(train_idx)

    positives = train_labels.sum(axis=0)
    negatives = n_train - positives

    pos_weights = (
        negatives
        / np.maximum(positives, 1)
    )

    print()
    print("=" * 82)
    print("IR TRAINING SET CLASS BALANCE")
    print("=" * 82)

    print()
    print("Training samples:", n_train)
    print()

    print(
        f"{'label':15s}"
        f"{'positive':>12s}"
        f"{'negative':>12s}"
        f"{'positive %':>14s}"
        f"{'pos_weight':>14s}"
    )

    print("-" * 67)

    for (
        name,
        pos,
        neg,
        weight
    ) in zip(
        label_names,
        positives,
        negatives,
        pos_weights
    ):

        percentage = (
            pos / n_train * 100
        )

        print(
            f"{name:15s}"
            f"{int(pos):12d}"
            f"{int(neg):12d}"
            f"{percentage:13.3f}%"
            f"{weight:14.2f}"
        )


if __name__ == "__main__":
    main()