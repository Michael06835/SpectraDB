from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class QM9SSpectrumDataset(Dataset):
    """
    QM9S single-modality spectrum dataset.

    Input:
        spectrum: [1, L], float32

    Target:
        functional-group multi-label vector: [N_labels], float32

    The stored row_index is preserved so that spectra, molecular
    structures, and annotations remain aligned.
    """

    def __init__(
        self,
        spectra_path,
        labels_path,
        indices,
        normalization=None,
    ):
        self.spectra_path = Path(spectra_path)
        self.labels_path = Path(labels_path)

        # mmap avoids loading the entire spectral matrix into RAM.
        self.spectra = np.load(
            self.spectra_path,
            mmap_mode="r"
        )

        self.labels = np.load(
            self.labels_path,
            mmap_mode="r"
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

        self.normalization = normalization

        if self.spectra.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Row mismatch: spectra={self.spectra.shape[0]}, "
                f"labels={self.labels.shape[0]}"
            )

        if np.any(self.indices < 0):
            raise ValueError("Negative sample index detected.")

        if np.any(self.indices >= self.spectra.shape[0]):
            raise ValueError("Sample index out of range.")

    def __len__(self):
        return len(self.indices)

    def _normalize(self, x):
        if self.normalization is None:
            return x

        if self.normalization == "max":
            vmax = np.max(np.abs(x))
            if vmax > 0:
                x = x / vmax
            return x

        raise ValueError(
            f"Unknown normalization mode: {self.normalization}"
        )

    def __getitem__(self, item):
        row_index = int(self.indices[item])

        x = np.asarray(
            self.spectra[row_index],
            dtype=np.float32
        ).copy()

        y = np.asarray(
            self.labels[row_index],
            dtype=np.float32
        ).copy()

        x = self._normalize(x)

        # Conv1d expects [channels, length].
        x = torch.from_numpy(x).unsqueeze(0)

        # BCEWithLogitsLoss expects floating-point targets.
        y = torch.from_numpy(y)

        return {
            "spectrum": x,
            "labels": y,
            "row_index": row_index,
        }


def load_label_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
