"""Shared paths, hashing and validation helpers for Line B B1/B3.

Only Validation-split access is exposed by the Line B export entry points.
The module contains no Test-split loading helper.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "processed" / "qm9s" / "prepared"
RUNS = ROOT / "runs"

LABEL_SETS = {
    "shared14": {
        "task": "Shared14",
        "phase": "B1",
        "names": PREPARED / "functional_group_label_names.json",
        "labels": PREPARED / "functional_group_labels.npy",
        "n_labels": 14,
    },
    "uvspec15": {
        "task": "UVSPEC15",
        "phase": "B3",
        "names": PREPARED / "uvspec15_label_names.json",
        "labels": PREPARED / "uvspec15_labels.npy",
        "n_labels": 15,
    },
}

# Data configuration is selected by (modality, label_set), not by modality alone.
# Shared14 keeps each modality's native B1 split.  UVSPEC15 uses the common UV
# split for all three modalities, as fixed by B0 v0.2.
DATA_CONFIGS = {
    ("ir", "shared14"): {
        "spectra": PREPARED / "ir_float32.npy",
        "split": PREPARED / "ir_scaffold_split_valid.npz",
        "expected_train_size": 103855,
        "input_points": 3501,
    },
    ("raman", "shared14"): {
        "spectra": PREPARED / "raman_float32.npy",
        "split": PREPARED / "raman_scaffold_split_valid.npz",
        "expected_train_size": 103855,
        "input_points": 3501,
    },
    ("uvvis", "shared14"): {
        "spectra": PREPARED / "uvvis_float32.npy",
        "split": PREPARED / "uvvis_scaffold_split_valid.npz",
        "expected_train_size": 103853,
        "input_points": 701,
    },
    ("ir", "uvspec15"): {
        "spectra": PREPARED / "ir_float32.npy",
        "split": PREPARED / "uvvis_scaffold_split_valid.npz",
        "expected_train_size": 103853,
        "input_points": 3501,
    },
    ("raman", "uvspec15"): {
        "spectra": PREPARED / "raman_float32.npy",
        "split": PREPARED / "uvvis_scaffold_split_valid.npz",
        "expected_train_size": 103853,
        "input_points": 3501,
    },
    ("uvvis", "uvspec15"): {
        "spectra": PREPARED / "uvvis_float32.npy",
        "split": PREPARED / "uvvis_scaffold_split_valid.npz",
        "expected_train_size": 103853,
        "input_points": 701,
    },
}

# Exact registered model matrix for each task.  These names are used to reject
# WBCE/recalibrated checkpoints and partial or foreign run sets in full-matrix
# evaluation and summarization.
TASK_RUNS = {
    "shared14": {
        "ir": (
            "EXP-IR-001-FULL",
            "EXP-IR-001-SEED123",
            "EXP-IR-001-SEED2026",
        ),
        "raman": (
            "EXP-RAMAN-001-FULL",
            "EXP-RAMAN-001-SEED123",
            "EXP-RAMAN-001-SEED2026",
        ),
        "uvvis": (
            "EXP-UVVIS-001-FULL",
            "EXP-UVVIS-001-SEED123",
            "EXP-UVVIS-001-SEED2026",
        ),
    },
    "uvspec15": {
        "ir": (
            "EXP-IR-003-UVSPEC15-FULL",
            "EXP-IR-003-UVSPEC15-SEED123",
            "EXP-IR-003-UVSPEC15-SEED2026",
        ),
        "raman": (
            "EXP-RAMAN-003-UVSPEC15-FULL",
            "EXP-RAMAN-003-UVSPEC15-SEED123",
            "EXP-RAMAN-003-UVSPEC15-SEED2026",
        ),
        "uvvis": (
            "EXP-UVVIS-003-UVSPEC15-FULL",
            "EXP-UVVIS-003-UVSPEC15-SEED123",
            "EXP-UVVIS-003-UVSPEC15-SEED2026",
        ),
    },
}

MODALITIES = {
    "ir": {
        "spectra": PREPARED / "ir_float32.npy",
        "split": PREPARED / "ir_scaffold_split_valid.npz",
        "expected_train_size": 103855,
        "input_points": 3501,
        "display": "IR",
    },
    "raman": {
        "spectra": PREPARED / "raman_float32.npy",
        "split": PREPARED / "raman_scaffold_split_valid.npz",
        "expected_train_size": 103855,
        "input_points": 3501,
        "display": "Raman",
    },
    "uvvis": {
        "spectra": PREPARED / "uvvis_float32.npy",
        "split": PREPARED / "uvvis_scaffold_split_valid.npz",
        "expected_train_size": 103853,
        "input_points": 701,
        "display": "UV-Vis",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash dtype, shape and contiguous bytes (not just the raw payload)."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def relative_to_root(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def load_label_names(path: Path) -> list[str]:
    names = read_json(path)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"Invalid label-name list: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate label names in {path}")
    return names


def validate_label_names(actual: list[str], expected: list[str], where: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Label order mismatch in {where}:\n"
            f"  actual  = {actual}\n"
            f"  expected= {expected}"
        )


def load_validation_indices(
    modality: str,
    split_path: Path | None = None,
    expected_n: int | None = 12981,
) -> tuple[np.ndarray, Path]:
    """Load only the frozen `val` key.  The Test key is never read here."""
    config = MODALITIES[modality]
    path = Path(split_path) if split_path is not None else config["split"]

    with np.load(path, allow_pickle=False) as archive:
        if "val" not in archive.files:
            raise KeyError(f"Missing 'val' key in {path}")
        indices = np.asarray(archive["val"], dtype=np.int64)

    if expected_n is not None and len(indices) != expected_n:
        raise ValueError(f"Validation size {len(indices)} != expected {expected_n}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate validation row_index values")
    if int(indices.min()) < 0 or int(indices.max()) >= 129817:
        raise ValueError("Validation row_index out of range")

    return indices, path


def ensure_validation_split(split: str) -> None:
    if split != "validation":
        raise ValueError(
            "Line B export is validation-only; Test export/inference is disabled."
        )


def resolve_label_set(label_set: str) -> dict:
    if label_set not in LABEL_SETS:
        raise KeyError(f"Unknown label set: {label_set}")
    return LABEL_SETS[label_set]


def resolve_data_config(modality: str, label_set: str) -> dict:
    if modality not in MODALITIES:
        raise KeyError(f"Unknown modality: {modality}")
    if label_set not in LABEL_SETS:
        raise KeyError(f"Unknown label set: {label_set}")
    key = (modality, label_set)
    if key not in DATA_CONFIGS:
        raise KeyError(f"No data configuration for modality={modality}, label_set={label_set}")
    config = dict(DATA_CONFIGS[key])
    config["modality"] = modality
    config["label_set"] = label_set
    config["task"] = LABEL_SETS[label_set]["task"]
    config["n_labels"] = LABEL_SETS[label_set]["n_labels"]
    return config


def resolve_task_runs(label_set: str) -> dict[str, tuple[str, ...]]:
    if label_set not in TASK_RUNS:
        raise KeyError(f"Unknown label set: {label_set}")
    return TASK_RUNS[label_set]


def expected_seed_from_run_name(run_name: str) -> int:
    if run_name.endswith("-SEED2026"):
        return 2026
    if run_name.endswith("-SEED123"):
        return 123
    if run_name.endswith("-FULL"):
        return 42
    raise ValueError(f"Cannot derive a registered B3 seed from run name: {run_name}")


def load_validation_truth(modality: str, label_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only the validation row indices and their authoritative labels."""
    data_config = resolve_data_config(modality, label_set)
    with np.load(data_config["split"], allow_pickle=False) as archive:
        if "val" not in archive.files:
            raise KeyError(f"Missing 'val' key in {data_config['split']}")
        row_index = np.asarray(archive["val"], dtype=np.int64)
    label_config = resolve_label_set(label_set)
    labels_matrix = np.load(label_config["labels"], mmap_mode="r")
    if labels_matrix.ndim != 2 or labels_matrix.shape[1] != label_config["n_labels"]:
        raise ValueError(f"Unexpected label matrix shape: {labels_matrix.shape}")
    labels = np.asarray(labels_matrix[row_index], dtype=np.uint8)
    return row_index, labels


def validate_repeat_manifest(manifest: dict, where: str) -> dict:
    repeat = manifest.get("repeat_check")
    if not isinstance(repeat, dict):
        raise ValueError(f"Missing repeat_check in {where}")
    if repeat.get("bitwise_equal") is not True:
        raise ValueError(f"repeat_check.bitwise_equal is not true in {where}")
    max_abs = repeat.get("max_abs_diff")
    if max_abs is None or float(max_abs) != 0.0:
        raise ValueError(f"repeat_check.max_abs_diff is not 0.0 in {where}")
    return repeat


REQUIRED_MANIFEST_TRUE = (
    "n_rows_equals_validation_size",
    "row_index_equals_frozen_validation",
    "labels_equal_frozen_label_matrix",
    "probabilities_finite",
    "probabilities_in_unit_interval",
    "head_dim_equals_label_count",
    "checkpoint_label_order_verified",
    "checkpoint_file_unchanged",
    "model_state_unchanged",
    "bn_buffers_unchanged",
)
REQUIRED_MANIFEST_FALSE = (
    "test_split_accessed",
    "checkpoint_mutated",
    "bn_buffers_mutated",
)


def validate_manifest_checks(manifest: dict, where: str) -> None:
    checks = manifest.get("checks") or {}
    for key in REQUIRED_MANIFEST_TRUE:
        if checks.get(key) is not True:
            raise ValueError(f"Manifest check {key} is not true in {where}")
    for key in REQUIRED_MANIFEST_FALSE:
        if checks.get(key) is not False:
            raise ValueError(f"Manifest check {key} is not false in {where}")
    validate_repeat_manifest(manifest, where)


def validate_prediction_arrays(
    pred_dir: Path,
    manifest: dict,
    modality: str,
    label_set: str,
    expected_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-hash loaded arrays and compare them with manifest and authoritative data."""
    pred_dir = Path(pred_dir)
    label_config = resolve_label_set(label_set)
    names = expected_names or load_label_names(label_config["names"])
    if manifest.get("label_names") != names:
        raise ValueError(f"Manifest label order mismatch in {pred_dir}")
    if manifest.get("n_samples") != 12981:
        raise ValueError(f"Manifest n_samples is not 12981 in {pred_dir}")

    arrays = manifest.get("arrays") or {}
    for key in ("probabilities", "row_index", "labels"):
        if key not in arrays or "sha256" not in arrays[key]:
            raise ValueError(f"Manifest missing arrays.{key}.sha256 in {pred_dir}")

    probabilities = np.load(pred_dir / "probabilities.npy", allow_pickle=False)
    row_index = np.load(pred_dir / "row_index.npy", allow_pickle=False)
    labels = np.load(pred_dir / "labels.npy", allow_pickle=False)

    if sha256_array(probabilities) != arrays["probabilities"]["sha256"]:
        raise ValueError(f"probabilities SHA256 mismatch in {pred_dir}")
    if sha256_array(row_index) != arrays["row_index"]["sha256"]:
        raise ValueError(f"row_index SHA256 mismatch in {pred_dir}")
    if sha256_array(labels) != arrays["labels"]["sha256"]:
        raise ValueError(f"labels SHA256 mismatch in {pred_dir}")

    expected_row, expected_labels = load_validation_truth(modality, label_set)
    if probabilities.shape != (len(expected_row), len(names)):
        raise ValueError(f"Unexpected probability shape in {pred_dir}: {probabilities.shape}")
    if labels.shape != (len(expected_row), len(names)):
        raise ValueError(f"Unexpected label shape in {pred_dir}: {labels.shape}")
    if row_index.shape != (len(expected_row),):
        raise ValueError(f"Unexpected row_index shape in {pred_dir}: {row_index.shape}")
    if not np.array_equal(row_index, expected_row):
        raise ValueError(f"row_index is not the authoritative validation split in {pred_dir}")
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"labels are not the authoritative validation labels in {pred_dir}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities in {pred_dir}")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(f"Probabilities outside [0,1] in {pred_dir}")

    label_files = manifest.get("label_files") or {}
    if label_files.get("names_sha256") != sha256_file(label_config["names"]):
        raise ValueError(f"Label-name SHA256 mismatch in {pred_dir}")
    if label_files.get("matrix_sha256") != sha256_file(label_config["labels"]):
        raise ValueError(f"Label-matrix SHA256 mismatch in {pred_dir}")
    return probabilities, row_index, labels


@contextmanager
def atomic_output_dir(out_dir: Path) -> Iterator[Path]:
    """Create a fresh output dir atomically; never overwrite an existing one."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{out_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    if tmp.exists():
        raise FileExistsError(f"Temporary directory already exists: {tmp}")
    tmp.mkdir()

    try:
        yield tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    else:
        if out_dir.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise FileExistsError(f"Output directory appeared during export: {out_dir}")
        os.replace(tmp, out_dir)
