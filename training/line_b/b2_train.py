"""Shared B2 training logic for IR/Raman -> UVSPEC15.

Six thin seed/modality entry points call ``main`` from this module.  The trainer
is Validation-only: it never reads ``split["test"]``, never constructs a Test
Dataset/DataLoader, and never writes Test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.models.spectrum_cnn1d import SpectrumStructureModel

PREPARED = ROOT / "processed" / "qm9s" / "prepared"
SPLIT_PATH = PREPARED / "uvvis_scaffold_split_valid.npz"
LABEL_PATH = PREPARED / "uvspec15_labels.npy"
LABEL_NAMES_PATH = PREPARED / "uvspec15_label_names.json"
RUNS = ROOT / "runs"

MODALITIES = {
    "ir": {
        "spectra": PREPARED / "ir_float32.npy",
        "display": "IR",
        "input_points": 3501,
    },
    "raman": {
        "spectra": PREPARED / "raman_float32.npy",
        "display": "Raman",
        "input_points": 3501,
    },
}

SEEDS = (42, 123, 2026)
RUN_IDS = {
    ("ir", 42): "EXP-IR-003-UVSPEC15-FULL",
    ("ir", 123): "EXP-IR-003-UVSPEC15-SEED123",
    ("ir", 2026): "EXP-IR-003-UVSPEC15-SEED2026",
    ("raman", 42): "EXP-RAMAN-003-UVSPEC15-FULL",
    ("raman", 123): "EXP-RAMAN-003-UVSPEC15-SEED123",
    ("raman", 2026): "EXP-RAMAN-003-UVSPEC15-SEED2026",
}

BATCH_SIZE = 512
NUM_WORKERS = 4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 256
NORMALIZATION = "max"
THRESHOLD = 0.5
TEST_SIZE_EXPECTED = 12981
LABEL_SET = "uvspec15"

LOG_FIELDS = [
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
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split() -> tuple[np.ndarray, np.ndarray]:
    with np.load(SPLIT_PATH, allow_pickle=False) as archive:
        if "train" not in archive.files or "val" not in archive.files:
            raise KeyError(f"{SPLIT_PATH} lacks train/val keys")
        train_idx = np.asarray(archive["train"], dtype=np.int64)
        val_idx = np.asarray(archive["val"], dtype=np.int64)
    if len(train_idx) != 103853 or len(val_idx) != 12981:
        raise ValueError(f"Unexpected B2 split sizes: train={len(train_idx)}, val={len(val_idx)}")
    if len(np.unique(train_idx)) != len(train_idx) or len(np.unique(val_idx)) != len(val_idx):
        raise ValueError("Duplicate indices in B2 split")
    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise ValueError("Train/Validation index overlap")
    return train_idx, val_idx


def load_labels() -> list[str]:
    if not LABEL_NAMES_PATH.is_file():
        raise FileNotFoundError(LABEL_NAMES_PATH)
    names = read_json(LABEL_NAMES_PATH)
    if not isinstance(names, list) or len(names) != 15:
        raise ValueError(f"Expected 15 UVSPEC15 labels, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("UVSPEC15 label names contain duplicates")
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(LABEL_PATH)
    label_shape = np.load(LABEL_PATH, mmap_mode="r").shape
    if tuple(label_shape) != (129817, 15):
        raise ValueError(f"Unexpected UVSPEC15 label shape: {label_shape}")
    return names


def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str], threshold: float = 0.5) -> dict:
    targets = np.asarray(targets).astype(np.int64)
    probabilities = np.asarray(probabilities).astype(np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    per_label = []
    ap_values = []
    auroc_values = []
    for index, name in enumerate(label_names):
        truth = targets[:, index]
        prob = probabilities[:, index]
        pred = predictions[:, index]
        support = int(truth.sum())
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, pred, average="binary", zero_division=0
        )
        if np.unique(truth).size == 2:
            ap = float(average_precision_score(truth, prob))
            auroc = float(roc_auc_score(truth, prob))
            ap_values.append(ap)
            auroc_values.append(auroc)
        else:
            ap = None
            auroc = None
        per_label.append({
            "label_index": index,
            "label": name,
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_precision": ap,
            "auroc": auroc,
        })
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    macro_p = float(np.mean([row["precision"] for row in per_label]))
    macro_r = float(np.mean([row["recall"] for row in per_label]))
    macro_f1 = float(np.mean([row["f1"] for row in per_label]))
    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "per_label": per_label,
    }


def evaluate(model, loader, criterion, device, use_amp: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    targets = []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)["logits"]
                loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            n_samples += x.size(0)
            targets.append(y.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    metrics = calculate_metrics(np.concatenate(targets), np.concatenate(probabilities), load_labels(), THRESHOLD)
    metrics["loss"] = total_loss / max(n_samples, 1)
    return metrics, np.concatenate(targets), np.concatenate(probabilities)


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        x = batch["spectrum"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)["logits"]
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(n_samples, 1)


def build_config(modality: str, seed: int, run_id: str, train_idx: np.ndarray, val_idx: np.ndarray, labels: list[str], use_amp: bool) -> dict:
    return {
        "experiment": run_id,
        "run_id": run_id,
        "phase": "B2",
        "modality": modality,
        "seed": seed,
        "label_set": LABEL_SET,
        "n_labels": len(labels),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "embedding_dim": EMBEDDING_DIM,
        "normalization": NORMALIZATION,
        "threshold": THRESHOLD,
        "selection_metric": "validation_mAP",
        "use_pos_weight": False,
        "loss": "BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "amp": bool(use_amp),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": TEST_SIZE_EXPECTED,
        "test_split_accessed": False,
        "split_file": str(SPLIT_PATH.relative_to(ROOT)),
        "label_file": str(LABEL_PATH.relative_to(ROOT)),
        "label_names_file": str(LABEL_NAMES_PATH.relative_to(ROOT)),
        "spectra_file": str(MODALITIES[modality]["spectra"].relative_to(ROOT)),
        "label_names": labels,
    }


def prepare_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        if not run_dir.is_dir():
            raise FileExistsError(f"Run path is not a directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def write_validation_per_label(path: Path, metrics: dict, n_samples: int) -> None:
    fieldnames = [
        "label_index", "label", "positive_val_samples", "negative_val_samples",
        "precision", "recall", "f1", "average_precision", "auroc",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in metrics["per_label"]:
            writer.writerow({
                "label_index": row["label_index"],
                "label": row["label"],
                "positive_val_samples": row["support"],
                "negative_val_samples": n_samples - row["support"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "average_precision": "" if row["average_precision"] is None else row["average_precision"],
                "auroc": "" if row["auroc"] is None else row["auroc"],
            })


def write_run_manifest(run_dir: Path, config: dict, label_names: list[str], best_epoch: int, best_val_map: float, entrypoint: Path) -> None:
    wrapper = Path(entrypoint).resolve()
    if not wrapper.is_file():
        wrapper = Path(__file__).resolve()
    code_files = [Path(__file__).resolve(), wrapper, ROOT / "training/datasets/qm9s_spectrum_dataset.py", ROOT / "training/models/spectrum_cnn1d.py"]
    manifest = {
        "phase": "B2",
        "experiment": config["experiment"],
        "modality": config["modality"],
        "seed": config["seed"],
        "label_set": LABEL_SET,
        "n_labels": len(label_names),
        "label_names": label_names,
        "train_size": config["train_size"],
        "val_size": config["val_size"],
        "test_size": TEST_SIZE_EXPECTED,
        "test_split_accessed": False,
        "best_epoch": best_epoch,
        "best_val_mAP": best_val_map,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "split": {"path": config["split_file"], "sha256": sha256_file(SPLIT_PATH)},
            "labels": {"path": config["label_file"], "sha256": sha256_file(LABEL_PATH)},
            "label_names": {"path": config["label_names_file"], "sha256": sha256_file(LABEL_NAMES_PATH)},
            "spectra": {"path": config["spectra_file"], "shape": list(np.load(MODALITIES[config["modality"]]["spectra"], mmap_mode="r").shape)},
        },
        "outputs": {
            "config": {"path": "config.json", "sha256": sha256_file(run_dir / "config.json")},
            "training_log": {"path": "training_log.csv", "sha256": sha256_file(run_dir / "training_log.csv")},
            "checkpoint": {"path": "best_model.pt", "sha256": sha256_file(run_dir / "best_model.pt")},
            "validation_metrics": {"path": "validation_metrics.json", "sha256": sha256_file(run_dir / "validation_metrics.json")},
            "validation_per_label": {"path": "validation_per_label_metrics.csv", "sha256": sha256_file(run_dir / "validation_per_label_metrics.csv")},
        },
        "code": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_sanity(modality: str, seed: int, max_batches: int = 2) -> int:
    device = resolve_device()
    use_amp = device.type == "cuda"
    set_seed(seed)
    labels = load_labels()
    train_idx, val_idx = load_split()
    spectra = MODALITIES[modality]["spectra"]
    if not spectra.is_file():
        raise FileNotFoundError(spectra)
    if int(np.load(spectra, mmap_mode="r").shape[1]) != MODALITIES[modality]["input_points"]:
        raise ValueError(f"Unexpected input length for {modality}")
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(LABEL_PATH)

    with tempfile.TemporaryDirectory(prefix=f"b2-sanity-{modality}-{seed}-") as tmp:
        tmp_dir = Path(tmp)
        train_subset = train_idx[:max_batches * BATCH_SIZE]
        val_subset = val_idx[:max_batches * BATCH_SIZE]
        train_dataset = QM9SSpectrumDataset(spectra, LABEL_PATH, train_subset, NORMALIZATION)
        val_dataset = QM9SSpectrumDataset(spectra, LABEL_PATH, val_subset, NORMALIZATION)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_amp)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_amp)
        model = SpectrumStructureModel(num_labels=len(labels), embedding_dim=EMBEDDING_DIM).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        metrics, targets, probabilities = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - start
        if probabilities.shape[1] != 15:
            raise AssertionError(f"Expected 15 logits, got {probabilities.shape[1]}")
        config = build_config(modality, seed, RUN_IDS[(modality, seed)], train_idx, val_idx, labels, use_amp)
        checkpoint = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mAP": metrics["macro_ap"],
            "val_metrics": {
                "loss": metrics["loss"],
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "mAP": metrics["macro_ap"],
                "macro_AUROC": metrics["macro_auroc"],
            },
            "label_names": labels,
            "config": config,
        }
        ckpt_path = tmp_dir / "best_model.pt"
        torch.save(checkpoint, ckpt_path)
        reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config_ok = reloaded["config"] == config
        labels_ok = reloaded["label_names"] == labels

        (tmp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (tmp_dir / "training_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow({
                "epoch": 1,
                "train_loss": loss,
                "val_loss": metrics["loss"],
                "val_micro_f1": metrics["micro_f1"],
                "val_macro_f1": metrics["macro_f1"],
                "val_micro_precision": metrics["micro_precision"],
                "val_micro_recall": metrics["micro_recall"],
                "val_mAP": metrics["macro_ap"],
                "val_macro_AUROC": metrics["macro_auroc"],
                "learning_rate": LEARNING_RATE,
                "epoch_seconds": elapsed,
                "peak_cuda_memory_GB": 0.0,
            })
        validation_payload = {
            "experiment": config["experiment"],
            "phase": "B2",
            "modality": modality,
            "seed": seed,
            "label_set": LABEL_SET,
            "n_labels": len(labels),
            "n_samples": len(val_subset),
            "threshold": THRESHOLD,
            "best_epoch": 1,
            "best_val_mAP": metrics["macro_ap"],
            "val_loss": metrics["loss"],
            "val_micro_f1": metrics["micro_f1"],
            "val_macro_f1": metrics["macro_f1"],
            "val_mAP": metrics["macro_ap"],
            "val_macro_AUROC": metrics["macro_auroc"],
            "test_split_accessed": False,
        }
        (tmp_dir / "validation_metrics.json").write_text(
            json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_validation_per_label(tmp_dir / "validation_per_label_metrics.csv", metrics, len(val_subset))
        write_run_manifest(tmp_dir, config, labels, 1, metrics["macro_ap"], Path(sys.argv[0]))
        manifest = read_json(tmp_dir / "run_manifest.json")
        manifest_ok = (
            manifest["test_split_accessed"] is False
            and manifest["n_labels"] == 15
            and all((tmp_dir / item["path"]).is_file() for item in manifest["outputs"].values())
        )

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        cuda_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )
        report = {
            "status": "PASS",
            "modality": modality,
            "seed": seed,
            "run_id": RUN_IDS[(modality, seed)],
            "device": str(device),
            "spectra_shape": list(np.load(spectra, mmap_mode="r").shape),
            "label_shape": list(np.load(LABEL_PATH, mmap_mode="r").shape),
            "n_labels": len(labels),
            "train_rows_loaded": len(train_subset),
            "val_rows_loaded": len(val_subset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "logits_shape": list(probabilities.shape),
            "train_loss": loss,
            "val_loss": metrics["loss"],
            "val_mAP": metrics["macro_ap"],
            "val_macro_AUROC": metrics["macro_auroc"],
            "elapsed_seconds": elapsed,
            "process_rss_mb": rss_mb,
            "cuda_peak_mb": cuda_peak_mb,
            "checkpoint_reload_config_equal": config_ok,
            "checkpoint_reload_labels_equal": labels_ok,
            "run_manifest_ok": manifest_ok,
            "test_split_accessed": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0


def train(modality: str, seed: int, run_id: str, entrypoint: str) -> int:
    if (modality, seed) not in RUN_IDS or RUN_IDS[(modality, seed)] != run_id:
        raise ValueError(f"Unexpected run mapping: {modality}/{seed}/{run_id}")
    device = resolve_device()
    use_amp = device.type == "cuda"
    set_seed(seed)
    labels = load_labels()
    train_idx, val_idx = load_split()
    spectra = MODALITIES[modality]["spectra"]
    for path in (spectra, LABEL_PATH, LABEL_NAMES_PATH, SPLIT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    spectra_shape = np.load(spectra, mmap_mode="r").shape
    if tuple(spectra_shape) != (129817, MODALITIES[modality]["input_points"]):
        raise ValueError(f"Unexpected spectra shape for {modality}: {spectra_shape}")
    run_dir = RUNS / run_id
    prepare_run_dir(run_dir)
    config = build_config(modality, seed, run_id, train_idx, val_idx, labels, use_amp)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_dataset = QM9SSpectrumDataset(spectra, LABEL_PATH, train_idx, NORMALIZATION)
    val_dataset = QM9SSpectrumDataset(spectra, LABEL_PATH, val_idx, NORMALIZATION)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_amp)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_amp)

    model = SpectrumStructureModel(num_labels=len(labels), embedding_dim=EMBEDDING_DIM).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True

    log_path = run_dir / "training_log.csv"
    best_checkpoint_path = run_dir / "best_model.pt"
    best_val_map = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    with log_path.open("w", encoding="utf-8-sig", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_metrics, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
            current_map = val_metrics["macro_ap"]
            if current_map is None:
                raise RuntimeError("Validation mAP is undefined")
            epoch_seconds = time.time() - epoch_start
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_micro_f1": val_metrics["micro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_micro_precision": val_metrics["micro_precision"],
                "val_micro_recall": val_metrics["micro_recall"],
                "val_mAP": current_map,
                "val_macro_AUROC": val_metrics["macro_auroc"],
                "learning_rate": LEARNING_RATE,
                "epoch_seconds": epoch_seconds,
                "peak_cuda_memory_GB": peak_gb,
            })
            log_file.flush()
            print(f"Epoch {epoch:03d}/{MAX_EPOCHS} train_loss={train_loss:.5f} val_mAP={current_map:.4f}")
            if current_map > best_val_map:
                best_val_map = current_map
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mAP": best_val_map,
                    "val_metrics": {
                        "loss": val_metrics["loss"],
                        "micro_f1": val_metrics["micro_f1"],
                        "macro_f1": val_metrics["macro_f1"],
                        "mAP": current_map,
                        "macro_AUROC": val_metrics["macro_auroc"],
                    },
                    "label_names": labels,
                    "config": config,
                }
                torch.save(checkpoint, best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    break

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
    validation_payload = {
        "experiment": run_id,
        "phase": "B2",
        "modality": modality,
        "seed": seed,
        "label_set": LABEL_SET,
        "n_labels": len(labels),
        "n_samples": int(len(val_idx)),
        "threshold": THRESHOLD,
        "best_epoch": int(best_epoch),
        "best_val_mAP": float(best_val_map),
        "val_loss": val_metrics["loss"],
        "val_micro_f1": val_metrics["micro_f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_micro_precision": val_metrics["micro_precision"],
        "val_micro_recall": val_metrics["micro_recall"],
        "val_mAP": val_metrics["macro_ap"],
        "val_macro_AUROC": val_metrics["macro_auroc"],
        "test_split_accessed": False,
    }
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(validation_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_validation_per_label(run_dir / "validation_per_label_metrics.csv", val_metrics, len(val_idx))
    write_run_manifest(run_dir, config, labels, best_epoch, best_val_map, Path(entrypoint))
    print(json.dumps({
        "status": "COMPLETED",
        "run_id": run_id,
        "modality": modality,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mAP": best_val_map,
        "output": str(run_dir),
        "test_split_accessed": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main(modality: str, seed: int, run_id: str, entrypoint: str | None = None, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"B2 {modality} seed {seed} UVSPEC15 training")
    parser.add_argument("--sanity", action="store_true", help="Run a small temporary sanity check; never writes a formal run directory")
    parser.add_argument("--max-batches", type=int, default=2, help="Sanity batches per split")
    args = parser.parse_args(argv)
    if args.sanity:
        return run_sanity(modality, seed, max_batches=args.max_batches)
    return train(modality, seed, run_id, entrypoint or __file__)
