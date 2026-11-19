"""Line B prediction export (FP32; Shared14 and UVSPEC15).

Validation export is the frozen B1/B3 path.  B4 Test export is gated:

* ``--split validation`` keeps the exact frozen Validation behaviour (no Freeze
  requirement, ``test_split_accessed=false``);
* ``--split test`` requires ``--require-freeze`` plus an explicit FREEZE SHA256
  pin, verifies the model identity/checkpoint/config/split/labels/threshold
  against ``FREEZE.json``, writes into ``runs/LINE-B-TEST-PREDICTIONS/`` only,
  and appends one entry to ``runs/LINE-B-MANIFESTS/test_access_log.json``.
  The access log discloses the earlier B0 Test exposure and never claims a
  blind test.

Key safety properties
---------------------
* Data paths, split and labels come from an explicit (modality, label_set)
  configuration table; no arbitrary CLI path override is accepted.
* Checkpoint weights and BN buffers are never modified.
* FP32 forward, `model.eval()`, autocast disabled.
* Output directory must not exist; partial outputs are removed.
* Every array/checkpoint/split/label/code file is SHA256-recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.datasets.qm9s_spectrum_dataset import QM9SSpectrumDataset
from training.line_b.common import (
    LABEL_SETS,
    MODALITIES,
    ROOT as PROJECT_ROOT,
    atomic_output_dir,
    ensure_validation_split,
    expected_seed_from_run_name,
    load_label_names,
    load_validation_indices,
    read_json,
    relative_to_root,
    resolve_data_config,
    resolve_label_set,
    resolve_task_runs,
    sha256_array,
    sha256_file,
    validate_label_names,
    write_json,
)
from training.models.spectrum_cnn1d import SpectrumStructureModel


DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_WORKERS = 0
TEST_ROOT = PROJECT_ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
TEST_N = 12981
HISTORICAL_TEST_DISCLOSURE = {
    "historical_test_information_read_before_b4": True,
    "historical_read_scope": (
        "B0 stage consistency audit of existing per-label Test CSVs and existing "
        "Raman Test probabilities"
    ),
    "user_attestation_no_operational_interference": True,
    "blind_test_claim": False,
    "result_classification": "exploratory/post-hoc",
}


@dataclass
class ExportRequest:
    run_dir: Path
    modality: str
    label_set: str
    split: str
    out_dir: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    device: str = "auto"
    repeat_check: bool = True
    checkpoint_name: str = "best_model.pt"
    expected_n: int | None = 12981
    require_freeze: bool = False
    freeze_path: Path | None = None
    freeze_sha256: str | None = None
    test_access_log: Path | None = None


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def _checkpoint_state(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model_state_dict'")
    return checkpoint["model_state_dict"]


def _state_dict_sha256(state: dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(b"|")
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"|")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"|")
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
        digest.update(b"||")
    return digest.hexdigest()


def _build_model(
    checkpoint: dict,
    expected_label_names: list[str],
    device: torch.device,
) -> tuple[SpectrumStructureModel, dict]:
    state = _checkpoint_state(checkpoint)
    head = state.get("feature_head.weight")
    if head is None:
        raise KeyError("Missing feature_head.weight in checkpoint")

    head_dim = int(head.shape[0])
    if head_dim != len(expected_label_names):
        raise ValueError(
            f"Head dim {head_dim} != expected labels {len(expected_label_names)}"
        )

    checkpoint_names = checkpoint.get("label_names")
    if checkpoint_names is None:
        raise KeyError("Checkpoint is missing label_names")
    validate_label_names(checkpoint_names, expected_label_names, "checkpoint.label_names")

    config = checkpoint.get("config") or {}
    config_names = config.get("label_names")
    if config_names is not None:
        validate_label_names(config_names, expected_label_names, "checkpoint.config.label_names")

    embedding_dim = int(config.get("embedding_dim", head.shape[1]))
    if int(head.shape[1]) != embedding_dim:
        raise ValueError(
            f"Head width {head.shape[1]} != embedding_dim {embedding_dim}"
        )

    model = SpectrumStructureModel(
        num_labels=len(expected_label_names),
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.float()
    model.eval()

    bn_layers = sum(
        1 for key, value in state.items()
        if key.endswith("running_mean") and hasattr(value, "shape")
    )
    nbt_values = sorted(
        {
            int(value)
            for key, value in state.items()
            if key.endswith("num_batches_tracked")
        }
    )

    info = {
        "head_dim": head_dim,
        "embedding_dim": embedding_dim,
        "bn_running_mean_layers": bn_layers,
        "num_batches_tracked": nbt_values,
    }
    return model, info


def _forward_pass(
    model: SpectrumStructureModel,
    dataset: QM9SSpectrumDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    all_probabilities: list[np.ndarray] = []
    all_row_indices: list[np.ndarray] = []

    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    start_available = psutil.virtual_memory().available
    min_available = start_available

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            spectra = batch["spectrum"].to(device=device, dtype=torch.float32)
            if torch.is_autocast_enabled():
                raise RuntimeError("Autocast must be disabled for Line B FP32 export")

            logits = model(spectra)["logits"]
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()

            all_probabilities.append(probabilities.astype(np.float32, copy=False))
            all_row_indices.append(batch["row_index"].numpy().astype(np.int64, copy=False))

            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, psutil.virtual_memory().available)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    probabilities = np.concatenate(all_probabilities, axis=0).astype(np.float32, copy=False)
    row_indices = np.concatenate(all_row_indices, axis=0).astype(np.int64, copy=False)

    resources = {
        "elapsed_seconds": round(elapsed, 6),
        "process_rss_start_mb": round(start_rss / (1024**2), 3),
        "process_peak_rss_mb": round(peak_rss / (1024**2), 3),
        "system_available_start_mb": round(start_available / (1024**2), 3),
        "system_available_min_mb": round(min_available / (1024**2), 3),
    }
    if device.type == "cuda":
        resources.update(
            {
                "gpu_peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / (1024**2), 3
                ),
                "gpu_peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved(device) / (1024**2), 3
                ),
            }
        )
    return probabilities, row_indices, resources


def _array_metadata(array: np.ndarray) -> dict:
    return {
        "shape": [int(x) for x in array.shape],
        "dtype": str(array.dtype),
        "sha256": sha256_array(array),
        "finite": bool(np.isfinite(array).all()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _validate_checkpoint_config(
    checkpoint: dict,
    modality: str,
    label_set: str,
    run_name: str,
    data_config: dict,
    expected_label_names: list[str],
) -> dict:
    config = checkpoint.get("config") or {}
    expected_train_size = int(data_config["expected_train_size"])
    expected_seed = expected_seed_from_run_name(run_name)

    required = (
        "seed", "train_size", "val_size", "loss", "normalization",
        "embedding_dim", "label_names", "use_pos_weight", "threshold",
    )
    missing = [key for key in required if key not in config or config[key] is None]
    if missing:
        raise ValueError(f"Checkpoint config missing required fields: {missing}")
    if int(config["seed"]) != expected_seed:
        raise ValueError(
            f"Checkpoint config.seed {config['seed']} != expected {expected_seed}"
        )
    if int(config["train_size"]) != expected_train_size:
        raise ValueError(
            f"Checkpoint train_size {config['train_size']} != {expected_train_size}"
        )
    if int(config["val_size"]) != 12981:
        raise ValueError(f"Checkpoint val_size {config['val_size']} != 12981")
    if int(config["embedding_dim"]) != 256:
        raise ValueError(f"Checkpoint embedding_dim {config['embedding_dim']} != 256")
    if float(config["threshold"]) != 0.5:
        raise ValueError(f"Checkpoint threshold {config['threshold']} != 0.5")
    if config["use_pos_weight"] is not False:
        raise ValueError("B3 expects an ordinary BCE checkpoint with use_pos_weight=false")

    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type is not None and checkpoint_type != "original":
        raise ValueError(f"Expected Original BN checkpoint, got {checkpoint_type}")
    if "modality" in config and config["modality"] is not None:
        if config["modality"] != modality:
            raise ValueError(
                f"Checkpoint modality {config['modality']} != requested {modality}"
            )
    elif not (label_set == "shared14" and modality == "ir" and run_name.startswith("EXP-IR-001-")):
        # The frozen B1 IR checkpoints are the only documented historical format
        # without config.modality; their run registration still fixes modality=ir.
        raise ValueError("Checkpoint config.modality is missing")
    if "label_set" in config and config["label_set"] is not None:
        if config["label_set"] != label_set:
            raise ValueError(
                f"Checkpoint label_set {config['label_set']} != requested {label_set}"
            )
    if "n_labels" in config and config["n_labels"] is not None:
        if int(config["n_labels"]) != len(expected_label_names):
            raise ValueError(
                f"Checkpoint n_labels {config['n_labels']} != {len(expected_label_names)}"
            )
    if config["normalization"] != "max":
        raise ValueError("Checkpoint normalization is not 'max'")
    loss = str(config["loss"]).lower()
    if "bce" not in loss or "weighted" in loss or "wbce" in loss:
        raise ValueError(f"Unexpected/weighted loss: {config['loss']}")
    validate_label_names(
        config["label_names"], expected_label_names, "checkpoint.config.label_names"
    )
    return {
        "seed": config["seed"],
        "expected_seed": expected_seed,
        "modality": config.get("modality"),
        "label_set": config.get("label_set"),
        "n_labels": config.get("n_labels"),
        "loss": config["loss"],
        "normalization": config["normalization"],
        "threshold": config.get("threshold"),
        "train_size": config["train_size"],
        "val_size": config["val_size"],
        "test_size": config.get("test_size"),
        "embedding_dim": config["embedding_dim"],
        "use_pos_weight": config.get("use_pos_weight"),
        "amp": config.get("amp"),
        "label_names": config["label_names"],
    }


def _load_test_indices(split_path: Path, expected_n: int = TEST_N) -> np.ndarray:
    """Load only the frozen `test` key.  Used exclusively by the Freeze-gated B4 export."""
    with np.load(split_path, allow_pickle=False) as archive:
        if "test" not in archive.files:
            raise KeyError(f"Missing 'test' key in {split_path}")
        indices = np.asarray(archive["test"], dtype=np.int64)
    if len(indices) != expected_n:
        raise ValueError(f"Test size {len(indices)} != expected {expected_n}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate Test row_index values")
    if int(indices.min()) < 0 or int(indices.max()) >= 129817:
        raise ValueError("Test row_index out of range")
    return indices


def _verify_freeze_gate(
    freeze_path: Path,
    freeze_sha256: str | None,
    task: str,
    label_set: str,
    modality: str,
    run_name: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_sha256: str,
    split_path: Path,
    expected_names: list[str],
) -> dict:
    """Verify FREEZE pin and the exact frozen identity of the model being exported."""
    freeze_path = Path(freeze_path)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"FREEZE not found: {freeze_path}")
    actual_freeze_sha = sha256_file(freeze_path)
    if not freeze_sha256:
        raise ValueError("A FREEZE SHA256 pin is required for Test export")
    if actual_freeze_sha != freeze_sha256:
        raise ValueError(
            f"FREEZE SHA256 mismatch: actual {actual_freeze_sha} != pinned {freeze_sha256}"
        )

    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise ValueError("FREEZE is not a frozen B3 manifest")
    protocol_path = PROJECT_ROOT / freeze["protocol_path"]
    if not protocol_path.is_file() or sha256_file(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("Protocol file does not match the FREEZE protocol SHA256")

    models = freeze.get("models") or []
    if len(models) != 18:
        raise ValueError(f"FREEZE model count {len(models)} != 18")
    matches = [
        record
        for record in models
        if record.get("task") == task
        and record.get("label_set") == label_set
        and record.get("modality") == modality
        and record.get("run_name") == run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Run {task}/{run_name} is not uniquely registered in FREEZE")
    record = matches[0]
    if int(record["seed"]) != expected_seed_from_run_name(run_name):
        raise ValueError("FREEZE seed does not match the registered run seed")

    label_set_record = freeze["labels_and_data"]["label_sets"][label_set]

    def _norm(value: str) -> str:
        return str(value).replace("\\", "/")

    checks = {
        "checkpoint_path": _norm(record["checkpoint"]["path"]) == _norm(relative_to_root(checkpoint_path)),
        "checkpoint_sha256": record["checkpoint"]["sha256"] == checkpoint_sha256,
        "config_sha256": record["config"]["sha256"] == config_sha256,
        "split_path": _norm(record["split_file"]["path"]) == _norm(relative_to_root(split_path)),
        "split_sha256": record["split_file"]["sha256"] == sha256_file(split_path),
        "label_names_sha256": label_set_record["label_names"]["sha256"]
        == sha256_file(PROJECT_ROOT / label_set_record["label_names"]["path"]),
        "label_matrix_sha256": label_set_record["label_matrix"]["sha256"]
        == sha256_file(PROJECT_ROOT / label_set_record["label_matrix"]["path"]),
        "threshold_sha256": record["threshold"]["sha256"]
        == sha256_file(PROJECT_ROOT / record["threshold"]["path"]),
        "calibration_manifest_sha256": record["calibration_manifest"]["sha256"]
        == sha256_file(PROJECT_ROOT / record["calibration_manifest"]["path"]),
        "label_order": label_set_record["label_names"]["names"] == expected_names,
    }
    if not all(checks.values()):
        raise ValueError(f"FREEZE identity gate failed: {checks}")

    return {
        "path": relative_to_root(freeze_path),
        "sha256": actual_freeze_sha,
        "status": freeze["status"],
        "phase": freeze["phase"],
        "protocol_version": freeze["protocol_version"],
        "protocol_sha256": freeze["protocol_sha256"],
        "model_identity": {
            "task": task,
            "label_set": label_set,
            "modality": modality,
            "run_name": run_name,
            "seed": int(record["seed"]),
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "config_sha256": record["config"]["sha256"],
            "split_sha256": record["split_file"]["sha256"],
            "threshold_sha256": record["threshold"]["sha256"],
        },
        "checks": checks,
    }


def _test_access_log_path(request: ExportRequest) -> Path:
    if request.test_access_log is not None:
        return Path(request.test_access_log)
    return Path(request.freeze_path).parent / "test_access_log.json"


def _append_test_access_log(log_path: Path, entry: dict, freeze_info: dict) -> None:
    log_path = Path(log_path)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if log_path.exists():
        log = read_json(log_path)
        existing_freeze = (log.get("freeze") or {}).get("sha256")
        if existing_freeze != freeze_info["sha256"]:
            raise ValueError("Existing test access log belongs to a different FREEZE")
    else:
        log = {
            "schema_version": "1.0",
            "phase": "B4.1",
            "created_at": now,
            "freeze": {
                "path": freeze_info["path"],
                "sha256": freeze_info["sha256"],
                "status": freeze_info["status"],
                "protocol_version": freeze_info["protocol_version"],
                "protocol_sha256": freeze_info["protocol_sha256"],
            },
            "historical_exposure": dict(HISTORICAL_TEST_DISCLOSURE),
            "access_policy": {
                "one_export_per_model": True,
                "test_driven_tuning": False,
                "threshold_refit_on_test": False,
                "bootstrap_run": False,
                "freeze_hash_required": True,
            },
            "entries": [],
            "n_entries": 0,
        }
    entries = log.setdefault("entries", [])
    if any(
        item.get("task") == entry["task"] and item.get("run_name") == entry["run_name"]
        for item in entries
    ):
        raise ValueError(
            f"Test export already logged for {entry['task']}/{entry['run_name']}"
        )
    entries.append(entry)
    log["n_entries"] = len(entries)
    log["models_exported"] = sorted(
        {f"{item['task']}/{item['run_name']}" for item in entries}
    )
    log["updated_at"] = now
    write_json(log_path, log)


def export_predictions(request: ExportRequest) -> dict:
    if request.split == "validation":
        ensure_validation_split("validation")
    elif request.split == "test":
        if not request.require_freeze:
            raise ValueError("Test export requires --require-freeze")
        if not request.freeze_sha256:
            raise ValueError("Test export requires a --freeze-sha256 pin")
        out_resolved = Path(request.out_dir).resolve()
        if not out_resolved.is_relative_to(TEST_ROOT.resolve()):
            raise ValueError(f"Test output must live under {TEST_ROOT}")
    else:
        raise ValueError(f"Unsupported split: {request.split}")

    if request.modality not in MODALITIES:
        raise KeyError(f"Unknown modality: {request.modality}")

    data_config = resolve_data_config(request.modality, request.label_set)
    label_config = resolve_label_set(request.label_set)
    expected_names = load_label_names(label_config["names"])
    if len(expected_names) != label_config["n_labels"]:
        raise ValueError(f"Expected {label_config['n_labels']} labels")
    if data_config["n_labels"] != label_config["n_labels"]:
        raise ValueError("Data/label configuration mismatch")

    spectra_path = data_config["spectra"]
    split_path = data_config["split"]
    labels_path = label_config["labels"]

    if request.split == "validation":
        indices, split_path = load_validation_indices(
            request.modality,
            split_path=split_path,
            expected_n=request.expected_n,
        )
    else:
        split_path = Path(split_path)
        indices = _load_test_indices(
            split_path, expected_n=request.expected_n or TEST_N
        )

    labels_matrix = np.load(labels_path, allow_pickle=False)
    if labels_matrix.ndim != 2 or labels_matrix.shape[1] != len(expected_names):
        raise ValueError(
            f"Label matrix shape {labels_matrix.shape} incompatible with {len(expected_names)} names"
        )
    if labels_matrix.shape[0] != 129817:
        raise ValueError(f"Unexpected label matrix row count: {labels_matrix.shape[0]}")
    expected_labels = np.asarray(labels_matrix[indices], dtype=np.uint8)
    if not set(np.unique(expected_labels).tolist()).issubset({0, 1}):
        raise ValueError("Labels are not binary")

    run_dir = Path(request.run_dir)
    registered = resolve_task_runs(request.label_set)
    if request.modality not in registered or run_dir.name not in registered[request.modality]:
        raise ValueError(
            f"Run {run_dir.name} is not registered for "
            f"{request.label_set}/{request.modality}"
        )
    checkpoint_path = run_dir / request.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint_path)

    freeze_info = None
    if request.split == "test":
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        freeze_info = _verify_freeze_gate(
            freeze_path=Path(request.freeze_path),
            freeze_sha256=request.freeze_sha256,
            task=label_config["task"],
            label_set=request.label_set,
            modality=request.modality,
            run_name=run_dir.name,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha_before,
            config_sha256=sha256_file(config_path),
            split_path=Path(split_path),
            expected_names=expected_names,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _validate_checkpoint_config(
        checkpoint,
        request.modality,
        request.label_set,
        run_dir.name,
        data_config,
        expected_names,
    )
    if request.repeat_check is not True:
        raise ValueError("Formal Line B export requires the repeat inference check")
    state_before_hash = _state_dict_sha256(_checkpoint_state(checkpoint))

    device = _resolve_device(request.device)
    model, model_info = _build_model(checkpoint, expected_names, device)

    dataset = QM9SSpectrumDataset(
        spectra_path=spectra_path,
        labels_path=labels_path,
        indices=indices,
        normalization="max",
    )

    probabilities, row_index, resources = _forward_pass(
        model,
        dataset,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        device=device,
    )

    repeat_info = None
    if request.repeat_check:
        repeat_probabilities, repeat_rows, repeat_resources = _forward_pass(
            model,
            dataset,
            batch_size=request.batch_size,
            num_workers=request.num_workers,
            device=device,
        )
        if not np.array_equal(row_index, repeat_rows):
            raise ValueError("Repeat pass produced different row_index order")
        difference = np.abs(probabilities - repeat_probabilities)
        repeat_info = {
            "max_abs_diff": float(difference.max()) if difference.size else 0.0,
            "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
            "bitwise_equal": bool(np.array_equal(probabilities, repeat_probabilities)),
            "allclose_atol_1e-6": bool(np.allclose(probabilities, repeat_probabilities, atol=1e-6)),
            "second_pass_resources": repeat_resources,
        }
        if repeat_info["bitwise_equal"] is not True or repeat_info["max_abs_diff"] != 0.0:
            raise RuntimeError(
                "Repeat inference failed the B0/B1 acceptance standard "
                f"(max_abs_diff={repeat_info['max_abs_diff']}, "
                f"bitwise_equal={repeat_info['bitwise_equal']})"
            )

    # Hard alignment / sanity checks.
    if not np.array_equal(row_index, indices):
        raise ValueError("Exported row_index does not match frozen validation split")
    if probabilities.shape != (len(indices), len(expected_names)):
        raise ValueError(f"Unexpected probability shape {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError("Probabilities outside [0,1]")
    if not np.array_equal(expected_labels, np.asarray(labels_matrix[indices], dtype=np.uint8)):
        raise ValueError("Internal label alignment failure")

    checkpoint_sha_after = sha256_file(checkpoint_path)
    state_after_hash = _state_dict_sha256(model.state_dict())
    bn_after = sorted(
        {
            int(value)
            for key, value in model.state_dict().items()
            if key.endswith("num_batches_tracked")
        }
    )
    checkpoint_file_unchanged = checkpoint_sha_before == checkpoint_sha_after
    model_state_unchanged = state_before_hash == state_after_hash
    bn_buffers_unchanged = bn_after == model_info["num_batches_tracked"]
    if not checkpoint_file_unchanged:
        raise RuntimeError("Checkpoint file changed during export")
    if not model_state_unchanged:
        raise RuntimeError("Model state changed during export")
    if not bn_buffers_unchanged:
        raise RuntimeError("BN buffers changed during export")

    checkpoint_config = checkpoint.get("config") or {}
    checkpoint_val_metrics = checkpoint.get("val_metrics") or {}
    spectra_sha256 = sha256_file(spectra_path)

    manifest = {
        "task": label_config["task"],
        "label_set": request.label_set,
        "modality": request.modality,
        "run_name": run_dir.name,
        "split": "test" if request.split == "test" else "validation",
        "split_key": "test" if request.split == "test" else "val",
        "n_samples": int(len(indices)),
        "label_names": expected_names,
        "label_order_verified": True,
        "task_config": {
            "expected_train_size": int(data_config["expected_train_size"]),
            "input_points": int(data_config["input_points"]),
            "n_labels": int(data_config["n_labels"]),
        },
        "checkpoint": {
            "path": relative_to_root(checkpoint_path),
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "sha256": checkpoint_sha_after,
            "checkpoint_type": "original",
            "best_val_mAP": checkpoint.get("best_val_mAP"),
            "val_metrics": checkpoint_val_metrics,
            "config": checkpoint_config,
        },
        "split_file": {
            "path": relative_to_root(split_path),
            "sha256": sha256_file(split_path),
            "key": "test" if request.split == "test" else "val",
        },
        "label_files": {
            "names_path": relative_to_root(label_config["names"]),
            "names_sha256": sha256_file(label_config["names"]),
            "matrix_path": relative_to_root(labels_path),
            "matrix_sha256": sha256_file(labels_path),
        },
        "spectra_file": {
            "path": relative_to_root(spectra_path),
            "sha256": spectra_sha256,
            "shape": [int(x) for x in dataset.spectra.shape],
            "dtype": str(dataset.spectra.dtype),
            "size_bytes": int(spectra_path.stat().st_size),
        },
        "model": model_info,
        "inference": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "batch_size": int(request.batch_size),
            "num_workers": int(request.num_workers),
            "dtype": "float32",
            "autocast": False,
            "model_eval": True,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": sys.version,
            "resources": resources,
        },
        "repeat_check": repeat_info,
        "arrays": {
            "probabilities": _array_metadata(probabilities),
            "row_index": _array_metadata(row_index),
            "labels": _array_metadata(expected_labels),
        },
        "checks": (
            {
                "n_rows_equals_validation_size": bool(len(indices) == len(probabilities)),
                "row_index_equals_frozen_validation": bool(np.array_equal(row_index, indices)),
                "labels_equal_frozen_label_matrix": bool(
                    np.array_equal(expected_labels, np.asarray(labels_matrix[indices], dtype=np.uint8))
                ),
                "probabilities_finite": bool(np.isfinite(probabilities).all()),
                "probabilities_in_unit_interval": bool(
                    probabilities.min() >= 0.0 and probabilities.max() <= 1.0
                ),
                "head_dim_equals_label_count": bool(model_info["head_dim"] == len(expected_names)),
                "checkpoint_label_order_verified": True,
                "test_split_accessed": False,
                "checkpoint_file_unchanged": bool(checkpoint_file_unchanged),
                "model_state_unchanged": bool(model_state_unchanged),
                "bn_buffers_unchanged": bool(bn_buffers_unchanged),
                "checkpoint_mutated": bool(not checkpoint_file_unchanged),
                "bn_buffers_mutated": bool(not bn_buffers_unchanged),
            }
            if request.split == "validation"
            else {
                "n_rows_equals_test_size": bool(len(indices) == len(probabilities)),
                "row_index_equals_frozen_test": bool(np.array_equal(row_index, indices)),
                "row_index_unique": bool(len(np.unique(row_index)) == len(row_index)),
                "labels_equal_frozen_label_matrix": bool(
                    np.array_equal(expected_labels, np.asarray(labels_matrix[indices], dtype=np.uint8))
                ),
                "probabilities_finite": bool(np.isfinite(probabilities).all()),
                "probabilities_in_unit_interval": bool(
                    probabilities.min() >= 0.0 and probabilities.max() <= 1.0
                ),
                "label_order_equals_freeze": True,
                "head_dim_equals_label_count": bool(model_info["head_dim"] == len(expected_names)),
                "checkpoint_label_order_verified": True,
                "checkpoint_sha256_equals_freeze": bool(
                    checkpoint_sha_after == freeze_info["model_identity"]["checkpoint_sha256"]
                ),
                "freeze_sha256_verified": True,
                "test_split_accessed": True,
                "checkpoint_file_unchanged": bool(checkpoint_file_unchanged),
                "model_state_unchanged": bool(model_state_unchanged),
                "bn_buffers_unchanged": bool(bn_buffers_unchanged),
                "checkpoint_mutated": bool(not checkpoint_file_unchanged),
                "bn_buffers_mutated": bool(not bn_buffers_unchanged),
            }
        ),
    }

    if request.split == "test":
        manifest["freeze"] = freeze_info
        manifest["test_access"] = {
            **HISTORICAL_TEST_DISCLOSURE,
            "access_logged": True,
            "access_log_path": relative_to_root(_test_access_log_path(request)),
        }

    # Code hashes are recorded after all code is fixed.
    manifest["code"] = {
        "exporter_path": relative_to_root(Path(__file__)),
        "exporter_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(Path(__file__).with_name("common.py")),
        "dataset_sha256": sha256_file(PROJECT_ROOT / "training" / "datasets" / "qm9s_spectrum_dataset.py"),
        "model_sha256": sha256_file(PROJECT_ROOT / "training" / "models" / "spectrum_cnn1d.py"),
    }

    with atomic_output_dir(Path(request.out_dir)) as temporary:
        np.save(temporary / "probabilities.npy", probabilities, allow_pickle=False)
        np.save(temporary / "row_index.npy", row_index, allow_pickle=False)
        np.save(temporary / "labels.npy", expected_labels, allow_pickle=False)
        if request.split == "test":
            for key, name in (
                ("probabilities", "probabilities.npy"),
                ("row_index", "row_index.npy"),
                ("labels", "labels.npy"),
            ):
                saved = temporary / name
                manifest["arrays"][key]["file_sha256"] = sha256_file(saved)
                manifest["arrays"][key]["file_size_bytes"] = int(saved.stat().st_size)
        write_json(temporary / "prediction_manifest.json", manifest)

    if request.split == "test":
        out_dir = Path(request.out_dir)
        log_path = _test_access_log_path(request)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task": label_config["task"],
            "label_set": request.label_set,
            "modality": request.modality,
            "run_name": run_dir.name,
            "seed": int(checkpoint_config["seed"]),
            "split": "test",
            "n_samples": int(len(indices)),
            "freeze_sha256": freeze_info["sha256"],
            "checkpoint_sha256": checkpoint_sha_after,
            "split_path": relative_to_root(split_path),
            "split_sha256": sha256_file(split_path),
            "test_row_index_sha256": sha256_array(indices),
            "probabilities_sha256": manifest["arrays"]["probabilities"]["sha256"],
            "probabilities_file_sha256": manifest["arrays"]["probabilities"]["file_sha256"],
            "labels_sha256": manifest["arrays"]["labels"]["sha256"],
            "output_dir": relative_to_root(out_dir),
            "manifest_sha256": sha256_file(out_dir / "prediction_manifest.json"),
            "exporter_sha256": manifest["code"]["exporter_sha256"],
            "command": " ".join(sys.argv),
            "historical_exposure_disclosed": True,
            "blind_test_claim": False,
            "test_driven_tuning": False,
            "thresholds_modified": False,
        }
        _append_test_access_log(log_path, entry, freeze_info)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    parser.add_argument("--label-set", default="shared14", choices=sorted(LABEL_SETS))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--num-workers", default=DEFAULT_NUM_WORKERS, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument(
        "--require-freeze",
        action="store_true",
        help="Require the B3 FREEZE hash pin and verify the frozen model identity.",
    )
    parser.add_argument(
        "--freeze-path",
        type=Path,
        default=PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json",
    )
    parser.add_argument("--freeze-sha256", default=None)
    parser.add_argument("--test-access-log", type=Path, default=None)
    parser.add_argument(
        "--repeat-check",
        dest="repeat_check",
        action="store_true",
        default=True,
        help="Run a second validation forward pass and report numerical differences.",
    )
    parser.add_argument(
        "--no-repeat-check",
        dest="repeat_check",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split == "test":
        if not args.require_freeze:
            raise SystemExit("--split test requires --require-freeze")
        if not args.freeze_sha256:
            raise SystemExit("--split test requires --freeze-sha256")
    request = ExportRequest(
        run_dir=args.run_dir,
        modality=args.modality,
        label_set=args.label_set,
        split=args.split,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        repeat_check=args.repeat_check,
        checkpoint_name=args.checkpoint_name,
        require_freeze=args.require_freeze,
        freeze_path=args.freeze_path,
        freeze_sha256=args.freeze_sha256,
        test_access_log=args.test_access_log,
    )
    manifest = export_predictions(request)
    summary = {
        "run_name": manifest["run_name"],
        "modality": manifest["modality"],
        "split": manifest["split"],
        "n_samples": manifest["n_samples"],
        "out_dir": str(args.out_dir),
        "probabilities_sha256": manifest["arrays"]["probabilities"]["sha256"],
        "repeat_check": manifest["repeat_check"],
        "resources": manifest["inference"]["resources"],
    }
    if manifest["split"] == "test":
        summary["freeze_sha256"] = manifest["freeze"]["sha256"]
        summary["test_access_log"] = manifest["test_access"]["access_log_path"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
