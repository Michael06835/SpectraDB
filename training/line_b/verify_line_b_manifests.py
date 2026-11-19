"""Unified, read-only validation-only asset verification for Line B.

Coverage:
* 18 registered Shared14 / UVSPEC15 models;
* checkpoints, configs, FP32 validation arrays, prediction manifests;
* 261 Validation-F1 threshold records;
* B3-A formal complementarity outputs;
* B1/B3 audit manifests and B3.5-A code provenance;
* raw prepared spectra, labels, splits, manifest and structure-library hashes.

This tool never loads Test indices, never constructs a Test dataset, and never
rewrites an existing formal asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import torch
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    MODALITIES,
    PREPARED,
    ROOT as PROJECT_ROOT,
    TASK_RUNS,
    expected_seed_from_run_name,
    load_label_names,
    read_json,
    resolve_data_config,
    resolve_label_set,
    sha256_array,
    sha256_file,
    validate_manifest_checks,
    validate_prediction_arrays,
    write_json,
)

PHASE = "B3.5-B"
DEFAULT_JSON = PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B3" / "b3_5b_manifest_verification_v2.json"
PRIOR_JSON = PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B3" / "b3_5b_manifest_verification.json"
N_SAMPLES = 12981
N_LABEL_ROWS = 129817
CODE_FILES = (
    "training/line_b/common.py",
    "training/line_b/metrics.py",
    "training/line_b/export_line_b_predictions.py",
    "training/line_b/evaluate_line_b_multilabel.py",
    "training/line_b/summarize_b1_exports.py",
    "training/line_b/calibrate_line_b_thresholds.py",
    "training/line_b/analyze_line_b_complementarity.py",
)


B3A_ANALYSIS_ROOT_SUFFIX = "runs/LINE-B-ANALYSIS"
B3A_ADDITIVE_SUBTREE = "validation_development"


class Checks:
    def __init__(self):
        self.items: list[dict[str, Any]] = []

    def add(self, check_id: str, status: str, detail: str = "", **extra: Any) -> None:
        item = {"id": check_id, "status": status, "detail": detail}
        item.update(extra)
        self.items.append(item)

    def passed(self, check_id: str, detail: str = "", **extra: Any) -> None:
        self.add(check_id, "PASS", detail, **extra)

    def failed(self, check_id: str, detail: str = "", **extra: Any) -> None:
        self.add(check_id, "FAILED", detail, **extra)

    def accepted(self, check_id: str, detail: str, **extra: Any) -> None:
        self.add(check_id, "ACCEPTED_HISTORICAL_LIMITATION", detail, **extra)

    def missing(self, check_id: str, detail: str, **extra: Any) -> None:
        self.add(check_id, "MISSING", detail, **extra)

    def has_failures(self) -> bool:
        return any(item["status"] == "FAILED" for item in self.items)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def _stream_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(root: Path, exclude: set[str] | None = None) -> dict[str, Any]:
    files = {}
    root = Path(root)
    excluded = set(exclude or [])
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and str(p.relative_to(root)) not in excluded:
                files[str(p.relative_to(root))] = _stream_sha(p)
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(files[rel].encode("ascii")); h.update(b"\n")
    return {"tree_sha256": h.hexdigest(), "n_files": len(files)}


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cwd": str(PROJECT_ROOT),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short").splitlines(),
    }


def _check_config(config: dict, expected_seed: int, label_set: str, data_config: dict, expected_names: list[str], checks: Checks, prefix: str, run_name: str) -> None:
    required = ("seed", "train_size", "val_size", "loss", "normalization", "embedding_dim", "label_names", "use_pos_weight", "threshold")
    missing = [k for k in required if k not in config or config[k] is None]
    if missing:
        checks.failed(f"{prefix}.required_fields", f"missing={missing}")
        return
    if int(config["seed"]) != expected_seed:
        checks.failed(f"{prefix}.seed", f"{config['seed']} != {expected_seed}")
    else:
        checks.passed(f"{prefix}.seed", str(expected_seed))
    if int(config["train_size"]) != int(data_config["expected_train_size"]):
        checks.failed(f"{prefix}.train_size", f"{config['train_size']} != {data_config['expected_train_size']}")
    else:
        checks.passed(f"{prefix}.train_size")
    if int(config["val_size"]) != N_SAMPLES:
        checks.failed(f"{prefix}.val_size", str(config["val_size"]))
    if int(config["embedding_dim"]) != 256:
        checks.failed(f"{prefix}.embedding_dim", str(config["embedding_dim"]))
    if float(config["threshold"]) != 0.5:
        checks.failed(f"{prefix}.threshold", str(config["threshold"]))
    if config["use_pos_weight"] is not False:
        checks.failed(f"{prefix}.use_pos_weight", str(config["use_pos_weight"]))
    if "bce" not in str(config["loss"]).lower() or "weighted" in str(config["loss"]).lower() or "wbce" in str(config["loss"]).lower():
        checks.failed(f"{prefix}.loss", str(config["loss"]))
    if config["normalization"] != "max":
        checks.failed(f"{prefix}.normalization", str(config["normalization"]))
    if config.get("label_names") != expected_names:
        checks.failed(f"{prefix}.label_names", "label order differs")
    if "label_set" in config and config["label_set"] is not None and config["label_set"] != label_set:
        checks.failed(f"{prefix}.label_set", f"{config['label_set']} != {label_set}")
    if "n_labels" in config and config["n_labels"] is not None and int(config["n_labels"]) != len(expected_names):
        checks.failed(f"{prefix}.n_labels", str(config["n_labels"]))
    if config.get("modality") is None:
        if label_set == "shared14" and run_name.startswith("EXP-IR-001-"):
            checks.accepted(f"{prefix}.modality", "B1 IR historical config has no modality; registration fixes modality=ir")
        else:
            checks.failed(f"{prefix}.modality", "missing modality")
    elif config["modality"] != data_config["modality"]:
        checks.failed(f"{prefix}.modality", f"{config['modality']} != {data_config['modality']}")


def _resolve_record_path(record: dict) -> Path | None:
    value = record.get("path")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _verify_calibration_assets(calibration: dict, model_dir: Path, checks: Checks, prefix: str) -> None:
    if calibration.get("all_checks_pass") is not True:
        checks.failed(f"{prefix}.calibration_all_checks")
    expected_inputs = {"checkpoint", "label_matrix", "label_names", "labels", "prediction_manifest", "probabilities", "row_index", "split"}
    inputs = calibration.get("inputs") or {}
    if set(inputs) != expected_inputs:
        checks.failed(f"{prefix}.calibration_inputs_keys", str(sorted(inputs)))
    for name, record in inputs.items():
        path = _resolve_record_path(record)
        if path is None or not path.is_file():
            checks.failed(f"{prefix}.calibration_input_{name}.exists", str(path))
        elif record.get("sha256") != _stream_sha(path):
            checks.failed(f"{prefix}.calibration_input_{name}.sha", "actual != recorded")
    expected_outputs = {
        "optimal_thresholds": model_dir / "optimal_thresholds.json",
        "validation_overall": model_dir / "validation_overall.json",
        "validation_per_label_comparison": model_dir / "validation_per_label_comparison.csv",
    }
    outputs = calibration.get("outputs") or {}
    if set(outputs) != set(expected_outputs):
        checks.failed(f"{prefix}.calibration_outputs_keys", str(sorted(outputs)))
    for name, path in expected_outputs.items():
        if not path.is_file():
            checks.failed(f"{prefix}.calibration_output_{name}.exists", str(path))
        elif _stream_sha(path) != outputs.get(name):
            checks.failed(f"{prefix}.calibration_output_{name}.sha", "actual != recorded")


def _check_threshold(
    threshold_path: Path,
    calibration_path: Path,
    task_summary: dict,
    manifest: dict,
    expected_names: list[str],
    expected_seed: int,
    checkpoint_sha: str,
    checks: Checks,
    prefix: str,
) -> dict:
    if not threshold_path.is_file():
        checks.failed(f"{prefix}.threshold_exists", str(threshold_path))
        return {}
    actual = _stream_sha(threshold_path)
    summary_rows = [row for row in task_summary.get("models", []) if row.get("run_name") == manifest["run_name"]]
    if len(summary_rows) != 1:
        checks.failed(f"{prefix}.threshold_summary_unique", f"{len(summary_rows)} records")
    elif summary_rows[0].get("thresholds_sha256") != actual:
        checks.failed(f"{prefix}.threshold_summary_sha", "threshold summary hash mismatch")
    if not calibration_path.is_file():
        checks.failed(f"{prefix}.calibration_manifest_exists", str(calibration_path))
        return {}
    calibration = read_json(calibration_path)
    _verify_calibration_assets(calibration, threshold_path.parent, checks, prefix)
    if (calibration.get("outputs") or {}).get("optimal_thresholds") != actual:
        checks.failed(f"{prefix}.calibration_manifest_sha", "calibration manifest hash mismatch")
    if calibration.get("test_split_accessed") is not False:
        checks.failed(f"{prefix}.calibration_test_flag", "not false")
    payload = read_json(threshold_path)
    if payload.get("task") != manifest.get("task") or payload.get("label_set") != manifest.get("label_set"):
        checks.failed(f"{prefix}.threshold_task", "task/label_set mismatch")
    if payload.get("run_name") != manifest.get("run_name") or payload.get("modality") != manifest.get("modality"):
        checks.failed(f"{prefix}.threshold_identity", "run/modality mismatch")
    if int(payload.get("seed")) != expected_seed:
        checks.failed(f"{prefix}.threshold_seed", str(payload.get("seed")))
    if payload.get("split") != "validation" or payload.get("test_split_accessed") is not False:
        checks.failed(f"{prefix}.threshold_split_test", "split/test flag invalid")
    if payload.get("label_names") != expected_names:
        checks.failed(f"{prefix}.threshold_label_names", "label order mismatch")
    threshold_map = payload.get("thresholds") or {}
    if list(threshold_map.keys()) != expected_names:
        checks.failed(f"{prefix}.threshold_keys", "threshold keys mismatch")
    values = np.asarray(list(threshold_map.values()), dtype=float)
    if len(values) != len(expected_names) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        checks.failed(f"{prefix}.threshold_values", "invalid threshold values")
    if payload.get("probabilities_sha256") != manifest["arrays"]["probabilities"]["sha256"]:
        checks.failed(f"{prefix}.threshold_probability_sha")
    if payload.get("row_index_sha256") != manifest["arrays"]["row_index"]["sha256"]:
        checks.failed(f"{prefix}.threshold_row_index_sha")
    if payload.get("labels_sha256") != manifest["arrays"]["labels"]["sha256"]:
        checks.failed(f"{prefix}.threshold_labels_sha")
    if payload.get("checkpoint_sha256") != checkpoint_sha:
        checks.failed(f"{prefix}.threshold_checkpoint_sha")
    return {"path": str(threshold_path), "sha256": actual, "calibration_path": str(calibration_path), "calibration_sha256": _stream_sha(calibration_path), "count": len(values)}


def verify_model(task: str, label_set: str, modality: str, run_name: str, config_root: Path, predictions_root: Path, thresholds_root: Path, provenance_assets: dict | None = None) -> dict:
    checks = Checks()
    data_config = resolve_data_config(modality, label_set)
    expected_names = load_label_names(resolve_label_set(label_set)["names"])
    expected_seed = expected_seed_from_run_name(run_name)
    run_root = config_root / run_name
    config_path = run_root / "config.json"
    checkpoint_path = run_root / "best_model.pt"
    pred_dir = predictions_root / task / modality.upper() / run_name / "validation"
    manifest_path = pred_dir / "prediction_manifest.json"
    result: dict[str, Any] = {
        "task": task, "label_set": label_set, "modality": modality, "run_name": run_name,
        "seed_expected": expected_seed, "checks": checks.items,
    }
    try:
        if not config_path.is_file(): checks.failed("config.exists", str(config_path))
        if not checkpoint_path.is_file(): checks.failed("checkpoint.exists", str(checkpoint_path))
        if not manifest_path.is_file(): checks.failed("prediction_manifest.exists", str(manifest_path))
        if checks.has_failures():
            result["status"] = "FAILED"; return result

        config = read_json(config_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config") or {}
        if config != checkpoint_config:
            checks.failed("config.checkpoint_equal", "config.json != checkpoint.config")
        _check_config(config, expected_seed, label_set, data_config, expected_names, checks, "config", run_name)

        checkpoint_sha = _stream_sha(checkpoint_path)
        state = checkpoint.get("model_state_dict") or {}
        head = state.get("feature_head.weight")
        if head is None or tuple(head.shape) != (len(expected_names), 256):
            checks.failed("checkpoint.head", str(None if head is None else tuple(head.shape)))
        bn = sum(1 for key in state if key.endswith("running_mean"))
        if bn != 12:
            checks.failed("checkpoint.bn_layers", str(bn))
        if checkpoint.get("checkpoint_type") not in (None, "original"):
            checks.failed("checkpoint.type", str(checkpoint.get("checkpoint_type")))
        if run_name == "EXP-UVVIS-001-SEED2026" and checkpoint_sha != "9e2b0dd81cb5319e5c4f1aa6f2bb550bf23f6b6a3045e478ac1aa2f97f88e01d":
            checks.failed("uv_seed2026.correct_checkpoint", checkpoint_sha)

        manifest = read_json(manifest_path)
        validate_manifest_checks(manifest, str(manifest_path))
        probabilities, row_index, labels = validate_prediction_arrays(pred_dir, manifest, modality, label_set, expected_names)
        if manifest.get("task") != task or manifest.get("label_set") != label_set or manifest.get("modality") != modality:
            checks.failed("prediction_identity", "task/label_set/modality mismatch")
        if manifest.get("run_name") != run_name:
            checks.failed("prediction_run_name", str(manifest.get("run_name")))
        if manifest.get("split") != "validation":
            checks.failed("prediction_split", str(manifest.get("split")))
        seed_config = (manifest.get("checkpoint", {}).get("config") or {}).get("seed")
        if seed_config is None or int(seed_config) != expected_seed:
            checks.failed("prediction_seed", str(seed_config))
        if manifest.get("checkpoint", {}).get("sha256_after") != checkpoint_sha:
            checks.failed("prediction_checkpoint_sha")
        if manifest.get("checkpoint", {}).get("path", "").replace("\\", "/").split("/")[-1] != "best_model.pt":
            checks.failed("prediction_checkpoint_path")
        split_file = manifest.get("split_file") or {}
        split_path = Path(split_file.get("path", ""))
        if not split_path.is_absolute(): split_path = ROOT / split_path
        if not split_path.is_file() or split_file.get("sha256") != _stream_sha(split_path):
            checks.failed("prediction_split_sha")
        if probabilities.dtype != np.float32:
            checks.failed("probabilities.dtype", str(probabilities.dtype))
        if row_index.dtype != np.int64:
            checks.failed("row_index.dtype", str(row_index.dtype))
        if labels.dtype != np.uint8:
            checks.failed("labels.dtype", str(labels.dtype))
        if len(row_index) != N_SAMPLES or len(np.unique(row_index)) != N_SAMPLES:
            checks.failed("row_index.count_unique", f"{len(row_index)} rows, {len(np.unique(row_index))} unique")
        if row_index.min() < 0 or row_index.max() >= N_LABEL_ROWS:
            checks.failed("row_index.range")
        if probabilities.shape != (N_SAMPLES, len(expected_names)) or labels.shape != (N_SAMPLES, len(expected_names)):
            checks.failed("array.shapes")
        if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
            checks.failed("probabilities.finite_range")
        if not set(np.unique(labels).tolist()).issubset({0, 1}):
            checks.failed("labels.binary")

        threshold = _check_threshold(
            thresholds_root / task / run_name / "optimal_thresholds.json",
            thresholds_root / task / run_name / "calibration_manifest.json",
            read_json(thresholds_root / task / "summary.json"),
            manifest, expected_names, expected_seed, checkpoint_sha, checks, "threshold",
        )
        if threshold:
            result["threshold"] = threshold
        if provenance_assets is not None:
            identity = (task, run_name)
            for key, actual in [
                ("manifests", _stream_sha(manifest_path)),
                ("checkpoints", checkpoint_sha),
                ("thresholds", threshold.get("sha256")),
                ("calibrations", threshold.get("calibration_sha256")),
            ]:
                expected = provenance_assets.get(key, {}).get(identity)
                if expected is None or expected != actual:
                    checks.failed(f"provenance.{key}.binding", f"{task}/{run_name}")
            array_rec = provenance_assets.get("arrays", {}).get(identity)
            if array_rec is None:
                checks.failed("provenance.arrays.binding", f"{task}/{run_name}")
            else:
                for field, actual in [
                    ("probabilities", sha256_array(probabilities)),
                    ("row_index", sha256_array(row_index)),
                    ("labels", sha256_array(labels)),
                ]:
                    if array_rec.get(field) != actual:
                        checks.failed("provenance.arrays.binding", f"{task}/{run_name}/{field}")
        result.update({
            "config_sha256": _stream_sha(config_path),
            "checkpoint_sha256": checkpoint_sha,
            "prediction_manifest_sha256": _stream_sha(manifest_path),
            "probabilities_sha256": sha256_array(probabilities),
            "row_index_sha256": sha256_array(row_index),
            "labels_sha256": sha256_array(labels),
            "split_sha256": _stream_sha(split_path) if split_path.is_file() else None,
            "n_samples": int(len(row_index)),
            "n_labels": int(len(expected_names)),
            "config_seed": int(config.get("seed", -1)),
            "label_names": expected_names,
        })
    except Exception as exc:
        checks.failed("exception", f"{type(exc).__name__}: {exc}")
    result["status"] = "FAILED" if checks.has_failures() else "PASS"
    return result


def _verify_task_aggregate(label_set: str, task_models: list[dict], checks: Checks) -> None:
    expected = TASK_RUNS[label_set]
    names = [m["run_name"] for m in task_models]
    if len(task_models) != 9 or len(set(names)) != 9:
        checks.failed(f"{label_set}.unique_models", f"{len(task_models)} records, {len(set(names))} unique")
    if set(names) != {r for runs in expected.values() for r in runs}:
        checks.failed(f"{label_set}.registered_models", str(sorted(set(names))))
    for modality, runs in expected.items():
        actual = {m["run_name"] for m in task_models if m["modality"] == modality}
        if actual != set(runs):
            checks.failed(f"{label_set}.{modality}.runs", str(sorted(actual)))
        if len([m for m in task_models if m["modality"] == modality]) != 3:
            checks.failed(f"{label_set}.{modality}.count", "expected 3 records")
    if not task_models:
        return
    ref = task_models[0]
    for m in task_models[1:]:
        if not np.array_equal(np.asarray(m.get("row_index")), np.asarray(ref.get("row_index"))):
            checks.failed(f"{label_set}.row_index_alignment", m["run_name"])
        if not np.array_equal(np.asarray(m.get("labels")), np.asarray(ref.get("labels"))):
            checks.failed(f"{label_set}.labels_alignment", m["run_name"])
        if m.get("label_names") != ref.get("label_names"):
            checks.failed(f"{label_set}.label_names_alignment", m["run_name"])


def _verify_metrics(task: str, label_set: str, metrics_root: Path, checks: Checks) -> dict:
    root = metrics_root / task
    summary_path = root / "summary.json"
    model_path = root / "model_metrics.csv"
    if not summary_path.is_file() or not model_path.is_file():
        checks.failed(f"metrics.{task}.exists", "summary/model_metrics missing")
        return {}
    summary = read_json(summary_path)
    import csv
    rows = list(csv.DictReader(model_path.open(encoding="utf-8-sig")))
    expected = TASK_RUNS[label_set]
    expected_runs = {run for runs in expected.values() for run in runs}
    expected_names = load_label_names(resolve_label_set(label_set)["names"])
    if summary.get("no_test_access") is not True:
        checks.failed(f"metrics.{task}.no_test_access")
    if summary.get("n_models") != 9 or len(rows) != 9:
        checks.failed(f"metrics.{task}.n_models", f"{summary.get('n_models')}/{len(rows)}")
    if summary.get("split") != "validation" or summary.get("threshold") != 0.5:
        checks.failed(f"metrics.{task}.summary_split_threshold")
    if summary.get("n_samples") != N_SAMPLES:
        checks.failed(f"metrics.{task}.summary_n_samples", str(summary.get("n_samples")))
    if summary.get("label_names") != expected_names:
        checks.failed(f"metrics.{task}.summary_label_names")
    if "label_set" in summary and summary["label_set"] != label_set:
        checks.failed(f"metrics.{task}.summary_label_set")
    if summary.get("n_per_modality") != {"ir": 3, "raman": 3, "uvvis": 3}:
        checks.failed(f"metrics.{task}.summary_per_modality", str(summary.get("n_per_modality")))
    names = [r["run_name"] for r in rows]
    if len(set(names)) != 9 or set(names) != expected_runs:
        checks.failed(f"metrics.{task}.run_matrix", str(sorted(set(names))))
    by_modality = {modality: 0 for modality in MODALITIES}
    for r in rows:
        run_name = r["run_name"]
        if run_name not in expected_runs:
            checks.failed(f"metrics.{task}.unknown_run", run_name)
            continue
        modality = r.get("modality")
        if modality not in expected or run_name not in expected[modality]:
            checks.failed(f"metrics.{task}.run_modality", run_name)
        else:
            by_modality[modality] += 1
        expected_seed = expected_seed_from_run_name(run_name)
        if int(r.get("seed", -1)) != expected_seed or int(r.get("seed_config", -1)) != expected_seed:
            checks.failed(f"metrics.{task}.seed", run_name)
        if r.get("seed_metadata_mismatch", "").lower() != "false":
            checks.failed(f"metrics.{task}.seed_mismatch", run_name)
        if float(r.get("threshold", -1)) != 0.5:
            checks.failed(f"metrics.{task}.threshold", run_name)
        if int(r.get("n_samples", -1)) != N_SAMPLES:
            checks.failed(f"metrics.{task}.n_samples", run_name)
        if "task" in r and r["task"] not in ("", task):
            checks.failed(f"metrics.{task}.task_field", run_name)
        if "label_set" in r and r["label_set"] not in ("", label_set):
            checks.failed(f"metrics.{task}.label_set_field", run_name)
    if any(count != 3 for count in by_modality.values()):
        checks.failed(f"metrics.{task}.modality_counts", str(by_modality))
    return {"summary_path": str(summary_path), "summary_sha256": _stream_sha(summary_path), "model_metrics_sha256": _stream_sha(model_path), "n_models": len(rows)}


def _verify_b3a(task: str, label_set: str, analysis_root: Path, checks: Checks) -> dict:
    root = analysis_root / task / "complementarity"
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        checks.failed(f"b3a.{task}.summary_exists", str(summary_path))
        return {}
    summary = read_json(summary_path)
    if summary.get("phase") != "B3-A" or summary.get("task") != task or summary.get("label_set") != label_set:
        checks.failed(f"b3a.{task}.identity")
    if summary.get("partial") is not False or summary.get("no_test_access") is not True:
        checks.failed(f"b3a.{task}.flags")
    if summary.get("n_models") != 9 or summary.get("n_labels") != len(load_label_names(resolve_label_set(label_set)["names"])):
        checks.failed(f"b3a.{task}.coverage")
    if summary.get("n_model_rows") != (252 if task == "Shared14" else 270):
        checks.failed(f"b3a.{task}.model_rows", str(summary.get("n_model_rows")))
    if summary.get("n_pairwise_rows") != (504 if task == "Shared14" else 540):
        checks.failed(f"b3a.{task}.pairwise_rows", str(summary.get("n_pairwise_rows")))
    if summary.get("n_three_modality_rows") != (84 if task == "Shared14" else 90):
        checks.failed(f"b3a.{task}.three_rows", str(summary.get("n_three_modality_rows")))
    expected_best = "ir" if task == "Shared14" else "raman"
    if summary.get("best_single_modality") != expected_best:
        checks.failed(f"b3a.{task}.best_single", f"{summary.get('best_single_modality')} != {expected_best}")
    for name, expected_sha in (summary.get("outputs") or {}).items():
        p = root / name
        if not p.is_file() or _stream_sha(p) != expected_sha:
            checks.failed(f"b3a.{task}.output_sha", name)
    if len(summary.get("outputs") or {}) != 6:
        checks.failed(f"b3a.{task}.output_count")
    frozen = (summary.get("provenance") or {}).get("frozen_b3_metrics") or {}
    frozen_path = Path(frozen.get("path", ""))
    if not frozen_path.is_absolute(): frozen_path = PROJECT_ROOT / frozen_path
    if not frozen_path.is_file() or frozen.get("sha256") != _stream_sha(frozen_path):
        checks.failed(f"b3a.{task}.frozen_b3_metrics_sha")
    if "code" in summary or "code_sha256" in summary:
        checks.accepted(f"b3a.{task}.code_provenance", "code field present but not used as historical proof")
    else:
        checks.accepted(f"b3a.{task}.code_provenance", "B3-A execution code SHA is unknown; B3.5-A reproduction evidence exists")
    return {"summary_path": str(summary_path), "summary_sha256": _stream_sha(summary_path), "n_outputs": len(summary.get("outputs") or {}), "frozen_b3_metrics_sha256": frozen.get("sha256")}


def _raw_asset(path: Path, checks: Checks, historical_hashes: dict | None = None, name: str | None = None, cache: dict | None = None) -> dict:
    if not path.is_file():
        checks.failed(f"raw.{name or path.name}.exists", str(path))
        return {"path": str(path), "exists": False}
    cache = cache if cache is not None else {}
    cache_key = str(Path(path).resolve())
    actual = cache.get(cache_key)
    if actual is None:
        actual = _stream_sha(path)
        cache[cache_key] = actual
    record = {"path": str(path), "exists": True, "sha256": actual, "historical_hash_available": False, "historical_match": None}
    if historical_hashes and name in historical_hashes:
        record["historical_hash_available"] = True
        record["historical_match"] = historical_hashes[name] == actual
        if not record["historical_match"]:
            checks.failed(f"raw.{name}.historical_sha", "historical hash mismatch")
    if name and path.suffix == ".npy":
        shape = tuple(np.load(path, mmap_mode="r").shape)
        record["shape"] = list(shape)
    return record


def _verify_b3a_analysis_root(root: Path, record: dict, assets: dict, checks: Checks) -> dict:
    """Verify the B3-A analysis root against its original 14 formal files.

    The B3.5-A tree digest was recorded when ``runs/LINE-B-ANALYSIS`` contained
    exactly the 14 B3-A formal files.  The later B3.5-C1/C2/D1/D2
    Validation-development assets were appended under
    ``validation_development/`` after that snapshot.  The historical digest is
    therefore reconstructed from the authoritative ``b3a_files`` project-relative
    paths and SHA256 values using the exact ``_tree_digest`` semantics, and the
    wider current tree is never used as the historical record.

    A missing, empty, moved or altered original file, an unexpected file outside
    ``validation_development/``, or a digest that cannot be reconstructed from the
    authoritative set is a FAILURE.  None of these conditions is downgraded to PASS.
    """
    root = Path(root).resolve()
    originals = list(assets.get("b3a_files") or [])
    expected_count = record.get("after_file_count")
    info: dict[str, Any] = {
        "root": str(root),
        "recorded_tree_sha256": record.get("after_tree_sha256"),
        "recorded_file_count": expected_count,
        "authoritative_file_count": len(originals),
        "additive_subtree": B3A_ADDITIVE_SUBTREE,
    }
    if not originals or expected_count is None:
        checks.failed("provenance.b3a_root.authoritative_set", "b3a_files / after_file_count missing from provenance")
        return info
    if len(originals) != int(expected_count):
        checks.failed("provenance.b3a_root.authoritative_count", f"{len(originals)} records vs recorded {expected_count} files")
        return info

    rel_hashes: dict[str, str] = {}
    per_task: dict[str, int] = defaultdict(int)
    for rec in originals:
        path = _resolve_record_path(rec)
        if path is None:
            checks.failed("provenance.b3a_root.path", str(rec))
            return info
        resolved = path.resolve()
        try:
            relative = str(resolved.relative_to(root))
        except ValueError:
            checks.failed("provenance.b3a_root.outside_root", str(resolved))
            return info
        if not resolved.is_file():
            checks.failed("provenance.b3a_root.missing", relative)
            return info
        if resolved.stat().st_size <= 0:
            checks.failed("provenance.b3a_root.empty", relative)
            return info
        actual = _stream_sha(resolved)
        if actual != rec.get("sha256"):
            checks.failed("provenance.b3a_root.sha", relative)
            return info
        if relative in rel_hashes:
            checks.failed("provenance.b3a_root.duplicate", relative)
            return info
        rel_hashes[relative] = actual
        per_task[str(rec.get("task"))] += 1

    reconstructed = hashlib.sha256()
    for relative in sorted(rel_hashes):
        reconstructed.update(relative.encode("utf-8"))
        reconstructed.update(b"\0")
        reconstructed.update(rel_hashes[relative].encode("ascii"))
        reconstructed.update(b"\n")
    info["files"] = sorted(rel_hashes)
    info["files_per_task"] = dict(per_task)
    info["reconstructed_tree_sha256"] = reconstructed.hexdigest()

    if record.get("before_tree_sha256") != record.get("after_tree_sha256"):
        checks.failed("provenance.b3a_root.before_after", str(root))
        return info
    if reconstructed.hexdigest() != record.get("after_tree_sha256"):
        checks.failed("provenance.b3a_root.reconstructed_digest", reconstructed.hexdigest())
        return info

    original_paths = {(root / relative).resolve() for relative in rel_hashes}
    additive: dict[str, int] = defaultdict(int)
    unexpected: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in original_paths:
            continue
        relative = resolved.relative_to(root)
        if relative.parts and relative.parts[0] == B3A_ADDITIVE_SUBTREE:
            additive[relative.parts[1] if len(relative.parts) > 1 else "."] += 1
        else:
            unexpected.append(str(relative))
    info["additive_file_counts_by_top_dir"] = dict(additive)
    info["additive_file_count"] = sum(additive.values())
    info["unexpected_files"] = sorted(unexpected)
    if unexpected:
        checks.failed("provenance.b3a_root.unexpected_files", ", ".join(sorted(unexpected)[:5]))
        return info

    checks.passed(
        "provenance.b3a_root",
        f"{len(rel_hashes)} authoritative B3-A files verified individually; reconstructed digest equals the "
        f"B3.5-A record; {info['additive_file_count']} post-snapshot {B3A_ADDITIVE_SUBTREE}/ files excluded "
        f"as later analysis assets",
        verified_files=info["files"],
        reconstructed_tree_sha256=info["reconstructed_tree_sha256"],
        recorded_tree_sha256=record.get("after_tree_sha256"),
        additive_file_count=info["additive_file_count"],
    )
    return info


def _verify_provenance_assets(provenance: dict, checks: Checks) -> dict:
    assets = provenance.get("formal_assets") or {}
    result = {"manifests": {}, "checkpoints": {}, "arrays": {}, "thresholds": {}, "calibrations": {}, "b3a_files": set()}
    expected_runs = {run for runs in TASK_RUNS["shared14"].values() for run in runs} | {run for runs in TASK_RUNS["uvspec15"].values() for run in runs}
    for key, list_name in [
        ("prediction_manifests", "manifests"),
        ("checkpoints", "checkpoints"),
        ("validation_arrays", "arrays"),
        ("threshold_files", "thresholds"),
        ("calibration_manifests", "calibrations"),
    ]:
        records = assets.get(key) or []
        if len(records) != 18:
            checks.failed(f"provenance.{key}.count", str(len(records)))
        seen = set()
        for rec in records:
            task = rec.get("task")
            path = _resolve_record_path(rec)
            run_name = rec.get("run_name")
            if key == "threshold_files" and path is not None:
                run_name = path.parent.name
            identity = (task, run_name)
            if task not in ("Shared14", "UVSPEC15") or not run_name or identity in seen:
                checks.failed(f"provenance.{key}.identity", str(identity))
                continue
            seen.add(identity)
            if key != "validation_arrays":
                if path is None or not path.is_file():
                    checks.failed(f"provenance.{key}.missing", str(path))
                elif rec.get("sha256") != _stream_sha(path):
                    checks.failed(f"provenance.{key}.sha", f"{task}/{run_name}")
                else:
                    result[list_name][identity] = rec["sha256"]
            else:
                result[list_name][identity] = rec
        if len(seen) != 18:
            checks.failed(f"provenance.{key}.unique", str(len(seen)))
        if {identity[1] for identity in seen} != expected_runs:
            checks.failed(f"provenance.{key}.run_matrix")
    for identity, rec in result["arrays"].items():
        task, run_name = identity
        if not run_name:
            continue
        pred_dir = PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS" / task
        candidates = list(pred_dir.glob(f"*/{run_name}/validation"))
        if len(candidates) != 1:
            checks.failed("provenance.arrays.path", str(identity))
            continue
        d = candidates[0]
        for field, filename in [("probabilities", "probabilities.npy"), ("row_index", "row_index.npy"), ("labels", "labels.npy")]:
            actual = sha256_array(np.load(d / filename, allow_pickle=False))
            if rec.get(field) != actual:
                checks.failed("provenance.arrays.sha", f"{task}/{run_name}/{field}")
    b3a = assets.get("b3a_files") or []
    if len(b3a) != 14:
        checks.failed("provenance.b3a_files.count", str(len(b3a)))
    b3a_seen = set()
    for rec in b3a:
        path = _resolve_record_path(rec)
        rel = str(path.resolve()) if path is not None else ""
        if rel in b3a_seen:
            checks.failed("provenance.b3a_files.duplicate", rel)
        b3a_seen.add(rel)
        if not path.is_file() or rec.get("sha256") != _stream_sha(path):
            checks.failed("provenance.b3a_files.sha", rel)
        else:
            result["b3a_files"].add(rel)

    roots = assets.get("roots_before_after") or {}
    b3a_analysis_root: dict[str, Any] = {}
    for root_key, record in roots.items():
        root = Path(root_key)
        if not root.is_absolute(): root = PROJECT_ROOT / root
        key_normalized = str(root_key).replace("\\", "/").rstrip("/")
        if key_normalized.endswith(B3A_ANALYSIS_ROOT_SUFFIX):
            # Scope fixed: the whole root now also holds later B3.5-C1/C2/D1/D2
            # Validation-development assets.  Verify the recorded 14-file
            # snapshot instead of treating the wider tree as the historical record.
            b3a_analysis_root = _verify_b3a_analysis_root(root, record, assets, checks)
            continue
        allowed = set()
        if key_normalized.endswith("runs/LINE-B-MANIFESTS/B3"):
            allowed = {
                "code_provenance.json",
                "b3_5b_manifest_verification.json",
                "b3_5b_manifest_verification_v2.json",
            }
        actual = _tree_digest(root, exclude=allowed)
        if record.get("before_tree_sha256") != record.get("after_tree_sha256"):
            checks.failed("provenance.roots.before_after", root_key)
        if actual["tree_sha256"] != record.get("after_tree_sha256"):
            checks.failed("provenance.roots.sha", root_key)
        elif actual["n_files"] != record.get("after_file_count"):
            checks.failed("provenance.roots.count", root_key)
        else:
            checks.passed("provenance.roots", root_key)
    if len(roots) != 6:
        checks.failed("provenance.roots.count", str(len(roots)))
    result["b3a_analysis_root"] = b3a_analysis_root
    return result


def verify_project() -> dict:
    checks = Checks()
    formal_roots = [
        PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS",
        PROJECT_ROOT / "runs" / "LINE-B-METRICS",
        PROJECT_ROOT / "runs" / "LINE-B-THRESHOLDS",
        PROJECT_ROOT / "runs" / "LINE-B-ANALYSIS",
        PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B1",
        PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / "B3",
    ]
    formal_before = {str(root): _tree_digest(root) for root in formal_roots}
    provenance_path = PROJECT_ROOT / "runs/LINE-B-MANIFESTS/B3/code_provenance.json"
    provenance = {}
    provenance_assets = {}
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        provenance_assets = _verify_provenance_assets(provenance, checks)
    else:
        checks.failed("code_provenance.exists")
    results: list[dict] = []
    task_models: dict[str, list[dict]] = {"Shared14": [], "UVSPEC15": []}
    for label_set, task in [("shared14", "Shared14"), ("uvspec15", "UVSPEC15")]:
        for modality, runs in TASK_RUNS[label_set].items():
            for run in runs:
                model = verify_model(task, label_set, modality, run, PROJECT_ROOT / "runs", PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS", PROJECT_ROOT / "runs" / "LINE-B-THRESHOLDS", provenance_assets)
                results.append(model)
                if model.get("status") == "PASS":
                    # Re-load arrays only for cross-model checks; this is validation-only.
                    from training.line_b.common import validate_prediction_arrays
                    pred_dir = PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS" / task / modality.upper() / run / "validation"
                    try:
                        _, row, lab = validate_prediction_arrays(pred_dir, read_json(pred_dir / "prediction_manifest.json"), modality, label_set)
                        task_models[task].append({"run_name": run, "modality": modality, "row_index": row, "labels": lab, "label_names": model["label_names"]})
                    except Exception as exc:
                        checks.failed(f"{task}.{run}.cross_load", str(exc))

    for label_set, task in [("shared14", "Shared14"), ("uvspec15", "UVSPEC15")]:
        _verify_task_aggregate(label_set, task_models[task], checks)
        if task_models[task]:
            if task_models[task][0]["label_names"] != load_label_names(resolve_label_set(label_set)["names"]):
                checks.failed(f"{task}.label_names.authority")
    if len(task_models["Shared14"]) == 9 and len(task_models["UVSPEC15"]) == 9:
        if not np.array_equal(task_models["Shared14"][0]["row_index"], task_models["UVSPEC15"][0]["row_index"]):
            checks.failed("cross_task.row_index", "Shared14/UVSPEC15 validation row_index mismatch")
        else:
            checks.passed("cross_task.row_index")
        # Labels are intentionally not required to match across tasks.

    # Per-task threshold record counts.
    threshold_summary = {}
    for task in ["Shared14", "UVSPEC15"]:
        path = PROJECT_ROOT / "runs" / "LINE-B-THRESHOLDS" / task / "summary.json"
        if not path.is_file():
            checks.failed(f"thresholds.{task}.summary_exists")
            continue
        s = read_json(path)
        threshold_summary[task] = {"path": str(path), "sha256": _stream_sha(path), "n_models": s.get("n_models"), "n_labels": s.get("n_labels"), "n_threshold_records": s.get("n_threshold_records"), "all_checks_pass": s.get("all_checks_pass"), "no_test_access": s.get("no_test_access")}
        if s.get("all_checks_pass") is not True or s.get("no_test_access") is not True:
            checks.failed(f"thresholds.{task}.flags")
        if s.get("n_models") != 9 or s.get("n_threshold_records") != (126 if task == "Shared14" else 135):
            checks.failed(f"thresholds.{task}.counts")
        model_entries = s.get("models") or []
        model_names = [r.get("run_name") for r in model_entries]
        if len(model_entries) != 9 or len(set(model_names)) != 9:
            checks.failed(f"thresholds.{task}.summary_models_unique", f"{len(model_entries)}/{len(set(model_names))}")
        output_hashes = s.get("output_hashes") or {}
        expected_outputs = {"threshold_records.csv", "per_label_calibration.csv", "model_metrics.csv", "modality_mean_sd.csv", "per_label_mean_sd.csv"}
        if set(output_hashes) != expected_outputs:
            checks.failed(f"thresholds.{task}.output_hashes_keys", str(sorted(output_hashes)))
        for name, expected in output_hashes.items():
            p = path.parent / name
            if not p.is_file() or _stream_sha(p) != expected:
                checks.failed(f"thresholds.{task}.output_hash", name)
        rec_path = path.parent / "threshold_records.csv"
        import csv
        rec_rows = list(csv.DictReader(rec_path.open(encoding="utf-8-sig"))) if rec_path.is_file() else []
        rec_count = len(rec_rows)
        if rec_count != (126 if task == "Shared14" else 135):
            checks.failed(f"thresholds.{task}.records", str(rec_count))
        expected_runs = {r for runs in TASK_RUNS["shared14" if task == "Shared14" else "uvspec15"].values() for r in runs}
        expected_labels = load_label_names(resolve_label_set("shared14" if task == "Shared14" else "uvspec15")["names"])
        pairs = [(r.get("run_name"), r.get("label")) for r in rec_rows]
        if len(set(pairs)) != len(pairs):
            checks.failed(f"thresholds.{task}.records_unique")
        if {r.get("run_name") for r in rec_rows} != expected_runs:
            checks.failed(f"thresholds.{task}.records_run_matrix")
        if {r.get("label") for r in rec_rows} != set(expected_labels):
            checks.failed(f"thresholds.{task}.records_labels")

    metrics_info = {}
    b3a_info = {}
    for task, label_set in [("Shared14", "shared14"), ("UVSPEC15", "uvspec15")]:
        metrics_info[task] = _verify_metrics(task, label_set, PROJECT_ROOT / "runs" / "LINE-B-METRICS", checks)
        b3a_info[task] = _verify_b3a(task, label_set, PROJECT_ROOT / "runs" / "LINE-B-ANALYSIS", checks)
        if metrics_info[task] and b3a_info[task]:
            if metrics_info[task].get("model_metrics_sha256") != b3a_info[task].get("frozen_b3_metrics_sha256"):
                checks.failed(f"metrics.{task}.b3a_source_sha", "B3 metrics differs from B3-A frozen source")

    # Audit manifests.
    for name, path in [("b1_export_audit", PROJECT_ROOT / "runs/LINE-B-MANIFESTS/B1/b1_export_audit.json"), ("b3_export_audit", PROJECT_ROOT / "runs/LINE-B-MANIFESTS/B3/b3_export_audit.json")]:
        if not path.is_file():
            checks.failed(f"audit.{name}.exists")
            continue
        audit = read_json(path)
        if audit.get("all_checks_pass") is not True or audit.get("no_test_access") is not True:
            checks.failed(f"audit.{name}.flags")
        else:
            checks.passed(f"audit.{name}")

    # Code provenance.
    provenance_info = {}
    if provenance:
        provenance_info = {"path": str(provenance_path), "sha256": _stream_sha(provenance_path), "no_test_access": provenance.get("no_test_access"), "unknown_historical": provenance.get("code_sha256", {}).get("unknown_historical_versions")}
        if provenance.get("no_test_access") is not True:
            checks.failed("code_provenance.no_test_access")
        current = provenance.get("code_sha256", {}).get("current_source_sha256", {})
        for rel, expected in current.items():
            p = PROJECT_ROOT / rel
            if not p.is_file() or _stream_sha(p) != expected:
                checks.failed("code_provenance.current_source_sha", rel)
        if not provenance.get("code_sha256", {}).get("unknown_historical_versions"):
            checks.failed("code_provenance.unknown_historical_missing")
        else:
            checks.accepted("code_provenance.historical_unknown", "B3-A/threshold historical execution code SHA is unknown; reproduction evidence only")
        if provenance.get("code_sha256", {}).get("b1_historical_vs_current"):
            checks.accepted("code_provenance.b1_historical_diff", "B1 exporter/common historical code SHA differs from current; historical manifest preserved")

    # Raw assets and historical hash comparison.  Historical hashes are
    # collected as sets so conflicting records cannot be silently overwritten.
    historical_splits = defaultdict(set)
    historical_labels = defaultdict(set)
    historical_names = defaultdict(set)
    spectra_hashes = defaultdict(set)
    spectra_records = []
    sha_cache = {}
    pred_roots = [PROJECT_ROOT / "runs/LINE-B-PREDICTIONS" / t for t in ("Shared14", "UVSPEC15")]
    for root in pred_roots:
        for mp in root.glob("*/*/validation/prediction_manifest.json"):
            m = read_json(mp)
            task = m.get("task")
            label_set = m.get("label_set")
            modality = m.get("modality")
            data_config = resolve_data_config(modality, label_set)
            sf = m.get("spectra_file") or {}
            if sf.get("path"):
                spath = _resolve_record_path(sf)
                if spath is None or not spath.is_file():
                    checks.failed("spectra.exists", str(spath))
                else:
                    if str(spath.resolve()) != str(Path(data_config["spectra"]).resolve()):
                        checks.failed("spectra.path_config", f"{task}/{m.get('run_name')}")
                    actual = sha_cache.get(str(spath.resolve()))
                    if actual is None:
                        actual = _stream_sha(spath)
                        sha_cache[str(spath.resolve())] = actual
                    if sf.get("shape") is not None and list(np.load(spath, mmap_mode="r").shape) != list(sf["shape"]):
                        checks.failed("spectra.shape", f"{task}/{m.get('run_name')}")
                    if sf.get("dtype") is not None and str(np.load(spath, mmap_mode="r").dtype) != str(sf["dtype"]):
                        checks.failed("spectra.dtype", f"{task}/{m.get('run_name')}")
                    if sf.get("size_bytes") is not None and int(spath.stat().st_size) != int(sf["size_bytes"]):
                        checks.failed("spectra.size", f"{task}/{m.get('run_name')}")
                    if sf.get("sha256"):
                        spectra_hashes[str(spath.resolve())].add(sf["sha256"])
                        spectra_records.append({"task": task, "run_name": m.get("run_name"), "path": str(spath), "historical_sha256": sf["sha256"], "actual_sha256": actual})
            lf = m.get("label_files") or {}
            for key, target, hash_key in [
                ("matrix_path", historical_labels, "matrix_sha256"),
                ("names_path", historical_names, "names_sha256"),
            ]:
                if lf.get(key) and lf.get(hash_key):
                    path = _resolve_record_path({"path": lf[key]})
                    if path is not None:
                        target[str(path.resolve())].add(lf[hash_key])
            split_file = m.get("split_file") or {}
            if split_file.get("path") and split_file.get("sha256"):
                path = _resolve_record_path({"path": split_file["path"]})
                if path is not None:
                    historical_splits[str(path.resolve())].add(split_file["sha256"])

    def _single_history(mapping, path):
        values = mapping.get(str(Path(path).resolve()), set())
        if len(values) > 1:
            checks.failed("historical_hash.conflict", f"{path}: {sorted(values)}")
            return None
        return next(iter(values)) if values else None

    for path, values in list(spectra_hashes.items()) + [(path, values) for path, values in historical_splits.items()] + [(path, values) for path, values in historical_labels.items()] + [(path, values) for path, values in historical_names.items()]:
        if len(values) > 1:
            checks.failed("historical_hash.conflict", f"{path}: {sorted(values)}")

    raw_paths = [
        (PREPARED / "ir_float32.npy", "ir_float32"),
        (PREPARED / "raman_float32.npy", "raman_float32"),
        (PREPARED / "uvvis_float32.npy", "uvvis_float32"),
        (PREPARED / "functional_group_labels.npy", "functional_group_labels"),
        (PREPARED / "functional_group_label_names.json", "functional_group_label_names"),
        (PREPARED / "uvspec15_labels.npy", "uvspec15_labels"),
        (PREPARED / "uvspec15_label_names.json", "uvspec15_label_names"),
        (PREPARED / "ir_scaffold_split_valid.npz", "ir_scaffold_split_valid"),
        (PREPARED / "raman_scaffold_split_valid.npz", "raman_scaffold_split_valid"),
        (PREPARED / "uvvis_scaffold_split_valid.npz", "uvvis_scaffold_split_valid"),
        (PREPARED / "scaffold_split_80_10_10.npz", "scaffold_split_80_10_10"),
        (PREPARED / "multimodal_complete_scaffold_split.npz", "multimodal_complete_scaffold_split"),
        (PREPARED / "modality_valid_masks.npz", "modality_valid_masks"),
        (PREPARED / "qm9s_manifest_prepared.csv", "qm9s_manifest_prepared"),
        (PROJECT_ROOT / "processed/qm9s/structures/qm9s_structure_library.csv", "qm9s_structure_library"),
        (PROJECT_ROOT / "processed/qm9s/structures/qm9s_invalid_structures.csv", "qm9s_invalid_structures"),
        (PROJECT_ROOT / "processed/qm9s/structures/build_structure_library_report.json", "build_structure_library_report"),
    ]
    raw = []
    for path, name in raw_paths:
        key = str(Path(path).resolve())
        historical_hash = None
        if key in spectra_hashes:
            historical_hash = _single_history(spectra_hashes, path)
        elif key in historical_splits:
            historical_hash = _single_history(historical_splits, path)
        elif key in historical_labels:
            historical_hash = _single_history(historical_labels, path)
        elif key in historical_names:
            historical_hash = _single_history(historical_names, path)
        his = {name: historical_hash} if historical_hash else {}
        rec = _raw_asset(path, checks, his, name, cache=sha_cache)
        if not his:
            rec["historical_hash_available"] = False
            rec["historical_match"] = None
        raw.append(rec)

    # Validation-only split contents; never read the Test key.
    from training.line_b.common import load_validation_indices
    split_checks = {}
    for modality in MODALITIES:
        try:
            idx, path = load_validation_indices(modality, expected_n=N_SAMPLES)
            split_checks[modality] = {"path": str(path), "sha256": _stream_sha(path), "n_val": len(idx), "unique": len(np.unique(idx)), "min": int(idx.min()), "max": int(idx.max()), "test_key_accessed": False}
            if len(idx) != N_SAMPLES or len(np.unique(idx)) != N_SAMPLES:
                checks.failed(f"split.{modality}.size_unique")
        except Exception as exc:
            checks.failed(f"split.{modality}", str(exc))

    # Future assets are recorded, not treated as B3.5-B failure.
    future_missing = []
    for rel in ["training/line_b/bootstrap_line_b_scaffold.py", "training/line_b/analyze_line_b_structure.py", "training/line_b/build_line_b_report.py"]:
        if not (PROJECT_ROOT / rel).is_file(): future_missing.append(rel)
    if not (PROJECT_ROOT / "runs/LINE-B-MANIFESTS/FREEZE.json").is_file():
        future_missing.append("runs/LINE-B-MANIFESTS/FREEZE.json")
    future_gaps = [{"path": rel, "status": "MISSING", "gate": "B4/FREEZE"} for rel in future_missing]

    model_failed = [m for m in results if m.get("status") != "PASS"]
    if model_failed:
        for m in model_failed:
            checks.failed(f"model.{m['task']}.{m['run_name']}", m.get("status", "FAILED"))
    if not task_models["Shared14"] or not task_models["UVSPEC15"]:
        checks.failed("cross_task.models", "model cross-load incomplete")

    code_current = {rel: _stream_sha(PROJECT_ROOT / rel) for rel in CODE_FILES}
    prior_verification = {
        "path": str(PRIOR_JSON),
        "exists": PRIOR_JSON.is_file(),
        "sha256": _stream_sha(PRIOR_JSON) if PRIOR_JSON.is_file() else None,
    }
    formal_after = {str(root): _tree_digest(root) for root in formal_roots}
    formal_trees = {
        "roots": {
            root: {
                "before": formal_before[root],
                "after": formal_after[root],
                "unchanged": formal_before[root] == formal_after[root],
            }
            for root in formal_before
        },
        "all_unchanged": all(formal_before[root] == formal_after[root] for root in formal_before),
    }
    if not formal_trees["all_unchanged"]:
        checks.failed("formal_assets.tree_unchanged", "formal asset tree changed during verification")
    else:
        checks.passed("formal_assets.tree_unchanged")
    result = {
        "phase": PHASE,
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_status": "FAILED" if checks.has_failures() else "PASS",
        "exit_code": 1 if checks.has_failures() else 0,
        "no_test_access": True,
        "environment": _environment(),
        "code_sha256": {
            "current_source_sha256": code_current,
            "verification_script_sha256": _stream_sha(Path(__file__)),
        },
        "prior_verification": prior_verification,
        "new_verification_code_sha256": _stream_sha(Path(__file__)),
        "summary": {
            "n_models": len(results),
            "n_passed_models": sum(1 for m in results if m.get("status") == "PASS"),
            "n_failed_models": len(model_failed),
            "threshold_records": sum(v.get("n_threshold_records", 0) for v in threshold_summary.values()),
        },
        "models": results,
        "task_aggregate": {task: {"n_models": len(task_models[task])} for task in task_models},
        "thresholds": threshold_summary,
        "metrics": metrics_info,
        "b3a": b3a_info,
        "formal_assets": formal_trees,
        "spectra_history": spectra_records,
        "raw_assets": raw,
        "splits_validation_only": split_checks,
        "code_provenance": provenance_info,
        "b3a_analysis_root": provenance_assets.get("b3a_analysis_root") or {},
        "verifier_scope": {
            "analysis_root_check": "per-file verification of the 14 authoritative B3-A formal files plus reconstruction of the B3.5-A recorded tree digest",
            "analysis_root_additive_subtree_excluded": f"{B3A_ANALYSIS_ROOT_SUFFIX}/{B3A_ADDITIVE_SUBTREE} (post-snapshot B3.5-C1/C2/D1/D2 analysis assets)",
            "all_other_roots_check": "full-root tree digest equality against the B3.5-A provenance record",
        },
        "checks": checks.items,
        "future_missing": future_missing,
        "future_gaps": future_gaps,
        "accepted_historical_limitations": [
            "B1 exporter/common historical code SHA differs from current source; historical manifest preserved.",
            "B3-A execution code SHA is unknown; B3.5-A reproduction evidence exists.",
            "threshold calibration execution code SHA is unknown; B3.5-A reproduction evidence exists.",
            "B1 IR config.modality is absent in the frozen historical format; registration fixes modality=ir.",
        ],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--split", choices=["validation"], default="validation", help="Validation-only; Test is not exposed.")
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test:
        raise SystemExit("Test verification is not authorized in B3.5-B")
    if args.json_out.exists():
        raise SystemExit(f"Refusing to overwrite existing verification output: {args.json_out}")
    result = verify_project()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, result)
    print(json.dumps({
        "phase": result["phase"],
        "overall_status": result["overall_status"],
        "exit_code": result["exit_code"],
        "json_out": str(args.json_out),
        "n_models": result["summary"]["n_models"],
        "n_passed_models": result["summary"]["n_passed_models"],
        "n_failed_models": result["summary"]["n_failed_models"],
        "threshold_records": result["summary"]["threshold_records"],
        "future_missing": result["future_missing"],
    }, ensure_ascii=False, indent=2))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
