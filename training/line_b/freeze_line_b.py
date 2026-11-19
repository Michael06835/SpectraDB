"""Generate the Line B B3 FREEZE manifest (``runs/LINE-B-MANIFESTS/FREEZE.json``).

This script only performs the B3 freeze:

1. run the existing read-only verifier ``verify_line_b_manifests.verify_project()``
   and save its complete result as the workspace pre-flight report (outside the
   ``runs/LINE-B-MANIFESTS/B3`` root, so the frozen B3 tree stays unchanged);
2. aggregate the frozen assets (18 models, labels, splits, raw prepared data,
   analysis outputs) with relative path, size and SHA256;
3. record the accepted provenance exceptions and the historical Test status;
4. validate every reference, then atomically write ``FREEZE.json``;
5. re-read ``FREEZE.json`` and independently re-hash every referenced asset.

It never reads the Test split, Test indices, Test labels or Test probabilities,
never retrains, never re-exports, never re-fits thresholds, and never overwrites
an existing ``FREEZE.json`` or any pre-flight report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch

from training.line_b.common import (
    LABEL_SETS,
    MODALITIES,
    PREPARED,
    TASK_RUNS,
    expected_seed_from_run_name,
    load_label_names,
    read_json,
    sha256_array,
    sha256_file,
    write_json,
)
from training.line_b.verify_line_b_manifests import _tree_digest, verify_project

FREEZE_PATH = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
PREFLIGHT_DIR = ROOT / "runs" / "LINE-B-MANIFESTS"
B3_ROOT = ROOT / "runs" / "LINE-B-MANIFESTS" / "B3"
RECHECK_DIR = ROOT / "runs" / "LINE-B-MANIFESTS"
PROTOCOL_PATH = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
PROVENANCE_PATH = ROOT / "runs" / "LINE-B-MANIFESTS" / "B3" / "code_provenance.json"
PREDICTIONS_ROOT = ROOT / "runs" / "LINE-B-PREDICTIONS"
THRESHOLDS_ROOT = ROOT / "runs" / "LINE-B-THRESHOLDS"
METRICS_ROOT = ROOT / "runs" / "LINE-B-METRICS"
ANALYSIS_ROOT = ROOT / "runs" / "LINE-B-ANALYSIS"
STRUCTURE_ROOT = ROOT / "processed" / "qm9s" / "structures"

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
LABEL_COUNTS = {"shared14": 14, "uvspec15": 15}
EXPECTED_THRESHOLD_RECORDS = {"Shared14": 126, "UVSPEC15": 135}
CODE_FILES = (
    "training/line_b/common.py",
    "training/line_b/metrics.py",
    "training/line_b/export_line_b_predictions.py",
    "training/line_b/evaluate_line_b_multilabel.py",
    "training/line_b/summarize_b1_exports.py",
    "training/line_b/calibrate_line_b_thresholds.py",
    "training/line_b/analyze_line_b_complementarity.py",
    "training/line_b/verify_line_b_manifests.py",
    "training/line_b/freeze_line_b.py",
)
B4_SCRIPTS = (
    "training/line_b/bootstrap_line_b_scaffold.py",
    "training/line_b/analyze_line_b_structure.py",
    "training/line_b/build_line_b_report.py",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    """Project-relative POSIX path; refuses anything outside the project root."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Asset outside project root: {resolved}") from exc


def file_ref(path: Path, **extra: Any) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Freeze asset is missing: {path}")
    record = {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    record.update(extra)
    return record


def npy_ref(path: Path, array_hash: bool = True) -> dict:
    path = Path(path)
    record = file_ref(path)
    array = np.load(path, mmap_mode="r")
    record["shape"] = [int(v) for v in array.shape]
    record["dtype"] = str(array.dtype)
    if array_hash:
        record["array_sha256"] = sha256_array(np.asarray(array))
    return record


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=ROOT).stdout.strip()


def capture_git() -> dict:
    status = [line for line in git("status", "--short").splitlines() if line.strip()]
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(status),
        "status_short": status,
        "commit_created_by_this_round": False,
        "push_performed_by_this_round": False,
    }


def capture_environment() -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cwd": str(ROOT),
    }


def find_prediction_dir(task: str, run_name: str) -> Path:
    candidates = [p for p in (PREDICTIONS_ROOT / task).glob(f"*/{run_name}/validation") if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Validation prediction dir for {task}/{run_name}, got {candidates}")
    return candidates[0]


def collect_models() -> list[dict]:
    models: list[dict] = []
    for task, label_set in TASKS:
        expected_names = load_label_names(LABEL_SETS[label_set]["names"])
        for modality in ("ir", "raman", "uvvis"):
            for run_name in TASK_RUNS[label_set][modality]:
                expected_seed = expected_seed_from_run_name(run_name)
                run_root = ROOT / "runs" / run_name
                pred_dir = find_prediction_dir(task, run_name)
                config = read_json(run_root / "config.json")
                manifest = read_json(pred_dir / "prediction_manifest.json")
                if int(config.get("seed", -1)) != expected_seed:
                    raise ValueError(f"config.seed mismatch for {run_name}")
                if len(expected_names) != LABEL_COUNTS[label_set]:
                    raise ValueError(f"label count mismatch for {label_set}")
                if manifest.get("task") != task or manifest.get("label_set") != label_set or manifest.get("modality") != modality:
                    raise ValueError(f"prediction manifest identity mismatch for {run_name}")
                if manifest.get("split") != "validation" or manifest.get("checks", {}).get("test_split_accessed") is not False:
                    raise ValueError(f"prediction manifest is not validation-only for {run_name}")
                if manifest.get("label_names") != expected_names:
                    raise ValueError(f"prediction label order mismatch for {run_name}")

                probabilities = npy_ref(pred_dir / "probabilities.npy")
                row_index = npy_ref(pred_dir / "row_index.npy")
                labels = npy_ref(pred_dir / "labels.npy")
                for name, ref, key in [
                    ("probabilities", probabilities, "probabilities"),
                    ("row_index", row_index, "row_index"),
                    ("labels", labels, "labels"),
                ]:
                    if ref["array_sha256"] != manifest["arrays"][key]["sha256"]:
                        raise ValueError(f"{name} array SHA256 differs from prediction manifest for {run_name}")

                split_file = manifest["split_file"]
                split_path = ROOT / split_file["path"]
                threshold_path = THRESHOLDS_ROOT / task / run_name / "optimal_thresholds.json"
                calibration_path = THRESHOLDS_ROOT / task / run_name / "calibration_manifest.json"
                threshold_payload = read_json(threshold_path)
                threshold_status = threshold_payload.get("threshold_status") or {}
                models.append({
                    "task": task,
                    "label_set": label_set,
                    "modality": modality,
                    "run_name": run_name,
                    "seed": expected_seed,
                    "n_labels": len(expected_names),
                    "split": "validation",
                    "config": file_ref(run_root / "config.json"),
                    "checkpoint": file_ref(run_root / "best_model.pt"),
                    "prediction_manifest": file_ref(pred_dir / "prediction_manifest.json"),
                    "probabilities": probabilities,
                    "row_index": row_index,
                    "labels": labels,
                    "split_file": file_ref(split_path, key=split_file.get("key", "val"), n_validation_indices=int(manifest["n_samples"])),
                    "threshold": file_ref(threshold_path, n_threshold_records=len(threshold_payload.get("thresholds") or {}),
                                          n_undefined_records=sum(1 for v in threshold_status.values() if v != "defined")),
                    "calibration_manifest": file_ref(calibration_path),
                })
    return models


def collect_labels_and_data() -> dict:
    label_sets = {}
    for label_set, config in LABEL_SETS.items():
        names_path = config["names"]
        matrix_path = config["labels"]
        names = load_label_names(names_path)
        label_sets[label_set] = {
            "task": config["task"],
            "label_names": file_ref(names_path, n_labels=len(names), names=names),
            "label_matrix": npy_ref(matrix_path),
        }

    splits = {}
    for modality, config in MODALITIES.items():
        splits[modality] = file_ref(config["split"], key="val", n_validation_indices=12981)

    spectra = {modality: npy_ref(config["spectra"], array_hash=False) for modality, config in MODALITIES.items()}

    prepared_manifest = PREPARED / "qm9s_manifest_prepared.csv"
    supporting = [
        PREPARED / "scaffold_split_80_10_10.npz",
        PREPARED / "multimodal_complete_scaffold_split.npz",
        PREPARED / "modality_valid_masks.npz",
    ]
    return {
        "label_sets": label_sets,
        "validation_splits": splits,
        "prepared_spectra": spectra,
        "prepared_manifest": file_ref(prepared_manifest),
        "supporting_split_assets": [file_ref(path) for path in supporting],
        "structure_library": {
            "path": rel(STRUCTURE_ROOT / "qm9s_structure_library.csv"),
            "sha256": sha256_file(STRUCTURE_ROOT / "qm9s_structure_library.csv"),
            "size_bytes": int((STRUCTURE_ROOT / "qm9s_structure_library.csv").stat().st_size),
        },
        "structure_library_supplementary": [
            file_ref(STRUCTURE_ROOT / "qm9s_invalid_structures.csv"),
            file_ref(STRUCTURE_ROOT / "build_structure_library_report.json"),
        ],
    }


def collect_analysis() -> dict:
    metrics = {}
    for task, label_set in TASKS:
        summary_path = METRICS_ROOT / task / "summary.json"
        summary = read_json(summary_path)
        model_metrics_path = METRICS_ROOT / task / "model_metrics.csv"
        metrics[task] = {
            "summary": file_ref(summary_path),
            "model_metrics": file_ref(model_metrics_path),
            "n_models": summary.get("n_models"),
            "n_samples": summary.get("n_samples"),
            "split": summary.get("split"),
            "primary_threshold": summary.get("threshold"),
            "n_labels": len(summary.get("label_names") or []),
            "no_test_access": summary.get("no_test_access"),
            "per_model_validation_metrics": summary.get("per_model") or [],
            "modality_mean_sd": summary.get("modality_metrics") or [],
        }

    b3a = {}
    for task, _label_set in TASKS:
        root = ANALYSIS_ROOT / task / "complementarity"
        summary_path = root / "summary.json"
        summary = read_json(summary_path)
        outputs = {
            name: file_ref(root / name)
            for name in sorted((summary.get("outputs") or {}).keys())
        }
        b3a[task] = {
            "summary": file_ref(summary_path),
            "phase": summary.get("phase"),
            "split": summary.get("split"),
            "n_models": summary.get("n_models"),
            "n_labels": summary.get("n_labels"),
            "seeds": summary.get("seeds"),
            "threshold_policies": summary.get("threshold_policies"),
            "best_single_modality": summary.get("best_single_modality"),
            "modality_macro_ap": summary.get("modality_macro_ap"),
            "n_model_rows": summary.get("n_model_rows"),
            "n_pairwise_rows": summary.get("n_pairwise_rows"),
            "n_three_modality_rows": summary.get("n_three_modality_rows"),
            "n_error_jaccard_undefined": summary.get("n_error_jaccard_undefined"),
            "all_checks_pass": summary.get("all_checks_pass"),
            "no_test_access": summary.get("no_test_access"),
            "partial": summary.get("partial"),
            "outputs": outputs,
        }

    thresholds = {}
    for task, _label_set in TASKS:
        root = THRESHOLDS_ROOT / task
        summary_path = root / "summary.json"
        summary = read_json(summary_path)
        thresholds[task] = {
            "summary": file_ref(summary_path),
            "n_models": summary.get("n_models"),
            "n_labels": summary.get("n_labels"),
            "n_threshold_records": summary.get("n_threshold_records"),
            "n_undefined_threshold_records": summary.get("n_undefined_threshold_records"),
            "default_threshold": summary.get("default_threshold"),
            "primary_threshold": summary.get("primary_threshold"),
            "secondary_threshold": summary.get("secondary_threshold"),
            "all_checks_pass": summary.get("all_checks_pass"),
            "no_test_access": summary.get("no_test_access"),
            "outputs": {
                name: file_ref(root / name)
                for name in sorted((summary.get("output_hashes") or {}).keys())
            },
        }

    audits = {}
    for name, path in (
        ("b1_export_audit", ROOT / "runs/LINE-B-MANIFESTS/B1/b1_export_audit.json"),
        ("b3_export_audit", ROOT / "runs/LINE-B-MANIFESTS/B3/b3_export_audit.json"),
    ):
        payload = read_json(path)
        audits[name] = file_ref(path, all_checks_pass=payload.get("all_checks_pass"), no_test_access=payload.get("no_test_access"))

    return {
        "validation_metrics": metrics,
        "b3a_complementarity": b3a,
        "thresholds": thresholds,
        "audit_manifests": audits,
        "code_provenance_manifest": file_ref(PROVENANCE_PATH),
    }


def collect_code_sha256() -> dict:
    provenance = read_json(PROVENANCE_PATH)
    historical = (provenance.get("code_sha256") or {}).get("historical_execution_code_sha256") or {}
    b1 = historical.get("B1_prediction_manifest") or {}
    b3 = historical.get("B3_UVSPEC15_prediction_manifest") or {}
    return {
        "current_worktree_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "historical_execution_sha256": {
            "b1_prediction_export": {
                "exporter_sha256": b1.get("exporter_sha256"),
                "common_sha256": b1.get("common_sha256"),
                "dataset_sha256": b1.get("dataset_sha256"),
                "model_sha256": b1.get("model_sha256"),
                "source": b1.get("source"),
            },
            "b3_uvspec15_prediction_export": {
                "exporter_sha256": b3.get("exporter_sha256"),
                "common_sha256": b3.get("common_sha256"),
                "dataset_sha256": b3.get("dataset_sha256"),
                "model_sha256": b3.get("model_sha256"),
                "source": b3.get("source"),
            },
            "b3a_complementarity_execution": None,
            "threshold_calibration_execution": None,
        },
        "raw_historical_unknown_markers": {
            "b3a_complementarity_execution": historical.get("B3_A_execution_code_sha256"),
            "threshold_calibration_execution": historical.get("threshold_calibration_execution_code_sha256"),
        },
        "b1_historical_vs_current": provenance["code_sha256"].get("b1_historical_vs_current") or [],
        "source": rel(PROVENANCE_PATH),
        "note": (
            "current_worktree_sha256 is the hash of the files present in the working tree at freeze time. "
            "It is NOT the historical execution SHA256 of B3-A, threshold calibration or B1 export. "
            "historical_execution_sha256 is copied from the B3.5-A code provenance record and is null where "
            "the historical execution version is unrecoverable."
        ),
    }


def collect_provenance_exceptions(code_sha: dict) -> dict:
    current = code_sha["current_worktree_sha256"]
    return {
        "b3a_execution_code": {
            "historical_execution_code_sha256": None,
            "status": "ACCEPTED_BY_USER",
            "reproduction_evidence": "B3.5-A",
            "reproduction_evidence_path": "runs/LINE-B-MANIFESTS/B3/code_provenance.json",
            "current_source_sha256": {
                "training/line_b/analyze_line_b_complementarity.py": current["training/line_b/analyze_line_b_complementarity.py"],
            },
            "limitation": "The B3-A execution code SHA256 is unrecoverable; B3-A outputs are bound by input hashes and re-verified by B3.5-A reproduction.",
        },
        "threshold_calibration_execution_code": {
            "historical_execution_code_sha256": None,
            "status": "ACCEPTED_BY_USER",
            "reproduction_evidence": "B3.5-A",
            "reproduction_evidence_path": "runs/LINE-B-MANIFESTS/B3/code_provenance.json",
            "current_source_sha256": {
                "training/line_b/calibrate_line_b_thresholds.py": current["training/line_b/calibrate_line_b_thresholds.py"],
            },
            "limitation": "The threshold-calibration execution code SHA256 is unrecoverable; 261 threshold records are bound by input hashes and re-verified by B3.5-A reproduction.",
        },
        "b1_historical_code": {
            "historical_code_difference": True,
            "historical_manifest_preserved": True,
            "files": code_sha["b1_historical_vs_current"],
            "limitation": "B1 historical exporter/common SHA256 differs from the current source; the historical manifests are preserved unchanged and were not rewritten during B3 freeze.",
        },
        "note": "These accepted exceptions do not change any raw data, label, threshold, checkpoint, probability array or statistic.",
        "does_not_change": ["raw_data", "labels", "thresholds", "checkpoints", "probabilities", "statistics"],
    }


def collect_test_status() -> dict:
    return {
        "historical_test_information_read_before_freeze": True,
        "historical_read_scope": "B0 stage consistency audit of existing per-label Test CSVs and existing Raman Test probabilities",
        "user_attestation_no_operational_interference": True,
        "test_access_during_freeze": False,
        "new_test_inference_during_freeze": False,
        "test_split_read_during_freeze": False,
        "test_indices_read_during_freeze": False,
        "test_labels_read_during_freeze": False,
        "test_probabilities_read_during_freeze": False,
        "result_classification": "exploratory/post-hoc",
    }


def collect_downstream() -> dict:
    b4 = {}
    for name in B4_SCRIPTS + ("training/line_b/analyze_line_b_complementarity.py", "training/line_b/export_line_b_predictions.py"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        b4[name] = {
            "present": path.is_file(),
            "freeze_gate_reference": "--require-freeze" in text,
        }
    return {
        "b4_status": "NOT_STARTED",
        "b5_status": "NOT_STARTED",
        "formal_2000_bootstrap": "NOT_RUN",
        "test_inference_during_freeze": "NONE",
        "b4_scripts": b4,
        "todos": [
            "B4 must implement the Test export gate (FREEZE.json presence + hash match), one export per model, and test_access_log.json; export_line_b_predictions.py currently exposes validation only.",
            "B4 must run the pre-registered paired scaffold-cluster 2,000-repetition Bootstrap; it was NOT run during B3.",
            "B4 is classified exploratory/post-hoc: historical Test information was already seen in B0.",
            "The pre-flight report and FREEZE.json live outside runs/LINE-B-MANIFESTS/B3, so the B3 root digest still equals the B3.5-A record and the verifier stays re-runnable; B4 gating consumes FREEZE.json.",
        ],
    }


def formal_tree_integrity(result: dict, b3_tree: dict) -> dict:
    roots = {}
    for root_key, record in (result.get("formal_assets", {}).get("roots") or {}).items():
        roots[rel(Path(root_key))] = {
            "before_tree_sha256": record.get("before", {}).get("tree_sha256"),
            "after_tree_sha256": record.get("after", {}).get("tree_sha256"),
            "n_files": record.get("after", {}).get("n_files"),
            "unchanged_during_preflight": record.get("unchanged"),
        }
    digest = hashlib.sha256()
    for key in sorted(roots):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(roots[key]["after_tree_sha256"]).encode("ascii"))
        digest.update(b"\n")
    provenance_roots = read_json(PROVENANCE_PATH).get("formal_assets", {}).get("roots_before_after") or {}
    recorded_b3 = (provenance_roots.get("runs/LINE-B-MANIFESTS/B3") or {}).get("after_tree_sha256")
    compatible = b3_tree.get("verifier_compatible") or {}
    return {
        "formal_asset_roots": roots,
        "formal_asset_tree_aggregate_sha256": digest.hexdigest(),
        "b3_manifests_root": {
            "verifier_compatible_tree_sha256": compatible.get("tree_sha256"),
            "verifier_compatible_file_count": compatible.get("n_files"),
            "code_provenance_recorded_tree_sha256": recorded_b3,
            "matches_code_provenance": compatible.get("tree_sha256") == recorded_b3,
            "b3_root_modified_by_freeze": False,
            "artifacts_added_by_freeze_outside_b3": [
                b3_tree.get("preflight_path", "runs/LINE-B-MANIFESTS/freeze_preflight_verification_<timestamp>.json"),
                "runs/LINE-B-MANIFESTS/FREEZE.json",
            ],
            "note": "The pre-flight report and FREEZE.json are written to runs/LINE-B-MANIFESTS/, outside the B3 root, so the B3 root and its B3.5-A digest are unchanged. The verifier stays re-runnable after the freeze.",
        },
    }


def build_payload(result: dict, preflight_ref: dict, b3_tree: dict) -> dict:
    code_sha = collect_code_sha256()
    checks = result.get("checks") or []
    models = collect_models()
    analysis = collect_analysis()
    threshold_total = sum(analysis["thresholds"][task]["n_threshold_records"] for task, _ in TASKS)
    return {
        "schema_version": "1.0",
        "phase": "B3",
        "status": "FROZEN",
        "generated_at": now_iso(),
        "project_root": str(ROOT),
        "protocol_version": "B0 v0.2",
        "protocol_path": rel(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "preflight_verification_path": preflight_ref["path"],
        "preflight_verification_sha256": preflight_ref["sha256"],
        "verification": {
            "verifier": "training/line_b/verify_line_b_manifests.py",
            "verifier_phase": result.get("phase"),
            "overall_status": result.get("overall_status"),
            "exit_code": result.get("exit_code"),
            "no_test_access": result.get("no_test_access"),
            "n_checks": len(checks),
            "n_passed": sum(1 for c in checks if c["status"] == "PASS"),
            "n_failed": sum(1 for c in checks if c["status"] == "FAILED"),
            "n_accepted_historical_limitation": sum(1 for c in checks if c["status"] == "ACCEPTED_HISTORICAL_LIMITATION"),
            "failed_check_ids": [c["id"] for c in checks if c["status"] == "FAILED"],
            "accepted_check_ids": [c["id"] for c in checks if c["status"] == "ACCEPTED_HISTORICAL_LIMITATION"],
            "n_models": result.get("summary", {}).get("n_models"),
            "n_passed_models": result.get("summary", {}).get("n_passed_models"),
            "n_failed_models": result.get("summary", {}).get("n_failed_models"),
            "threshold_records": result.get("summary", {}).get("threshold_records"),
            "future_missing_at_preflight": result.get("future_missing"),
            "future_gaps_are_downstream_only": True,
            "accepted_historical_limitations": result.get("accepted_historical_limitations"),
            "verifier_scope": result.get("verifier_scope"),
            "b3a_analysis_root_scope": result.get("b3a_analysis_root"),
        },
        "verification_matrix": {
            "shared14_models": 9,
            "uvspec15_models": 9,
            "total_models": 18,
            "shared14_threshold_records": analysis["thresholds"]["Shared14"]["n_threshold_records"],
            "uvspec15_threshold_records": analysis["thresholds"]["UVSPEC15"]["n_threshold_records"],
            "total_threshold_records": threshold_total,
            "b3a_outputs_per_task": {task: len(analysis["b3a_complementarity"][task]["outputs"]) for task, _ in TASKS},
            "test_accessed_during_freeze": False,
        },
        "models": models,
        "labels_and_data": collect_labels_and_data(),
        "analysis": analysis,
        "code_sha256": code_sha,
        "accepted_provenance_exceptions": collect_provenance_exceptions(code_sha),
        "test_status": collect_test_status(),
        "freeze_integrity": formal_tree_integrity(result, b3_tree),
        "environment": capture_environment(),
        "git": capture_git(),
        "downstream": collect_downstream(),
        "artifact_policy": {
            "overwrite_existing_freeze": False,
            "overwrite_historical_assets": False,
            "commit_created": False,
            "push_performed": False,
            "test_inference": "NONE",
            "formal_bootstrap_2000": "NOT_RUN",
        },
    }


def assert_preflight_ready(result: dict) -> None:
    problems: list[str] = []
    checks = result.get("checks") or []
    failed = [c["id"] for c in checks if c["status"] == "FAILED"]

    if result.get("overall_status") != "PASS":
        problems.append(f"overall_status={result.get('overall_status')}")
    if result.get("exit_code") != 0:
        problems.append(f"exit_code={result.get('exit_code')}")
    if result.get("no_test_access") is not True:
        problems.append("no_test_access is not true")
    if failed:
        problems.append(f"failed checks: {failed}")

    models = result.get("models") or []
    if len(models) != 18:
        problems.append(f"model records = {len(models)}")
    for task, _label_set in TASKS:
        task_models = [m for m in models if m.get("task") == task]
        passed = [m for m in task_models if m.get("status") == "PASS"]
        if len(task_models) != 9 or len(passed) != 9:
            problems.append(f"{task} models = {len(task_models)} (passed {len(passed)})")
    summary = result.get("summary") or {}
    if summary.get("n_models") != 18 or summary.get("n_passed_models") != 18 or summary.get("n_failed_models") != 0:
        problems.append(f"summary={summary}")

    thresholds = result.get("thresholds") or {}
    for task, _label_set in TASKS:
        record = thresholds.get(task) or {}
        if record.get("n_threshold_records") != EXPECTED_THRESHOLD_RECORDS[task]:
            problems.append(f"{task} threshold records = {record.get('n_threshold_records')}")
        if record.get("all_checks_pass") is not True or record.get("no_test_access") is not True:
            problems.append(f"{task} threshold flags invalid")
    if summary.get("threshold_records") != 261:
        problems.append(f"total threshold records = {summary.get('threshold_records')}")

    b3a = result.get("b3a") or {}
    for task, _label_set in TASKS:
        record = b3a.get(task) or {}
        if record.get("n_outputs") != 6:
            problems.append(f"{task} B3-A outputs = {record.get('n_outputs')}")

    scope = result.get("b3a_analysis_root") or {}
    if scope.get("authoritative_file_count") != 14:
        problems.append(f"B3-A analysis root authoritative files = {scope.get('authoritative_file_count')}")
    if scope.get("reconstructed_tree_sha256") != scope.get("recorded_tree_sha256"):
        problems.append("B3-A analysis root reconstructed digest != recorded digest")

    metrics = result.get("metrics") or {}
    for task, _label_set in TASKS:
        record = metrics.get(task) or {}
        if record.get("n_models") != 9:
            problems.append(f"{task} metrics n_models = {record.get('n_models')}")

    if (result.get("formal_assets") or {}).get("all_unchanged") is not True:
        problems.append("formal asset trees changed during pre-flight")

    if problems:
        print(json.dumps({"preflight": "FAILED", "problems": problems}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


def iter_asset_refs(node: Any) -> Iterator[dict]:
    if isinstance(node, dict):
        path = node.get("path")
        sha = node.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and "size_bytes" in node:
            yield node
        for value in node.values():
            yield from iter_asset_refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_asset_refs(value)


def validate_payload(payload: dict) -> None:
    """Every reference must stay inside the project and resolve to an existing file."""
    problems = []
    refs = list(iter_asset_refs(payload))
    for ref in refs:
        resolved = (ROOT / ref["path"]).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            problems.append(f"outside project: {ref['path']}")
            continue
        if not resolved.is_file():
            problems.append(f"missing: {ref['path']}")
            continue
        if not isinstance(ref.get("size_bytes"), int) or resolved.stat().st_size != ref["size_bytes"]:
            problems.append(f"size mismatch: {ref['path']}")
    for pair in [("protocol_path", "protocol_sha256"), ("preflight_verification_path", "preflight_verification_sha256")]:
        path_key, sha_key = pair
        resolved = (ROOT / payload[path_key]).resolve()
        if not resolved.is_file() or sha256_file(resolved) != payload[sha_key]:
            problems.append(f"top-level reference invalid: {payload[path_key]}")
    if len(refs) < 100:
        problems.append(f"unexpectedly few asset references: {len(refs)}")
    if problems:
        raise SystemExit("FREEZE payload validation failed:\n" + "\n".join(problems))


def recheck(freeze_path: Path) -> dict:
    """Re-read FREEZE.json from disk and re-hash every referenced asset independently."""
    reread = read_json(freeze_path)
    refs = list(iter_asset_refs(reread))
    missing, mismatched = [], []
    for ref in refs:
        resolved = (ROOT / ref["path"]).resolve()
        if not resolved.is_file():
            missing.append(ref["path"])
        elif sha256_file(resolved) != ref["sha256"] or resolved.stat().st_size != ref["size_bytes"]:
            mismatched.append(ref["path"])
    protocol_ok = sha256_file(ROOT / reread["protocol_path"]) == reread["protocol_sha256"]
    preflight_ok = sha256_file(ROOT / reread["preflight_verification_path"]) == reread["preflight_verification_sha256"]
    models = reread.get("models") or []
    counts = {
        "n_assets_rechecked": len(refs),
        "n_missing": len(missing),
        "n_sha256_mismatch": len(mismatched),
        "n_models": len(models),
        "n_shared14_models": sum(1 for m in models if m["task"] == "Shared14"),
        "n_uvspec15_models": sum(1 for m in models if m["task"] == "UVSPEC15"),
        "n_checkpoints": sum(1 for m in models if m.get("checkpoint", {}).get("path")),
        "n_probability_arrays": sum(1 for m in models if m.get("probabilities", {}).get("path")),
        "n_threshold_files": sum(1 for m in models if m.get("threshold", {}).get("path")),
        "n_calibration_manifests": sum(1 for m in models if m.get("calibration_manifest", {}).get("path")),
        "threshold_records": reread.get("verification_matrix", {}).get("total_threshold_records"),
        "b3a_csv_files": sum(len(v.get("outputs") or {}) for v in (reread.get("analysis", {}).get("b3a_complementarity") or {}).values()),
    }
    checks = {
        "json_parses": True,
        "status_frozen": reread.get("status") == "FROZEN",
        "phase_b3": reread.get("phase") == "B3",
        "protocol_sha256_matches": protocol_ok,
        "preflight_sha256_matches": preflight_ok,
        "all_assets_exist": not missing,
        "all_sha256_match": not mismatched,
        "n_assets_rechecked": counts["n_assets_rechecked"],
        "n_models_18": counts["n_models"] == 18,
        "threshold_records_261": counts["threshold_records"] == 261,
        "b3a_csv_files_12": counts["b3a_csv_files"] == 12,
        "test_access_during_freeze_false": (reread.get("test_status") or {}).get("test_access_during_freeze") is False,
        "no_new_test_inference": (reread.get("test_status") or {}).get("new_test_inference_during_freeze") is False,
    }
    all_pass = all(bool(v) for k, v in checks.items() if k != "n_assets_rechecked")
    return {
        "phase": "B3-FREEZE-postwrite-recheck",
        "generated_at": now_iso(),
        "freeze_path": rel(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "counts": counts,
        "checks": checks,
        "missing": missing,
        "sha256_mismatch": mismatched,
        "all_checks_pass": all_pass,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=FREEZE_PATH)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT_DIR)
    parser.add_argument("--recheck-dir", type=Path, default=RECHECK_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    freeze_path = args.json_out.resolve()
    if freeze_path.exists():
        raise SystemExit(f"Refusing to overwrite existing FREEZE file: {freeze_path}")
    if freeze_path == ROOT or not freeze_path.is_relative_to(ROOT):
        raise SystemExit(f"FREEZE path must stay inside the project: {freeze_path}")

    started = now_iso()
    print(json.dumps({"stage": "preflight-start", "at": started}, ensure_ascii=False))

    result = verify_project()
    assert_preflight_ready(result)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    preflight_path = args.preflight_dir.resolve() / f"freeze_preflight_verification_{timestamp}.json"
    if preflight_path.exists():
        raise SystemExit(f"Refusing to overwrite existing pre-flight report: {preflight_path}")
    write_json(preflight_path, result)
    preflight_ref = file_ref(preflight_path)
    print(json.dumps({"stage": "preflight-saved", "path": rel(preflight_path), "sha256": preflight_ref["sha256"],
                      "overall_status": result["overall_status"], "checks": len(result["checks"])}, ensure_ascii=False))

    # The pre-flight report and FREEZE.json are written to runs/LINE-B-MANIFESTS/,
    # outside the B3 root, so the B3 tree and its B3.5-A digest stay unchanged.
    verifier_exclude = {
        "code_provenance.json",
        "b3_5b_manifest_verification.json",
        "b3_5b_manifest_verification_v2.json",
    }
    b3_tree = {
        "verifier_compatible": _tree_digest(B3_ROOT, exclude=verifier_exclude),
        "preflight_path": preflight_ref["path"],
    }

    payload = build_payload(result, preflight_ref, b3_tree)
    payload["freeze_integrity"]["n_referenced_assets"] = len(list(iter_asset_refs(payload)))
    validate_payload(payload)

    tmp = freeze_path.with_name(f".{freeze_path.name}.tmp-{timestamp}-{hashlib.sha256(str(started).encode()).hexdigest()[:8]}")
    write_json(tmp, payload)
    reread_tmp = read_json(tmp)
    if reread_tmp.get("status") != "FROZEN" or reread_tmp.get("phase") != "B3":
        tmp.unlink(missing_ok=True)
        raise SystemExit("Temporary FREEZE file failed to round-trip")
    if freeze_path.exists():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Refusing to overwrite existing FREEZE file: {freeze_path}")
    tmp.replace(freeze_path)
    print(json.dumps({"stage": "freeze-written", "path": rel(freeze_path), "sha256": sha256_file(freeze_path)}, ensure_ascii=False))

    report = recheck(freeze_path)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    recheck_path = args.recheck_dir.resolve() / f"freeze_postwrite_recheck_{stamp}.json"
    if recheck_path.exists():
        raise SystemExit(f"Refusing to overwrite existing post-write recheck: {recheck_path}")
    write_json(recheck_path, report)
    print(json.dumps({
        "stage": "postwrite-recheck",
        "path": rel(recheck_path),
        "all_checks_pass": report["all_checks_pass"],
        "counts": report["counts"],
        "missing": report["missing"],
        "sha256_mismatch": report["sha256_mismatch"],
    }, ensure_ascii=False, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
