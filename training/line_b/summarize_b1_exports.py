"""Summarize and audit Line B validation exports (Shared14 / UVSPEC15).

Reads manifests and arrays only; writes a compact inventory and resource table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    LABEL_SETS,
    REQUIRED_MANIFEST_FALSE,
    REQUIRED_MANIFEST_TRUE,
    ROOT as PROJECT_ROOT,
    atomic_output_dir,
    expected_seed_from_run_name,
    load_label_names,
    read_json,
    relative_to_root,
    resolve_label_set,
    resolve_task_runs,
    sha256_array,
    validate_manifest_checks,
    validate_prediction_arrays,
    write_json,
)


REQUIRED_TRUE = list(REQUIRED_MANIFEST_TRUE)
REQUIRED_FALSE = list(REQUIRED_MANIFEST_FALSE)


def _seed_from_run_name(run_name: str):
    if run_name.endswith("-SEED123"):
        return 123
    if run_name.endswith("-SEED2026"):
        return 2026
    if run_name.endswith("-FULL"):
        return 42
    return None


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(
    predictions_root: Path,
    label_set: str = "shared14",
    phase: str = "B1",
) -> tuple[list[dict], list[dict], dict]:
    label_config = resolve_label_set(label_set)
    expected_runs = resolve_task_runs(label_set)
    expected_n_labels = int(label_config["n_labels"])
    expected_task = label_config["task"]
    expected_names = load_label_names(label_config["names"])
    manifests = sorted(Path(predictions_root).glob("*/*/validation/prediction_manifest.json"))
    expected_count = sum(len(run_names) for run_names in expected_runs.values())
    if len(manifests) != expected_count:
        raise ValueError(f"Expected {expected_count} prediction manifests, found {len(manifests)}")

    inventory: list[dict] = []
    resources: list[dict] = []
    reference_row = None
    reference_labels = None
    reference_names = None
    failures: list[str] = []
    found_by_modality = {modality: set() for modality in expected_runs}
    actual_counts = {modality: 0 for modality in expected_runs}
    seen_run_names: set[str] = set()

    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        pred_dir = manifest_path.parent
        run_name = manifest.get("run_name")
        if run_name != pred_dir.parent.name:
            failures.append(f"{manifest_path}: run_name does not match directory")
        if run_name in seen_run_names:
            failures.append(f"{manifest_path}: duplicate run_name {run_name}")
        seen_run_names.add(run_name)
        if manifest.get("task") != expected_task:
            failures.append(f"{manifest_path}: task {manifest.get('task')} != {expected_task}")
        if manifest.get("label_set") != label_set:
            failures.append(f"{manifest_path}: label_set {manifest.get('label_set')} != {label_set}")
        if manifest.get("split") != "validation":
            failures.append(f"{manifest_path}: split is not validation")
        if len(manifest.get("label_names") or []) != expected_n_labels:
            failures.append(f"{manifest_path}: unexpected label count")
        modality = manifest.get("modality")
        if modality in expected_runs and run_name in expected_runs[modality]:
            found_by_modality[modality].add(run_name)
            actual_counts[modality] += 1
        else:
            failures.append(f"{manifest_path}: run {run_name} not registered for {label_set}/{modality}")

        try:
            validate_manifest_checks(manifest, str(manifest_path))
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
            continue

        seed_config = (manifest.get("checkpoint", {}).get("config") or {}).get("seed")
        try:
            expected_seed = expected_seed_from_run_name(run_name)
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
            expected_seed = None
        if seed_config is None:
            failures.append(f"{manifest_path}: missing checkpoint seed")
            continue
        elif expected_seed is not None and int(seed_config) != expected_seed:
            failures.append(
                f"{manifest_path}: checkpoint seed {seed_config} != {expected_seed}"
            )

        try:
            probabilities, row_index, labels = validate_prediction_arrays(
                pred_dir,
                manifest,
                modality,
                label_set,
                expected_names,
            )
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
            continue

        if reference_row is None:
            reference_row = row_index
            reference_labels = labels
            reference_names = manifest["label_names"]
        else:
            if not np.array_equal(row_index, reference_row):
                failures.append(f"{manifest_path}: row_index mismatch")
            if not np.array_equal(labels, reference_labels):
                failures.append(f"{manifest_path}: labels mismatch")
            if manifest["label_names"] != reference_names:
                failures.append(f"{manifest_path}: label names mismatch")

        inventory.append(
            {
                "run_name": manifest["run_name"],
                "modality": manifest["modality"],
                "seed_run_label": expected_seed,
                "seed_config": seed_config,
                "seed_metadata_mismatch": bool(
                    expected_seed is not None
                    and seed_config is not None
                    and int(expected_seed) != int(seed_config)
                ),
                "n_samples": int(len(row_index)),
                "head_dim": int(manifest["model"]["head_dim"]),
                "prediction_dir": relative_to_root(pred_dir),
                "probabilities_sha256": sha256_array(probabilities),
                "row_index_sha256": sha256_array(row_index),
                "labels_sha256": sha256_array(labels),
                "checkpoint_sha256": manifest["checkpoint"]["sha256_after"],
                "split_sha256": manifest["split_file"]["sha256"],
                "label_names_sha256": manifest["label_files"]["names_sha256"],
                "repeat_max_abs_diff": manifest["repeat_check"]["max_abs_diff"],
                "repeat_bitwise_equal": manifest["repeat_check"]["bitwise_equal"],
                "checks_all_true": all(
                    manifest.get("checks", {}).get(key) is True for key in REQUIRED_TRUE
                )
                and all(
                    manifest.get("checks", {}).get(key) is False for key in REQUIRED_FALSE
                )
                and manifest["repeat_check"]["bitwise_equal"] is True
                and float(manifest["repeat_check"]["max_abs_diff"]) == 0.0,
            }
        )

        inference = manifest["inference"]
        resources.append(
            {
                "run_name": manifest["run_name"],
                "modality": manifest["modality"],
                "device": inference["device"],
                "device_name": inference["device_name"],
                "batch_size": inference["batch_size"],
                "num_workers": inference["num_workers"],
                "torch_version": inference["torch_version"],
                "cuda_version": inference["cuda_version"],
                "elapsed_seconds": inference["resources"]["elapsed_seconds"],
                "process_peak_rss_mb": inference["resources"]["process_peak_rss_mb"],
                "system_available_min_mb": inference["resources"]["system_available_min_mb"],
                "gpu_peak_allocated_mb": inference["resources"].get("gpu_peak_allocated_mb"),
                "gpu_peak_reserved_mb": inference["resources"].get("gpu_peak_reserved_mb"),
            }
        )

    for modality, run_names in expected_runs.items():
        if found_by_modality[modality] != set(run_names):
            failures.append(
                f"{label_set}/{modality}: found {sorted(found_by_modality[modality])}, "
                f"expected {sorted(run_names)}"
            )
        if actual_counts[modality] != 3:
            failures.append(
                f"{label_set}/{modality}: expected 3 records, found {actual_counts[modality]}"
            )
        expected_seeds = sorted(
            _seed_from_run_name(run_name) for run_name in run_names
        )
        actual_seeds = sorted(
            row["seed_config"]
            for row in inventory
            if row["modality"] == modality
        )
        if expected_seeds != [42, 123, 2026]:
            failures.append(
                f"{label_set}/{modality}: unexpected registered seed matrix {expected_seeds}"
            )
        if actual_seeds != [42, 123, 2026]:
            failures.append(
                f"{label_set}/{modality}: actual seed matrix {actual_seeds} != [42, 123, 2026]"
            )
    if len(seen_run_names) != expected_count:
        failures.append(
            f"{label_set}: expected {expected_count} unique runs, found {len(seen_run_names)}"
        )

    if failures:
        raise RuntimeError(f"{phase} export audit failed:\n" + "\n".join(failures))

    summary = {
        "phase": phase,
        "task": expected_task,
        "label_set": label_set,
        "split": "validation",
        "n_models": len(manifests),
        "n_labels": expected_n_labels,
        "n_per_modality": {
            modality: sum(1 for row in inventory if row["modality"] == modality)
            for modality in ("ir", "raman", "uvvis")
        },
        "n_samples": int(len(reference_row)),
        "label_names": reference_names,
        "row_index_sha256": sha256_array(reference_row),
        "labels_sha256": sha256_array(reference_labels),
        "no_test_access": True,
        "all_checks_pass": True,
        "max_repeat_abs_diff": max(
            float(row["repeat_max_abs_diff"] or 0.0) for row in inventory
        ),
        "seed_metadata_mismatches": [
            {
                "run_name": row["run_name"],
                "modality": row["modality"],
                "seed_run_label": row["seed_run_label"],
                "seed_config": row["seed_config"],
            }
            for row in inventory
            if row["seed_metadata_mismatch"]
        ],
        "unique_config_seeds": {
            modality: sorted(
                {
                    int(row["seed_config"])
                    for row in inventory
                    if row["modality"] == modality and row["seed_config"] is not None
                }
            )
            for modality in ("ir", "raman", "uvvis")
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return inventory, resources, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SETS),
        default="shared14",
        help="Label task to summarize; Shared14 remains the backward-compatible default.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=None,
        help="Defaults to runs/LINE-B-PREDICTIONS/<Task>.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to runs/LINE-B-MANIFESTS/<phase>.",
    )
    parser.add_argument("--phase", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_config = resolve_label_set(args.label_set)
    phase = args.phase or label_config["phase"]
    predictions_root = args.predictions_root or (
        PROJECT_ROOT / "runs" / "LINE-B-PREDICTIONS" / label_config["task"]
    )
    out_dir = args.out_dir or (
        PROJECT_ROOT / "runs" / "LINE-B-MANIFESTS" / phase
    )
    inventory, resources, summary = summarize(
        predictions_root,
        label_set=args.label_set,
        phase=phase,
    )

    with atomic_output_dir(out_dir) as temporary:
        _write_csv(
            temporary / "prediction_inventory.csv",
            inventory,
            [
                "run_name",
                "modality",
                "seed_run_label",
                "seed_config",
                "seed_metadata_mismatch",
                "n_samples",
                "head_dim",
                "prediction_dir",
                "probabilities_sha256",
                "row_index_sha256",
                "labels_sha256",
                "checkpoint_sha256",
                "split_sha256",
                "label_names_sha256",
                "repeat_max_abs_diff",
                "repeat_bitwise_equal",
                "checks_all_true",
            ],
        )
        _write_csv(
            temporary / "resource_summary.csv",
            resources,
            [
                "run_name",
                "modality",
                "device",
                "device_name",
                "batch_size",
                "num_workers",
                "torch_version",
                "cuda_version",
                "elapsed_seconds",
                "process_peak_rss_mb",
                "system_available_min_mb",
                "gpu_peak_allocated_mb",
                "gpu_peak_reserved_mb",
            ],
        )
        write_json(temporary / f"{phase.lower()}_export_audit.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
