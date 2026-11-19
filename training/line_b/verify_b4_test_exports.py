"""Independent acceptance verification for B4.1 Line B Test predictions.

Verifies, against the pinned ``FREEZE.json`` and the authoritative frozen
splits/labels:

* 18/18 frozen models exported exactly once with correct shapes/dtypes/ranges;
* Test ``row_index`` equals the frozen Test split (unique, no missing rows);
* labels equal the authoritative label matrix rows and label order is frozen;
* checkpoint/config/split/threshold identities match FREEZE;
* every output file has a SHA256 and matches its manifest;
* ``test_access_log.json`` has one truthful entry per model and discloses the
  earlier B0 Test exposure (no blind-test claim);
* Validation / threshold / B3-A / FREEZE assets are unchanged.

Read-only; writes only its own JSON report.  Never trains, never re-fits
thresholds, never runs Bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    read_json,
    relative_to_root,
    sha256_array,
    sha256_file,
    write_json,
)

MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
TEST_N = 12981
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
DEFAULT_PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check_id: str, ok: bool, detail: str = "") -> None:
        self.items.append({"id": check_id, "ok": bool(ok), "detail": detail})

    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["ok"]]


def _load_freeze(path: Path, pin: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"FREEZE not found: {path}")
    actual = sha256_file(path)
    if actual != pin:
        raise SystemExit(f"FREEZE SHA256 mismatch: {actual} != {pin}")
    freeze = read_json(path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise SystemExit("FREEZE is not a frozen B3 manifest")
    protocol = ROOT / freeze["protocol_path"]
    if not protocol.is_file() or sha256_file(protocol) != freeze["protocol_sha256"]:
        raise SystemExit("Protocol file does not match the FREEZE protocol SHA256")
    if len(freeze.get("models") or []) != 18:
        raise SystemExit("FREEZE does not register 18 models")
    return freeze


def _test_indices(split_path: Path) -> np.ndarray:
    with np.load(split_path, allow_pickle=False) as archive:
        if "test" not in archive.files:
            raise KeyError(f"Missing test key in {split_path}")
        return np.asarray(archive["test"], dtype=np.int64)


def _entry_for(log: dict, task: str, run_name: str) -> dict | None:
    for entry in log.get("entries") or []:
        if entry.get("task") == task and entry.get("run_name") == run_name:
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--access-log", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    freeze = _load_freeze(args.freeze, args.freeze_sha256)
    access_log_path = args.access_log or (args.freeze.parent / "test_access_log.json")
    checks = Checks()
    label_names = {
        label_set: record["label_names"]["names"]
        for label_set, record in freeze["labels_and_data"]["label_sets"].items()
    }
    models: list[dict] = []
    all_rows: list[np.ndarray] = []
    root = args.predictions_root.resolve()

    for record in freeze["models"]:
        task = record["task"]
        label_set = record["label_set"]
        modality = record["modality"]
        run_name = record["run_name"]
        prefix = f"{task}/{run_name}"
        out_dir = root / task / MODALITY_DIRS[modality] / run_name / "test"
        manifest_path = out_dir / "prediction_manifest.json"
        checks.add(f"{prefix}.dir", out_dir.is_dir(), str(out_dir))
        checks.add(f"{prefix}.manifest", manifest_path.is_file(), str(manifest_path))
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        expected_names = label_names[label_set]

        identity_ok = (
            manifest.get("task") == task
            and manifest.get("label_set") == label_set
            and manifest.get("modality") == modality
            and manifest.get("run_name") == run_name
            and manifest.get("split") == "test"
            and manifest.get("split_key") == "test"
            and manifest.get("n_samples") == TEST_N
            and manifest.get("label_names") == expected_names
        )
        checks.add(f"{prefix}.identity", identity_ok)
        checks.add(f"{prefix}.freeze_pin", (manifest.get("freeze") or {}).get("sha256") == args.freeze_sha256)

        checkpoint_path = ROOT / manifest["checkpoint"]["path"]
        checks.add(
            f"{prefix}.checkpoint_freeze",
            manifest["checkpoint"]["sha256"] == record["checkpoint"]["sha256"],
        )
        checks.add(
            f"{prefix}.checkpoint_disk",
            checkpoint_path.is_file() and sha256_file(checkpoint_path) == record["checkpoint"]["sha256"],
        )
        config_path = ROOT / record["config"]["path"]
        checks.add(
            f"{prefix}.config_disk",
            config_path.is_file() and sha256_file(config_path) == record["config"]["sha256"],
        )
        split_path = ROOT / manifest["split_file"]["path"]
        checks.add(
            f"{prefix}.split_freeze",
            str(manifest["split_file"]["path"]).replace("\\", "/")
            == str(record["split_file"]["path"]).replace("\\", "/")
            and manifest["split_file"]["sha256"] == record["split_file"]["sha256"],
        )
        checks.add(
            f"{prefix}.split_disk",
            split_path.is_file() and sha256_file(split_path) == record["split_file"]["sha256"],
        )

        probabilities = np.load(out_dir / "probabilities.npy", allow_pickle=False)
        row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
        labels = np.load(out_dir / "labels.npy", allow_pickle=False)
        arrays = manifest["arrays"]
        expected_test = _test_indices(split_path)
        expected_labels = np.load(ROOT / freeze["labels_and_data"]["label_sets"][label_set]["label_matrix"]["path"], allow_pickle=False)[expected_test]

        shapes_ok = (
            probabilities.shape == (TEST_N, len(expected_names))
            and row_index.shape == (TEST_N,)
            and labels.shape == (TEST_N, len(expected_names))
        )
        checks.add(f"{prefix}.shapes", shapes_ok, f"{probabilities.shape}/{row_index.shape}/{labels.shape}")
        checks.add(f"{prefix}.dtypes", probabilities.dtype == np.float32 and row_index.dtype == np.int64 and labels.dtype == np.uint8)
        checks.add(f"{prefix}.probabilities_finite", bool(np.isfinite(probabilities).all()))
        checks.add(
            f"{prefix}.probabilities_range",
            bool(probabilities.min() >= 0.0 and probabilities.max() <= 1.0),
        )
        checks.add(f"{prefix}.row_index_unique", len(np.unique(row_index)) == TEST_N)
        checks.add(f"{prefix}.row_index_frozen", bool(np.array_equal(row_index, expected_test)))
        checks.add(f"{prefix}.labels_frozen", bool(np.array_equal(labels, expected_labels)))
        checks.add(f"{prefix}.array_sha", sha256_array(probabilities) == arrays["probabilities"]["sha256"] and sha256_array(row_index) == arrays["row_index"]["sha256"] and sha256_array(labels) == arrays["labels"]["sha256"])
        for key, name in (("probabilities", "probabilities.npy"), ("row_index", "row_index.npy"), ("labels", "labels.npy")):
            file_path = out_dir / name
            checks.add(
                f"{prefix}.file_sha.{key}",
                file_path.is_file()
                and sha256_file(file_path) == arrays[key].get("file_sha256")
                and file_path.stat().st_size == arrays[key].get("file_size_bytes"),
            )
        required_true = (
            "n_rows_equals_test_size",
            "row_index_equals_frozen_test",
            "row_index_unique",
            "labels_equal_frozen_label_matrix",
            "probabilities_finite",
            "probabilities_in_unit_interval",
            "label_order_equals_freeze",
            "head_dim_equals_label_count",
            "checkpoint_label_order_verified",
            "checkpoint_sha256_equals_freeze",
            "freeze_sha256_verified",
            "test_split_accessed",
            "checkpoint_file_unchanged",
            "model_state_unchanged",
            "bn_buffers_unchanged",
        )
        required_false = ("checkpoint_mutated", "bn_buffers_mutated")
        manifest_checks_ok = (
            set(manifest["checks"]) == set(required_true) | set(required_false)
            and all(manifest["checks"][key] is True for key in required_true)
            and all(manifest["checks"][key] is False for key in required_false)
        )
        checks.add(
            f"{prefix}.manifest_checks",
            manifest_checks_ok,
            json.dumps(manifest["checks"], ensure_ascii=False),
        )
        checks.add(
            f"{prefix}.repeat_bitwise",
            manifest["repeat_check"]["bitwise_equal"] is True and manifest["repeat_check"]["max_abs_diff"] == 0.0,
        )
        checks.add(
            f"{prefix}.test_access_disclosed",
            manifest["test_access"]["historical_test_information_read_before_b4"] is True
            and manifest["test_access"]["blind_test_claim"] is False
            and manifest["test_access"]["result_classification"] == "exploratory/post-hoc",
        )

        models.append({
            "task": task,
            "label_set": label_set,
            "modality": modality,
            "run_name": run_name,
            "seed": int(record["seed"]),
            "n_samples": int(TEST_N),
            "probabilities_sha256": arrays["probabilities"]["sha256"],
            "probabilities_file_sha256": arrays["probabilities"].get("file_sha256"),
            "row_index_sha256": arrays["row_index"]["sha256"],
            "labels_sha256": arrays["labels"]["sha256"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "output_dir": relative_to_root(out_dir),
        })
        all_rows.append(row_index)

    checks.add("models.count_18", len(models) == 18, str(len(models)))
    if all_rows:
        checks.add("cross_model.row_index_identical", all(np.array_equal(all_rows[0], row) for row in all_rows[1:]))

    # Access log.
    checks.add("access_log.exists", access_log_path.is_file(), str(access_log_path))
    access_summary = {}
    if access_log_path.is_file():
        log = read_json(access_log_path)
        entries = log.get("entries") or []
        access_summary = {
            "path": relative_to_root(access_log_path),
            "sha256": sha256_file(access_log_path),
            "n_entries": len(entries),
            "n_unique_models": len({(e.get("task"), e.get("run_name")) for e in entries}),
            "freeze_sha256": (log.get("freeze") or {}).get("sha256"),
            "historical_exposure": log.get("historical_exposure"),
            "access_policy": log.get("access_policy"),
        }
        checks.add("access_log.freeze_pin", (log.get("freeze") or {}).get("sha256") == args.freeze_sha256)
        checks.add("access_log.n_entries_18", len(entries) == 18, str(len(entries)))
        checks.add("access_log.unique_models_18", access_summary["n_unique_models"] == 18, str(access_summary["n_unique_models"]))
        exposure = log.get("historical_exposure") or {}
        checks.add(
            "access_log.historical_disclosure",
            exposure.get("historical_test_information_read_before_b4") is True
            and exposure.get("blind_test_claim") is False
            and exposure.get("result_classification") == "exploratory/post-hoc"
            and exposure.get("user_attestation_no_operational_interference") is True,
        )
        policy = log.get("access_policy") or {}
        checks.add(
            "access_log.policy",
            policy.get("one_export_per_model") is True
            and policy.get("test_driven_tuning") is False
            and policy.get("threshold_refit_on_test") is False
            and policy.get("bootstrap_run") is False,
        )
        for record in freeze["models"]:
            task, run_name = record["task"], record["run_name"]
            entry = _entry_for(log, task, run_name)
            checks.add(f"access_log.entry.{task}.{run_name}", entry is not None)
            if entry is None:
                continue
            out_dir = root / task / MODALITY_DIRS[record["modality"]] / run_name / "test"
            manifest = read_json(out_dir / "prediction_manifest.json")
            entry_ok = (
                entry.get("freeze_sha256") == args.freeze_sha256
                and entry.get("checkpoint_sha256") == record["checkpoint"]["sha256"]
                and entry.get("probabilities_sha256") == manifest["arrays"]["probabilities"]["sha256"]
                and entry.get("test_row_index_sha256") == manifest["arrays"]["row_index"]["sha256"]
                and entry.get("labels_sha256") == manifest["arrays"]["labels"]["sha256"]
                and entry.get("manifest_sha256") == sha256_file(out_dir / "prediction_manifest.json")
                and entry.get("n_samples") == TEST_N
                and entry.get("split") == "test"
            )
            checks.add(f"access_log.entry_match.{task}.{run_name}", entry_ok)

    # Frozen assets unchanged.
    frozen = {}
    for index, record in enumerate(freeze["models"]):
        task, run_name = record["task"], record["run_name"]
        validation_manifest = ROOT / record["prediction_manifest"]["path"]
        frozen[f"validation_manifest.{task}.{run_name}"] = (
            validation_manifest.is_file()
            and sha256_file(validation_manifest) == record["prediction_manifest"]["sha256"]
        )
        for key in ("probabilities", "row_index", "labels"):
            validation_array = validation_manifest.parent / f"{key}.npy"
            frozen[f"validation_array.{task}.{run_name}.{key}"] = (
                validation_array.is_file()
                and sha256_array(np.load(validation_array, allow_pickle=False))
                == record[key]["array_sha256"]
            )
        frozen[f"threshold.{task}.{run_name}"] = (
            (ROOT / record["threshold"]["path"]).is_file()
            and sha256_file(ROOT / record["threshold"]["path"]) == record["threshold"]["sha256"]
        )
        frozen[f"calibration.{task}.{run_name}"] = (
            (ROOT / record["calibration_manifest"]["path"]).is_file()
            and sha256_file(ROOT / record["calibration_manifest"]["path"]) == record["calibration_manifest"]["sha256"]
        )
    checks.add("frozen.validation_manifests", all(v for k, v in frozen.items() if k.startswith("validation_manifest.")))
    checks.add("frozen.validation_arrays", all(v for k, v in frozen.items() if k.startswith("validation_array.")))
    checks.add("frozen.thresholds", all(v for k, v in frozen.items() if k.startswith("threshold.")))
    checks.add("frozen.calibrations", all(v for k, v in frozen.items() if k.startswith("calibration.")))

    for task in ("Shared14", "UVSPEC15"):
        summary_path = ROOT / "runs" / "LINE-B-THRESHOLDS" / task / "summary.json"
        checks.add(
            f"frozen.threshold_summary.{task}",
            summary_path.is_file()
            and sha256_file(summary_path) == freeze["analysis"]["thresholds"][task]["summary"]["sha256"],
        )
    checks.add(
        "frozen.b3a_files",
        all(
            (ROOT / item["path"]).is_file() and sha256_file(ROOT / item["path"]) == item["sha256"]
            for item in [
                freeze["analysis"]["b3a_complementarity"][task]["outputs"][name]
                for task in ("Shared14", "UVSPEC15")
                for name in freeze["analysis"]["b3a_complementarity"][task]["outputs"]
            ]
        ),
    )

    # Output inventory.
    manifests = sorted(root.glob("*/*/*/test/prediction_manifest.json"))
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0
    checks.add("inventory.manifest_count_18", len(manifests) == 18, str(len(manifests)))
    file_counts = {}
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.name
            file_counts[suffix] = file_counts.get(suffix, 0) + 1

    failures = checks.failed()
    status = "PASS" if not failures else "FAILED"
    report = {
        "phase": "B4.1-verification",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze": {
            "path": relative_to_root(args.freeze),
            "sha256": args.freeze_sha256,
            "protocol_version": freeze["protocol_version"],
            "protocol_sha256": freeze["protocol_sha256"],
        },
        "predictions_root": relative_to_root(root),
        "summary": {
            "status": status,
            "n_models": len(models),
            "n_checks": len(checks.items),
            "n_failed": len(failures),
            "test_samples": TEST_N,
            "n_files": sum(file_counts.values()),
            "total_bytes": int(total_bytes),
            "file_counts": file_counts,
        },
        "models": models,
        "access_log": access_summary,
        "frozen_assets_checked": len(frozen),
        "checks": checks.items,
        "failures": failures,
        "status_counts": dict(Counter(("PASS" if item["ok"] else "FAILED") for item in checks.items)),
    }
    out_path = args.json_out or (
        ROOT / "runs" / "LINE-B-MANIFESTS" / f"b4_1_verification_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing verification: {out_path}")
    write_json(out_path, report)
    print(json.dumps({
        "status": status,
        "n_models": len(models),
        "n_checks": len(checks.items),
        "n_failed": len(failures),
        "failed_ids": [item["id"] for item in failures],
        "n_files": report["summary"]["n_files"],
        "total_bytes": report["summary"]["total_bytes"],
        "json_out": relative_to_root(out_path),
    }, ensure_ascii=False, indent=1))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
