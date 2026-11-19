"""B4.1 Test prediction export driver for the 18 frozen Line B models.

The model list, seeds, checkpoints, splits and thresholds are taken exclusively
from the pinned ``FREEZE.json``.  Each model is exported exactly once into
``runs/LINE-B-TEST-PREDICTIONS/<Task>/<Modality>/<run>/test/``; the exporter
itself enforces the Freeze gate and appends to
``runs/LINE-B-MANIFESTS/test_access_log.json``.

This driver never reads Test labels directly, never trains, never re-fits
thresholds, never runs Bootstrap and never touches B0/B3/Validation assets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_file, write_json
from training.line_b.export_line_b_predictions import (
    TEST_ROOT,
    ExportRequest,
    export_predictions,
)

MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--out-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser


def _verify_freeze(freeze_path: Path, pin: str) -> dict:
    if not freeze_path.is_file():
        raise SystemExit(f"FREEZE not found: {freeze_path}")
    actual = sha256_file(freeze_path)
    if actual != pin:
        raise SystemExit(f"FREEZE SHA256 mismatch: {actual} != {pin}")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise SystemExit("FREEZE is not a frozen B3 manifest")
    if len(freeze.get("models") or []) != 18:
        raise SystemExit("FREEZE does not register exactly 18 models")
    protocol = ROOT / freeze["protocol_path"]
    if not protocol.is_file() or sha256_file(protocol) != freeze["protocol_sha256"]:
        raise SystemExit("Protocol file does not match the FREEZE protocol SHA256")
    return freeze


def _existing_entry(log_path: Path, task: str, run_name: str) -> dict | None:
    if not log_path.is_file():
        return None
    log = read_json(log_path)
    for entry in log.get("entries") or []:
        if entry.get("task") == task and entry.get("run_name") == run_name:
            return entry
    return None


def _append_recovered(log_path: Path, manifest: dict, out_dir: Path, command: str) -> None:
    """Re-create a missing access-log entry from a completed Test manifest."""
    from training.line_b.export_line_b_predictions import _append_test_access_log

    arrays = manifest["arrays"]
    entry = {
        "timestamp_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": manifest["task"],
        "label_set": manifest["label_set"],
        "modality": manifest["modality"],
        "run_name": manifest["run_name"],
        "seed": int(manifest["checkpoint"]["config"]["seed"]),
        "split": "test",
        "n_samples": int(manifest["n_samples"]),
        "freeze_sha256": manifest["freeze"]["sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "split_path": manifest["split_file"]["path"],
        "split_sha256": manifest["split_file"]["sha256"],
        "test_row_index_sha256": arrays["row_index"]["sha256"],
        "probabilities_sha256": arrays["probabilities"]["sha256"],
        "probabilities_file_sha256": arrays["probabilities"]["file_sha256"],
        "labels_sha256": arrays["labels"]["sha256"],
        "output_dir": relative_to_root(out_dir),
        "manifest_sha256": sha256_file(out_dir / "prediction_manifest.json"),
        "exporter_sha256": manifest["code"]["exporter_sha256"],
        "command": command,
        "historical_exposure_disclosed": True,
        "blind_test_claim": False,
        "test_driven_tuning": False,
        "thresholds_modified": False,
        "recovered_from_manifest": True,
    }
    _append_test_access_log(log_path, entry, manifest["freeze"])


def main() -> None:
    args = build_parser().parse_args()
    freeze = _verify_freeze(args.freeze, args.freeze_sha256)
    log_path = args.freeze.parent / "test_access_log.json"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict] = []

    for record in freeze["models"]:
        task = record["task"]
        label_set = record["label_set"]
        modality = record["modality"]
        run_name = record["run_name"]
        out_dir = (
            args.out_root / task / MODALITY_DIRS[modality] / run_name / "test"
        )
        row = {
            "task": task,
            "label_set": label_set,
            "modality": modality,
            "run_name": run_name,
            "seed": int(record["seed"]),
            "out_dir": relative_to_root(out_dir),
            "expected_checkpoint_sha256": record["checkpoint"]["sha256"],
        }
        if out_dir.exists():
            manifest_path = out_dir / "prediction_manifest.json"
            if not manifest_path.is_file():
                raise SystemExit(f"Existing Test dir without manifest: {out_dir}")
            manifest = read_json(manifest_path)
            if (
                manifest.get("split") != "test"
                or manifest.get("run_name") != run_name
                or manifest.get("task") != task
                or manifest.get("modality") != modality
                or (manifest.get("freeze") or {}).get("sha256") != args.freeze_sha256
                or manifest["checkpoint"]["sha256"] != record["checkpoint"]["sha256"]
            ):
                raise SystemExit(f"Existing Test manifest is inconsistent: {manifest_path}")
            if _existing_entry(log_path, task, run_name) is None:
                _append_recovered(log_path, manifest, out_dir, " ".join(sys.argv))
            row.update(
                {
                    "status": "SKIPPED_EXISTS",
                    "n_samples": int(manifest["n_samples"]),
                    "probabilities_sha256": manifest["arrays"]["probabilities"]["sha256"],
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
            rows.append(row)
            print(json.dumps({"progress": len(rows), "status": "SKIPPED_EXISTS", "run": run_name}, ensure_ascii=False))
            continue

        began = time.perf_counter()
        manifest = export_predictions(
            ExportRequest(
                run_dir=ROOT / "runs" / run_name,
                modality=modality,
                label_set=label_set,
                split="test",
                out_dir=out_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                repeat_check=True,
                expected_n=12981,
                require_freeze=True,
                freeze_path=args.freeze,
                freeze_sha256=args.freeze_sha256,
                test_access_log=log_path,
            )
        )
        elapsed = time.perf_counter() - began
        row.update(
            {
                "status": "EXPORTED",
                "n_samples": int(manifest["n_samples"]),
                "probabilities_sha256": manifest["arrays"]["probabilities"]["sha256"],
                "row_index_sha256": manifest["arrays"]["row_index"]["sha256"],
                "labels_sha256": manifest["arrays"]["labels"]["sha256"],
                "manifest_sha256": sha256_file(out_dir / "prediction_manifest.json"),
                "elapsed_seconds": round(elapsed, 3),
                "inference": manifest["inference"],
                "repeat_check": {
                    "bitwise_equal": manifest["repeat_check"]["bitwise_equal"],
                    "max_abs_diff": manifest["repeat_check"]["max_abs_diff"],
                },
                "freeze_sha256_verified": manifest["freeze"]["sha256"] == args.freeze_sha256,
            }
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": len(rows),
                    "status": "EXPORTED",
                    "run": run_name,
                    "n_samples": row["n_samples"],
                    "repeat_bitwise_equal": row["repeat_check"]["bitwise_equal"],
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "phase": "B4.1",
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_path": relative_to_root(args.freeze),
        "freeze_sha256": args.freeze_sha256,
        "protocol_version": freeze["protocol_version"],
        "protocol_sha256": freeze["protocol_sha256"],
        "test_root": relative_to_root(args.out_root),
        "test_access_log": relative_to_root(log_path),
        "n_models": len(rows),
        "n_exported": sum(1 for row in rows if row["status"] == "EXPORTED"),
        "n_skipped_existing": sum(1 for row in rows if row["status"] == "SKIPPED_EXISTS"),
        "rows": rows,
    }
    summary_path = args.summary_out or (
        ROOT / "runs" / "LINE-B-MANIFESTS" / f"b4_1_export_summary_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    write_json(summary_path, summary)
    print(json.dumps({"summary": relative_to_root(summary_path), "n_exported": summary["n_exported"], "n_skipped": summary["n_skipped_existing"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
