"""Preserve IR train-only BN-recalibrated checkpoints without replacing originals.

IR port of the validated training/save_uvspec15_bn_checkpoint.py. The BN
recalibration itself is NOT reimplemented here: it is imported from
training/diagnose_ir_batchnorm_full.py (reset_recompute_bn), i.e. exactly the
algorithm that the six approved Full BN Diagnostics already used. Only the data
interface, label set, experiment mapping and output location differ.

Recomputes BN from the IR TRAINING split (103855 samples) because the diagnostic
JSON files do NOT retain BN buffers. Only the 36 BN state tensors may change; the
original best_model.pt is never written.

Run from SpectraDB root in (spectra-db), one model per invocation:
    python training/save_ir_bn_checkpoint.py --loss bce --seed 42
    python training/save_ir_bn_checkpoint.py --loss bce --seed 123
    python training/save_ir_bn_checkpoint.py --loss bce --seed 2026
    python training/save_ir_bn_checkpoint.py --loss wbce --seed 42
    python training/save_ir_bn_checkpoint.py --loss wbce --seed 123
    python training/save_ir_bn_checkpoint.py --loss wbce --seed 2026

Outputs (default: refuse to overwrite):
    runs/IR-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt
    runs/IR-BN-RECALIBRATED/<run_name>/bn_recalibration_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import diagnose_ir_batchnorm_full as diag

OUTPUT_ROOT = ROOT / "runs" / "IR-BN-RECALIBRATED"
DIAGNOSTIC_ROOT = ROOT / "runs" / "IR-BN-DIAGNOSTICS"

EXPECTED_N_LABELS = 14
EXPECTED_TRAIN_SAMPLES = 103855
BN_METRIC_ATOL = 1e-3       # validation metrics vs the approved diagnostic record
BN_STAT_RTOL = 1e-3         # BN summary statistics vs the approved diagnostic record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_with_diagnostic(model: torch.nn.Module, device: torch.device,
                            run_name: str, val_idx: np.ndarray) -> dict:
    """Saved checkpoint must reproduce the approved Full BN Diagnostic record."""
    diag_path = DIAGNOSTIC_ROOT / run_name / "batchnorm_full_diagnostic.json"
    if not diag_path.is_file():
        raise FileNotFoundError(
            f"Approved Full BN Diagnostic record is missing: {diag_path}. "
            "The saved checkpoint cannot be cross-checked against it."
        )
    record = json.loads(diag_path.read_text(encoding="utf-8"))

    saved_bn = diag.summarize_bn(model)
    expected_bn = record["batchnorm_after"]
    if len(saved_bn) != len(expected_bn):
        raise RuntimeError(
            f"BN layer count mismatch vs diagnostic: {len(saved_bn)} != {len(expected_bn)}"
        )
    problems = []
    for got, want in zip(saved_bn, expected_bn):
        if got["name"] != want["name"]:
            problems.append(f"layer order mismatch: {got['name']} != {want['name']}")
            continue
        if got["num_batches_tracked"] != want["num_batches_tracked"]:
            problems.append(f"{got['name']}: nbt {got['num_batches_tracked']} != {want['num_batches_tracked']}")
        for key in ("mean_abs_mean", "min_running_var", "max_running_var"):
            if not np.isclose(got[key], want[key], rtol=BN_STAT_RTOL, atol=1e-9):
                problems.append(f"{got['name']}.{key}: {got[key]:.6e} != {want[key]:.6e}")

    val_loader = diag.make_loader(val_idx, device)
    labels, _, probabilities = diag.collect(model, val_loader, device)
    metrics = diag.get_metrics(labels, probabilities)
    diffs = {}
    for key in ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "mAP", "macro_AUROC"):
        expected = record["after"]["validation"][key]
        observed = metrics[key]
        if expected is None or observed is None:
            problems.append(f"validation metric {key} is undefined")
            continue
        diffs[key] = float(abs(observed - expected))
        if diffs[key] > BN_METRIC_ATOL:
            problems.append(f"validation {key}: {observed:.6f} vs {expected:.6f} (diff {diffs[key]:.3e})")

    if problems:
        raise RuntimeError(
            "Saved checkpoint does not match the approved Full BN Diagnostic:\n  "
            + "\n  ".join(problems)
            + "\nNo manifest was written for this model."
        )

    return {
        "path": str(diag_path),
        "bn_statistics_match": True,
        "validation_metrics_match": True,
        "max_validation_metric_abs_diff": max(diffs.values()),
        "validation_metrics_recomputed": metrics,
        "validation_metrics_recorded": record["after"]["validation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save IR train-only BN-recalibrated checkpoint (never overwrite)."
    )
    parser.add_argument("--loss", choices=["bce", "wbce"], default="bce")
    parser.add_argument("--seed", choices=[42, 123, 2026], type=int, default=42)
    args = parser.parse_args()

    if args.seed not in diag.RUN_IDS[args.loss]:
        parser.error(f"loss={args.loss} has no configured seed {args.seed}")

    run_name = diag.RUN_IDS[args.loss][args.seed]
    source_dir = ROOT / "runs" / run_name
    original_ckpt = source_dir / "best_model.pt"
    dest_dir = OUTPUT_ROOT / run_name
    dest_ckpt = dest_dir / "best_model_bn_recalibrated.pt"
    manifest_path = dest_dir / "bn_recalibration_manifest.json"

    # Hard stop BEFORE expensive BN passes, preventing accidental replacement.
    if dest_ckpt.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Destination already exists: {dest_dir}. "
            "No file will be overwritten; inspect or move existing files manually."
        )
    diag.require_inputs(source_dir)
    if not diag.LABEL_NAMES_PATH.is_file() or not diag.SPLIT_PATH.is_file():
        raise FileNotFoundError("IR label/split files are missing")
    with diag.LABEL_NAMES_PATH.open("r", encoding="utf-8") as f:
        label_names = json.load(f)
    if len(label_names) != EXPECTED_N_LABELS:
        raise RuntimeError(f"Expected exactly {EXPECTED_N_LABELS} IR labels")

    source_sha256 = sha256_file(original_ckpt)
    checkpoint = torch.load(original_ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    canonical_experiment = "EXP-IR-001-FULL" if args.loss == "bce" else "EXP-IR-002-FULL"
    if config.get("experiment") not in (canonical_experiment, run_name):
        raise RuntimeError("Checkpoint experiment metadata does not match the selected run")
    if int(config.get("seed", -1)) != args.seed:
        raise RuntimeError(
            f"Wrong checkpoint seed: expected {args.seed}, got {config.get('seed')}"
        )
    if config.get("label_names") != label_names or checkpoint.get("label_names") != label_names:
        raise RuntimeError("Checkpoint label names/order differs from the IR label schema")
    if config.get("normalization") != "max":
        raise RuntimeError("Unexpected normalization (expected max)")
    if int(config.get("batch_size", -1)) != diag.BATCH_SIZE or int(config.get("num_workers", -1)) != diag.NUM_WORKERS:
        raise RuntimeError("Batch size/workers do not match the established baseline")
    if float(config.get("threshold", -1)) != diag.THRESHOLD:
        raise RuntimeError("Unexpected default classification threshold")
    if bool(config.get("use_pos_weight")) != (args.loss == "wbce"):
        raise RuntimeError("Wrong BCE/WBCE checkpoint selected")
    if args.loss == "wbce" and (
        config.get("pos_weight_transform") != "sqrt"
        or float(config.get("pos_weight_cap", -1)) != 20.0
    ):
        raise RuntimeError("Weighted BCE configuration mismatch")

    with np.load(diag.SPLIT_PATH) as split_file:
        train_idx = np.asarray(split_file["train"], dtype=np.int64)
        val_idx = np.asarray(split_file["val"], dtype=np.int64)
        test_idx = np.asarray(split_file["test"], dtype=np.int64)
    if len(train_idx) != EXPECTED_TRAIN_SAMPLES or len(np.unique(train_idx)) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError("Unexpected TRAIN index count or duplicates")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = diag.make_model(checkpoint, len(label_names), device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    params_before = {k: v.detach().clone() for k, v in model.named_parameters()}
    train_loader = diag.make_loader(train_idx, device)

    print("=" * 78)
    print("IR TRAIN-only BN recalibration and checkpoint preservation")
    print(f"Run: {run_name} | Loss: {args.loss} | Seed: {args.seed} | Device: {device}")
    print(f"Original: {original_ckpt}")
    print(f"Source SHA256: {source_sha256}")
    print("No training/validation/test-based model selection will occur.")
    print("=" * 78)

    info = diag.reset_recompute_bn(model, train_loader, device)
    if info["train_samples_used"] != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(
            f"BN recalibration used {info['train_samples_used']} TRAIN samples, "
            f"expected {EXPECTED_TRAIN_SAMPLES}"
        )
    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Only the 36 BN state tensors may change; every parameter must be identical.
    bn_keys = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                bn_keys.add(f"{mod_name}.{field}" if mod_name else field)
    if set(new_state) != set(before) or not bn_keys.issubset(new_state):
        raise RuntimeError("Unexpected state_dict schema changed")
    bad_keys = [k for k in new_state if k not in bn_keys and not torch.equal(new_state[k], before[k])]
    if bad_keys:
        raise RuntimeError(f"Non-BN tensor(s) unexpectedly changed: {bad_keys}")
    changed_bn = [k for k in bn_keys if not torch.equal(new_state[k], before[k])]
    if not changed_bn:
        raise RuntimeError("No BN buffers changed; refusing to save an ineffective result")
    for key in bn_keys:
        if not torch.isfinite(new_state[key].float()).all():
            raise RuntimeError(f"Non-finite BN buffer found: {key}")
    bad_params = [k for k, v in model.named_parameters() if not torch.equal(params_before[k], v.detach())]
    if bad_params:
        raise RuntimeError(f"Trainable parameters changed during BN recalibration: {bad_params}")

    # Keep the original checkpoint's architecture/config/epoch/labels; replace ONLY
    # the state_dict and append explicit provenance of the additional processing.
    info = dict(info)
    info.update({
        "source_checkpoint": str(original_ckpt),
        "source_sha256": source_sha256,
        "run_directory": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "checkpoint_config_seed": int(config["seed"]),
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "normalization": config["normalization"],
        "default_threshold": float(config["threshold"]),
        "changed_bn_buffer_count": len(changed_bn),
        "changed_bn_buffer_names": sorted(changed_bn),
        "checkpoint_type": "postprocessed_bn_running_statistics",
        "note": "Training-only BN buffer recalibration; no gradient or optimizer updates",
        "config_summary": {
            "experiment": config.get("experiment"),
            "seed": int(config.get("seed", -1)),
            "batch_size": int(config.get("batch_size", -1)),
            "num_workers": int(config.get("num_workers", -1)),
            "normalization": config.get("normalization"),
            "threshold": float(config.get("threshold", -1)),
            "embedding_dim": int(config.get("embedding_dim", -1)),
            "num_labels": len(label_names),
            "use_pos_weight": bool(config.get("use_pos_weight")),
            "pos_weight_transform": config.get("pos_weight_transform"),
            "pos_weight_cap": config.get("pos_weight_cap"),
        },
        "data_scale": {
            "spectra_length": 3501,
            "num_labels": len(label_names),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "train_batches": info["train_batches_used"],
            "split_file": str(diag.SPLIT_PATH),
        },
    })
    output = dict(checkpoint)
    output["model_state_dict"] = new_state
    output["bn_recalibration"] = info

    dest_dir.mkdir(parents=True, exist_ok=True)
    # 'xb' refuses to overwrite an existing file even if another process creates it.
    with dest_ckpt.open("xb") as fh:
        torch.save(output, fh)
    reloaded = torch.load(dest_ckpt, map_location="cpu", weights_only=False)
    if (reloaded.get("bn_recalibration", {}).get("source_sha256") != source_sha256
            or any(not torch.equal(new_state[k], reloaded["model_state_dict"][k]) for k in new_state)):
        raise RuntimeError(f"Saved checkpoint failed verification: {dest_ckpt}")
    saved_sha256 = sha256_file(dest_ckpt)
    print(f"PASS: {info['batchnorm_layers']} BN layers, {info['train_samples_used']} TRAIN samples")
    print(f"PASS: only BN running buffers changed ({len(changed_bn)}/{len(bn_keys)} buffers)")
    print(f"PASS: trainable parameters unchanged; checkpoint reloaded and verified")

    # Cross-check the saved artifact against the approved Full BN Diagnostic record.
    reloaded_model = diag.make_model(reloaded, len(label_names), device)
    diagnostic_check = compare_with_diagnostic(
        reloaded_model, device, run_name, val_idx
    )
    print(
        f"PASS: BN statistics and Validation metrics match the approved diagnostic "
        f"(max metric diff {diagnostic_check['max_validation_metric_abs_diff']:.2e})"
    )

    manifest_content = dict(info)
    manifest_content.update({
        "saved_checkpoint": str(dest_ckpt),
        "saved_sha256": saved_sha256,
        "saved_size_bytes": dest_ckpt.stat().st_size,
        "verification": {
            "only_bn_buffers_changed": True,
            "trainable_parameters_unchanged": True,
            "bn_buffers_finite": True,
            "reload_ok": True,
            "diagnostic_consistent": True,
        },
        "diagnostic_reference": diagnostic_check,
        "original_checkpoint_untouched": True,
    })
    with manifest_path.open("x", encoding="utf-8") as fh:
        json.dump(manifest_content, fh, ensure_ascii=False, indent=2)

    if sha256_file(original_ckpt) != source_sha256:
        raise RuntimeError("Original checkpoint changed during the run (CRITICAL)")

    print(f"SAVED: {dest_ckpt}")
    print(f"MANIFEST: {manifest_path}")
    print(f"ORIGINAL UNTOUCHED: {original_ckpt}")
    print("Note: this saves the model; thresholds for the new probabilities must be re-fitted on Validation.")


if __name__ == "__main__":
    main()
