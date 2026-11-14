"""Preserve UVSPEC15 train-only BN-recalibrated checkpoints without replacing originals.

Run from SpectraDB root in (spectra-db):
    python training/save_uvspec15_bn_checkpoint.py --run 003 --seed 42
    python training/save_uvspec15_bn_checkpoint.py --run 003 --seed 123
    python training/save_uvspec15_bn_checkpoint.py --run 003 --seed 2026
    python training/save_uvspec15_bn_checkpoint.py --run 004 --seed 42
    python training/save_uvspec15_bn_checkpoint.py --run 004 --seed 123
    python training/save_uvspec15_bn_checkpoint.py --run 004 --seed 2026

Requires training/diagnose_uvspec15_batchnorm_multiseed.py in the same project.
Recomputes BN from TRAIN because earlier diagnostic JSON files do NOT retain BN buffers.
Only BN buffers are modified, and the original best_model.pt is never written.
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

from training import diagnose_uvspec15_batchnorm_multiseed as diag

OUTPUT_ROOT = ROOT / "runs" / "UVSPEC15-BN-RECALIBRATED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save train-only BN-recalibrated UVSPEC15 checkpoint (never overwrite)."
    )
    parser.add_argument("--run", choices=["003", "004"], default="003")
    parser.add_argument("--seed", choices=[42, 123, 2026], type=int, default=42)
    args = parser.parse_args()

    if args.seed not in diag.RUN_IDS[args.run]:
        parser.error(f"run {args.run} does not have a configured seed {args.seed} checkpoint")
    run_name = diag.RUN_IDS[args.run][args.seed]
    source_dir = ROOT / "runs" / run_name
    original_ckpt = source_dir / "best_model.pt"
    dest_dir = OUTPUT_ROOT / run_name
    dest_ckpt = dest_dir / "best_model_bn_recalibrated.pt"
    manifest = dest_dir / "bn_recalibration_manifest.json"

    # Hard stop BEFORE expensive BN passes, preventing accidental replacement.
    if dest_ckpt.exists() or manifest.exists():
        raise FileExistsError(
            f"Destination already exists: {dest_dir}. "
            "No file will be overwritten; inspect or move existing files manually."
        )
    diag.require_inputs(source_dir)
    if not diag.LABEL_NAMES_PATH.is_file() or not diag.SPLIT_PATH.is_file():
        raise FileNotFoundError("UVSPEC15 label/split files are missing")
    with diag.LABEL_NAMES_PATH.open("r", encoding="utf-8") as f:
        label_names = json.load(f)
    if len(label_names) != 15:
        raise RuntimeError("Expected exactly 15 UVSPEC15 labels")

    checkpoint = torch.load(original_ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected_experiment = (
        "EXP-UVVIS-003-UVSPEC15-FULL" if args.run == "003"
        else "EXP-UVVIS-004-UVSPEC15-WBCE"
    )
    if config.get("experiment") not in (expected_experiment, run_name):
        raise RuntimeError("Checkpoint experiment metadata does not match the selected run")
    if int(config.get("seed", -1)) != args.seed:
        raise RuntimeError(
            f"Wrong checkpoint seed: expected {args.seed}, got {config.get('seed')}"
        )
    if config.get("label_names") != label_names or checkpoint.get("label_names") != label_names:
        raise RuntimeError("Checkpoint label names/order differs from UVSPEC15 schema")
    if config.get("normalization") != "max":
        raise RuntimeError("Unexpected normalization (expected max)")
    if int(config.get("batch_size", -1)) != 512 or int(config.get("num_workers", -1)) != 4:
        raise RuntimeError("Batch size/workers do not match the established baseline")
    if bool(config.get("use_pos_weight")) != (args.run == "004"):
        raise RuntimeError("Wrong BCE/WBCE checkpoint selected")
    if args.run == "004" and (
        config.get("pos_weight_transform") != "sqrt"
        or float(config.get("pos_weight_cap", -1)) != 20.0
    ):
        raise RuntimeError("Weighted BCE configuration mismatch")

    with np.load(diag.SPLIT_PATH) as split_file:
        train_idx = np.asarray(split_file["train"], dtype=np.int64)
    if len(train_idx) != 103853 or len(np.unique(train_idx)) != 103853:
        raise RuntimeError("Unexpected TRAIN index count or duplicates")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = diag.make_model(checkpoint, len(label_names), device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_loader = diag.make_loader(train_idx, device)

    print("=" * 78)
    print("TRAIN-only BN recalibration and checkpoint preservation")
    print(f"Run: {run_name} | Seed: {args.seed} | Device: {device}")
    print(f"Original: {original_ckpt}")
    print("No training/validation/test-based model selection will occur.")
    print("=" * 78)

    info = diag.reset_recompute_bn(model, train_loader, device)
    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bn_keys = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                bn_keys.add(f"{mod_name}.{field}")
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

    # Keep the original checkpoint's architecture/config/epoch/labels; replace ONLY
    # the state_dict and append explicit provenance of the additional processing.
    source_sha256 = sha256_file(original_ckpt)
    info = dict(info)
    info.update({
        "source_checkpoint": str(original_ckpt),
        "source_sha256": source_sha256,
        "run_directory": run_name,
        "seed": args.seed,
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "changed_bn_buffer_count": len(changed_bn),
        "checkpoint_type": "postprocessed_bn_running_statistics",
        "note": "Training-only BN buffer recalibration; no gradient or optimizer updates",
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
    manifest_content = dict(info)
    manifest_content["saved_checkpoint"] = str(dest_ckpt)
    manifest_content["saved_sha256"] = sha256_file(dest_ckpt)
    with manifest.open("x", encoding="utf-8") as fh:
        json.dump(manifest_content, fh, ensure_ascii=False, indent=2)

    print(f"PASS: {info['batchnorm_layers']} BN layers, {info['train_samples_used']} TRAIN samples")
    print(f"PASS: only BN running buffers changed ({len(changed_bn)} buffers)")
    print(f"SAVED: {dest_ckpt}")
    print(f"MANIFEST: {manifest}")
    print(f"ORIGINAL UNTOUCHED: {original_ckpt}")
    print("Note: this saves the model; thresholds for the new probabilities must be re-fitted on Validation.")


if __name__ == "__main__":
    main()
