"""CPU-only tests for Gate A check 09 (`checkpoint_reload_integrity`).

Incident this pins down
-----------------------
`cmd_save()` runs Gate A *before* the checkpoint file exists, so check 09 is by
construction still pending at that moment (`pass=False`, `method="pending"`). The
pre-write gate nevertheless evaluated the whole checklist, so every `cmd_save`
invocation aborted with `09_checkpoint_reload_integrity` - which is exactly what
happened for BCE seed 123 on 2026-11-16. `cmd_finalize` never hit this because it
finishes check 09 (two loads from disk) before looking at the status.

The tests below pin (A) the corrected ordering rule, (B) the serialization round-trip on
a synthetic checkpoint, (C) the same round-trip on the REAL seed-123 checkpoint, and
(D) that the comparison still detects a genuine difference.

CPU only: every tensor is loaded with map_location="cpu"; nothing is written outside a
temporary directory; no checkpoint is modified.

Run:  python training\\test_raman_bn_reload_integrity.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.save_raman_bn_checkpoint import (  # noqa: E402
    Checklist,
    finish_reload_check,
)

CHECK09 = "09_checkpoint_reload_integrity"
PENDING_BEFORE_WRITE = {CHECK09}
SEED123 = ROOT / "runs" / "EXP-RAMAN-001-SEED123" / "best_model.pt"

failures = []


def check(condition: bool, message: str) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================== #
print("A. gate ordering: a pending check 09 must not decide the pre-write verdict")
print("=" * 78)

lst = Checklist("artifact_integrity")
lst.add("01_source_sha256_matches_original_checkpoint", True)
lst.add("08_bn_summary_matches_diagnostic", True)
lst.add(CHECK09, False, method="pending", note="to be filled by the save path")

check(lst.status == "FAIL", "full status is FAIL while check 09 is pending")
check(lst.status_ignoring(PENDING_BEFORE_WRITE) == "PASS",
      "pre-write status ignores the pending check 09")
check("09_checkpoint_reload_integrity" in lst.failures,
      "the pending check is still recorded as a failure (fail-closed if never finished)")

finish_reload_check(lst, True, method="reload the written file and compare every tensor")
check(lst.status == "PASS", "after a successful reload the full status is PASS")
check(lst.status_ignoring(PENDING_BEFORE_WRITE) == "PASS", "pre-write status stays PASS")

lst_fail = Checklist("artifact_integrity")
lst_fail.add("01_source_sha256_matches_original_checkpoint", True)
lst_fail.add(CHECK09, False, method="pending")
finish_reload_check(lst_fail, False, method="reload", problem="tensor mismatch")
check(lst_fail.status == "FAIL", "a genuine reload failure keeps the full status FAIL")

# a real pre-write failure must still block
lst_block = Checklist("artifact_integrity")
lst_block.add("01_source_sha256_matches_original_checkpoint", True)
lst_block.add("02_trainable_parameters_bit_identical", False)
lst_block.add(CHECK09, False, method="pending")
check(lst_block.status_ignoring(PENDING_BEFORE_WRITE) == "FAIL",
      "a genuine Gate A failure still blocks the pre-write gate")
check("02_trainable_parameters_bit_identical" in lst_block.failures,
      "the blocking check is reported by name")

# =========================================================================== #
print()
print("B. synthetic checkpoint: save -> reload -> tensor-by-tensor comparison")
print("=" * 78)

tmp = Path(tempfile.mkdtemp(prefix="raman-reload-test-"))
try:
    synth_state = {
        "encoder.stem.0.weight": torch.arange(32, dtype=torch.float32) / 7.0,
        "encoder.stem.1.running_mean": torch.linspace(-1, 1, 32),
        "encoder.stem.1.running_var": torch.linspace(0.5, 2.0, 32),
        "encoder.stem.1.num_batches_tracked": torch.tensor(203, dtype=torch.int64),
        "encoder.blocks.0.conv1.weight": torch.randn(32, 32, 7, dtype=torch.float16),
        "feature_head.weight": torch.randn(14, 256),
        "feature_head.bias": torch.zeros(14),
    }
    synth = {
        "config": {"experiment": "SYNTH", "seed": 123, "embedding_dim": 256},
        "label_names": ["a", "b"],
        "epoch": 7,
        "best_val_mAP": 0.9,
        "model_state_dict": synth_state,
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(3)}},
                                 "param_groups": [{"lr": 1e-3}]},
    }
    src = tmp / "synthetic_source.pt"
    with src.open("xb") as fh:
        torch.save(synth, fh)

    dst = tmp / "synthetic_saved.pt"
    out = dict(synth)
    out["model_state_dict"] = {k: v.detach().cpu().clone() for k, v in synth_state.items()}
    with dst.open("xb") as fh:
        torch.save(out, fh)
    reloaded = torch.load(dst, map_location="cpu", weights_only=False)

    check(sorted(reloaded["model_state_dict"]) == sorted(out["model_state_dict"]),
          "state_dict keys preserved")
    same = all(torch.equal(out["model_state_dict"][k], reloaded["model_state_dict"][k])
               for k in out["model_state_dict"])
    check(same, f"all {len(out['model_state_dict'])} tensors equal after reload")
    check(reloaded["model_state_dict"]["encoder.blocks.0.conv1.weight"].dtype
          == torch.float16, "float16 dtype preserved (no silent cast)")
    check(reloaded["model_state_dict"]["encoder.stem.1.num_batches_tracked"].dtype
          == torch.int64, "int64 buffer dtype preserved")
    check(reloaded["config"] == out["config"], "config metadata preserved")
    check(reloaded["label_names"] == out["label_names"], "label order preserved")
    check(reloaded["epoch"] == out["epoch"], "epoch preserved")

    # byte identity is neither required nor checked
    check(sha256_file(dst) != "" and sha256_file(src) != "",
          "file hashes are recorded separately; the check compares tensors, not bytes")

    # negative control: a real difference must be detected
    tampered = {k: v.clone() for k, v in reloaded["model_state_dict"].items()}
    tampered["feature_head.bias"][0] = 1e-6
    detected = not all(torch.equal(out["model_state_dict"][k], tampered[k])
                       for k in out["model_state_dict"])
    check(detected, "a 1e-6 change in one element is detected by torch.equal")

    # ======================================================================= #
    print()
    print("C. REAL seed-123 checkpoint: save -> reload -> tensor-by-tensor comparison")
    print("=" * 78)
    if not SEED123.is_file():
        check(False, f"missing checkpoint: {SEED123}")
    else:
        original = torch.load(SEED123, map_location="cpu", weights_only=False)
        real_out = dict(original)
        real_out["model_state_dict"] = {
            k: v.detach().cpu().clone() for k, v in original["model_state_dict"].items()}
        real_dst = tmp / "seed123_rewritten.pt"
        with real_dst.open("xb") as fh:
            torch.save(real_out, fh)
        real_reloaded = torch.load(real_dst, map_location="cpu", weights_only=False)

        keys = sorted(real_out["model_state_dict"])
        mismatched = [k for k in keys
                      if not torch.equal(real_out["model_state_dict"][k],
                                         real_reloaded["model_state_dict"][k])]
        check(not mismatched,
              f"all {len(keys)} tensors of the real checkpoint round-trip exactly"
              + (f" (mismatched: {mismatched[:3]})" if mismatched else ""))
        check(real_reloaded["config"] == real_out["config"], "real config preserved")
        check(real_reloaded["label_names"] == real_out["label_names"],
              "real label order preserved")
        bn_buffers = [k for k in keys if k.endswith(
            ("running_mean", "running_var", "num_batches_tracked"))]
        check(len(bn_buffers) == 36, f"36 BN buffers round-trip ({len(bn_buffers)} found)")
        print(f"     source sha256 : {sha256_file(SEED123)}")
        print(f"     rewritten sha : {sha256_file(real_dst)}")
        print("     (the two differ by design: torch.save is not byte-stable, and the")
        print("      check compares tensors only - it never compares file hashes)")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"FAIL: {len(failures)} assertion(s) failed")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("PASS: check 09 ordering fixed; synthetic and real checkpoints round-trip exactly; "
      "genuine differences are still detected")
