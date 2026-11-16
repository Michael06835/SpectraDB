"""Preserve Raman train-only BN-recalibrated checkpoints without replacing originals.

Raman port of training/save_ir_bn_checkpoint.py (itself the UVSPEC15 port), with a
rewritten ACCEPTANCE layer (gate policy v2). The BN recalibration itself is still NOT
reimplemented here: it is imported from training/diagnose_raman_batchnorm_full.py
(reset_recompute_bn), i.e. exactly the algorithm the six Full BN Diagnostics used.
Only the data interface, label set, experiment mapping, output location and the
acceptance gates differ.

WHY THE ACCEPTANCE LAYER WAS REWRITTEN
--------------------------------------
The v1 gate required ONE cross-process Validation-mAP reading to land within 1e-3 of
the Phase 2 record. The T1/T2 experiment (2026-11-16) proved that requirement is not
achievable for this metric, with this checkpoint, on this machine:

    official FP16 sigmoid   mAP = 0.945208  and  0.946288   (same file, two processes)
    auxiliary FP32 sigmoid  mAP = 0.946809  and  0.946807   (same two processes)
    in-process repeat       probabilities bitwise identical

The FP16 variation (~1.1e-3) is produced by quantising the sigmoid output onto the
FP16 grid (~4.9e-4 spacing near 1.0), which re-orders near-tied samples and swings the
AP of `nitro` (n = 11 Validation positives). It is an evaluation-numerics property, not
an artifact defect: the recalibrated BN buffers match the Phase 2 record to 6.3e-7.

WHAT THAT DOES *NOT* LICENSE
----------------------------
An earlier revision (gate policy v2, withdrawn before any use) tried to absorb the
variation by widening the mAP tolerance to 3e-3 and reporting a
PASS_WITH_RECORDED_MAP_VARIATION state. That was rejected: it converts a failure into a
pass, and a Raman-only tolerance would break methodological parity with IR v1 and
UVSPEC15 v1. Gate policy v3 accordingly keeps 1e-3 on ALL SIX metrics, on every split,
exactly as in IR and UVSPEC15.

Gate policy v3 SEPARATES two questions that the original single gate conflated:

  1. artifact_integrity   -- deterministic, and the ONLY gate that decides whether the
                             saved file is the intended artifact. 10 named checks
                             (hashes, bit-identical parameters, BN buffers, provenance,
                             BN summary, reload integrity).
  2. numeric_reproduction -- official FP16 protocol metrics vs the Phase 2 record,
                             1e-3 on all six. PASS or FAIL, nothing in between. A FAIL is
                             reported verbatim and is never re-rolled.

The two verdicts are recorded side by side and never merged into one word:
`acceptance.status` reads ARTIFACT_PASS__NUMERIC_PASS or ARTIFACT_PASS__NUMERIC_FAIL.

Why a numeric FAIL can coexist with an intact artifact: the Phase 2 record was produced
with cudnn.benchmark=True and cudnn.deterministic unset, so its FP16 mAP is a single draw
from a nondeterministic ensemble and cannot be bit-reproduced. That is a property of the
reference configuration - the actual source of the inconsistency - and it is addressed at
the methodology level (pinned algorithms, see
training/verify_raman_bn_reproducibility.py), never by touching a tolerance.

Manifest rules (gate policy v4):
  * Gate A PASS + Gate B PASS -> manifest with acceptance.status
    ARTIFACT_PASS__NUMERIC_PASS and ablation_release.status ELIGIBLE;
  * Gate A PASS + Gate B FAIL -> manifest with acceptance.status
    ARTIFACT_PASS__NUMERIC_FAIL, ablation_release.status NOT_ELIGIBLE, and the exceeding
    metrics with observed value, reference value and absolute difference. It exists for
    provenance and human review only. Exit code 3;
  * Gate A FAIL -> no manifest at all; stop and report;
  * manifest existence is never approval, and a verdict is never re-rolled;
  * never overwritten (exclusive create), never hand-edited;
  * records the full gate policy and the measured basis.

`--finalize-existing` re-admits a checkpoint that a previous run wrote but could not
manifest (the Phase 3 orphan). It performs NO BN recalibration and NEVER rewrites the
checkpoint: it re-verifies the artifact from disk and, if everything passes, writes the
manifest. The manifest records that the checkpoint was not regenerated.

This operation trains nothing and never writes a checkpoint outside --output-root.
Existing runs are read-only.

Usage (from the SpectraDB root in the spectra-db environment), one model per invocation:

    # normal path (the five models that still need their BN checkpoint)
    python training/save_raman_bn_checkpoint.py --loss bce  --seed 123
    python training/save_raman_bn_checkpoint.py --loss bce  --seed 2026
    python training/save_raman_bn_checkpoint.py --loss wbce --seed 42
    python training/save_raman_bn_checkpoint.py --loss wbce --seed 123
    python training/save_raman_bn_checkpoint.py --loss wbce --seed 2026

    # re-admit the existing Phase 3 orphan (no BN recomputation, no rewrite)
    python training/save_raman_bn_checkpoint.py --loss bce --seed 42 \
        --finalize-existing \
        --expect-checkpoint-sha256 a9f9d55f1239abd7c91f52a4900f3ba5f23a9f4ced799b626802d956197022f5

    # CPU-only artifact audit of that orphan (no GPU, no manifest)
    python training/save_raman_bn_checkpoint.py --loss bce --seed 42 \
        --finalize-existing --dry-run

    # print the gate policy only
    python training/save_raman_bn_checkpoint.py --show-policy

Outputs (both refuse to overwrite):
    runs/RAMAN-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt   (normal path only)
    runs/RAMAN-BN-RECALIBRATED/<run_name>/bn_recalibration_manifest.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import diagnose_raman_batchnorm_full as diag  # noqa: E402
from training import raman_numeric_audit  # noqa: E402

OUTPUT_ROOT = ROOT / "runs" / "RAMAN-BN-RECALIBRATED"
DIAGNOSTIC_ROOT = ROOT / "runs" / "RAMAN-BN-DIAGNOSTICS"

EXPECTED_N_LABELS = 14
EXPECTED_TRAIN_SAMPLES = 103855
EXPECTED_TRAIN_BATCHES = 203
EXPECTED_BN_LAYERS = 12
EXPECTED_BN_BUFFERS = 36          # 12 layers x (running_mean, running_var, num_batches_tracked)

METRIC_KEYS = ("micro_f1", "macro_f1", "micro_precision", "micro_recall",
               "mAP", "macro_AUROC")

# --------------------------------------------------------------------------- #
# GATE POLICY v2 - single source of truth, copied verbatim into every manifest
# --------------------------------------------------------------------------- #
GATE_POLICY = {
    "version": "raman-bn-gate-v5",
    "supersedes": (
        "raman-bn-gate-v4 (same tolerances and gates; the audit label "
        "KNOWN_FP16_MAP_DEVIATION_ONLY was renamed to the neutral MAP_ONLY_NUMERIC_MISMATCH "
        "because it named a cause the check cannot establish, and because that label alone "
        "could be approved at any magnitude; release now requires an explicit approval "
        "bound to the checkpoint sha256). raman-bn-gate-v3 (no manifest on numeric failure) "
        "and raman-bn-gate-v2 (mAP-only 3e-3 envelope, withdrawn before use) are superseded."
    ),
    "principle": (
        "Artifact integrity and numerical reproducibility are two independent verdicts. "
        "No per-model and no per-metric tolerance adjustment: every one of the six official "
        "metrics is held to 1e-3, exactly as in IR v1 / UVSPEC15 v1. The BN summary "
        "tolerance is also 1e-3. A numerical failure is recorded as a failure, is never "
        "re-rolled, and is never converted into a pass by a label or by a file existing."
    ),
    "numeric_tolerance": 1e-3,
    "bn_summary_tolerance": 1e-3,
    "rationale_experiment": (
        "T1/T2 Validation-only reproducibility probe, EXP-RAMAN-001-FULL, 2026-11-16"
    ),
    # DESCRIPTIVE METADATA ONLY. Nothing in this block is read by Gate A, Gate B, any
    # tolerance, any verdict classification or any release decision; the only consumer is
    # official_metrics_report(), which copies the spread into the manifest for the record.
    #
    # Arithmetic provenance (do not interchange these two gaps):
    #   T1/T2 pair  [0.945208, 0.946288] -> 0.001080  == the cross-process spread below
    #   Gate B fail [0.945208, 0.946294] -> 0.001086  == the original Phase 3 incident,
    #                                                    a DIFFERENT reading pair
    "measured": {
        "official_fp16_map_same_file_two_processes": [0.945208, 0.946288],
        "official_fp16_map_cross_process_spread": 1.080e-3,
        "auxiliary_fp32_sigmoid_map_same_file_two_processes": [0.946809, 0.946807],
        "auxiliary_fp32_sigmoid_map_cross_process_spread": 2.0e-6,
        "threshold_metrics_max_abs_diff_vs_phase2_record": 2.3e-4,
        "bn_summary_max_relative_diff_vs_phase2_record": 6.3e-7,
        "in_process_repeat": "bitwise identical probabilities",
        "localisation": "the mAP variation is carried almost entirely by nitro AP (n=11)",
    },
    "known_limitation": (
        "The Phase 2 record was produced with torch.backends.cudnn.benchmark=True and "
        "cudnn.deterministic unset, i.e. under a configuration whose convolution algorithm "
        "selection is not pinned. The recorded FP16 mAP is therefore one sample from a "
        "nondeterministic ensemble and cannot be bit-reproduced by any later run. This is a "
        "property of the reference, not of the saved artifact, and it is the actual source "
        "of the seed-42 failure. It is measured, not waived: see gate policy v4's manifest "
        "policy and training/verify_raman_bn_reproducibility.py."
    ),
    "artifact_integrity": {
        "gate": "hard",
        "blocks_manifest": True,
        "bn_summary_rtol_gate": 1e-3,     # IR v1 parity (restored; measured residual 6.3e-7)
        "bn_summary_rtol_warn": 1e-5,     # record-only tier, never blocks
        "checks": 10,
    },
    "numeric_reproduction": {
        "channel": "official FP16 sigmoid at threshold 0.5 (historical protocol, unchanged)",
        "reference": (
            "runs/RAMAN-BN-DIAGNOSTICS/<run>/batchnorm_full_diagnostic.json :: after.validation"
        ),
        "tolerance": 1e-3,
        "tolerance_by_metric": {
            "micro_f1": 1e-3,
            "macro_f1": 1e-3,
            "micro_precision": 1e-3,
            "micro_recall": 1e-3,
            "macro_AUROC": 1e-3,
            "mAP": 1e-3,
        },
        "statuses": {
            "PASS": "all six official metrics within 1e-3 of the Phase 2 record",
            "FAIL": "at least one official metric deviates by more than 1e-3; recorded "
                    "verbatim as a numerical reproduction failure",
        },
        "manifest_policy": {
            "ARTIFACT_PASS__NUMERIC_PASS": "release ELIGIBLE; manifest written",
            "ARTIFACT_PASS__NUMERIC_FAIL": (
                "manifest written for provenance and human review. Release is NOT_ELIGIBLE "
                "unless an explicit approval bound to this checkpoint's sha256 is supplied "
                "(--manual-approval-note + --approve-checkpoint-sha256), in which case the "
                "release becomes ELIGIBLE_WITH_MANUAL_APPROVAL. The numeric verdict stays "
                "FAIL in all cases."
            ),
            "ARTIFACT_FAIL": "no manifest; stop and report",
            "audit_label": (
                "MAP_ONLY_NUMERIC_MISMATCH is descriptive (which metrics moved), never "
                "causal, and by itself authorises nothing at any magnitude"
            ),
            "manifest_existence_is_not_approval": True,
        },
        "retry_policy": (
            "single shot; a verdict is never re-rolled. Re-running to obtain a different "
            "verdict is forbidden."
        ),
    },
    "auxiliary_channel": {
        "name": ("FP32 sigmoid metrics plus a fixed-algorithm (deterministic) evaluation "
                 "path: cudnn.benchmark=False, cudnn.deterministic=True, "
                 "use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8"),
        "status": "RECORDED_ONLY_NOT_GATED",
        "script": "training/verify_raman_bn_reproducibility.py",
        "note": (
            "No Phase 2 reference exists for this channel, so it cannot gate. It measures "
            "cross-process stability of the artifact and is never compared with, merged into, "
            "or substituted for the official FP16 numbers."
        ),
    },
    "label_policy": {
        "all_labels_retained": True,
        "labels_excluded_from_mAP": [],
        "nitro": "reported per label, never used to redefine mAP",
    },
    "test_split": "not evaluated in this script (Phase 3 is a Validation-only gate)",
}


class ArtifactIntegrityError(RuntimeError):
    """A deterministic artifact check failed: the saved file is not the intended artifact."""


class NumericReproductionError(RuntimeError):
    """Numeric reproduction failed beyond the documented envelope."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checklist:
    """Ordered named checks with an explicit PASS/FAIL status."""

    def __init__(self, kind: str):
        self.kind = kind
        self.checks: dict[str, dict] = {}
        self.failures: list[str] = []

    def add(self, name: str, ok: bool, **details) -> bool:
        self.checks[name] = {"pass": bool(ok), **details}
        if not ok:
            self.failures.append(name)
        return bool(ok)

    @property
    def status(self) -> str:
        return "PASS" if not self.failures else "FAIL"

    def status_ignoring(self, names) -> str:
        """PASS if no failure outside `names`.

        Used for the pre-write gate in cmd_save, where check 09 is deliberately still
        pending (its file does not exist yet) and must not decide the verdict.
        """
        skip = set(names)
        return "FAIL" if [f for f in self.failures if f not in skip] else "PASS"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "status": self.status,
            "failures": list(self.failures),
            "checks": self.checks,
        }


def bn_buffer_keys(state_dict: dict) -> list[str]:
    return sorted(
        k for k in state_dict
        if any(k.endswith("." + f) for f in
               ("running_mean", "running_var", "num_batches_tracked"))
    )


def bn_layer_names(state_dict: dict) -> list[str]:
    return sorted(k.rsplit(".", 1)[0] for k in state_dict if k.endswith(".running_mean"))


def collect_dual(model, loader, device):
    """One Validation pass.

    `official` reproduces diag.collect op-for-op: sigmoid on the logits dtype, then
    float(). It is the historical protocol and must not change.
    `auxiliary` is sigmoid(logits.float()): an FP32-sigmoid view of the same logits,
    stored for cross-process stability analysis only.
    """
    y_l, off_l, aux_l, logit_l = [], [], [], []
    logits_dtype = official_dtype = None
    with torch.no_grad():
        for batch in loader:
            x = batch["spectrum"].to(device, non_blocking=True)
            y = batch["labels"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(x)["logits"]
            official = torch.sigmoid(logits)
            auxiliary = torch.sigmoid(logits.float())
            if logits_dtype is None:
                logits_dtype = str(logits.dtype)
                official_dtype = str(official.dtype)
            y_l.append(y.detach().cpu().numpy())
            off_l.append(official.detach().float().cpu().numpy())
            aux_l.append(auxiliary.detach().float().cpu().numpy())
            logit_l.append(logits.detach().float().cpu().numpy())

    labels = np.concatenate(y_l, axis=0).astype(np.int32)
    official_p = np.concatenate(off_l, axis=0)
    auxiliary_p = np.concatenate(aux_l, axis=0)
    logits_np = np.concatenate(logit_l, axis=0)
    for name, arr in (("official probabilities", official_p),
                      ("auxiliary probabilities", auxiliary_p),
                      ("logits", logits_np)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite {name}; investigate model numerics.")
    return {
        "labels": labels,
        "official": official_p,
        "auxiliary": auxiliary_p,
        "logits": logits_np,
        "logits_dtype": logits_dtype,
        "official_sigmoid_dtype": official_dtype,
    }


def classify_numeric(official_metrics: dict, recorded: dict) -> dict:
    """Two-state numeric verdict for Phase 3 (PASS / FAIL), plus its audit class.

    The judging rule lives in training/raman_numeric_audit.py so that Phase 3, Phase 4 and
    Phase 5 cannot drift apart. Gate B itself stays strictly binary: tolerance 1e-3 on all
    six official metrics, and any deviation beyond it is a FAIL. The audit class only
    labels *what kind* of failure it is, for the manifest and for the human reviewer.
    """
    audit = raman_numeric_audit.classify(
        official_metrics, recorded, context="Phase 3 Gate B (Validation, official FP16)"
    )
    verdict = "PASS" if audit["status"] == raman_numeric_audit.PASS else "FAIL"
    return {
        "status": verdict,
        "audit_class": audit["status"],
        "audit_reason": audit["reason"],
        "channel": GATE_POLICY["numeric_reproduction"]["channel"],
        "reference": GATE_POLICY["numeric_reproduction"]["reference"],
        "tolerance": audit["tolerance"],
        "metrics": audit["metrics"],
        "exceeding_metrics": audit["exceeding_metrics"],
        "evaluated_once": True,
        "retry_policy": GATE_POLICY["numeric_reproduction"]["retry_policy"],
        "scope": ("this verdict concerns numerical reproducibility only; it says nothing "
                  "about artifact integrity and must not be read as one"),
    }


def official_metrics_report(numeric: dict) -> dict:
    """The official FP16 numbers, reported verbatim, never smoothed into a pass."""
    return {
        "status": numeric["status"],
        "tolerance": numeric["tolerance"],
        "exceeding_metrics": numeric["exceeding_metrics"],
        "official_fp16_metrics": {k: v["observed"] for k, v in numeric["metrics"].items()},
        "phase2_recorded_metrics": {k: v["recorded"] for k, v in numeric["metrics"].items()},
        "absolute_differences": {k: v["abs_diff"] for k, v in numeric["metrics"].items()},
        "note": (
            "official FP16 sigmoid protocol at threshold 0.5, all 14 labels. A difference "
            "above 1e-3 is a numerical reproduction FAILURE, not a pass and not a rounding "
            "artefact to be tolerated."
        ),
        "measured_cross_process_spread_of_this_metric_family": GATE_POLICY["measured"][
            "official_fp16_map_cross_process_spread"],
        "known_limitation": GATE_POLICY["known_limitation"],
    }


# --------------------------------------------------------------------------- #
# Gate A - artifact integrity (deterministic)
# --------------------------------------------------------------------------- #
def verify_artifact(original: dict, recalibrated: dict, *, original_path: Path,
                    recal_path: Path, source_sha256: str, recal_sha256: str,
                    diagnostic: dict, label_names: list, mode: str) -> Checklist:
    """Ten deterministic checks. Every failure is collected, then raised together."""
    lst = Checklist("artifact_integrity")
    osd = original["model_state_dict"]
    rsd = recalibrated["model_state_dict"]
    bn_keys = bn_buffer_keys(rsd)
    bn_names = bn_layer_names(rsd)
    rec = recalibrated.get("bn_recalibration", {})
    diag_after = diagnostic.get("batchnorm_after", [])
    diag_recal = diagnostic.get("recalibration", {})

    # 1. source sha256 recorded in the artifact equals the current original file
    lst.add(
        "01_source_sha256_matches_original_checkpoint",
        rec.get("source_sha256") == source_sha256,
        recorded=rec.get("source_sha256"),
        observed=source_sha256,
        source_checkpoint=rec.get("source_checkpoint"),
        diagnostic_source_checkpoint=diagnostic.get("source_checkpoint"),
    )

    # 2. trainable parameters bit-identical (BN affine weight/bias are parameters too)
    param_keys = [k for k in rsd if k.endswith((".weight", ".bias")) and k not in bn_keys]
    param_diff = [k for k in param_keys if not torch.equal(osd[k], rsd[k])]
    lst.add(
        "02_trainable_parameters_bit_identical",
        not param_diff and sorted(osd) == sorted(rsd),
        tensors_compared=len(param_keys),
        differing=param_diff,
    )

    # 3. every non-BN state_dict tensor bit-identical (non-BN buffers included)
    others = [k for k in rsd if k not in bn_keys]
    other_diff = [k for k in others if not torch.equal(osd[k], rsd[k])]
    lst.add(
        "03_non_bn_state_tensors_bit_identical",
        not other_diff,
        tensors_compared=len(others),
        differing=other_diff,
    )

    # 4. exactly 36 BN buffers, all of them changed
    changed = [k for k in bn_keys if not torch.equal(osd[k], rsd[k])]
    lst.add(
        "04_only_36_bn_buffers_changed",
        len(bn_keys) == EXPECTED_BN_BUFFERS and len(changed) == EXPECTED_BN_BUFFERS,
        bn_buffers=len(bn_keys),
        changed=len(changed),
        expected=EXPECTED_BN_BUFFERS,
        unchanged_bn_buffers=[k for k in bn_keys if k not in changed],
    )

    # 5. BN layer count / names / num_batches_tracked match the diagnostic record
    recorded_names = [row["name"] for row in diag_after]
    recorded_nbt = sorted({int(row["num_batches_tracked"]) for row in diag_after})
    live_nbt = sorted({int(rsd[k].item()) for k in bn_keys
                       if k.endswith("num_batches_tracked")})
    lst.add(
        "05_bn_layer_names_and_nbt_match_diagnostic",
        len(bn_names) == EXPECTED_BN_LAYERS and bn_names == sorted(recorded_names)
        and live_nbt == recorded_nbt == [EXPECTED_TRAIN_BATCHES],
        bn_layers=len(bn_names),
        names_match=bn_names == sorted(recorded_names),
        nbt_live=live_nbt,
        nbt_diagnostic=recorded_nbt,
        expected_nbt=EXPECTED_TRAIN_BATCHES,
    )

    # 6. Training-only recalibration scope matches the diagnostic record
    lst.add(
        "06_training_only_scope_matches_diagnostic",
        int(rec.get("train_samples_used", -1)) == EXPECTED_TRAIN_SAMPLES
        and int(diag_recal.get("train_samples_used", -1)) == EXPECTED_TRAIN_SAMPLES
        and int(rec.get("train_batches_used", -1)) == EXPECTED_TRAIN_BATCHES
        and int(diag_recal.get("train_batches_used", -1)) == EXPECTED_TRAIN_BATCHES
        and int(rec.get("batchnorm_layers", -1)) == EXPECTED_BN_LAYERS,
        artifact_train_samples=int(rec.get("train_samples_used", -1)),
        diagnostic_train_samples=int(diag_recal.get("train_samples_used", -1)),
        artifact_train_batches=int(rec.get("train_batches_used", -1)),
        diagnostic_train_batches=int(diag_recal.get("train_batches_used", -1)),
        expected_samples=EXPECTED_TRAIN_SAMPLES,
        expected_batches=EXPECTED_TRAIN_BATCHES,
        policy=rec.get("recalibration"),
    )

    # 7. BN buffers finite, running variance strictly positive
    nonfinite = [k for k in bn_keys if not torch.isfinite(rsd[k].float()).all()]
    bad_var = [k for k in bn_keys if k.endswith("running_var") and bool((rsd[k] <= 0).any())]
    lst.add(
        "07_bn_buffers_finite_and_var_positive",
        not nonfinite and not bad_var,
        nonfinite=nonfinite,
        nonpositive_variance=bad_var,
    )

    # 8. BN summary vs the Phase 2 record (hard gate 1e-4, warning tier 1e-5)
    gate_rtol = GATE_POLICY["artifact_integrity"]["bn_summary_rtol_gate"]
    warn_rtol = GATE_POLICY["artifact_integrity"]["bn_summary_rtol_warn"]
    model = diag.make_model(recalibrated, len(label_names), torch.device("cpu"))
    live = diag.summarize_bn(model)
    worst = 0.0
    worst_at = ""
    mismatched = []
    if len(live) != len(diag_after):
        mismatched.append(f"layer count {len(live)} != {len(diag_after)}")
    for got, want in zip(live, diag_after):
        for key in ("mean_abs_mean", "min_running_var", "max_running_var"):
            rel = abs(got[key] - want[key]) / max(abs(want[key]), 1e-30)
            if rel > worst:
                worst, worst_at = rel, f"{got['name']}.{key}"
            if rel > gate_rtol:
                mismatched.append(f"{got['name']}.{key} rel={rel:.3e}")
    if worst <= warn_rtol:
        tier = "EXACT_OR_TIGHT"
    elif worst <= gate_rtol:
        tier = "GATE_PASS_WARN"
    else:
        tier = "FAIL"
    lst.add(
        "08_bn_summary_matches_diagnostic",
        not mismatched,
        worst_relative_diff=worst,
        worst_at=worst_at,
        gate_rtol=gate_rtol,
        warn_rtol=warn_rtol,
        tier=tier,
        mismatches=mismatched,
    )

    # 10. original checkpoint untouched during this invocation (Filled by callers as
    #     check 10; recorded here as the loaded-vs-rehashed comparison).
    lst.add(
        "10_original_checkpoint_untouched",
        sha256_file(original_path) == source_sha256,
        path=str(original_path),
        expected=source_sha256,
        observed=sha256_file(original_path),
    )
    # 9. reload integrity (callers supply the result of the mode-specific check).
    #    Deliberately starts as FAIL so a code path that forgets to finish it cannot
    #    accidentally report PASS.
    lst.add(
        "09_checkpoint_reload_integrity",
        False,
        method="pending",
        note=f"to be filled by the {mode} path",
        recal_path=str(recal_path),
        recal_sha256=recal_sha256,
    )
    return lst


def finish_reload_check(lst: Checklist, ok: bool, **details) -> None:
    name = "09_checkpoint_reload_integrity"
    lst.checks[name].update({"pass": bool(ok), **details})
    if ok:
        while name in lst.failures:
            lst.failures.remove(name)
    elif name not in lst.failures:
        lst.failures.append(name)


# --------------------------------------------------------------------------- #
# Gate B - numeric reproduction (one Validation pass)
# --------------------------------------------------------------------------- #
def evaluate_numeric(model, label_names, val_idx, device) -> tuple:
    loader = diag.make_loader(val_idx, device)
    data = collect_dual(model, loader, device)
    labels = data["labels"]
    official = diag.get_metrics(labels, data["official"])
    auxiliary = diag.get_metrics(labels, data["auxiliary"])

    nitro_j = label_names.index("nitro")
    nitro_official = data["official"][:, nitro_j]
    nitro_aux = data["auxiliary"][:, nitro_j]
    nitro_pos = int(labels[:, nitro_j].sum())
    tie_official = int(np.unique(nitro_official).size)
    tie_aux = int(np.unique(nitro_aux).size)

    diagnostics = {
        "n_validation_samples": int(len(labels)),
        "logits_dtype": data["logits_dtype"],
        "official_sigmoid_dtype": data["official_sigmoid_dtype"],
        "nitro": {
            "positives": nitro_pos,
            "unique_official_probabilities": tie_official,
            "unique_auxiliary_probabilities": tie_aux,
            "note": ("few distinct FP16 sigmoid values => near-tie re-ordering under "
                     "tiny logit perturbations is expected"),
        },
    }
    return official, auxiliary, diagnostics


# --------------------------------------------------------------------------- #
# shared context
# --------------------------------------------------------------------------- #
def resolve_run(loss_key: str, seed: int) -> tuple:
    if seed not in diag.RUN_IDS[loss_key]:
        raise SystemExit(f"STOP: loss={loss_key} has no configured seed {seed}")
    run_name = diag.RUN_IDS[loss_key][seed]
    source_dir = ROOT / "runs" / run_name
    dest_dir = OUTPUT_ROOT / run_name
    return run_name, source_dir, dest_dir


def load_context(loss_key: str, seed: int, label_names: list) -> tuple:
    diag.require_inputs(ROOT / "runs" / diag.RUN_IDS[loss_key][seed])
    with np.load(diag.SPLIT_PATH) as split_file:
        split = {k: np.asarray(split_file[k], dtype=np.int64) for k in ("train", "val", "test")}
    if len(split["train"]) != EXPECTED_TRAIN_SAMPLES:
        raise SystemExit("STOP: unexpected TRAIN index count")
    if len(np.unique(split["train"])) != EXPECTED_TRAIN_SAMPLES:
        raise SystemExit("STOP: duplicated TRAIN indices")
    return split


def check_output_free(dest_ckpt: Path, manifest_path: Path) -> None:
    if dest_ckpt.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Destination already exists under {dest_ckpt.parent}. No file is overwritten: "
            "inspect the existing files manually, then move them aside if a real retry is "
            "intended. Do NOT delete a checkpoint without recording its SHA256 first."
        )


def build_manifest(*, artifact: Checklist, numeric: dict, auxiliary: dict,
                   diagnostics: dict, mode: str, extra: dict) -> dict:
    """Assemble the manifest (called only when Gate A passed).

    Gate A PASS + Gate B PASS  -> ARTIFACT_PASS__NUMERIC_PASS, release ELIGIBLE.
    Gate A PASS + Gate B FAIL  -> ARTIFACT_PASS__NUMERIC_FAIL, release NOT_ELIGIBLE.
                                  A human release is recorded separately, as
                                  manual_approval.json bound to this file's sha256; it is
                                  never encoded as a pass anywhere in the manifest.
    """
    numeric_pass = numeric["status"] == "PASS"
    release = "ELIGIBLE" if numeric_pass else "NOT_ELIGIBLE"
    release_reason = (
        "artifact integrity PASS and numeric reproduction PASS" if numeric_pass else
        f"artifact integrity PASS but numeric reproduction FAILED for "
        f"{numeric['exceeding_metrics']}; a sha-bound manual_approval.json is required "
        "before the formal ablation may use this checkpoint"
    )

    body = dict(extra)
    body.update({
        "gate_policy": GATE_POLICY,
        "artifact_integrity": artifact.as_dict(),
        "numeric_reproduction": numeric,
        "official_fp16_metrics": official_metrics_report(numeric),
        "auxiliary_fp32_sigmoid": {
            "status": GATE_POLICY["auxiliary_channel"]["status"],
            "note": GATE_POLICY["auxiliary_channel"]["note"],
            "metrics": auxiliary,
            "recorded_only_reason": (
                "no Phase 2 reference exists for this channel; it is a stability indicator, "
                "not a reproduction check, and is never mixed with the FP16 numbers"
            ),
        },
        "numeric_diagnostics": diagnostics,
        "acceptance": {
            "mode": mode,
            "artifact_integrity": artifact.status,
            "numeric_reproduction": numeric["status"],
            "status": f"ARTIFACT_{artifact.status}__NUMERIC_{numeric['status']}",
            "all_metrics_agree": numeric_pass and artifact.status == "PASS",
            "manifest_written_because": (
                "artifact_integrity == PASS; the numeric verdict is recorded verbatim"
            ),
            "dual_status_manifest": not numeric_pass,
            "manifest_generated_by_code": True,
            "manifest_hand_edited": False,
        },
        "ablation_release": {
            "status": release,
            "reason": release_reason,
            "requires_manual_approval": not numeric_pass,
            "manual_approval_file": (str(raman_numeric_audit.APPROVAL_NAME)
                                     if not numeric_pass else None),
            "bound_checkpoint_sha256": str(extra.get("saved_sha256")) or None,
            "approval_scope": (
                "one checkpoint sha256, one audit verdict; not transferable; "
                "OTHER_ANOMALY is never approvable"
            ),
            "manifest_existence_is_not_approval": True,
        },
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    body["verification"]["numeric_reproduction_passed"] = numeric_pass
    body["verification"]["diagnostic_consistent"] = numeric_pass
    if not numeric_pass:
        body["verification"]["diagnostic_consistent_note"] = (
            "NUMERICAL REPRODUCTION FAILED for "
            f"{numeric['exceeding_metrics']} at tolerance {numeric['tolerance']:.1e} "
            f"(audit class {numeric.get('audit_class')}). Artifact integrity is unaffected "
            "and is evidenced separately; no claim of metric agreement is made anywhere in "
            "this manifest."
        )
    return body


def print_gate_summary(artifact: Checklist, numeric: dict) -> None:
    print("\n  ARTIFACT INTEGRITY GATE")
    for name, row in artifact.checks.items():
        print(f"    [{'PASS' if row['pass'] else 'FAIL'}] {name}")
    print(f"    -> {artifact.status}")
    print("\n  NUMERIC REPRODUCTION GATE (official FP16 sigmoid, 14 labels incl. nitro)")
    for key in METRIC_KEYS:
        row = numeric["metrics"][key]
        print(f"    {key:18s} observed={row['observed']:.6f} recorded={row['recorded']:.6f} "
              f"|d|={row['abs_diff']:.3e} tol={row['tolerance']:.1e} "
              f"{'ok' if row['within_tolerance'] else 'EXCEEDS'}")
    print(f"    -> {numeric['status']}")
    if numeric["status"] != "PASS":
        print("    NOTE: numerical reproducibility FAILED for: "
              f"{numeric['exceeding_metrics']}. This is reported as a failure and is not "
              "converted into a pass. The artifact verdict is independent and is stated "
              "separately above.")


# --------------------------------------------------------------------------- #
# normal path
# --------------------------------------------------------------------------- #
def cmd_save(args, label_names: list) -> int:
    run_name, source_dir, dest_dir = resolve_run(args.loss, args.seed)
    original_ckpt = source_dir / "best_model.pt"
    dest_ckpt = dest_dir / "best_model_bn_recalibrated.pt"
    manifest_path = dest_dir / "bn_recalibration_manifest.json"

    check_output_free(dest_ckpt, manifest_path)
    split = load_context(args.loss, args.seed, label_names)
    train_idx, val_idx = split["train"], split["val"]

    source_sha256 = sha256_file(original_ckpt)
    checkpoint = torch.load(original_ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    canonical = "EXP-RAMAN-001-FULL" if args.loss == "bce" else "EXP-RAMAN-002-FULL"
    if config.get("experiment") not in (canonical, run_name):
        raise SystemExit("STOP: checkpoint experiment metadata does not match the selected run")
    if int(config.get("seed", -1)) != args.seed:
        raise SystemExit("STOP: wrong checkpoint seed")
    if config.get("label_names") != label_names or checkpoint.get("label_names") != label_names:
        raise SystemExit("STOP: checkpoint label names/order differs from the Raman schema")
    if config.get("normalization") != "max":
        raise SystemExit("STOP: unexpected normalization")
    if int(config.get("batch_size", -1)) != diag.BATCH_SIZE:
        raise SystemExit("STOP: batch size differs from the established baseline")
    if int(config.get("num_workers", -1)) != diag.NUM_WORKERS:
        raise SystemExit("STOP: num_workers differs from the established baseline")
    if float(config.get("threshold", -1)) != diag.THRESHOLD:
        raise SystemExit("STOP: unexpected default classification threshold")
    if bool(config.get("use_pos_weight")) != (args.loss == "wbce"):
        raise SystemExit("STOP: wrong BCE/WBCE checkpoint selected")
    if args.loss == "wbce" and (config.get("pos_weight_transform") != "sqrt"
                                or float(config.get("pos_weight_cap", -1)) != 20.0):
        raise SystemExit("STOP: weighted BCE configuration mismatch")

    diagnostic_path = DIAGNOSTIC_ROOT / run_name / "batchnorm_full_diagnostic.json"
    if not diagnostic_path.is_file():
        raise SystemExit(f"STOP: approved Full BN Diagnostic record missing: {diagnostic_path}")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dry_run:
        print(f"DRY RUN: would recalibrate BN for {run_name} on {device} using "
              f"{len(train_idx)} TRAIN samples (203 batches), then Gate A + Gate B.")
        print(f"  would write (exclusive): {dest_ckpt}")
        print(f"  would write (exclusive): {manifest_path}")
        print("  nothing was executed.")
        return 0
    if device.type != "cuda":
        raise SystemExit(
            "STOP: this acceptance path must run under the official protocol "
            "(CUDA AMP fp16, batch 512). No CUDA device is available, and evaluating on "
            "CPU would silently change the historical metric definition. Use --dry-run for "
            "a CPU-only artifact audit."
        )
    torch.backends.cudnn.benchmark = True

    model = diag.make_model(checkpoint, len(label_names), device)
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    params_before = {k: v.detach().clone() for k, v in model.named_parameters()}

    print("=" * 78)
    print("Raman TRAIN-only BN recalibration and checkpoint preservation")
    print(f"Run: {run_name} | Loss: {args.loss} | Seed: {args.seed} | Device: {device}")
    print(f"Source SHA256: {source_sha256}")
    print(f"Gate policy: {GATE_POLICY['version']}")
    print("=" * 78)

    train_loader = diag.make_loader(train_idx, device)
    info = diag.reset_recompute_bn(model, train_loader, device)
    if info["train_samples_used"] != EXPECTED_TRAIN_SAMPLES:
        raise SystemExit("STOP: BN recalibration used the wrong number of TRAIN samples")
    if info["train_batches_used"] != EXPECTED_TRAIN_BATCHES:
        raise SystemExit("STOP: BN recalibration used the wrong number of TRAIN batches")

    new_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_params = [k for k, v in model.named_parameters() if not torch.equal(params_before[k], v.detach())]
    if bad_params:
        raise ArtifactIntegrityError(f"trainable parameters changed during BN recalibration: {bad_params}")

    # Build the artifact dict that would be written, then Gate A on it (in memory).
    recal = {
        "config": config,
        "label_names": label_names,
        "model_state_dict": new_state,
        "bn_recalibration": {
            **info,
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
            "checkpoint_type": "postprocessed_bn_running_statistics",
            "note": "Training-only BN buffer recalibration; no gradient or optimizer updates",
        },
    }
    artifact_lst = verify_artifact(
        checkpoint, recal,
        original_path=original_ckpt, recal_path=dest_ckpt,
        source_sha256=source_sha256, recal_sha256="<not yet written>",
        diagnostic=diagnostic, label_names=label_names, mode="save",
    )
    # Pre-write gate A. Check 09 (reload integrity) is *pending* by construction here:
    # the file it has to reload does not exist until a few lines below, so it is excluded
    # from this verdict and finished after the write. Including it would make the gate
    # unsatisfiable in this path (regression fixed after the BCE seed-123 failure).
    PENDING_BEFORE_WRITE = {"09_checkpoint_reload_integrity"}
    if artifact_lst.status_ignoring(PENDING_BEFORE_WRITE) != "PASS":
        blocking = [f for f in artifact_lst.failures if f not in PENDING_BEFORE_WRITE]
        raise ArtifactIntegrityError(
            "Artifact integrity gate FAILED before writing; no checkpoint and no manifest "
            "were written.\n  " + "\n  ".join(blocking)
        )

    # Gate B before writing anything: an out-of-envelope result cannot leave an orphan.
    official, auxiliary, diagnostics = evaluate_numeric(model, label_names, val_idx, device)
    numeric = classify_numeric(official, diagnostic["after"]["validation"])
    if numeric["status"] != "PASS":
        print("\n  NUMERICAL REPRODUCTION FAILED (tolerance 1e-3 on all six official FP16 "
              f"metrics). Exceeding metrics: {numeric['exceeding_metrics']}.")
        print("  The checkpoint and its provenance manifest are still written: Gate A passed,")
        print("  and the manifest will state ARTIFACT_PASS__NUMERIC_FAIL with the observed,")
        print("  reference and absolute difference for every metric. This is NOT approval.")
        print("  Do not re-run this command to obtain a different number.")

    # Save (exclusive), reload, check 9, then manifest.
    output = dict(checkpoint)
    output["model_state_dict"] = new_state
    output["bn_recalibration"] = recal["bn_recalibration"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    with dest_ckpt.open("xb") as fh:
        torch.save(output, fh)
    reloaded = torch.load(dest_ckpt, map_location="cpu", weights_only=False)
    saved_sha256 = sha256_file(dest_ckpt)
    reload_ok = all(torch.equal(new_state[k], reloaded["model_state_dict"][k]) for k in new_state)
    finish_reload_check(artifact_lst, reload_ok,
                        method="reload the written file and compare every tensor",
                        tensors_compared=len(new_state),
                        recal_sha256=saved_sha256)
    if artifact_lst.status != "PASS":
        raise ArtifactIntegrityError(
            "Reload integrity failed after writing; no manifest was written.\n  "
            + "\n  ".join(artifact_lst.failures)
        )

    changed_bn = [k for k in bn_buffer_keys(new_state)
                  if not torch.equal(before_state[k], new_state[k])]
    print_gate_summary(artifact_lst, numeric)
    extra = {
        "source_checkpoint": str(original_ckpt),
        "source_sha256": source_sha256,
        "run_directory": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "checkpoint_config_seed": int(config["seed"]),
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "gate_device": str(device),
        "official_protocol": ("CUDA AMP fp16; sigmoid on the logits dtype; threshold 0.5; "
                              "14 labels including nitro"),
        "normalization": config["normalization"],
        "default_threshold": float(config["threshold"]),
        "batchnorm_layers": int(info["batchnorm_layers"]),
        "train_batches_used": int(info["train_batches_used"]),
        "train_samples_used": int(info["train_samples_used"]),
        "changed_bn_buffer_count": len(changed_bn),
        "changed_bn_buffer_names": sorted(changed_bn),
        "checkpoint_type": "postprocessed_bn_running_statistics",
        "note": ("Training-only BN buffer recalibration; no gradient or optimizer updates"),
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
            "test_samples": int(len(split["test"])),
            "train_batches": int(info["train_batches_used"]),
            "split_file": str(diag.SPLIT_PATH),
        },
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
        "diagnostic_reference": {
            "path": str(diagnostic_path),
            "bn_statistics_match": True,
            "validation_metrics_match": numeric["status"] != "FAIL",
            "max_validation_metric_abs_diff": max(
                v["abs_diff"] for v in numeric["metrics"].values()),
            "validation_metrics_recomputed": official,
            "validation_metrics_recorded": diagnostic["after"]["validation"],
        },
        "original_checkpoint_untouched": True,
    }
    manifest = build_manifest(artifact=artifact_lst, numeric=numeric, auxiliary=auxiliary,
                              diagnostics=diagnostics, mode="save", extra=extra)
    with manifest_path.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    if sha256_file(original_ckpt) != source_sha256:
        raise RuntimeError("Original checkpoint changed during the run (CRITICAL)")
    print(f"\n  SAVED: {dest_ckpt}")
    print(f"  MANIFEST: {manifest_path}")
    print(f"  ORIGINAL UNTOUCHED: {original_ckpt}")
    print(f"  ACCEPTANCE: {manifest['acceptance']['status']}")
    print("  Note: thresholds for the new probabilities must be re-fitted on Validation.")
    if manifest["ablation_release"]["status"] != "ELIGIBLE":
        print("  NOT RELEASED for the formal ablation: human decision required.")
        return 3
    return 0


# --------------------------------------------------------------------------- #
# finalize path - re-admit an existing checkpoint, never rewrite it
# --------------------------------------------------------------------------- #
def cmd_finalize(args, label_names: list) -> int:
    run_name, source_dir, dest_dir = resolve_run(args.loss, args.seed)
    original_ckpt = source_dir / "best_model.pt"
    dest_ckpt = dest_dir / "best_model_bn_recalibrated.pt"
    manifest_path = dest_dir / "bn_recalibration_manifest.json"

    if not dest_ckpt.is_file():
        raise SystemExit(f"STOP: no existing checkpoint to finalize: {dest_ckpt}")
    if manifest_path.exists():
        raise FileExistsError(
            f"STOP: a manifest already exists: {manifest_path}. Finalization never "
            "overwrites an existing manifest."
        )

    source_sha256 = sha256_file(original_ckpt)
    recal_sha256 = sha256_file(dest_ckpt)
    print("=" * 78)
    print("Raman BN checkpoint FINALIZATION (no BN recomputation, no checkpoint rewrite)")
    print(f"Run: {run_name} | Loss: {args.loss} | Seed: {args.seed}")
    print(f"Existing checkpoint: {dest_ckpt}")
    print(f"  sha256: {recal_sha256}")
    if args.expect_checkpoint_sha256:
        if recal_sha256 != args.expect_checkpoint_sha256.lower():
            raise SystemExit(
                "STOP: existing checkpoint sha256 does not match --expect-checkpoint-sha256.\n"
                f"  expected {args.expect_checkpoint_sha256.lower()}\n  observed {recal_sha256}"
            )
        print("  PASS: sha256 matches the audited value")
    print(f"Source checkpoint: {original_ckpt}")
    print(f"  sha256: {source_sha256}")
    print("=" * 78)

    original = torch.load(original_ckpt, map_location="cpu", weights_only=False)
    recalibrated = torch.load(dest_ckpt, map_location="cpu", weights_only=False)
    config = recalibrated.get("config", {})
    if config.get("experiment") not in (("EXP-RAMAN-001-FULL" if args.loss == "bce"
                                         else "EXP-RAMAN-002-FULL"), run_name):
        raise SystemExit("STOP: checkpoint experiment metadata does not match the selected run")
    if int(config.get("seed", -1)) != args.seed:
        raise SystemExit("STOP: wrong checkpoint seed")
    if config.get("label_names") != label_names or recalibrated.get("label_names") != label_names:
        raise SystemExit("STOP: checkpoint label names/order differs from the Raman schema")

    run_meta = recalibrated.get("bn_recalibration", {})
    if run_meta.get("loss") != args.loss or int(run_meta.get("seed", -1)) != args.seed:
        raise SystemExit("STOP: checkpoint provenance loss/seed does not match the CLI")
    if run_meta.get("run_directory") != run_name:
        raise SystemExit("STOP: checkpoint provenance run_directory does not match")

    diagnostic_path = DIAGNOSTIC_ROOT / run_name / "batchnorm_full_diagnostic.json"
    if not diagnostic_path.is_file():
        raise SystemExit(f"STOP: approved Full BN Diagnostic record missing: {diagnostic_path}")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    split = load_context(args.loss, args.seed, label_names)

    artifact_lst = verify_artifact(
        original, recalibrated,
        original_path=original_ckpt, recal_path=dest_ckpt,
        source_sha256=source_sha256, recal_sha256=recal_sha256,
        diagnostic=diagnostic, label_names=label_names, mode="finalize",
    )
    # Check 9 in finalize mode: two independent loads must agree tensor by tensor.
    again = torch.load(dest_ckpt, map_location="cpu", weights_only=False)
    load_stable = (sorted(again["model_state_dict"]) == sorted(recalibrated["model_state_dict"])
                   and all(torch.equal(again["model_state_dict"][k],
                                       recalibrated["model_state_dict"][k])
                           for k in recalibrated["model_state_dict"]))
    finish_reload_check(
        artifact_lst, load_stable,
        method="two independent loads from disk compared tensor by tensor "
               "(no write in this mode)",
        tensors_compared=len(recalibrated["model_state_dict"]),
    )

    print("\n  ARTIFACT INTEGRITY GATE (deterministic, CPU-only)")
    for name, row in artifact_lst.checks.items():
        print(f"    [{'PASS' if row['pass'] else 'FAIL'}] {name}")
    print(f"    -> {artifact_lst.status}")

    if artifact_lst.status != "PASS":
        raise ArtifactIntegrityError(
            "Artifact integrity gate FAILED; no manifest was written.\n  "
            + "\n  ".join(artifact_lst.failures)
            + "\nThe existing checkpoint was NOT modified."
        )

    if args.dry_run:
        print("\n  DRY RUN: artifact gate passed. Numeric reproduction (Gate B) needs a GPU "
              "Validation pass and was skipped; no manifest was written.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit(
            "STOP: Gate B must run under the official protocol (CUDA AMP fp16, batch 512). "
            "No CUDA device is available; evaluating on CPU would change the historical "
            "metric definition. Use --dry-run for a CPU-only artifact audit."
        )
    torch.backends.cudnn.benchmark = True
    model = diag.make_model(recalibrated, len(label_names), device)
    official, auxiliary, diagnostics = evaluate_numeric(
        model, label_names, split["val"], device)
    numeric = classify_numeric(official, diagnostic["after"]["validation"])
    print_gate_summary(artifact_lst, numeric)
    if numeric["status"] != "PASS":
        print("\n  NUMERICAL REPRODUCTION FAILED (tolerance 1e-3 on all six official FP16 "
              f"metrics). Exceeding metrics: {numeric['exceeding_metrics']}.")
        print("  A provenance manifest will still be written with the dual status")
        print("  ARTIFACT_PASS__NUMERIC_FAIL; the checkpoint is NOT released for the formal")
        print("  ablation. This is not approval, and re-running will not re-roll the verdict.")

    extra = {
        "source_checkpoint": str(original_ckpt),
        "source_sha256": source_sha256,
        "run_directory": run_name,
        "loss": args.loss,
        "seed": args.seed,
        "checkpoint_config_seed": int(config["seed"]),
        "batch_size": diag.BATCH_SIZE,
        "num_workers": diag.NUM_WORKERS,
        "gate_device": str(device),
        "official_protocol": ("CUDA AMP fp16; sigmoid on the logits dtype; threshold 0.5; "
                              "14 labels including nitro"),
        "normalization": config.get("normalization"),
        "default_threshold": float(config.get("threshold", -1)),
        "batchnorm_layers": int(run_meta.get("batchnorm_layers", -1)),
        "train_batches_used": int(run_meta.get("train_batches_used", -1)),
        "train_samples_used": int(run_meta.get("train_samples_used", -1)),
        "changed_bn_buffer_count": EXPECTED_BN_BUFFERS,
        "changed_bn_buffer_names": sorted(bn_buffer_keys(recalibrated["model_state_dict"])),
        "checkpoint_type": "postprocessed_bn_running_statistics",
        "note": ("Training-only BN buffer recalibration performed by the earlier run; this "
                 "manifest was generated by finalization without recomputation"),
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
            "train_samples": int(len(split["train"])),
            "val_samples": int(len(split["val"])),
            "test_samples": int(len(split["test"])),
            "train_batches": int(run_meta.get("train_batches_used", -1)),
            "split_file": str(diag.SPLIT_PATH),
        },
        "saved_checkpoint": str(dest_ckpt),
        "saved_sha256": recal_sha256,
        "saved_size_bytes": dest_ckpt.stat().st_size,
        "verification": {
            "only_bn_buffers_changed": True,
            "trainable_parameters_unchanged": True,
            "bn_buffers_finite": True,
            "reload_ok": True,
            "diagnostic_consistent": True,
        },
        "diagnostic_reference": {
            "path": str(diagnostic_path),
            "bn_statistics_match": True,
            "validation_metrics_match": numeric["status"] != "FAIL",
            "max_validation_metric_abs_diff": max(
                v["abs_diff"] for v in numeric["metrics"].values()),
            "validation_metrics_recomputed": official,
            "validation_metrics_recorded": diagnostic["after"]["validation"],
        },
        "original_checkpoint_untouched": True,
        "finalization": {
            "mode": "finalize_existing_checkpoint",
            "checkpoint_written_by": (
                "training/save_raman_bn_checkpoint.py (earlier run under the retired "
                "raman-bn-gate-v1 mAP rule)"
            ),
            "checkpoint_mtime": _dt.datetime.fromtimestamp(
                dest_ckpt.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "bn_recalibration_rerun": False,
            "checkpoint_rewritten": False,
            "checkpoint_sha256_before": recal_sha256,
            "checkpoint_sha256_after": sha256_file(dest_ckpt),
            "prior_manifest_existed": False,
            "expect_checkpoint_sha256": args.expect_checkpoint_sha256,
        },
    }
    manifest = build_manifest(artifact=artifact_lst, numeric=numeric, auxiliary=auxiliary,
                              diagnostics=diagnostics, mode="finalize", extra=extra)
    with manifest_path.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    if sha256_file(original_ckpt) != source_sha256:
        raise RuntimeError("Original checkpoint changed during finalization (CRITICAL)")
    if sha256_file(dest_ckpt) != recal_sha256:
        raise RuntimeError("Recalibrated checkpoint changed during finalization (CRITICAL)")
    print(f"\n  MANIFEST WRITTEN: {manifest_path}")
    print(f"  CHECKPOINT UNCHANGED: {dest_ckpt}")
    print(f"  ORIGINAL UNTOUCHED: {original_ckpt}")
    print(f"  ACCEPTANCE: {manifest['acceptance']['status']}")
    if manifest["ablation_release"]["status"] != "ELIGIBLE":
        print("  NOT RELEASED for the formal ablation: human decision required.")
        return 3
    return 0


# --------------------------------------------------------------------------- #
# manual release record for a NUMERIC_FAIL checkpoint
# --------------------------------------------------------------------------- #
def cmd_approve_release(args, label_names: list) -> int:
    """Record a human release bound to an existing checkpoint and its manifest.

    This never edits the manifest and never changes a numeric verdict. It writes
    manual_approval.json next to the checkpoint, bound to one sha256.
    """
    run_name, _source_dir, dest_dir = resolve_run(args.loss, args.seed)
    dest_ckpt = dest_dir / "best_model_bn_recalibrated.pt"
    manifest_path = dest_dir / "bn_recalibration_manifest.json"
    if not dest_ckpt.is_file() or not manifest_path.is_file():
        raise SystemExit("STOP: --approve-release needs both the checkpoint and its manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = sha256_file(dest_ckpt)
    if manifest.get("saved_sha256") != sha:
        raise SystemExit("STOP: manifest.saved_sha256 does not match the checkpoint on disk")
    acceptance = manifest.get("acceptance", {})
    if acceptance.get("status") == "ARTIFACT_PASS__NUMERIC_PASS":
        raise SystemExit(
            "STOP: numeric reproduction PASSED for this checkpoint; no manual release is "
            "needed and none is recorded.")
    if acceptance.get("artifact_integrity") != "PASS":
        raise SystemExit("STOP: artifact integrity did not pass; release is not approvable")
    audit_status = manifest.get("numeric_reproduction", {}).get("audit_class")
    if audit_status == raman_numeric_audit.OTHER:
        raise SystemExit(
            "STOP: the audit class is OTHER_ANOMALY (a threshold-based metric or AUROC is "
            "outside tolerance). That class can never be approved by a manual flag.")
    path = raman_numeric_audit.write_approval(
        dest_dir, args.manual_approval_note, args.approve_checkpoint_sha256, sha,
        audit_status, manifest.get("saved_sha256"))
    print("=" * 78)
    print("MANUAL RELEASE RECORDED (numeric verdict unchanged: FAIL)")
    print(f"  run         : {run_name}")
    print(f"  checkpoint  : {dest_ckpt}")
    print(f"  sha256      : {sha}")
    print(f"  audit class : {audit_status}")
    print(f"  note        : {args.manual_approval_note}")
    print(f"  record      : {path}")
    print("  The manifest is unchanged and still reports NUMERIC_FAIL.")
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
def show_policy() -> int:
    print(json.dumps(GATE_POLICY, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save (or re-admit) a Raman train-only BN-recalibrated checkpoint "
                    "under gate policy v4: artifact integrity and numeric reproduction are "
                    "reported as two independent verdicts."
    )
    parser.add_argument("--loss", choices=["bce", "wbce"], default="bce")
    parser.add_argument("--seed", choices=[42, 123, 2026], type=int, default=42)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="re-verify an already written recalibrated checkpoint and write "
                             "only its manifest; never recomputes BN, never rewrites the file")
    parser.add_argument("--expect-checkpoint-sha256", default=None,
                        help="in --finalize-existing mode, require this sha256 for the "
                             "existing checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs and print the plan; with --finalize-existing it "
                             "runs the CPU-only artifact gate and writes nothing")
    parser.add_argument("--show-policy", action="store_true",
                        help="print the gate policy and exit")
    parser.add_argument("--approve-release", action="store_true",
                        help="record a sha-bound human release for an existing "
                             "NUMERIC_FAIL checkpoint; never edits the manifest")
    parser.add_argument("--manual-approval-note", default=None,
                        help="with --approve-release: who/when/on what evidence")
    parser.add_argument("--approve-checkpoint-sha256", default=None,
                        help="with --approve-release: the checkpoint sha256 being approved")
    args = parser.parse_args()

    if args.show_policy:
        return show_policy()

    label_names = json.loads(diag.LABEL_NAMES_PATH.read_text(encoding="utf-8"))
    if len(label_names) != EXPECTED_N_LABELS:
        raise SystemExit(f"STOP: expected {EXPECTED_N_LABELS} Raman labels")

    if args.approve_release:
        return cmd_approve_release(args, label_names)
    if args.finalize_existing:
        return cmd_finalize(args, label_names)
    return cmd_save(args, label_names)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:          # e.g. piping into Select-Object / more
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
