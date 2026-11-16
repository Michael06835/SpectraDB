"""Shared numeric-audit logic for the Raman Phase 3-5 acceptance path (gate policy v5).

One place, three callers
------------------------
    training/save_raman_bn_checkpoint.py                  (Gate B, Validation)
    training/calibrate_raman_original_thresholds.py       (Phase 4, Test anomaly record)
    training/calibrate_raman_recalibrated_thresholds.py   (Phase 5, Validation release + Test record)

The tolerance is 1e-3 everywhere, identical to IR v1 and UVSPEC15 v1. This module never
changes a reference value and never widens a tolerance.

Three audit classes
-------------------
PASS
    every audited metric is within 1e-3 of its recorded value.

MAP_ONLY_NUMERIC_MISMATCH
    the only entries outside 1e-3 are mAP entries (ranking-only metric), while every
    threshold-based metric and AUROC stayed inside 1e-3. This is a DESCRIPTIVE label: it
    says which metrics moved, NOT why. In particular it does not assert that FP16
    quantisation is the cause, and the magnitude of the deviation is not interpreted here.
    It is still a NUMERIC_FAIL.

OTHER_ANOMALY
    anything else: a threshold-based metric or AUROC out of tolerance, a missing
    reference value, a non-finite number. This indicates a wrong artifact, wrong data or
    wrong split, and it can never be approved.

Release vs anomaly
------------------
`split_release_and_anomaly` partitions the audited metrics into:
    validation_release -- the Validation-side technical release condition;
    test_anomaly       -- a Test-side record that never selects a model, threshold or
                          inference run, and never authorises anything on its own.

Single-shot rule
----------------
The FIRST audit record written for a (run, phase) pair is authoritative. Later runs load
it instead of replacing it, so a re-run cannot convert a recorded deviation into a pass.

Probability artefacts
---------------------
The first successful inference writes the Validation/Test probability arrays plus a
sidecar `probability_artifacts.json` (exclusive create). A resumed run loads those arrays
instead of re-running inference, and a later run may never replace them: replacing the
first observation could otherwise manufacture a different verdict.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "runs" / "RAMAN-NUMERIC-AUDIT"

TOLERANCE = 1e-3
GATE_POLICY_VERSION = "raman-bn-gate-v5"

PASS = "PASS"
MAP_ONLY = "MAP_ONLY_NUMERIC_MISMATCH"
OTHER = "OTHER_ANOMALY"
STATUSES = (PASS, MAP_ONLY, OTHER)

THRESHOLD_METRIC_LEAVES = ("micro_f1", "macro_f1", "micro_precision", "micro_recall",
                           "macro_AUROC")
MAP_LEAF = "mAP"

PROBABILITY_MANIFEST_NAME = "probability_artifacts.json"
APPROVAL_NAME = "manual_approval.json"


class ProbabilityArtifactError(RuntimeError):
    """The frozen probability arrays are missing, inconsistent or would be replaced."""


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def array_sha256(a: np.ndarray) -> str:
    b = np.ascontiguousarray(np.asarray(a).astype("<f4", copy=False))
    return hashlib.sha256(b.tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def classify(observed: dict, recorded: dict, context: str,
             tolerance: float = TOLERANCE) -> dict:
    """Three-class audit of official FP16 metrics against a recorded reference.

    Keys may be namespaced ("test.mAP"). The class depends on which *leaf* names exceed
    the tolerance; the magnitude is recorded but never interpreted.
    """
    per_metric = {}
    exceeding = []
    missing = []
    nonfinite = []

    for key, obs in observed.items():
        ref = recorded.get(key)
        if ref is None:
            missing.append(key)
            continue
        obs_f, ref_f = float(obs), float(ref)
        if not (np.isfinite(obs_f) and np.isfinite(ref_f)):
            nonfinite.append(key)
            continue
        diff = abs(obs_f - ref_f)
        per_metric[key] = {
            "observed": obs_f,
            "recorded": ref_f,
            "delta": obs_f - ref_f,
            "abs_diff": diff,
            "tolerance": tolerance,
            "within_tolerance": diff <= tolerance,
        }
        if diff > tolerance:
            exceeding.append(key)

    if missing or nonfinite:
        status = OTHER
        reason = ("reference values missing or non-finite: "
                  f"missing={missing} nonfinite={nonfinite}")
    elif not exceeding:
        status = PASS
        reason = "all audited metrics within tolerance"
    elif all(_leaf(k) == MAP_LEAF for k in exceeding):
        status = MAP_ONLY
        reason = ("only ranking-only mAP entries are outside tolerance; every "
                  "threshold-based metric and AUROC is inside it. The label describes "
                  "WHICH metrics moved; it does not establish WHY.")
    else:
        status = OTHER
        reason = ("threshold-based metric or AUROC outside tolerance: "
                  f"{[k for k in exceeding if _leaf(k) != MAP_LEAF]}")

    return {
        "status": status,
        "reason": reason,
        "context": context,
        "tolerance": tolerance,
        "gate_policy_version": GATE_POLICY_VERSION,
        "metrics": per_metric,
        "exceeding_metrics": exceeding,
        "missing_metrics": missing,
        "nonfinite_metrics": nonfinite,
        "cause_not_established": status == MAP_ONLY,
        "magnitude_interpreted": False,
        "all_labels_retained": True,
        "evaluated_once": True,
        "retry_policy": "single shot; re-running to obtain a different verdict is forbidden",
    }


def subset_status(metrics: dict, keys) -> dict:
    """Re-apply the same rule to a subset of already-classified metrics."""
    part = {k: metrics[k] for k in keys if k in metrics}
    exceeding = [k for k, v in part.items() if not v["within_tolerance"]]
    if not part:
        status = "NOT_EVALUATED"
    elif not exceeding:
        status = PASS
    elif all(_leaf(k) == MAP_LEAF for k in exceeding):
        status = MAP_ONLY
    else:
        status = OTHER
    return {
        "status": status,
        "evaluated_metrics": sorted(part),
        "exceeding_metrics": exceeding,
        "metrics": part,
    }


def split_release_and_anomaly(audit: dict, release_prefix: str,
                              anomaly_prefix: str) -> dict:
    """Validation-side release condition vs Test-side anomaly record.

    `release_prefix`/`anomaly_prefix` are key prefixes such as "validation." and "test.".
    The Test side is a record only: it never releases, never selects and never blocks the
    Validation-side decision.
    """
    metrics = audit["metrics"]
    release_keys = [k for k in metrics if k.startswith(release_prefix)]
    anomaly_keys = [k for k in metrics if k.startswith(anomaly_prefix)]
    return {
        "validation_release": subset_status(metrics, release_keys),
        "test_anomaly": subset_status(metrics, anomaly_keys),
        "policy": (
            "the Validation-side subset is the technical release condition; the Test-side "
            "subset is recorded and is never used to select a model, a threshold or an "
            "inference run"
        ),
    }


# --------------------------------------------------------------------------- #
def validate_approval(note, approved_sha, expected_sha, audit_status: str) -> dict:
    """Manual-approval gate for a NUMERIC_FAIL manifest.

    Approval must be explicit, traceable and BOUND to the checkpoint it authorises.
    OTHER_ANOMALY cannot be approved at all.
    """
    problems = []
    if audit_status == OTHER:
        problems.append("OTHER_ANOMALY cannot be approved by any manual flag")
    if not note or not str(note).strip():
        problems.append("--manual-approval-note is required (who / when / on what evidence)")
    if not approved_sha:
        problems.append("--approve-checkpoint-sha256 is required to bind the approval")
    elif expected_sha and str(approved_sha).lower() != str(expected_sha).lower():
        problems.append("--approve-checkpoint-sha256 does not match the checkpoint "
                        f"({approved_sha} != {expected_sha})")
    return {
        "requested": bool(note) or bool(approved_sha),
        "valid": not problems,
        "problems": problems,
        "note": note,
        "checkpoint_sha256": (str(approved_sha).lower() if approved_sha else None),
        "expected_checkpoint_sha256": (str(expected_sha).lower() if expected_sha else None),
        "audit_status_at_approval": audit_status,
        "given_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": ("approval binds to one checkpoint sha256 and one audit verdict; it is "
                   "never transferable and never applies to OTHER_ANOMALY"),
    }


# --------------------------------------------------------------------------- #
def record_path(run_name: str, phase: str, root: Path = AUDIT_ROOT) -> Path:
    return root / run_name / f"{phase}.json"


def find_record(run_name: str, phase: str, root: Path = AUDIT_ROOT) -> dict | None:
    path = record_path(run_name, phase, root)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(run_name: str, phase: str, payload: dict,
                 root: Path = AUDIT_ROOT) -> Path:
    """Exclusive create: never overwrites an earlier observation."""
    path = record_path(run_name, phase, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.update({
        "phase": phase,
        "run_directory": run_name,
        "recorded_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "single_shot": True,
        "note": ("first observation for this (run, phase); later runs must reuse this "
                 "verdict instead of replacing it"),
    })
    with path.open("x", encoding="utf-8") as fh:
        json.dump(body, fh, ensure_ascii=False, indent=2)
    return path


def resolve_authoritative(run_name: str, phase: str, current: dict,
                          root: Path = AUDIT_ROOT) -> tuple[dict, bool, Path | None]:
    """Return (authoritative_audit, is_first_observation, record_path). First record wins."""
    existing = find_record(run_name, phase, root)
    if existing is not None:
        merged = dict(existing)
        merged["subsequent_reading"] = {
            "status": current["status"],
            "exceeding_metrics": current["exceeding_metrics"],
            "metrics": current["metrics"],
            "note": "informational only; it does not replace the first verdict",
        }
        return merged, False, record_path(run_name, phase, root)
    path = write_record(run_name, phase, current, root)
    return current, True, path


# --------------------------------------------------------------------------- #
def calibration_dir_state(out_dir: Path, thresholds_name: str, summary_name: str) -> str:
    """'fresh' | 'interrupted' (thresholds frozen, no summary) | 'complete' | 'unknown'."""
    if not out_dir.exists():
        return "fresh"
    if (out_dir / summary_name).exists():
        return "complete"
    if (out_dir / thresholds_name).exists():
        return "interrupted"
    return "unknown"


def persist_probabilities(out_dir: Path, tag: str, arrays: dict, meta: dict) -> dict:
    """Write the FIRST probability arrays and their sidecar; refuse to replace them.

    Raises ProbabilityArtifactError if the sidecar already exists: a later run must load
    the first arrays instead of producing new ones, because replacing the first
    observation could manufacture a different verdict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / PROBABILITY_MANIFEST_NAME
    if sidecar.exists():
        raise ProbabilityArtifactError(
            f"{sidecar} already exists: the first probability observation must be reused, "
            "not replaced. Resume from the frozen arrays instead."
        )
    manifest = {
        "tag": tag,
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": "first observation is authoritative; later runs must load it",
        "meta": meta,
        "arrays": {},
    }
    for name, array in arrays.items():
        target = out_dir / f"{tag}__{name}__probabilities.npy"
        if target.exists():
            raise ProbabilityArtifactError(
                f"{target} already exists; refusing to overwrite a frozen array."
            )
        with target.open("xb") as fh:
            np.save(fh, np.asarray(array, dtype=np.float32))
        reload = np.load(target)
        manifest["arrays"][name] = {
            "file": target.name,
            "sha256": array_sha256(reload),
            "shape": list(reload.shape),
            "dtype": str(reload.dtype),
            "min": float(reload.min()),
            "max": float(reload.max()),
        }
    with sidecar.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest


def load_probability_manifest(out_dir: Path) -> dict | None:
    sidecar = out_dir / PROBABILITY_MANIFEST_NAME
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def load_frozen_probabilities(out_dir: Path, expected_checkpoint_sha: str,
                              label_names: list, split_sizes: dict) -> dict:
    """Load and verify the first probability arrays. No inference is performed."""
    manifest = load_probability_manifest(out_dir)
    if manifest is None:
        raise ProbabilityArtifactError(f"no {PROBABILITY_MANIFEST_NAME} in {out_dir}")

    meta = manifest.get("meta", {})
    problems = []
    if meta.get("checkpoint_sha256") != expected_checkpoint_sha:
        problems.append("checkpoint sha256 mismatch "
                        f"({meta.get('checkpoint_sha256')} != {expected_checkpoint_sha})")
    if meta.get("label_names") != label_names:
        problems.append("label order/name mismatch in the frozen probability manifest")
    for split, size in split_sizes.items():
        recorded = (meta.get("splits") or {}).get(split)
        if recorded is not None and int(recorded) != int(size):
            problems.append(f"{split} split size mismatch ({recorded} != {size})")
    if problems:
        raise ProbabilityArtifactError("frozen probability manifest rejected:\n  "
                                       + "\n  ".join(problems))

    out = {}
    for name in ("validation", "test"):
        entry = manifest["arrays"].get(name)
        if entry is None:
            raise ProbabilityArtifactError(f"frozen arrays are missing the {name} split")
        path = out_dir / entry["file"]
        if not path.is_file():
            raise ProbabilityArtifactError(f"frozen array file missing: {path}")
        array = np.load(path)
        if array_sha256(array) != entry["sha256"]:
            raise ProbabilityArtifactError(f"frozen array {name} fails its recorded SHA256")
        if list(array.shape) != list(entry["shape"]):
            raise ProbabilityArtifactError(f"frozen array {name} shape changed")
        if int(array.shape[0]) != int(split_sizes[name]):
            raise ProbabilityArtifactError(f"frozen array {name} row count mismatch")
        out[name] = array
    out["manifest"] = manifest
    return out


def write_approval(out_dir: Path, note: str, approved_sha: str,
                   expected_sha: str, audit_status: str,
                   manifest_sha256: str) -> Path:
    """Record the human release for a NUMERIC_FAIL checkpoint (exclusive create).

    The record is bound to one checkpoint sha256 and one audit verdict, and OTHER_ANOMALY
    can never be approved.
    """
    check = validate_approval(note, approved_sha, expected_sha, audit_status)
    if not check["valid"]:
        raise ProbabilityArtifactError(
            "approval rejected:\n  " + "\n  ".join(check["problems"]))
    if manifest_sha256 and str(manifest_sha256).lower() != str(expected_sha).lower():
        raise ProbabilityArtifactError(
            "approval rejected: the manifest's saved_sha256 does not match the checkpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / APPROVAL_NAME
    if path.exists():
        raise ProbabilityArtifactError(
            f"{path} already exists: the release record is exclusive and is never "
            "rewritten. Inspect it instead of re-approving.")
    body = {
        "kind": "manual_ablation_release",
        "note": check["note"],
        "checkpoint_sha256": check["checkpoint_sha256"],
        "manifest_saved_sha256": str(manifest_sha256).lower() if manifest_sha256 else None,
        "audit_status_at_approval": audit_status,
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": GATE_POLICY_VERSION,
        "scope": ("authorises this checkpoint sha256 only; it does not change any numeric "
                  "verdict and cannot be reused for another artifact"),
    }
    with path.open("x", encoding="utf-8") as fh:
        json.dump(body, fh, ensure_ascii=False, indent=2)
    return path


def load_approval(out_dir: Path, expected_sha: str,
                  manifest_sha256: str | None) -> dict:
    """Read and validate the human release record, if any."""
    path = out_dir / APPROVAL_NAME
    if not path.is_file():
        return {"present": False, "valid": False, "path": str(path),
                "reason": "no manual approval record found"}
    body = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if str(body.get("checkpoint_sha256", "")).lower() != str(expected_sha).lower():
        problems.append("approval is bound to a different checkpoint sha256")
    if manifest_sha256 and str(body.get("manifest_saved_sha256", "")).lower() != \
            str(manifest_sha256).lower():
        problems.append("approval does not match the manifest's saved_sha256")
    if body.get("audit_status_at_approval") == OTHER:
        problems.append("OTHER_ANOMALY cannot be approved")
    if not str(body.get("note", "")).strip():
        problems.append("approval note is empty")
    return {
        "present": True,
        "valid": not problems,
        "path": str(path),
        "problems": problems,
        "note": body.get("note"),
        "checkpoint_sha256": body.get("checkpoint_sha256"),
        "audit_status_at_approval": body.get("audit_status_at_approval"),
        "created_at": body.get("created_at"),
    }


def decide_phase5_release(manifest: dict, approval: dict) -> dict:
    """Phase 5 release decision. Pure function, so it can be unit-tested.

    Order matters: artifact integrity first, then the manifest's ablation_release, then the
    sha-bound approval. A manifest's mere existence is never sufficient.
    """
    art = (manifest.get("artifact_integrity") or {}).get("status")
    if art != "PASS":
        return {"allowed": False, "mode": "artifact_fail",
                "reason": f"artifact_integrity={art!r}; Phase 5 must not run on it"}
    release = (manifest.get("ablation_release") or {}).get("status")
    if release == "ELIGIBLE":
        return {"allowed": True, "mode": "eligible",
                "reason": "manifest ablation_release == ELIGIBLE"}
    if release == "NOT_ELIGIBLE":
        if approval.get("valid"):
            return {"allowed": True, "mode": "manual_approval",
                    "reason": ("manifest ablation_release == NOT_ELIGIBLE, covered by a "
                               "valid sha-bound manual approval"),
                    "approval_note": approval.get("note"),
                    "approval_path": approval.get("path")}
        return {"allowed": False, "mode": "not_released",
                "reason": ("manifest ablation_release == NOT_ELIGIBLE and no valid, "
                           f"sha-bound approval ({approval.get('problems') or approval.get('reason')})")}
    return {"allowed": False, "mode": "unknown_release",
            "reason": f"unrecognised ablation_release.status={release!r}"}


def print_audit(audit: dict, record_file: Path | None, first: bool,
                extra: dict | None = None) -> None:
    print()
    print("=" * 84)
    print("NUMERIC AUDIT (official FP16 protocol, tolerance "
          f"{audit['tolerance']:.1e}, policy {audit['gate_policy_version']})")
    print("=" * 84)
    for key, row in audit["metrics"].items():
        flag = "ok" if row["within_tolerance"] else "EXCEEDS"
        print(f"  {key:22s} observed={row['observed']:.6f} recorded={row['recorded']:.6f} "
              f"|d|={row['abs_diff']:.3e} {flag}")
    print(f"  -> {audit['status']}")
    print(f"  {audit['reason']}")
    if audit.get("cause_not_established"):
        print("  NOTE: 'map only' describes WHICH metrics moved, not WHY; the cause is not")
        print("        established by this check and the magnitude is not interpreted here.")
    if extra:
        for name, block in extra.items():
            print(f"  {name:20s} -> {block['status']}"
                  + (f"  exceeding={block['exceeding_metrics']}"
                     if block["exceeding_metrics"] else ""))
    if record_file is not None:
        print(f"  audit record: {record_file}"
              f"{'  (first observation)' if first else '  (frozen from an earlier run)'}")
    if audit["status"] != PASS:
        print("  NOTE: this is NOT a pass. No file's existence authorises the checkpoint;")
        print("        only an explicit, sha-bound approval recorded in the manifest does.")
