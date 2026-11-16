"""Static, CPU-only unit test for gate policy raman-bn-gate-v5.

Covers: the neutral audit labels, the two-verdict manifest, the Phase 5 release decision,
sha-bound manual approval, frozen probability arrays and resume consistency, and the
cross-script tolerance contract. No GPU, no data, no checkpoint.

Run:  python training\\test_raman_bn_gate_policy.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.save_raman_bn_checkpoint as save  # noqa: E402
from training import raman_numeric_audit as na  # noqa: E402
from training.save_raman_bn_checkpoint import (  # noqa: E402
    GATE_POLICY,
    METRIC_KEYS,
    Checklist,
    build_manifest,
    classify_numeric,
    official_metrics_report,
)

RECORDED = {
    "micro_f1": 0.914329,
    "macro_f1": 0.871058,
    "micro_precision": 0.931940,
    "micro_recall": 0.897372,
    "mAP": 0.945208,
    "macro_AUROC": 0.988582,
}

failures = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def expect_raises(fn, exc, message):
    try:
        fn()
    except exc:
        return
    except Exception as other:                      # noqa: BLE001
        failures.append(f"{message}: raised {type(other).__name__} instead of {exc.__name__}")
        return
    failures.append(f"{message}: did not raise")


# ============================================================ policy shape ===
check(GATE_POLICY["version"] == "raman-bn-gate-v5", "policy version must be v5")
check(GATE_POLICY["numeric_tolerance"] == 1e-3, "numeric tolerance must be 1e-3")
check(GATE_POLICY["bn_summary_tolerance"] == 1e-3, "BN summary tolerance must be 1e-3")
check(GATE_POLICY["artifact_integrity"]["bn_summary_rtol_gate"] == 1e-3,
      "BN summary hard gate must be 1e-3 (IR parity)")
check(GATE_POLICY["artifact_integrity"]["bn_summary_rtol_warn"] == 1e-5,
      "BN summary warning tier stays 1e-5 and must not block")
check(GATE_POLICY["artifact_integrity"]["checks"] == 10, "artifact gate must have 10 checks")
check(GATE_POLICY["label_policy"]["all_labels_retained"] is True, "all 14 labels retained")
check(GATE_POLICY["label_policy"]["labels_excluded_from_mAP"] == [],
      "no label may be excluded from mAP")
tolerances = GATE_POLICY["numeric_reproduction"]["tolerance_by_metric"]
check(set(tolerances) == set(METRIC_KEYS), "every metric must declare a tolerance")
check(set(tolerances.values()) == {1e-3}, "no metric may be widened")
check("manifest_existence_is_not_approval" in
      GATE_POLICY["numeric_reproduction"]["manifest_policy"],
      "the policy must state that a manifest is not an approval")
check(GATE_POLICY["auxiliary_channel"]["status"] == "RECORDED_ONLY_NOT_GATED",
      "auxiliary channel must be non-gating")

# the withdrawn causal label must be gone from the audit module
check(not hasattr(na, "KNOWN_MAP"), "the old KNOWN_MAP symbol must be removed")
check(na.MAP_ONLY == "MAP_ONLY_NUMERIC_MISMATCH", "the neutral label must be used")
check("KNOWN_FP16" not in Path(na.__file__).read_text(encoding="utf-8"),
      "the causal label must not survive in the audit module")

# ============================================================ classifier ===
CASES = {
    "record reproduced": ({**RECORDED}, na.PASS),
    "T1-B cross-process reading": ({**RECORDED, "mAP": 0.946288}, na.MAP_ONLY),
    "Phase 3 failing reading": ({**RECORDED, "mAP": 0.946294}, na.MAP_ONLY),
    "mAP +4e-3 (still mAP only)": ({**RECORDED, "mAP": 0.9493}, na.MAP_ONLY),
    "mAP inside tolerance": ({**RECORDED, "mAP": 0.946108}, na.PASS),
    "threshold metric moved": ({**RECORDED, "micro_f1": 0.9163}, na.OTHER),
    "AUROC moved": ({**RECORDED, "macro_AUROC": 0.9900}, na.OTHER),
    "everything slightly off but inside": ({**RECORDED, "micro_f1": 0.9148,
                                            "mAP": 0.9459,
                                            "macro_AUROC": 0.98888}, na.PASS),
}
print(f"{'case':40s} {'expected':22s} {'got':22s} ok")
print("-" * 92)
for name, (observed, expected) in CASES.items():
    audit = na.classify(observed, RECORDED, "test")
    ok = audit["status"] == expected
    print(f"{name:40s} {expected:22s} {audit['status']:22s} {'OK' if ok else 'MISMATCH'}")
    check(ok, f"{name}: expected {expected}, got {audit['status']}")
    check(audit["status"] == na.PASS or not audit["metrics"][
        next(iter(audit["exceeding_metrics"]))]["within_tolerance"],
        f"{name}: an exceeding metric must not be marked within tolerance")

# neutrality of the map-only label
audit = na.classify({**RECORDED, "mAP": 0.9493}, RECORDED, "test")
check(audit["cause_not_established"] is True, "map-only label must not assert a cause")
check(audit["magnitude_interpreted"] is False, "map-only label must not interpret magnitude")
check("does not establish WHY" in audit["reason"], "the reason text must be causal-neutral")
check(audit["exceeding_metrics"] == ["mAP"], "map-only label names mAP")
check(na.classify({**RECORDED, "test.mAP": 0.95},
                  {"micro_f1": 1.0}, "test")["status"] == na.OTHER,
      "a missing reference must be OTHER_ANOMALY")

split = na.split_release_and_anomaly(
    na.classify({"validation.micro_f1": 0.92, "validation.mAP": 0.90,
                 "test.micro_f1": 0.90, "test.mAP": 0.99},
                {"validation.micro_f1": 0.92, "validation.mAP": 0.90,
                 "test.micro_f1": 0.90, "test.mAP": 0.95}, "t"),
    "validation.", "test.")
check(split["validation_release"]["status"] == na.PASS, "val release evaluated separately")
check(split["test_anomaly"]["status"] == na.MAP_ONLY, "test anomaly evaluated separately")

# ============================================================ manifest ===
artifact = Checklist("artifact_integrity")
artifact.add("01_test", True)
artifact.add("09_test", True)
aux = {k: 0.9 for k in METRIC_KEYS}
extra = {"verification": {"diagnostic_consistent": True}, "saved_sha256": "a" * 64,
         "gate_device": "cuda"}

m_pass = build_manifest(artifact=artifact, numeric=classify_numeric(RECORDED, RECORDED),
                        auxiliary=aux, diagnostics={}, mode="finalize", extra=dict(extra))
check(m_pass["acceptance"]["status"] == "ARTIFACT_PASS__NUMERIC_PASS", "dual status pass")
check(m_pass["acceptance"]["all_metrics_agree"] is True, "agreement true on full pass")
check(m_pass["ablation_release"]["status"] == "ELIGIBLE", "full pass releases")
check(m_pass["ablation_release"]["requires_manual_approval"] is False, "no approval needed")
check(m_pass["verification"]["diagnostic_consistent"] is True, "consistent on pass")

m_fail = build_manifest(artifact=artifact,
                        numeric=classify_numeric({**RECORDED, "mAP": 0.946294}, RECORDED),
                        auxiliary=aux, diagnostics={}, mode="finalize", extra=dict(extra))
check(m_fail["acceptance"]["status"] == "ARTIFACT_PASS__NUMERIC_FAIL", "dual status fail")
check(m_fail["acceptance"]["all_metrics_agree"] is False, "no agreement on numeric fail")
check(m_fail["verification"]["diagnostic_consistent"] is False, "inconsistent on fail")
check(m_fail["verification"]["numeric_reproduction_passed"] is False, "numeric flag false")
check(m_fail["ablation_release"]["status"] == "NOT_ELIGIBLE", "numeric fail blocks release")
check(m_fail["ablation_release"]["requires_manual_approval"] is True, "approval required")
check(m_fail["ablation_release"]["manual_approval_file"] == na.APPROVAL_NAME,
      "manifest points at the approval file")
check(m_fail["official_fp16_metrics"]["status"] == "FAIL", "official metrics report FAIL")
check(official_metrics_report(
    classify_numeric({**RECORDED, "mAP": 0.946294}, RECORDED)
)["absolute_differences"]["mAP"] > 1e-3, "deviation recorded, not rounded away")

# ============================================================ phase 5 release ===
approval_ok = {"valid": True, "note": "x", "path": "p", "problems": []}
approval_no = {"valid": False, "problems": ["no record"]}


def manifest_v5(art="PASS", release="ELIGIBLE"):
    return {"artifact_integrity": {"status": art},
            "ablation_release": {"status": release}}


d = na.decide_phase5_release(manifest_v5(), approval_no)
check(d["allowed"] is True and d["mode"] == "eligible", "release: ELIGIBLE allowed")
d = na.decide_phase5_release(manifest_v5(release="NOT_ELIGIBLE"), approval_no)
check(d["allowed"] is False and d["mode"] == "not_released",
      "release: NOT_ELIGIBLE without approval is refused")
d = na.decide_phase5_release(manifest_v5(release="NOT_ELIGIBLE"), approval_ok)
check(d["allowed"] is True and d["mode"] == "manual_approval",
      "release: NOT_ELIGIBLE with a valid approval is allowed")
d = na.decide_phase5_release(manifest_v5(art="FAIL"), approval_ok)
check(d["allowed"] is False and d["mode"] == "artifact_fail",
      "release: artifact failure is refused even with an approval")
d = na.decide_phase5_release(manifest_v5(release="SOMETHING_ELSE"), approval_ok)
check(d["allowed"] is False and d["mode"] == "unknown_release",
      "release: an unrecognised release status is refused")
d = na.decide_phase5_release({}, approval_ok)
check(d["allowed"] is False, "release: a manifest without integrity fields is refused")

# ============================================================ approval record ===
tmp = Path(tempfile.mkdtemp(prefix="raman-v5-test-"))
try:
    sha = "b" * 64
    ck = tmp / "ck"
    p1 = na.write_approval(ck, "operator: accepted after T1/T2", sha, sha, na.MAP_ONLY, sha)
    check(p1.is_file(), "approval: record written")
    loaded = na.load_approval(ck, sha, sha)
    check(loaded["valid"] is True, "approval: valid when sha and manifest match")
    check(loaded["note"] == "operator: accepted after T1/T2", "approval: note preserved")
    check(na.load_approval(ck, "c" * 64, sha)["valid"] is False,
          "approval: bound to one checkpoint sha256 only")
    check(na.load_approval(ck, sha, "d" * 64)["valid"] is False,
          "approval: must match the manifest saved_sha256")
    expect_raises(lambda: na.write_approval(ck, "again", sha, sha, na.MAP_ONLY, sha),
                  na.ProbabilityArtifactError, "approval: record is exclusive")
    expect_raises(lambda: na.write_approval(tmp / "ck2", "n", sha, sha, na.OTHER, sha),
                  na.ProbabilityArtifactError, "approval: OTHER_ANOMALY never approvable")
    expect_raises(lambda: na.write_approval(tmp / "ck3", "", sha, sha, na.MAP_ONLY, sha),
                  na.ProbabilityArtifactError, "approval: note is mandatory")
    expect_raises(lambda: na.write_approval(tmp / "ck4", "n", "e" * 64, sha,
                                            na.MAP_ONLY, sha),
                  na.ProbabilityArtifactError, "approval: sha must match the checkpoint")
    check(na.load_approval(tmp / "nowhere", sha, sha)["valid"] is False,
          "approval: absent record is invalid")

    # ======================================================== audit single-shot ===
    rec = na.classify({**RECORDED, "mAP": 0.946294}, RECORDED, "t")
    first, is_first, path = na.resolve_authoritative("RUN", "phase", rec, root=tmp)
    check(is_first and path.is_file(), "audit: first observation recorded")
    later, is_first2, _ = na.resolve_authoritative(
        "RUN", "phase", na.classify(RECORDED, RECORDED, "t"), root=tmp)
    check(not is_first2, "audit: later run is not a first observation")
    check(later["status"] == na.MAP_ONLY,
          "audit: a later PASS cannot replace a recorded mismatch")
    check(later["subsequent_reading"]["status"] == na.PASS,
          "audit: later reading kept for information only")

    # ======================================================== dir state ===
    cal = tmp / "cal"
    check(na.calibration_dir_state(cal, "t.json", "s.json") == "fresh", "dir: fresh")
    cal.mkdir()
    check(na.calibration_dir_state(cal, "t.json", "s.json") == "unknown", "dir: unknown")
    (cal / "t.json").write_text("{}", encoding="utf-8")
    check(na.calibration_dir_state(cal, "t.json", "s.json") == "interrupted",
          "dir: interrupted")
    (cal / "s.json").write_text("{}", encoding="utf-8")
    check(na.calibration_dir_state(cal, "t.json", "s.json") == "complete", "dir: complete")

    # ======================================================== frozen probabilities ===
    pdir = tmp / "prob"
    labels = ["a", "b"]
    meta = {"checkpoint": "ck.pt", "checkpoint_sha256": "f" * 64, "bn_state": "recalibrated",
            "splits": {"validation": 3, "test": 2}, "dtype": "float32",
            "label_names": labels, "metric_basis": "official fp16"}
    val = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(3, 2)
    tst = np.linspace(1.0, 0.0, 4, dtype=np.float32).reshape(2, 2)
    manifest = na.persist_probabilities(pdir, "RUN", {"validation": val, "test": tst}, meta)
    check((pdir / na.PROBABILITY_MANIFEST_NAME).is_file(), "probs: sidecar written")
    check(manifest["arrays"]["validation"]["sha256"] == na.array_sha256(val),
          "probs: sha256 matches the array")
    frozen = na.load_frozen_probabilities(pdir, meta["checkpoint_sha256"], labels,
                                          {"validation": 3, "test": 2})
    check(np.array_equal(frozen["validation"], val), "probs: validation round-trips exactly")
    check(np.array_equal(frozen["test"], tst), "probs: test round-trips exactly")
    expect_raises(lambda: na.persist_probabilities(
        pdir, "RUN", {"validation": val, "test": tst}, meta),
        na.ProbabilityArtifactError,
        "probs: a second run may never replace the first observation")
    expect_raises(lambda: na.load_frozen_probabilities(pdir, "0" * 64, labels,
                                                       {"validation": 3, "test": 2}),
                  na.ProbabilityArtifactError, "probs: checkpoint binding enforced")
    expect_raises(lambda: na.load_frozen_probabilities(pdir, "f" * 64, ["a", "c"],
                                                       {"validation": 3, "test": 2}),
                  na.ProbabilityArtifactError, "probs: label order enforced")
    expect_raises(lambda: na.load_frozen_probabilities(pdir, "f" * 64, labels,
                                                       {"validation": 4, "test": 2}),
                  na.ProbabilityArtifactError, "probs: split size enforced")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ============================================================ downstream parity ===
import training.calibrate_raman_recalibrated_thresholds as cal  # noqa: E402
import training.calibrate_raman_original_thresholds as orig_cal  # noqa: E402
import training.summarize_raman_ablation as summ  # noqa: E402
import training.diagnose_raman_batchnorm_full as diag  # noqa: E402

check(cal.BASELINE_ATOL == 1e-3 and orig_cal.BASELINE_ATOL == 1e-3,
      "both calibrators must use 1e-3")
check(cal.GATE_POLICY_VERSION == GATE_POLICY["version"], "Phase 5 shares the policy version")
check(not hasattr(cal, "BASELINE_MAP_ATOL"), "no mAP exception in Phase 5")
check(not hasattr(orig_cal, "BASELINE_MAP_ATOL"), "no mAP exception in Phase 4")
check(summ.TOLERANCE == 0.0025, "summarize keeps 0.0025 (IR parity)")
check(not hasattr(summ, "TOLERANCE_BY_METRIC"), "no mAP exception in the summarizer")
check(diag.THRESHOLD == 0.5 and diag.BATCH_SIZE == 512 and diag.NUM_WORKERS == 4,
      "official protocol constants untouched")


def diag_after(mAP_val, mAP_test):
    f1 = {k: RECORDED[k] for k in
          ("micro_f1", "macro_f1", "micro_precision", "micro_recall")}
    return {"after": {"validation": {**f1, "mAP": mAP_val,
                                     "macro_AUROC": RECORDED["macro_AUROC"]},
                      "test": {**f1, "mAP": mAP_test,
                               "macro_AUROC": RECORDED["macro_AUROC"]}}}


val_ok = {k: RECORDED[k] for k in
          ("micro_f1", "macro_f1", "micro_precision", "micro_recall")}
default_ok = {**val_ok, "mAP": 0.950000, "macro_AUROC": RECORDED["macro_AUROC"]}

a = cal.verify_against_diagnostic(diag_after(0.952, 0.950), val_ok, default_ok,
                                  {"mAP": 0.950, "macro_AUROC": RECORDED["macro_AUROC"]},
                                  "EXP-RAMAN-001-FULL")
check(a["status"] == na.PASS, "Phase 5: identical -> PASS")
s = na.split_release_and_anomaly(a, "validation.", "test.")
check(s["validation_release"]["status"] == na.PASS, "Phase 5: validation release PASS")
check(s["test_anomaly"]["status"] == na.PASS, "Phase 5: test record PASS")

a = cal.verify_against_diagnostic(diag_after(0.952, 0.950), val_ok, default_ok,
                                  {"mAP": 0.951080,
                                   "macro_AUROC": RECORDED["macro_AUROC"]},
                                  "EXP-RAMAN-001-FULL")
s = na.split_release_and_anomaly(a, "validation.", "test.")
check(a["status"] == na.MAP_ONLY, "Phase 5: test mAP drift -> MAP_ONLY")
check(s["validation_release"]["status"] == na.PASS,
      "Phase 5: a test-side drift must NOT affect the validation release")
check(s["test_anomaly"]["status"] == na.MAP_ONLY, "Phase 5: drift lands in the test record")

a = cal.verify_against_diagnostic(diag_after(0.952, 0.950),
                                  {**val_ok, "micro_f1": 0.9163}, default_ok,
                                  {"mAP": 0.950,
                                   "macro_AUROC": RECORDED["macro_AUROC"]},
                                  "EXP-RAMAN-001-FULL")
s = na.split_release_and_anomaly(a, "validation.", "test.")
check(s["validation_release"]["status"] == na.OTHER,
      "Phase 5: a validation-side move fails the release condition")
check(a["status"] != na.PASS, "Phase 5: never reports PASS when a validation metric moved")

if failures:
    print(f"\nFAIL: {len(failures)} assertion(s) failed")
    for f in failures:
        print("  -", f)
    sys.exit(1)

print("\nPASS: gate policy v5 behaves as specified "
      f"({len(CASES)} classifier cases, neutral map-only label, two-verdict manifest, "
      "sha-bound release, frozen probability arrays, validation/test separation)")
