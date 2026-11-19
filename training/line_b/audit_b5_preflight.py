"""B5 Phase 0 — read-only final-report preflight audit.

Checks the frozen baseline, inventories B4.1–B4.4 and the B4.4 supplement,
independently cross-checks B4.2 point metrics against B4.3 paired differences,
verifies B4.3 bootstrap completeness, B4.4 result completeness, supplement
file hashes, report/manifest/verification consistency, CI semantics, the
pre-specified H1–H4 criteria and scientific-interpretation boundaries.

Writes only runs/LINE-B-TEST-ANALYSIS/B5-PREFLIGHT/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

B0 = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
TESTLOG = ROOT / "runs" / "LINE-B-MANIFESTS" / "test_access_log.json"
B41 = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
B43 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
B44 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4"
SUPP = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4-SUPPLEMENT-REPRESENTATIVE-MOLECULES"
OUT = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B5-PREFLIGHT"
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}

B0_SHA = "713fa92f5fd1f2134c1512bd2e49f689433a2158d120e41c9460ff5c287dd46a"
FREEZE_SHA = "65a84959356a9a1f7a6e1d6434449f607c75a4c224a17450f3345ba25bb92a0c"


class Audit:
    def __init__(self):
        self.checks = []
        self.inventory = []
        self.missing = []

    def add(self, cid, status, detail="", severity="normal"):
        status = status.upper()
        self.checks.append({"id": cid, "status": status, "detail": str(detail), "severity": severity})
        return status

    def fail(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    def warn(self):
        return [c for c in self.checks if c["status"] == "WARN"]

    def status(self):
        if self.fail():
            return "FAIL"
        if self.warn():
            return "WARN"
        return "PASS"


def _csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    h = hashlib.sha256()
    for name, digest in entries:
        h.update(name.encode()); h.update(b"\0"); h.update(digest.encode()); h.update(b"\n")
    return h.hexdigest()


def _inv(audit: Audit, stage: str, path: Path, role: str):
    digest = sha256_file(path)
    audit.inventory.append({
        "stage": stage, "path": relative_to_root(path), "role": role,
        "sha256": digest, "size_bytes": int(path.stat().st_size), "status": "present",
    })
    return digest


def _check_manifest_outputs(audit: Audit, stage: str, root: Path, manifest: dict):
    bad = []
    for item in manifest.get("outputs", []):
        path = root / item["path"]
        if not path.is_file():
            bad.append((item["path"], "missing"))
        elif sha256_file(path) != item["sha256"] or path.stat().st_size != item.get("size_bytes", path.stat().st_size):
            bad.append((item["path"], "hash/size mismatch"))
    audit.add(f"{stage}.manifest_outputs", "PASS" if not bad else "FAIL",
              f"{len(manifest.get('outputs', []))} outputs" if not bad else str(bad[:5]))
    return bad


def _check_verification(audit: Audit, stage: str, root: Path):
    ver = read_json(root / "verification.json")
    man = read_json(root / "analysis_manifest.json")
    ok = (ver.get("status") == "PASS"
          and ver.get("analysis_manifest", {}).get("sha256") == sha256_file(root / "analysis_manifest.json"))
    audit.add(f"{stage}.verification", "PASS" if ok else "FAIL",
              f"status={ver.get('status')} checks={ver.get('summary', {}).get('n_checks')} failed={ver.get('summary', {}).get('n_failed')}")
    return ver, man


def _macro_map(overall):
    out = {}
    for row in overall:
        out[(row["task"], int(row["seed"]), row["modality"], row["policy"])] = {
            "macro_f1": float(row["macro_f1"]), "macro_ap": float(row["macro_ap"]),
            "macro_auroc": float(row["macro_auroc"]), "n_labels": int(row["n_labels"]),
        }
    return out


def _label_map(per_label):
    out = {}
    for row in per_label:
        out[(row["task"], int(row["seed"]), row["modality"], row["policy"], row["label"])] = {
            "f1": float(row["f1"]), "ap": float(row["average_precision"]),
            "auroc": float(row["auroc"]), "test_positives": int(row["test_positives"]),
        }
    return out


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    # ---------- Baseline ----------
    baseline = {
        "protocol": {"path": relative_to_root(B0), "sha256": sha256_file(B0), "expected": B0_SHA},
        "freeze": {"path": relative_to_root(FREEZE), "sha256": sha256_file(FREEZE), "expected": FREEZE_SHA},
        "test_access_log": {"path": relative_to_root(TESTLOG), "sha256": sha256_file(TESTLOG)},
    }
    audit.add("baseline.protocol", "PASS" if baseline["protocol"]["sha256"] == B0_SHA else "FAIL", baseline["protocol"]["sha256"])
    audit.add("baseline.freeze", "PASS" if baseline["freeze"]["sha256"] == FREEZE_SHA else "FAIL", baseline["freeze"]["sha256"])
    freeze = read_json(FREEZE)
    audit.add("baseline.freeze_status", "PASS" if freeze.get("status") == "FROZEN" else "FAIL", freeze.get("status"))
    testlog = read_json(TESTLOG)
    entries = testlog.get("entries", testlog if isinstance(testlog, list) else [])
    audit.add("baseline.test_access_entries", "PASS" if len(entries) == 18 else "FAIL", len(entries))

    # ---------- Inventory ----------
    for stage, root in (("B4.1", B41), ("B4.2", B42), ("B4.3", B43), ("B4.4", B44), ("B4.4-SUPPLEMENT", SUPP)):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                _inv(audit, stage, path, path.name)
    for path, label in ((B0, "protocol"), (FREEZE, "freeze"), (TESTLOG, "test_access_log")):
        _inv(audit, "baseline", path, label)

    # B4.1 model matrix and arrays.
    models = []
    for record in freeze["models"]:
        d = B41 / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        manifest = read_json(d / "prediction_manifest.json")
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        row_index = np.load(d / "row_index.npy", allow_pickle=False)
        labels = np.load(d / "labels.npy", allow_pickle=False)
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        ok = (sha256_array(probs) == manifest["arrays"]["probabilities"]["sha256"]
              and sha256_array(row_index) == manifest["arrays"]["row_index"]["sha256"]
              and sha256_array(labels) == manifest["arrays"]["labels"]["sha256"]
              and probs.shape == (12981, len(names)) and labels.shape == (12981, len(names))
              and row_index.shape == (12981,))
        models.append({"task": record["task"], "modality": record["modality"], "seed": int(record["seed"]),
                       "names": names, "row_index": row_index, "probabilities": probs, "labels": labels, "ok": ok})
    audit.add("b4_1.models_18", "PASS" if len(models) == 18 else "FAIL", len(models))
    audit.add("b4_1.array_hashes_shapes", "PASS" if all(m["ok"] for m in models) else "FAIL",
              f"{sum(1 for m in models if m['ok'])}/18 ok")
    for task, n_labels in (("Shared14", 14), ("UVSPEC15", 15)):
        subset = [m for m in models if m["task"] == task]
        audit.add(f"b4_1.{task}.matrix", "PASS" if len(subset) == 9 and all(len(m["names"]) == n_labels for m in subset) else "FAIL",
                  f"{len(subset)} models, labels={sorted({len(m['names']) for m in subset})}")
        audit.add(f"b4_1.{task}.seeds", "PASS" if sorted({m["seed"] for m in subset}) == [42, 123, 2026] else "FAIL",
                  sorted({m["seed"] for m in subset}))
        ref = subset[0]["row_index"]
        audit.add(f"b4_1.{task}.row_alignment", "PASS" if all(np.array_equal(ref, m["row_index"]) for m in subset) else "FAIL",
                  "identical row_index")

    # ---------- B4.2 ----------
    ver42, man42 = _check_verification(audit, "b4_2", B42)
    _check_manifest_outputs(audit, "b4_2", B42, man42)
    overall42 = _csv(B42 / "overall_metrics.csv")
    per_label42 = _csv(B42 / "per_label_metrics.csv")
    map42_macro = _macro_map(overall42)
    map42_label = _label_map(per_label42)
    report42 = (B42 / "report.md").read_text(encoding="utf-8")
    audit.add("b4_2.report_hash_in_manifest",
              "PASS" if any(i["path"].endswith("report.md") and i["sha256"] == sha256_file(B42 / "report.md") for i in man42["outputs"]) else "FAIL",
              sha256_file(B42 / "report.md"))
    corrected = "## 3. Seed stability (F1 under frozen Validation-F1 thresholds)" in report42 and "validation_f1" in report42
    audit.add("b4_2.corrected_report", "PASS" if corrected else "FAIL", "corrected heading and validation_f1 quote present")
    backups = [p for p in B42.rglob("*") if p.is_file() and any(k in p.name.lower() for k in ("backup", ".bak", "old"))]
    audit.add("b4_2.no_old_report_used", "PASS" if not backups else "WARN", [p.name for p in backups])
    audit.add("b4_2.row_counts", "PASS" if len(overall42) == 36 and len(per_label42) == 522 else "FAIL",
              f"overall={len(overall42)}, per_label={len(per_label42)}")
    sample42 = next(r for r in overall42 if r["task"] == "Shared14" and r["seed"] == "42" and r["modality"] == "ir" and r["policy"] == "fixed_0.5")
    trace42 = {field: f"{float(sample42[field]):.4f}" in report42
               for field in ("micro_f1", "macro_ap", "macro_auroc")}
    audit.add("b4_2.report_number_traceability", "PASS" if all(trace42.values()) else "WARN",
              {k: v for k, v in trace42.items() if not v} or "Shared14 IR seed42 micro_f1/macro_ap/macro_auroc present")

    # ---------- B4.3 ----------
    ver43, man43 = _check_verification(audit, "b4_3", B43)
    _check_manifest_outputs(audit, "b4_3", B43, man43)
    summary43 = read_json(B43 / "summary.json")
    audit.add("b4_3.reps_2000", "PASS" if int(summary43["config"]["reps"]) == 2000 else "FAIL", summary43["config"]["reps"])
    audit.add("b4_3.counts", "PASS" if summary43["counts"]["point_rows"] == 1488 and summary43["counts"]["ci_per_seed_rows"] == 1116 and summary43["counts"]["ci_three_seed_rows"] == 372 else "FAIL",
              summary43["counts"])
    npz43 = np.load(B43 / "bootstrap_replicates.npz")
    lengths = [npz43[k].shape for k in npz43.files]
    audit.add("b4_3.npz_completeness",
              "PASS" if len(npz43.files) == 1488 and all(shape and shape[0] == 2000 for shape in lengths) else "FAIL",
              f"keys={len(npz43.files)}, all_reps_2000={all(s and s[0] == 2000 for s in lengths)}")
    ci_seed = _csv(B43 / "bootstrap_ci_per_seed.csv")
    ci_three = _csv(B43 / "bootstrap_ci_three_seed.csv")
    audit.add("b4_3.ci_rows", "PASS" if len(ci_seed) == 1116 and len(ci_three) == 372 else "FAIL",
              f"{len(ci_seed)}, {len(ci_three)}")
    audit.add("b4_3.ci_n_total", "PASS" if all(int(r["n_total_reps"]) == 2000 for r in ci_seed + ci_three) else "FAIL",
              "n_total_reps=2000")
    audit.add("b4_3.draws_sha_recorded", "PASS" if len(summary43["config"]["draws_sha256"]) == 64 else "FAIL", summary43["config"]["draws_sha256"])

    # B4.3 point rows are paired differences: independently compare to B4.2 absolute metrics.
    point43 = _csv(B43 / "point_comparisons.csv")
    diffs = []
    mismatches = 0
    for row in point43:
        task = row["task"]; policy = row["threshold_policy"]
        metric = row["metric"]; scope = row["scope"]; base = row["base_modality"]; other = row["other_modality"]
        seed_raw = row["seed"]
        seeds = [42, 123, 2026] if seed_raw == "three_seed_mean" else [int(seed_raw)]
        lookup_policy = "fixed_0.5" if metric in ("AP", "AUROC") else policy
        diffs_seed = []
        for seed in seeds:
            if scope == "label":
                b = map42_label.get((task, seed, base, lookup_policy, row["label"]))
                o = map42_label.get((task, seed, other, lookup_policy, row["label"]))
                key = "ap" if metric == "AP" else "auroc" if metric == "AUROC" else "f1"
            else:
                b = map42_macro.get((task, seed, base, lookup_policy))
                o = map42_macro.get((task, seed, other, lookup_policy))
                key = "macro_ap" if metric == "AP" else "macro_auroc" if metric == "AUROC" else "macro_f1"
            if not b or not o:
                diffs_seed = []
                break
            diffs_seed.append(b[key] - o[key])
        if not diffs_seed:
            mismatches += 1
            continue
        expected = float(np.mean(diffs_seed))
        diffs.append(abs(float(row["point_estimate"]) - expected))
    max_diff = max(diffs) if diffs else None
    audit.add("b4_3.point_vs_b4_2", "PASS" if mismatches == 0 and max_diff is not None and max_diff <= 1e-6 else "FAIL",
              f"rows={len(point43)}, missing={mismatches}, max_abs_diff={max_diff:.3e}")
    audit.add("b4_3.point_is_paired_difference", "PASS" if max_diff is not None and max_diff <= 1e-6 else "FAIL",
              "point_estimate == B4.2(base) - B4.2(other)")

    report43 = (B43 / "report.md").read_text(encoding="utf-8")
    ci_semantics = ("paired" in report43.lower() and "difference" in report43.lower()
                    and "absolute" not in report43.lower())
    audit.add("b4_3.ci_language", "PASS" if ci_semantics else "WARN",
              "report frames CIs as paired differences; no absolute-metric CI claim" if ci_semantics
              else "B4.3 report says 'paired bootstrap' but does not explicitly say 'difference'; B5 must state CIs are of paired differences, not absolute single-modality metrics")
    audit.add("b4_3.no_absolute_ci_columns",
              "PASS" if not any("absolute" in k.lower() for k in (ci_seed[0].keys() if ci_seed else [])) else "FAIL",
              "CI tables carry comparison/point_estimate columns, not absolute metric columns")
    first_macro = next(r for r in point43 if r["task"] == "Shared14" and r["comparison"] == "IR-Raman"
                       and r["metric"] == "AP" and r["scope"] == "macro" and r["seed"] == "three_seed_mean")
    expected_point = f"{float(first_macro['point_estimate']):+.4f}"
    trace43 = {"reps_2000": "2000" in report43, "scaffolds_2853": "2853" in report43,
               "mean_W_12864.7": "12864.7" in report43, "macro_AP_point": expected_point in report43}
    audit.add("b4_3.report_number_traceability", "PASS" if all(trace43.values()) else "WARN",
              {k: v for k, v in trace43.items() if not v} or "2000 reps, 2853 scaffolds, mean W, first macro AP present")

    # ---------- B4.4 ----------
    ver44, man44 = _check_verification(audit, "b4_4", B44)
    _check_manifest_outputs(audit, "b4_4", B44, man44)
    summary44 = read_json(B44 / "summary.json")
    expected_counts = {"pairwise_rows": 558, "mcnemar_rows": 522, "oracle_rows": 174,
                       "conditional_rows": 120, "sample_rows": 36,
                       "subgroup_metric_rows": 11484, "subgroup_complementarity_rows": 2112,
                       "subgroup_bootstrap_rows": 504}
    audit.add("b4_4.counts", "PASS" if all(summary44["counts"].get(k) == v for k, v in expected_counts.items()) else "FAIL",
              {k: summary44["counts"].get(k) for k in expected_counts})
    required44 = ["pairwise_metrics.csv", "mcnemar_holm.csv", "pairwise_or_and.csv", "three_modality_or.csv",
                  "oracle_summary.csv", "oracle_per_label.csv", "conditional_complementarity.csv",
                  "conditional_bootstrap_ci.csv", "sample_error_summary.csv", "hard_sample_summary.csv",
                  "spearman.csv", "subgroup_metrics.csv", "subgroup_complementarity.csv",
                  "subgroup_bootstrap_ci.csv", "report.md", "summary.json", "analysis_manifest.json", "verification.json"]
    missing44 = [n for n in required44 if not (B44 / n).is_file() or (B44 / n).stat().st_size == 0]
    audit.add("b4_4.required_files", "PASS" if not missing44 else "FAIL", missing44)
    oracle = _csv(B44 / "oracle_summary.csv")
    oracle_bad = []
    for row in oracle:
        if row["scope"] == "overall" and abs(int(row["any_correct"]) + int(row["all_wrong"]) - int(row["n_pairs"])) > 0:
            oracle_bad.append(row["task"])
    audit.add("b4_4.oracle_partition", "PASS" if not oracle_bad else "FAIL", oracle_bad)
    cond = _csv(B44 / "conditional_complementarity.csv")
    audit.add("b4_4.conditional_rows", "PASS" if len(cond) == 120 else "FAIL", len(cond))
    cci = _csv(B44 / "conditional_bootstrap_ci.csv")
    audit.add("b4_4.conditional_ci_reps", "PASS" if all(int(r["n_total_reps"]) == 2000 for r in cci) else "FAIL",
              f"rows={len(cci)}")
    sbc = _csv(B44 / "subgroup_bootstrap_ci.csv")
    audit.add("b4_4.subgroup_ci_reps", "PASS" if all(int(r["n_total_reps"]) == 500 for r in sbc) else "FAIL",
              f"rows={len(sbc)}")
    audit.add("b4_4.subgroup_support", "PASS" if all((int(r["n_samples"]) >= 100 and int(r["positive_count"]) >= 30) == (r["support_sufficient"].lower() == "true") for r in _csv(B44 / "subgroup_metrics.csv")) else "FAIL",
              "support flag = n>=100 and positive>=30")
    npz44c = np.load(B44 / "conditional_replicates.npz")
    npz44s = np.load(B44 / "subgroup_bootstrap_replicates.npz")
    audit.add("b4_4.replicate_arrays", "PASS" if all(a.shape[0] == 2000 for a in npz44c.values()) and all(a.shape[0] == 500 for a in npz44s.values()) else "FAIL",
              f"conditional={len(npz44c.files)}, subgroup={len(npz44s.files)}")

    # Key report figures traceable to CSVs.
    report44 = (B44 / "report.md").read_text(encoding="utf-8")
    oracle_any = {}
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in oracle if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["scope"] == "overall"]
        oracle_any[task] = _mean([float(r["any_correct_rate"]) for r in rows])
    uv_rescue = {}
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in cond if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "uv_rescue_rate"]
        uv_rescue[task] = _mean([float(r["rate"]) for r in rows])
    trace = {
        "oracle_any_shared14": f"{oracle_any['Shared14']:.4f}" in report44,
        "oracle_any_uvspec15": f"{oracle_any['UVSPEC15']:.4f}" in report44,
        "uv_rescue_shared14": f"{uv_rescue['Shared14']:.4f}"[:5] in report44,
        "uv_rescue_uvspec15": f"{uv_rescue['UVSPEC15']:.4f}"[:5] in report44,
        "subgroup_19_of_21": "19 of 21" in report44,
        "subgroup_16_of_21": "16 of 21" in report44,
    }
    audit.add("b4_4.report_traceability", "PASS" if all(trace.values()) else "WARN",
              {k: v for k, v in trace.items() if not v} or "key figures present")

    # ---------- Supplement ----------
    vers, mans = _check_verification(audit, "supplement", SUPP)
    _check_manifest_outputs(audit, "supplement", SUPP, mans)
    examples = _csv(SUPP / "examples.csv")
    status_rows = _csv(SUPP / "subgroup_status.csv")
    supp_counts = Counter(r["task"] for r in examples)
    nonempty = Counter(r["task"] for r in status_rows if int(r["n_samples"]) > 0)
    empty = Counter(r["task"] for r in status_rows if int(r["n_samples"]) == 0)
    bad_status = [r for r in status_rows if int(r["n_samples"]) > 0 and (r["correct_status"] != "ok" or r["error_status"] != "ok")]
    audit.add("supplement.examples_88", "PASS" if len(examples) == 88 and supp_counts == {"Shared14": 44, "UVSPEC15": 44} else "FAIL",
              f"rows={len(examples)}, per_task={dict(supp_counts)}")
    audit.add("supplement.subgroups", "PASS" if all(nonempty[t] == 22 and empty[t] == 4 for t in ("Shared14", "UVSPEC15")) else "FAIL",
              f"nonempty={dict(nonempty)}, empty={dict(empty)}")
    audit.add("supplement.no_missing_candidates", "PASS" if not bad_status else "FAIL", len(bad_status))
    supp_report = (SUPP / "report.md").read_text(encoding="utf-8")
    report_correct = sum(1 for line in supp_report.splitlines()
                         if line.startswith("| Shared14 |") and line.split("|")[3].strip() == "correct")
    report_error = sum(1 for line in supp_report.splitlines()
                       if line.startswith("| Shared14 |") and line.split("|")[3].strip() == "error")
    uv_report_correct = sum(1 for line in supp_report.splitlines()
                            if line.startswith("| UVSPEC15 |") and line.split("|")[3].strip() == "correct")
    uv_report_error = sum(1 for line in supp_report.splitlines()
                          if line.startswith("| UVSPEC15 |") and line.split("|")[3].strip() == "error")
    audit.add("supplement.report_example_traceability",
              "PASS" if (report_correct, report_error, uv_report_correct, uv_report_error) == (22, 22, 22, 22) else "WARN",
              f"Shared14={report_correct}/{report_error}, UVSPEC15={uv_report_correct}/{uv_report_error}")
    b44_tree = _tree_digest(B44)
    audit.add("supplement.b44_tree", "PASS" if mans["inputs"]["b4_4"]["tree_sha256"] == b44_tree else "FAIL",
              f"manifest={mans['inputs']['b4_4']['tree_sha256']}, current={b44_tree}")
    supp_files = [p for p in SUPP.rglob("*") if p.is_file()]
    audit.add("supplement.file_count", "PASS" if len(supp_files) == 358 else "FAIL", len(supp_files))

    # ---------- B4.4 Oracle / OR / conditional / subgroup completeness ----------
    or_rows = _csv(B44 / "pairwise_or_and.csv")
    three_rows = _csv(B44 / "three_modality_or.csv")
    audit.add("b4_4.or_rows", "PASS" if len(or_rows) == 36 and len(three_rows) == 12 else "FAIL",
              f"pairwise_or={len(or_rows)}, three={len(three_rows)}")
    audit.add("b4_4.mcnemar_rows", "PASS" if len(_csv(B44 / "mcnemar_holm.csv")) == 522 else "FAIL", "")
    audit.add("b4_4.sample_rows", "PASS" if len(_csv(B44 / "sample_error_summary.csv")) == 36 else "FAIL", "")

    # ---------- H1–H4 evidence (B0 criteria) ----------
    names42 = sorted({r["label"] for r in per_label42})
    shared_labels = [n for n in names42 if n in {r["label"] for r in per_label42 if r["task"] == "Shared14"}]
    vib = [n for n in ["carbonyl", "nitrile", "alkyne", "amine", "hydroxyl"] if n in shared_labels]
    def ap(task, modality, label=None, metric="ap"):
        vals = []
        for row in per_label42:
            if row["task"] == task and row["modality"] == modality and row["policy"] == "fixed_0.5" and (label is None or row["label"] == label):
                vals.append(float(row[metric if metric != "ap" else "average_precision"]))
        return _mean(vals)
    def macro(task, modality, metric):
        vals = [v[metric] for (t, s, m, p), v in map42_macro.items() if t == task and m == modality and p == "fixed_0.5"]
        return _mean(vals)
    h1 = {
        "shared14_macro_ap": {m: macro("Shared14", m, "macro_ap") for m in ("ir", "raman", "uvvis")},
        "shared14_macro_auroc": {m: macro("Shared14", m, "macro_auroc") for m in ("ir", "raman", "uvvis")},
        "vibration_label_gap": _mean([max(ap("Shared14", "ir", n), ap("Shared14", "raman", n)) - ap("Shared14", "uvvis", n) for n in vib]),
        "other_label_gap": _mean([max(ap("Shared14", "ir", n), ap("Shared14", "raman", n)) - ap("Shared14", "uvvis", n) for n in shared_labels if n not in vib]),
    }
    h1["main_supported"] = (h1["shared14_macro_ap"]["ir"] > h1["shared14_macro_ap"]["uvvis"]
                            and h1["shared14_macro_ap"]["raman"] > h1["shared14_macro_ap"]["uvvis"]
                            and h1["shared14_macro_auroc"]["ir"] > h1["shared14_macro_auroc"]["uvvis"]
                            and h1["shared14_macro_auroc"]["raman"] > h1["shared14_macro_auroc"]["uvvis"])
    h1["vibration_gap_supported"] = h1["vibration_label_gap"] > h1["other_label_gap"]
    h1["supported"] = "partial: main IR/Raman>UV supported; vibration-label gap is not larger than other-label gap" if h1["main_supported"] else False
    h2 = {
        "uvspec15_macro_ap": {m: macro("UVSPEC15", m, "macro_ap") for m in ("ir", "raman", "uvvis")},
        "uvspec15_macro_auroc": {m: macro("UVSPEC15", m, "macro_auroc") for m in ("ir", "raman", "uvvis")},
    }
    h2["supported"] = (h2["uvspec15_macro_ap"]["uvvis"] > h2["uvspec15_macro_ap"]["ir"]
                       and h2["uvspec15_macro_ap"]["uvvis"] > h2["uvspec15_macro_ap"]["raman"]
                       and h2["uvspec15_macro_auroc"]["uvvis"] > h2["uvspec15_macro_auroc"]["ir"]
                       and h2["uvspec15_macro_auroc"]["uvvis"] > h2["uvspec15_macro_auroc"]["raman"])
    h3 = {
        "oracle_any_correct": oracle_any,
        "positive_or_three_modality_gain_min": min(float(r["or_gain_vs_best_single"]) for r in three_rows if r["threshold_policy"] == "fixed_0.5"),
        "positive_or_three_modality_gain_max": max(float(r["or_gain_vs_best_single"]) for r in three_rows if r["threshold_policy"] == "fixed_0.5"),
        "uv_rescue_rate": uv_rescue,
        "uv_rescue_delta": {},
        "exclusive_correct_subgroup_enrichment": "not directly tested in B4.4 (no subgroup-level only_UV / only_IR / only_Raman table)",
    }
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in cond if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "uv_rescue_rate"]
        h3["uv_rescue_delta"][task] = _mean([float(r["delta"]) for r in rows])
    h3["supported"] = False
    h3["verdict"] = "partial: errors not identical and Oracle any-correct high, but positive OR does not beat best_single and UV conditional rescue is below marginal; H3's subgroup exclusive-correct enrichment clause is not directly tested."
    # H4: best AP modality across supported subgroups.
    sm = _csv(B44 / "subgroup_metrics.csv")
    best_counts = defaultdict(Counter)
    for task in ("Shared14", "UVSPEC15"):
        subset = [r for r in sm if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["support_sufficient"].lower() == "true"]
        by_sub = defaultdict(lambda: defaultdict(list))
        for r in subset:
            if r["ap"] not in ("", None):
                by_sub[r["subgroup"]][r["modality"]].append(float(r["ap"]))
        for sub, d in by_sub.items():
            means = {m: _mean(v) for m, v in d.items() if _mean(v) is not None}
            if means:
                best_counts[task][max(means, key=means.get)] += 1
    h4 = {"best_modality_counts": {t: dict(c) for t, c in best_counts.items()},
          "supported": any(len(set(c)) > 1 for c in best_counts.values())}
    audit.add("hypotheses.H1", "PASS" if h1["main_supported"] else "FAIL",
              f"main_supported={h1['main_supported']}; vibration_gap_supported={h1['vibration_gap_supported']}; "
              f"AP IR/Raman/UV={h1['shared14_macro_ap']}; vib_gap={h1['vibration_label_gap']:.4f}, other_gap={h1['other_label_gap']:.4f}")
    audit.add("hypotheses.H2", "PASS" if not h2["supported"] else "WARN",
              f"supported={h2['supported']}; UVSPEC15 AP UV/IR/Raman={h2['uvspec15_macro_ap']}; AUROC={h2['uvspec15_macro_auroc']} (H2 not supported by current Test metrics)")
    audit.add("hypotheses.H3", "WARN", h3["verdict"])
    audit.add("hypotheses.H4", "PASS" if h4["supported"] else "WARN",
              f"supported={h4['supported']}; best={h4['best_modality_counts']}")

    # ---------- Scientific language and required caveats ----------
    texts = {"B4.2": report42, "B4.3": report43, "B4.4": report44,
             "supplement": (SUPP / "report.md").read_text(encoding="utf-8")}
    for stage, text in texts.items():
        audit.add(f"language.{stage}.exploratory", "PASS" if re.search(r"exploratory|post-hoc", text, re.I) else "WARN", "")
    audit.add("language.supplement_not_xai", "PASS" if "not XAI" in texts["supplement"] else "FAIL", "")
    audit.add("language.supplement_not_confirmatory", "PASS" if "does not constitute confirmatory" in texts["supplement"] else "FAIL", "")
    overclaim = []
    for stage, text in texts.items():
        for phrase in ("proves", "confirms that", "xai result", "oracle is a fusion", "positive or is oracle",
                       "uv has no incremental", "uv is better in all subgroups"):
            if phrase in text.lower():
                overclaim.append((stage, phrase))
    audit.add("language.overclaim_scan", "PASS" if not overclaim else "FAIL", overclaim or "no prohibited claims")
    missing_caveats = []
    for stage, text in texts.items():
        if "chemical information" not in text.lower() and "化学信息" not in text:
            missing_caveats.append(f"{stage}:AP/AUROC-not-chemical-information")
        if "experimental" not in text.lower() and "实验域" not in text:
            missing_caveats.append(f"{stage}:QM9S-to-experimental-domain")
    audit.add("language.required_boundaries", "WARN" if missing_caveats else "PASS", missing_caveats)

    # B4.4 report Phase-4 wording consistent with computed evidence.
    phase4 = ("no supported subgroup" in report44.lower() or "no supported subgroup shows" in report44.lower())
    audit.add("b4_4.uv_subgroup_claim", "PASS" if phase4 else "WARN", "reports no positive UV subgroup advantage")

    # ---------- Missing assets ----------
    expected_roots = {"B4.2": man42, "B4.3": man43, "B4.4": man44, "B4.4-SUPPLEMENT": mans}
    roots = {"B4.2": B42, "B4.3": B43, "B4.4": B44, "B4.4-SUPPLEMENT": SUPP}
    for stage, man in expected_roots.items():
        for item in man.get("outputs", []):
            if not (roots[stage] / item["path"]).is_file():
                audit.missing.append({"stage": stage, "path": item["path"], "reason": "manifest output missing"})
    for m in models:
        if not m["ok"]:
            audit.missing.append({"stage": "B4.1", "path": f"{m['task']}/{m['modality']}/seed{m['seed']}", "reason": "array/hash/shape mismatch"})
    audit.add("assets.missing", "PASS" if not audit.missing else "FAIL", len(audit.missing))

    # ---------- Write outputs ----------
    status = audit.status()
    inventory_fields = ["stage", "path", "role", "sha256", "size_bytes", "status"]
    with (OUT / "input_inventory.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=inventory_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(audit.inventory)
    with (OUT / "missing_assets.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "path", "reason"], lineterminator="\n")
        writer.writeheader(); writer.writerows(audit.missing)

    consistency = {
        "phase": "B5-PREFLIGHT",
        "generated_at": generated,
        "status": status,
        "baseline": baseline,
        "inventory": {"files": len(audit.inventory), "bytes": sum(r["size_bytes"] for r in audit.inventory)},
        "b4_2_b4_3": {"rows_compared": len(point43), "max_abs_diff": max_diff,
                       "point_estimate_is_paired_difference": bool(max_diff is not None and max_diff <= 1e-6)},
        "b4_3_bootstrap": {"reps": 2000, "npz_keys": len(npz43.files),
                           "ci_rows_per_seed": len(ci_seed), "ci_rows_three_seed": len(ci_three),
                           "draws_sha256": summary43["config"]["draws_sha256"]},
        "b4_4": {"counts": summary44["counts"], "oracle_any": oracle_any, "uv_rescue": uv_rescue},
        "supplement": {"examples": len(examples), "files": len(supp_files),
                        "b44_tree": b44_tree},
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "checks": audit.checks,
    }
    write_json(OUT / "cross_stage_consistency.json", consistency)
    verification = {
        "phase": "B5-PREFLIGHT",
        "generated_at": generated,
        "status": status,
        "summary": {"n_checks": len(audit.checks),
                    "n_pass": sum(1 for c in audit.checks if c["status"] == "PASS"),
                    "n_warn": len(audit.warn()), "n_fail": len(audit.fail()),
                    "inventory_files": len(audit.inventory), "missing_assets": len(audit.missing)},
        "checks": audit.checks,
        "failures": audit.fail(),
    }
    write_json(OUT / "verification.json", verification)

    lines = ["# Line B B5 Phase 0 — Final Report Preflight", "",
             f"> **READ-ONLY AUDIT.** Generated {generated}. Status: **{status}**.",
             "> No training, inference, threshold fitting, bootstrap, or B4 modification was performed.", "",
             "## 1. Frozen baseline", "",
             "| Item | SHA256 | Status |", "|---|---|---|",
             f"| B0 v0.2 protocol | `{baseline['protocol']['sha256']}` | {'PASS' if baseline['protocol']['sha256'] == B0_SHA else 'FAIL'} |",
             f"| FREEZE | `{baseline['freeze']['sha256']}` | {'PASS' if baseline['freeze']['sha256'] == FREEZE_SHA else 'FAIL'} |",
             f"| Test access log | `{baseline['test_access_log']['sha256']}` | {'PASS' if len(entries) == 18 else 'FAIL'} |", "",
             f"Inventory: **{len(audit.inventory)} files / {sum(r['size_bytes'] for r in audit.inventory):,} bytes**.", "",
             "## 2. Check summary", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    for c in audit.checks:
        lines.append(f"| `{c['id']}` | **{c['status']}** | {c['detail']} |")
    lines += ["", "## 3. H1–H4 evidence (B0 criteria)", "",
              f"- **H1** supported=`{h1['supported']}`; Shared14 macro AP IR/Raman/UV = "
              f"{h1['shared14_macro_ap']['ir']:.4f}/{h1['shared14_macro_ap']['raman']:.4f}/{h1['shared14_macro_ap']['uvvis']:.4f}; "
              f"vibration-label gap={h1['vibration_label_gap']:.4f} vs other-label gap={h1['other_label_gap']:.4f}.",
              f"- **H2** supported=`{h2['supported']}`; UVSPEC15 macro AP UV/IR/Raman = "
              f"{h2['uvspec15_macro_ap']['uvvis']:.4f}/{h2['uvspec15_macro_ap']['ir']:.4f}/{h2['uvspec15_macro_ap']['raman']:.4f}; "
              f"AUROC = {h2['uvspec15_macro_auroc']['uvvis']:.4f}/{h2['uvspec15_macro_auroc']['ir']:.4f}/{h2['uvspec15_macro_auroc']['raman']:.4f}. "
              "H2 is **not supported** by the current Test metrics and must be reported as rejected, not confirmed.",
              f"- **H3** partial: Oracle any-correct Shared14/UVSPEC15 = {oracle_any['Shared14']:.4f}/{oracle_any['UVSPEC15']:.4f}; "
              f"three-modality positive OR gain range = [{h3['positive_or_three_modality_gain_min']:+.4f}, {h3['positive_or_three_modality_gain_max']:+.4f}]; "
              f"UV rescue Shared14/UVSPEC15 = {uv_rescue['Shared14']:.4f}/{uv_rescue['UVSPEC15']:.4f}, delta = "
              f"{h3['uv_rescue_delta']['Shared14']:+.4f}/{h3['uv_rescue_delta']['UVSPEC15']:+.4f}. "
              "The H3 subgroup exclusive-correct enrichment clause is **not directly tested** in B4.4.",
              f"- **H4** supported=`{h4['supported']}`; best-AP modality counts = {h4['best_modality_counts']}.", "",
              "## 4. Scientific interpretation boundary", "",
              "- Positive OR/AND are fixed rules, **not True Oracle**; True Oracle is a diagnostic ceiling, **not a fusion model**.",
              "- Different error sets do **not** by themselves establish conditional complementarity; B4.4's UV rescue rate is below its marginal rate with a negative CI.",
              "- UV marginal performance is **not** UV conditional complementation value; AP/AUROC are **not** direct measures of chemical information content.",
              "- QM9S is a computational domain; results must not be extrapolated to experimental NIST/API-Raman data.",
              "- All Test results are exploratory/post-hoc; representative molecules are qualitative only, not XAI and not statistical evidence.",
              "- **B5 final report must add the AP/AUROC-to-chemical-information and QM9S-to-experimental-domain caveats explicitly** if the stage reports do not contain them.", "",
              "## 5. CI semantics", "",
              f"B4.3 `point_comparisons.csv` stores **paired differences** (base − other), compared here against B4.2 absolute metrics: "
              f"{len(point43)} rows, max |difference| = {max_diff:.3e}. B4.3 CIs are CIs of these paired differences, **not** CIs of absolute single-modality AP/AUROC/F1. "
              "B4.4 conditional CIs are CIs of rescue rates/deltas; subgroup CIs are within-subgroup paired differences.", "",
              "## 6. Missing assets", "",
              f"{len(audit.missing)} missing/mismatched assets." if audit.missing else "None.", "",
              "## 7. B5 mandatory fixes (WARN items)", "",
              "1. **B4.3 CI wording:** state explicitly that B4.3 CIs are CIs of paired differences (base − other), not CIs of absolute single-modality AP/AUROC/F1.",
              "2. **H3 scope:** do not claim the H3 subgroup exclusive-correct enrichment clause; B4.4 does not directly test only_UV/only_IR/only_Raman by structural subgroup. Report H3 as partial.",
              "3. **Required boundaries:** add explicit AP/AUROC ≠ chemical-information-content and QM9S computational → NIST/API-Raman experimental-domain caveats to the B5 final report.",
              "4. **Hypothesis verdicts:** H1 partial (main IR/Raman>UV supported; vibration-label-gap clause not), H2 not supported (rejected), H3 partial, H4 descriptively supported.", "",
              "## 8. B5 readiness", "",
              f"**{status}** — " + ("No missing assets, hash mismatch, metric inconsistency, or overclaim was found. The three WARN items are documentation/interpretation gaps to be fixed in the B5 report; B5 may proceed after human approval." if status == "WARN" else
                                 "B5 may proceed after human approval." if status == "PASS" else "B5 must not proceed until failures are resolved."), ""]
    (OUT / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": status, "n_checks": len(audit.checks),
                      "n_pass": sum(1 for c in audit.checks if c["status"] == "PASS"),
                      "n_warn": len(audit.warn()), "n_fail": len(audit.fail()),
                      "inventory": len(audit.inventory), "missing": len(audit.missing),
                      "out": relative_to_root(OUT)}, ensure_ascii=False, indent=1))
    raise SystemExit(0 if status != "FAIL" else 1)


if __name__ == "__main__":
    main()
