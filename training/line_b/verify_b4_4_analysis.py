"""Independent acceptance verification for B4.4 Test error complementarity analysis.

Recomputes pairwise confusion/Jaccard/F1, McNemar+Holm, OR/AND, Oracle,
conditional counts, sample-level summaries and subgroup membership/metrics
from the frozen Test predictions.  Bootstrap CIs are re-derived from the stored
replicate arrays, and a sample of conditional replicates is recomputed with an
independent weighted-metric implementation.

Read-only; writes only verification.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
SEEDS = (42, 123, 2026)
POLICIES = ("fixed_0.5", "validation_f1")
PAIRS = (("ir", "raman"), ("uvvis", "ir"), ("uvvis", "raman"))
PAIR_LABELS = {("ir", "raman"): "IR-Raman", ("uvvis", "ir"): "UV-IR", ("uvvis", "raman"): "UV-Raman"}
DIRECTIONS = (("ir", "raman"), ("raman", "ir"), ("uvvis", "ir"), ("ir", "uvvis"), ("uvvis", "raman"), ("raman", "uvvis"))
DEFAULT_ANALYSIS = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4"
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
B43 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
STRUCTURE = ROOT / "processed" / "qm9s" / "structures" / "qm9s_structure_library.csv"
SCAFFOLD = ROOT / "processed" / "qm9s" / "prepared" / "qm9s_manifest_prepared.csv"
D1 = ROOT / "runs" / "LINE-B-ANALYSIS" / "validation_development" / "B3.5-D1" / "final_22subgroups_20reps"


class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, ok, detail=""):
        self.items.append({"id": cid, "ok": bool(ok), "detail": str(detail)})

    def failed(self):
        return [i for i in self.items if not i["ok"]]


def _rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def _conf(y, pred):
    y = y.astype(bool); pred = pred.astype(bool)
    tp = int(np.sum(y & pred)); fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred)); tn = int(np.sum(~y & ~pred))
    return tp, fp, fn, tn, _f1(tp, fp, fn)


def _mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_tail = np.logaddexp.reduce([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)])
    log_p = math.log(2.0) + float(log_tail) - n * math.log(2.0)
    return min(1.0, math.exp(log_p)) if log_p < 0 else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    analysis = args.analysis_dir.resolve()
    checks = Checks()
    if sha256_file(args.freeze) != args.freeze_sha256:
        raise SystemExit("FREEZE SHA256 mismatch")
    freeze = read_json(args.freeze)
    checks.add("freeze.status", freeze.get("status") == "FROZEN" and freeze.get("phase") == "B3")
    checks.add("freeze.protocol", freeze["protocol_sha256"] == sha256_file(ROOT / "docs/experiments/line_b_protocol_draft.md"))

    summary = read_json(analysis / "summary.json")
    manifest = read_json(analysis / "analysis_manifest.json")
    checks.add("summary.phase", summary.get("phase") == "B4.4")
    checks.add("summary.classification", summary["classification"]["status"] == "exploratory/post-hoc")
    bad = []
    for item in manifest["outputs"]:
        p = analysis / item["path"]
        if not p.is_file() or sha256_file(p) != item["sha256"] or p.stat().st_size != item["size_bytes"]:
            bad.append(item["path"])
    checks.add("manifest.outputs", not bad, ",".join(bad[:5]))

    models = []
    for record in freeze["models"]:
        d = PREDICTIONS / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        m = read_json(d / "prediction_manifest.json")
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        row = np.load(d / "row_index.npy", allow_pickle=False)
        lab = np.load(d / "labels.npy", allow_pickle=False)
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        t = read_json(ROOT / record["threshold"]["path"])
        models.append({
            "task": record["task"], "label_set": record["label_set"], "modality": record["modality"],
            "seed": int(record["seed"]), "probabilities": probs, "labels": lab, "row_index": row,
            "thresholds": np.asarray([t["thresholds"][n] for n in names]), "names": names,
            "manifest": m,
        })
    checks.add("models.count_18", len(models) == 18, len(models))
    labels_by_task = {}
    names_by_task = {}
    for model in models:
        labels_by_task.setdefault(model["task"], model["labels"])
        names_by_task.setdefault(model["task"], model["names"])
        if not np.array_equal(labels_by_task[model["task"]], model["labels"]):
            checks.add(f"labels.aligned.{model['task']}", False)
    for task in labels_by_task:
        checks.add(f"labels.count.{task}", labels_by_task[task].shape[0] == 12981)

    def pred(model, policy):
        thresholds = np.full(len(model["names"]), 0.5) if policy == "fixed_0.5" else model["thresholds"]
        return model["probabilities"] >= thresholds.reshape(1, -1)

    by_task = {task: {(m["seed"], m["modality"]): m for m in models if m["task"] == task} for task, _ in TASKS}

    # Pairwise metrics / jaccard / McNemar.
    pairwise = _rows(analysis / "pairwise_metrics.csv")
    mcnemar_csv = _rows(analysis / "mcnemar_holm.csv")
    max_pair = 0.0
    for row in pairwise:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]
        base, other = row["base_modality"], row["other_modality"]
        pa = pred(by_task[task][(seed, base)], policy)
        pb = pred(by_task[task][(seed, other)], policy)
        if row["scope"] == "label":
            index = int(row["label_index"])
            y = by_task[task][(seed, base)]["labels"][:, index]
            ca = _conf(y, pa[:, index]); cb = _conf(y, pb[:, index])
            base_ok = pa[:, index] == y; other_ok = pb[:, index] == y
            both_correct = int(np.sum(base_ok & other_ok))
            error_union = len(y) - both_correct
            both_wrong = int(np.sum(~base_ok & ~other_ok))
            expected = {
                "base_tp": ca[0], "base_fp": ca[1], "base_fn": ca[2], "base_tn": ca[3],
                "other_tp": cb[0], "other_fp": cb[1], "other_fn": cb[2], "other_tn": cb[3],
                "both_correct": both_correct, "both_wrong": both_wrong,
                "base_only_correct": int(np.sum(base_ok & ~other_ok)),
                "other_only_correct": int(np.sum(~base_ok & other_ok)),
                "error_union": error_union,
                "error_jaccard": 1.0 if error_union == 0 else both_wrong / error_union,
            }
            for key, value in expected.items():
                max_pair = max(max_pair, abs(float(row[key]) - float(value)))
            checks.add(f"pair.counts.{task}.{seed}.{policy}.{row['comparison']}.{row['label']}",
                       all(abs(float(row[k]) - float(v)) <= 1e-12 for k, v in expected.items()))
    checks.add("pairwise.max_abs_diff", max_pair <= 1e-9, f"{max_pair:.3e}")

    # McNemar / Holm.
    holm_max = 0.0
    families = {}
    for row in mcnemar_csv:
        families.setdefault((row["task"], int(row["seed"]), row["threshold_policy"]), []).append(row)
    for key, rows in families.items():
        checks.add(f"mcnemar.family_size.{key}", len(rows) == (42 if key[0] == "Shared14" else 45), len(rows))
        for row in rows:
            p = _mcnemar(int(row["base_only_correct"]), int(row["other_only_correct"]))
            holm_max = max(holm_max, abs(p - float(row["p_raw"])))
        ordered = sorted(rows, key=lambda r: (float(r["p_raw"]), int(r["label_index"]), r["comparison"]))
        m = len(ordered); adjusted = 1.0
        for i, row in enumerate(ordered):
            candidate = min(1.0, (m - i) * float(row["p_raw"]))
            adjusted = max(adjusted, candidate) if i else candidate
            holm_max = max(holm_max, abs(adjusted - float(row["p_holm"])))
    checks.add("mcnemar.max_abs_diff", holm_max <= 1e-12, f"{holm_max:.3e}")

    # OR / AND / Oracle.
    or_rows = _rows(analysis / "pairwise_or_and.csv")
    or_max = 0.0
    for row in or_rows:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]
        base, other = row["base_modality"], row["other_modality"]
        pa = pred(by_task[task][(seed, base)], policy)
        pb = pred(by_task[task][(seed, other)], policy)
        y = by_task[task][(seed, base)]["labels"]
        f1 = {}
        for mod, p in (("base", pa), ("other", pb)):
            f1[mod] = np.asarray([_conf(y[:, i], p[:, i])[4] for i in range(y.shape[1])])
        or_f1 = np.asarray([_conf(y[:, i], pa[:, i] | pb[:, i])[4] for i in range(y.shape[1])])
        and_f1 = np.asarray([_conf(y[:, i], pa[:, i] & pb[:, i])[4] for i in range(y.shape[1])])
        gains = or_f1 - np.maximum(f1["base"], f1["other"])
        expected = {"base_f1": f1["base"].mean(), "other_f1": f1["other"].mean(),
                    "or_f1": or_f1.mean(), "and_f1": and_f1.mean(), "or_gain_vs_max": gains.mean()}
        for k, v in expected.items():
            or_max = max(or_max, abs(float(row[k]) - float(v)))
    checks.add("or.max_abs_diff", or_max <= 1e-9, f"{or_max:.3e}")

    three_rows = _rows(analysis / "three_modality_or.csv")
    three_max = 0.0
    for row in three_rows:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]
        y = by_task[task][(seed, "ir")]["labels"]
        union = pred(by_task[task][(seed, "ir")], policy) | pred(by_task[task][(seed, "raman")], policy) | pred(by_task[task][(seed, "uvvis")], policy)
        or_f1 = np.mean([_conf(y[:, i], union[:, i])[4] for i in range(y.shape[1])])
        best = row["best_single"]
        best_f1 = np.mean([_conf(y[:, i], pred(by_task[task][(seed, best)], policy)[:, i])[4] for i in range(y.shape[1])])
        three_max = max(three_max, abs(float(row["three_or_f1"]) - or_f1), abs(float(row["best_single_f1"]) - best_f1),
                        abs(float(row["or_gain_vs_best_single"]) - (or_f1 - best_f1)))
    checks.add("three_or.max_abs_diff", three_max <= 1e-9, f"{three_max:.3e}")

    oracle = _rows(analysis / "oracle_summary.csv")
    oracle_max = 0.0
    for row in oracle:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]
        correct = {m: pred(by_task[task][(seed, m)], policy) == by_task[task][(seed, m)]["labels"] for m in MODALITIES}
        c = correct["ir"].astype(int) + correct["raman"].astype(int) + correct["uvvis"].astype(int)
        expected = {"any_correct": int(np.sum(c >= 1)), "exactly_one_correct": int(np.sum(c == 1)),
                    "all_wrong": int(np.sum(c == 0)), "all_correct": int(np.sum(c == 3)),
                    "only_ir_correct": int(np.sum((c == 1) & correct["ir"])),
                    "only_raman_correct": int(np.sum((c == 1) & correct["raman"])),
                    "only_uv_correct": int(np.sum((c == 1) & correct["uvvis"]))}
        for k, v in expected.items():
            oracle_max = max(oracle_max, abs(float(row[k]) - float(v)))
    checks.add("oracle.max_abs_diff", oracle_max <= 1e-9, f"{oracle_max:.3e}")

    # Conditional point + CI arrays + sample replicates.
    conditional = _rows(analysis / "conditional_complementarity.csv")
    cond_max = 0.0
    for row in conditional:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]
        correct = {m: pred(by_task[task][(seed, m)], policy) == by_task[task][(seed, m)]["labels"] for m in MODALITIES}
        C = np.stack([correct[m].reshape(-1) for m in MODALITIES], axis=1).astype(np.int64)
        W = 1 - C
        metric = row["metric"]
        if metric == "conditional_correct_given_base_wrong":
            i = MODALITIES.index(row["base_modality"]); j = MODALITIES.index(row["other_modality"])
            num = int((W[:, i] * C[:, j]).sum()); den = int(W[:, i].sum())
            rate = num / den if den else None
            p_other = float(C[:, j].mean())
            cond_max = max(cond_max, abs(float(row["count_num"]) - num), abs(float(row["count_den"]) - den))
            if rate is not None:
                cond_max = max(cond_max, abs(float(row["rate"]) - rate), abs(float(row["delta"]) - (rate - p_other)))
        elif metric == "uv_marginal_correct_rate":
            cond_max = max(cond_max, abs(float(row["rate"]) - float(C[:, 2].mean())))
    checks.add("conditional.point_max_abs_diff", cond_max <= 1e-9, f"{cond_max:.3e}")

    cond_ci = _rows(analysis / "conditional_bootstrap_ci.csv")
    cond_store = np.load(analysis / "conditional_replicates.npz")
    ci_max = 0.0
    for row in cond_ci:
        key = "|".join([row["task"], row["threshold_policy"], row["metric"], row["base_modality"], row["other_modality"]])
        arr = cond_store[key]
        valid = arr[np.isfinite(arr)]
        low, high = np.percentile(valid, (2.5, 97.5))
        ci_max = max(ci_max, abs(float(row["ci_low"]) - low), abs(float(row["ci_high"]) - high))
        checks.add(f"conditional.valid.{key}", int(row["n_valid_reps"]) == len(valid))
    checks.add("conditional.ci_max_abs_diff", ci_max <= 1e-9, f"{ci_max:.3e}")

    # Sample-level summary.
    sample = _rows(analysis / "sample_error_summary.csv")
    sample_max = 0.0
    for row in sample:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]; mod = row["modality"]
        model = by_task[task][(seed, mod)]
        errors = pred(model, policy) != model["labels"].astype(bool)
        n_errors = errors.sum(axis=1)
        expected = {
            "mean_n_errors": float(n_errors.mean()), "median_n_errors": float(np.median(n_errors)),
            "q1_n_errors": float(np.percentile(n_errors, 25, method="linear")),
            "q3_n_errors": float(np.percentile(n_errors, 75, method="linear")),
            "exact_match_count": int((n_errors == 0).sum()),
        }
        for k, v in expected.items():
            sample_max = max(sample_max, abs(float(row[k]) - v))
    checks.add("sample.max_abs_diff", sample_max <= 1e-9, f"{sample_max:.3e}")

    # Subgroup membership and a sample of metric rows.
    structure = {}
    with STRUCTURE.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            structure[int(row["row_index"])] = row
    subgroup_rows = _rows(analysis / "subgroup_metrics.csv")
    subgroup_max = 0.0
    support_mismatch = 0
    sampled = 0
    checked = 0
    for row in subgroup_rows[::37]:
        task = row["task"]; seed = int(row["seed"]); policy = row["threshold_policy"]; mod = row["modality"]
        model = by_task[task][(seed, mod)]
        index = int(row["label_index"])
        if row["subgroup"].startswith("functional_group="):
            group = row["subgroup"].split("=", 1)[1]
            tokens = {t.strip() for t in (structure[int(model["row_index"][0])].get("functional_group_types", "") or "").split(";") if t.strip()}
            mask = np.asarray([group in {t.strip() for t in (structure[int(i)].get("functional_group_types", "") or "").split(";") if t.strip()} for i in model["row_index"]])
        else:
            dimension, label = row["subgroup"].split("=", 1)
            quartiles = read_json(D1 / task / "structure_analysis" / "summary.json")["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
            def value(r):
                if dimension == "ring_count":
                    v = int(float(r["ring_count"])); return str(v) if v < 3 else ">=3"
                if dimension == "aromatic_ring_count":
                    v = int(float(r["aromatic_ring_count"])); return str(v) if v < 2 else ">=2"
                if dimension == "n_o_f":
                    v = int(float(r["nitrogen_count"])) + int(float(r["oxygen_count"])) + int(float(r["fluorine_count"]))
                    return str(v) if v < 3 else ">=3"
                if dimension == "rotatable_bond_count":
                    v = int(float(r["rotatable_bond_count"])); return str(v) if v < 2 else ">=2"
                if dimension == "fraction_csp3": return "<0.5" if float(r["fraction_csp3"]) < 0.5 else ">=0.5"
                if dimension == "heavy_atom_count":
                    v = int(float(r["heavy_atom_count"]))
                    if v <= quartiles["q1"]: return "Q1"
                    if v <= quartiles["q2"]: return "Q2"
                    if v <= quartiles["q3"]: return "Q3"
                    return "Q4"
                return ""
            mask = np.asarray([value(structure[int(i)]) == label for i in model["row_index"]])
        y = model["labels"][mask, index]; p = model["probabilities"][mask, index]
        ap = float(average_precision_score(y, p)) if np.unique(y).size == 2 else None
        pred_mask = (p >= (0.5 if policy == "fixed_0.5" else model["thresholds"][index])).astype(int)
        f1 = _conf(y.astype(int), pred_mask)[4]
        if ap is not None:
            subgroup_max = max(subgroup_max, abs(float(row["ap"]) - ap))
        subgroup_max = max(subgroup_max, abs(float(row["f1"]) - f1))
        sampled += 1
        expected_n = int(mask.sum()); expected_pos = int(y.sum())
        expected_support = expected_n >= 100 and expected_pos >= 30
        if int(row["n_samples"]) != expected_n or int(row["positive_count"]) != expected_pos or (row["support_sufficient"].lower() == "true") != expected_support:
            support_mismatch += 1
        checked += 1
    checks.add("subgroup.sample_metric_max_abs_diff", subgroup_max <= 1e-9, f"{subgroup_max:.3e} ({checked} rows)")
    checks.add("subgroup.support_membership", support_mismatch == 0, f"{support_mismatch}/{sampled} mismatches")

    # Subgroup bootstrap CIs from stored replicate arrays.
    boot_rows = _rows(analysis / "subgroup_bootstrap_ci.csv")
    boot_store = np.load(analysis / "subgroup_bootstrap_replicates.npz")
    boot_max = 0.0
    for row in boot_rows:
        key = "|".join([row["task"], row["subgroup"], row["policy"], row["comparison"], row["metric"]])
        arr = boot_store[key]
        valid = arr[np.isfinite(arr)]
        low, high = np.percentile(valid, (2.5, 97.5))
        boot_max = max(boot_max, abs(float(row["ci_low"]) - low), abs(float(row["ci_high"]) - high))
        checks.add(f"subgroup_boot.valid.{key}", int(row["n_valid_reps"]) == len(valid))
    checks.add("subgroup_boot.ci_max_abs_diff", boot_max <= 1e-9, f"{boot_max:.3e}")

    # Frozen assets.
    checks.add("frozen.freeze", sha256_file(args.freeze) == args.freeze_sha256)
    checks.add("frozen.b42", read_json(B42 / "verification.json")["status"] == "PASS")
    checks.add("frozen.b43", read_json(B43 / "verification.json")["status"] == "PASS")
    b42_bad = [i["path"] for i in read_json(B42 / "analysis_manifest.json")["outputs"]
               if sha256_file(B42 / i["path"]) != i["sha256"]]
    checks.add("frozen.b42_outputs", not b42_bad, ",".join(b42_bad[:3]))
    b43_bad = [i["path"] for i in read_json(B43 / "analysis_manifest.json")["outputs"]
               if sha256_file(B43 / i["path"]) != i["sha256"]]
    checks.add("frozen.b43_outputs", not b43_bad, ",".join(b43_bad[:3]))
    checks.add("frozen.test_access_log", sha256_file(ROOT / "runs/LINE-B-MANIFESTS/test_access_log.json") == summary["sources"]["test_access_log"]["sha256"])

    failures = checks.failed()
    status = "PASS" if not failures else "FAILED"
    report = {
        "phase": "B4.4-verification",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "analysis_manifest": {"path": relative_to_root(analysis / "analysis_manifest.json"),
                              "sha256": sha256_file(analysis / "analysis_manifest.json")},
        "freeze": {"path": relative_to_root(args.freeze), "sha256": args.freeze_sha256},
        "summary": {"n_checks": len(checks.items), "n_failed": len(failures),
                    "subgroup_rows_checked": checked},
        "checks": checks.items,
        "failures": failures,
    }
    out = args.json_out or (analysis / "verification.json")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite: {out}")
    write_json(out, report)
    print(json.dumps({"status": status, "n_checks": len(checks.items), "n_failed": len(failures),
                      "failed_ids": [i["id"] for i in failures], "verification": relative_to_root(out)},
                     ensure_ascii=False, indent=1))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
