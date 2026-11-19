"""Independent verification of the Line B B5 final archive.

Checks input hashes, output hashes, 18-model / 3-seed / 14–15-label coverage,
key numbers against the frozen CSV/JSON assets, figure and report references,
the three Phase-0 WARN fixes, H1–H4 consistency, and that B4/FREEZE/B0 are
byte-unchanged.  Writes verification.json only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_file, write_json

B0 = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
TESTLOG = ROOT / "runs" / "LINE-B-MANIFESTS" / "test_access_log.json"
B41 = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
B42 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
B43 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.3"
B44 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4"
SUPP = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4-SUPPLEMENT-REPRESENTATIVE-MOLECULES"
OUT = ROOT / "runs" / "LINE-B-FINAL"
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
TASKS = ("Shared14", "UVSPEC15")
MODALITIES = ("ir", "raman", "uvvis")


class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, ok, detail=""):
        self.items.append({"id": cid, "ok": bool(ok), "detail": str(detail)})

    def failed(self):
        return [i for i in self.items if not i["ok"]]


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


def _mean(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    return float(np.mean(vals)) if vals else None


def main():
    checks = Checks()
    man = read_json(OUT / "asset_manifest.json")
    report = (OUT / "line_b_final_report.md").read_text(encoding="utf-8")
    readme = (OUT / "README_summary.txt").read_text(encoding="utf-8")

    # Inputs.
    inputs_ok = True
    for key in ("protocol", "freeze", "test_access_log"):
        item = man["inputs"][key]
        path = ROOT / item["path"]
        ok = path.is_file() and sha256_file(path) == item["sha256"]
        inputs_ok &= ok
        checks.add(f"input.{key}", ok, item["sha256"])
    for stage in ("b4_2", "b4_3", "b4_4", "supplement"):
        item = man["inputs"][stage]
        for key in ("manifest", "verification"):
            path = {"b4_2": B42, "b4_3": B43, "b4_4": B44, "supplement": SUPP}[stage] / ("analysis_manifest.json" if key == "manifest" else "verification.json")
            checks.add(f"input.{stage}.{key}", sha256_file(path) == item[key], item[key])
    checks.add("input.b44_tree", _tree_digest(B44) == man["inputs"]["b4_4"]["tree_sha256"], man["inputs"]["b4_4"]["tree_sha256"])
    seed_models = man["inputs"]["b4_1_seed_models"]
    checks.add("input.seed_models_18", len(seed_models) == 18, len(seed_models))
    for row in seed_models:
        path = B41 / row["task"] / MODALITY_DIRS[row["modality"]] / next(
            r["run_name"] for r in read_json(FREEZE)["models"]
            if r["task"] == row["task"] and r["modality"] == row["modality"] and int(r["seed"]) == row["seed"]
        ) / "test" / "prediction_manifest.json"
        checks.add(f'input.pred_manifest.{row["task"]}.{row["modality"]}.{row["seed"]}',
                   path.is_file() and sha256_file(path) == row["prediction_manifest"])

    # Outputs.
    bad_outputs = []
    seen = set()
    for item in man["outputs"]:
        path = OUT / item["path"]
        if item["path"] in seen:
            bad_outputs.append((item["path"], "duplicate"))
            continue
        seen.add(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            bad_outputs.append((item["path"], "missing/hash/size"))
    checks.add("outputs.manifest_hashes", not bad_outputs, bad_outputs[:5])

    # Coverage.
    overall = _csv(OUT / "tables" / "overall_metrics.csv")
    per_label = _csv(OUT / "tables" / "per_label_metrics.csv")
    micro = _csv(OUT / "tables" / "micro_macro_summary.csv")
    three = _csv(OUT / "tables" / "paired_differences_three_seed.csv")
    per_seed = _csv(OUT / "tables" / "paired_differences_per_seed.csv")
    cond = _csv(OUT / "tables" / "conditional_complementarity.csv")
    cond_ci = _csv(OUT / "tables" / "conditional_bootstrap_ci.csv")
    sg = _csv(OUT / "tables" / "subgroup_metrics.csv")
    sgb = _csv(OUT / "tables" / "subgroup_bootstrap_ci.csv")
    rep = _csv(OUT / "tables" / "representative_molecules.csv")
    combos = {(r["task"], r["modality"], int(r["seed"])) for r in overall}
    checks.add("coverage.models_18", len(combos) == 18, len(combos))
    checks.add("coverage.tasks_labels", {t: len({r["label"] for r in per_label if r["task"] == t}) for t in TASKS} == {"Shared14": 14, "UVSPEC15": 15},
               {t: len({r["label"] for r in per_label if r["task"] == t}) for t in TASKS})
    checks.add("coverage.seeds", {int(r["seed"]) for r in overall} == {42, 123, 2026}, sorted({int(r["seed"]) for r in overall}))
    checks.add("coverage.policies", {r["policy"] for r in overall} == {"fixed_0.5", "validation_f1"}, sorted({r["policy"] for r in overall}))
    checks.add("coverage.per_label_rows", len(per_label) == 522, len(per_label))
    checks.add("coverage.overall_rows", len(overall) == 36, len(overall))
    checks.add("coverage.micro_rows", len(micro) == 36, len(micro))
    checks.add("coverage.paired_three_seed_rows", len(three) == 372, len(three))
    checks.add("coverage.paired_per_seed_rows", len(per_seed) == 1116, len(per_seed))
    checks.add("coverage.conditional_rows", len(cond) == 120 and len(cond_ci) == 64, f"{len(cond)}/{len(cond_ci)}")
    checks.add("coverage.subgroup_rows", len(sg) == 11484 and len(sgb) == 504, f"{len(sg)}/{len(sgb)}")
    checks.add("coverage.representative_88", len(rep) == 88, len(rep))

    # Micro AP/AUROC independent recomputation from B4.1 for all 18 models.
    freeze = read_json(FREEZE)
    micro_bad = []
    for record in freeze["models"]:
        d = B41 / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        truth = np.load(d / "labels.npy", allow_pickle=False)
        ap = float(average_precision_score(truth.reshape(-1), probs.reshape(-1)))
        auroc = float(roc_auc_score(truth.reshape(-1), probs.reshape(-1)))
        row = next(r for r in micro if r["task"] == record["task"] and int(r["seed"]) == int(record["seed"])
                   and r["modality"] == record["modality"] and r["threshold_policy"] == "fixed_0.5")
        if abs(float(row["micro_ap"]) - ap) > 1e-9 or abs(float(row["micro_auroc"]) - auroc) > 1e-9:
            micro_bad.append((record["task"], record["modality"], record["seed"]))
    checks.add("numbers.micro_ap_auroc", not micro_bad, micro_bad[:5])

    # Key report numbers vs B4.2.
    def mean_overall(task, mod, field, policy="fixed_0.5"):
        return _mean([r[field] for r in overall if r["task"] == task and r["modality"] == mod and r["policy"] == policy])
    key_values = {
        "Shared14 IR macro AP": (mean_overall("Shared14", "ir", "macro_ap"), 4),
        "Shared14 UV macro AP": (mean_overall("Shared14", "uvvis", "macro_ap"), 4),
        "UVSPEC15 UV macro AP": (mean_overall("UVSPEC15", "uvvis", "macro_ap"), 4),
        "UVSPEC15 Raman macro AUROC": (mean_overall("UVSPEC15", "raman", "macro_auroc"), 4),
    }
    missing_numbers = {}
    for name, (value, nd) in key_values.items():
        if f"{value:.{nd}f}" not in report:
            missing_numbers[name] = f"{value:.{nd}f}"
    checks.add("numbers.report_key_values", not missing_numbers, missing_numbers or "key values present")

    # Paired-difference CI semantics (WARN-1).
    checks.add("warn1.paired_difference",
               "paired differences" in report.lower() and "not for absolute" in report.lower()
               and "absolute single-modality" in report.lower(),
               "report states Δ=Metric(base)-Metric(other), not absolute CI")
    checks.add("warn1.absolute_point_estimates",
               "absolute single-modality ap/auroc/f1 are b4.2 point estimates" in report.lower(),
               "absolute performance referred to B4.2 point estimates")

    # WARN-2.
    checks.add("warn2.h3_not_tested",
               "not directly tested" in report.lower() and "only_uv/only_ir/only_raman" in report.lower(),
               "H3 subgroup only_* enrichment explicitly marked not directly tested")
    checks.add("warn2.no_uv_enrichment_claim",
               "no uv-conjugation enrichment claim" in report.lower(),
               "no UV-conjugation enrichment claim")

    # WARN-3 boundaries.
    checks.add("warn3.ap_not_chemical",
               "not the total chemical information content" in report.lower() or "≠ chemical-information" in report,
               "AP/AUROC != chemical information")
    checks.add("warn3.qm9s_not_experimental",
               "must **not** be extrapolated to nist" in report.lower() or "not be extrapolated to nist" in report.lower(),
               "QM9S not experimental domain")
    checks.add("warn3.representative_qualitative",
               "not xai" in report.lower() and "not statistical evidence" in report.lower(),
               "representative molecules qualitative only")

    # H1-H4.
    hyp = _csv(OUT / "tables" / "hypothesis_evaluation.csv")
    verdicts = {r["hypothesis"]: r["verdict"] for r in hyp}
    checks.add("hypotheses.table", verdicts == {"H1": "PARTIAL SUPPORT", "H2": "REJECTED", "H3": "PARTIAL SUPPORT", "H4": "DESCRIPTIVELY SUPPORTED"},
               verdicts)
    checks.add("hypotheses.report", all(f"{h} {v}" in report for h, v in verdicts.items()), verdicts)

    # Figures and report references.
    fig_paths = sorted((OUT / "figures").glob("*.png"))
    checks.add("figures.count", len(fig_paths) >= 10, len(fig_paths))
    bad_fig = []
    for path in fig_paths:
        try:
            with Image.open(path) as im:
                if path.stat().st_size < 5000 or im.size[0] < 300:
                    bad_fig.append(path.name)
        except Exception:
            bad_fig.append(path.name)
    checks.add("figures.valid_png", not bad_fig, bad_fig)

    refs = set(re.findall(r"(?:tables|figures|representative_molecules)/[A-Za-z0-9_./*-]+", report))
    bad_refs = []
    for ref in refs:
        if "*" in ref:
            if not list(OUT.glob(ref)):
                bad_refs.append(ref)
        elif not (OUT / ref).exists():
            bad_refs.append(ref)
    checks.add("report.references_exist", not bad_refs, bad_refs[:8])

    # Representative assets byte-identical to the supplement.
    rep_bad = []
    for row in rep:
        for name, key in (("structure.png", "structure_png_sha256"), ("structure.svg", "structure_svg_sha256"),
                          ("spectra.png", "spectra_png_sha256"), ("metadata.json", "metadata_json_sha256")):
            src = SUPP / row["example_dir"] / name
            dst = OUT / "representative_molecules" / row["example_dir"] / name
            if not dst.is_file() or sha256_file(dst) != sha256_file(src) or sha256_file(dst) != row[key]:
                rep_bad.append((row["task"], row["row_index"], name))
    checks.add("representative.byte_identical", not rep_bad, rep_bad[:5])

    # Three WARN fixes must be reflected in README too.
    checks.add("readme.warn_fixes", all(k in readme for k in ("paired-difference", "NOT directly tested", "chemical-information")), "")

    # Historical assets unchanged.
    for stage, path in (("B4.2", B42), ("B4.3", B43), ("B4.4", B44), ("Supplement", SUPP)):
        ver = read_json(path / "verification.json")
        checks.add(f"frozen.{stage}", ver.get("status") == "PASS" and
                   ver.get("analysis_manifest", {}).get("sha256") == sha256_file(path / "analysis_manifest.json"))
    checks.add("frozen.b44_tree", _tree_digest(B44) == man["inputs"]["b4_4"]["tree_sha256"])
    checks.add("frozen.freeze", sha256_file(FREEZE) == man["inputs"]["freeze"]["sha256"])
    checks.add("frozen.protocol", sha256_file(B0) == man["inputs"]["protocol"]["sha256"])

    failures = checks.failed()
    status = "PASS" if not failures else "FAIL"
    verification = {
        "phase": "B5",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "analysis_manifest": {"path": relative_to_root(OUT / "asset_manifest.json"),
                              "sha256": sha256_file(OUT / "asset_manifest.json")},
        "summary": {"n_checks": len(checks.items), "n_failed": len(failures),
                    "n_tables": len(list((OUT / "tables").glob("*.csv"))),
                    "n_figures": len(fig_paths), "n_representative": len(rep)},
        "checks": checks.items,
        "failures": failures,
    }
    target = OUT / "verification.json"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite: {target}")
    write_json(target, verification)
    print(json.dumps({"status": status, "n_checks": len(checks.items), "n_failed": len(failures),
                      "failed_ids": [i["id"] for i in failures], "verification": relative_to_root(target)},
                     ensure_ascii=False, indent=1))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
