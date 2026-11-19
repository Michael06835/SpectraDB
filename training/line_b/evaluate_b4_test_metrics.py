"""B4.2 Line B Test performance evaluation and modality comparison.

Reads only the frozen B4.1 Test probability arrays and the frozen
Validation-F1 thresholds registered in ``FREEZE.json``.  It never trains,
never re-fits thresholds, never runs Bootstrap and never modifies frozen
assets.

Metric definitions are reused from ``training/line_b/metrics.py`` (for the
fixed-0.5 policy directly, and per label for the frozen-threshold policy);
micro aggregation uses the same scikit-learn call as ``metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import PIL
import sklearn
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    atomic_output_dir,
    read_json,
    relative_to_root,
    sha256_array,
    sha256_file,
    write_json,
)
from training.line_b import b4_2_plots as plots
from training.line_b.metrics import compute_multilabel_metrics

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
POLICIES = ("fixed_0.5", "validation_f1")
OVERALL_METRICS = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_ap",
    "macro_auroc",
)
KEY_METRICS = ("micro_f1", "macro_f1", "macro_ap", "macro_auroc")
TEST_N = 12981
DEFAULT_FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
DEFAULT_PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
DEFAULT_OUT = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.2"
CLASSIFICATION = {
    "status": "exploratory/post-hoc",
    "blind_test_claim": False,
    "historical_test_information_read_before_b4": True,
    "test_driven_tuning": False,
    "threshold_refit_on_test": False,
    "bootstrap_run": False,
}


def _round(value, digits: int = 8):
    if value is None:
        return ""
    return round(float(value), digits)


def _mean_sd(values) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and value != ""]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.{digits}f}"


def _metrics_with_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    thresholds: np.ndarray,
) -> dict:
    """Same metric definitions as metrics.py, with one frozen threshold per label."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (y_true.shape[1],):
        raise ValueError("Threshold vector does not match label count")

    per_label: list[dict] = []
    for index, name in enumerate(label_names):
        result = compute_multilabel_metrics(
            y_true[:, [index]],
            y_prob[:, [index]],
            [name],
            threshold=float(thresholds[index]),
        )
        row = dict(result["per_label"][0])
        row["threshold"] = float(thresholds[index])
        per_label.append(row)

    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int64)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p = float(np.mean([row["precision"] for row in per_label]))
    macro_r = float(np.mean([row["recall"] for row in per_label]))
    macro_f1 = float(np.mean([row["f1"] for row in per_label]))
    ap_values = [row["average_precision"] for row in per_label if row["average_precision"] is not None]
    auroc_values = [row["auroc"] for row in per_label if row["auroc"] is not None]
    return {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "threshold": None,
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_ap": float(np.mean(ap_values)) if ap_values else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "n_defined_ap": len(ap_values),
        "n_defined_auroc": len(auroc_values),
        "undefined_ap_labels": [row["label"] for row in per_label if row["average_precision"] is None],
        "undefined_auroc_labels": [row["label"] for row in per_label if row["auroc"] is None],
        "per_label": per_label,
    }


def _support_tier(val_positives: int, test_positives: int) -> str:
    smallest = min(int(val_positives), int(test_positives))
    if smallest < 30:
        return "very_low"
    if smallest < 150:
        return "low"
    return "normal"


def _load_freeze(path: Path, pin: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"FREEZE not found: {path}")
    actual = sha256_file(path)
    if actual != pin:
        raise SystemExit(f"FREEZE SHA256 mismatch: {actual} != {pin}")
    freeze = read_json(path)
    if freeze.get("status") != "FROZEN" or freeze.get("phase") != "B3":
        raise SystemExit("FREEZE is not a frozen B3 manifest")
    if len(freeze.get("models") or []) != 18:
        raise SystemExit("FREEZE does not register 18 models")
    protocol = ROOT / freeze["protocol_path"]
    if not protocol.is_file() or sha256_file(protocol) != freeze["protocol_sha256"]:
        raise SystemExit("Protocol file does not match the FREEZE protocol SHA256")
    return freeze


def _verify_test_access_log(freeze: dict, pin: str) -> dict:
    log_path = ROOT / "runs" / "LINE-B-MANIFESTS" / "test_access_log.json"
    if not log_path.is_file():
        raise SystemExit(f"test_access_log.json missing: {log_path}")
    log = read_json(log_path)
    if (log.get("freeze") or {}).get("sha256") != pin:
        raise SystemExit("test_access_log.json does not belong to this FREEZE")
    entries = log.get("entries") or []
    if len(entries) != 18 or len({(e.get("task"), e.get("run_name")) for e in entries}) != 18:
        raise SystemExit("test_access_log.json does not contain 18 unique model exports")
    exposure = log.get("historical_exposure") or {}
    if exposure.get("blind_test_claim") is not False:
        raise SystemExit("test_access_log.json must not claim a blind test")
    return {
        "path": relative_to_root(log_path),
        "sha256": sha256_file(log_path),
        "n_entries": len(entries),
        "historical_exposure": exposure,
    }


def _load_model(model: dict, freeze: dict, pin: str, predictions_root: Path, label_names: list[str]) -> dict:
    task = model["task"]
    label_set = model["label_set"]
    modality = model["modality"]
    run_name = model["run_name"]
    out_dir = predictions_root / task / MODALITY_DIRS[modality] / run_name / "test"
    manifest_path = out_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing Test prediction manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if (
        manifest.get("task") != task
        or manifest.get("label_set") != label_set
        or manifest.get("modality") != modality
        or manifest.get("run_name") != run_name
        or manifest.get("split") != "test"
        or manifest.get("n_samples") != TEST_N
    ):
        raise SystemExit(f"Test manifest identity mismatch: {manifest_path}")
    if (manifest.get("freeze") or {}).get("sha256") != pin:
        raise SystemExit(f"Test manifest FREEZE pin mismatch: {manifest_path}")
    if manifest.get("label_names") != label_names:
        raise SystemExit(f"Test label order mismatch: {manifest_path}")
    if manifest["checkpoint"]["sha256"] != model["checkpoint"]["sha256"]:
        raise SystemExit(f"Test checkpoint differs from FREEZE: {manifest_path}")

    probabilities = np.load(out_dir / "probabilities.npy", allow_pickle=False)
    row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
    labels = np.load(out_dir / "labels.npy", allow_pickle=False)
    for key, array in (("probabilities", probabilities), ("row_index", row_index), ("labels", labels)):
        if sha256_array(array) != manifest["arrays"][key]["sha256"]:
            raise SystemExit(f"Test array hash mismatch: {out_dir}/{key}.npy")

    split_path = ROOT / model["split_file"]["path"]
    if sha256_file(split_path) != model["split_file"]["sha256"]:
        raise SystemExit(f"Frozen split hash mismatch: {split_path}")
    with np.load(split_path, allow_pickle=False) as archive:
        test_indices = np.asarray(archive["test"], dtype=np.int64)
    if not np.array_equal(row_index, test_indices):
        raise SystemExit(f"Test row_index differs from frozen split: {out_dir}")

    label_matrix = np.load(ROOT / freeze["labels_and_data"]["label_sets"][label_set]["label_matrix"]["path"], allow_pickle=False)
    if not np.array_equal(labels, np.asarray(label_matrix[row_index], dtype=np.uint8)):
        raise SystemExit(f"Test labels differ from authoritative matrix: {out_dir}")

    validation_labels_path = ROOT / model["labels"]["path"]
    validation_labels = np.load(validation_labels_path, allow_pickle=False)
    if sha256_file(validation_labels_path) != model["labels"]["sha256"]:
        raise SystemExit(f"Validation label matrix hash mismatch: {validation_labels_path}")

    threshold_path = ROOT / model["threshold"]["path"]
    if sha256_file(threshold_path) != model["threshold"]["sha256"]:
        raise SystemExit(f"Frozen threshold hash mismatch: {threshold_path}")
    threshold_payload = read_json(threshold_path)
    if threshold_payload.get("label_names") != label_names:
        raise SystemExit(f"Frozen threshold label order mismatch: {threshold_path}")
    thresholds = np.asarray([threshold_payload["thresholds"][name] for name in label_names], dtype=np.float64)
    if threshold_payload.get("threshold_status") and any(
        value != "defined" for value in threshold_payload["threshold_status"].values()
    ):
        raise SystemExit(f"Undefined frozen threshold in {threshold_path}")

    return {
        "task": task,
        "label_set": label_set,
        "modality": modality,
        "run_name": run_name,
        "seed": int(model["seed"]),
        "probabilities": probabilities,
        "row_index": row_index,
        "labels": labels,
        "thresholds": thresholds,
        "validation_labels": validation_labels,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "threshold_path": relative_to_root(threshold_path),
        "threshold_sha256": sha256_file(threshold_path),
        "split_path": relative_to_root(split_path),
        "split_sha256": sha256_file(split_path),
    }


def _build_figures(outputs: dict, task_rows: dict, labels_by_task: dict) -> list[dict]:
    figures_dir = outputs["figures_dir"]
    figures: list[dict] = []

    def _register(path: Path, category: str) -> None:
        figures.append({
            "path": f"figures/{path.name}",
            "category": category,
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        })

    plots.build_overall_heatmap(figures_dir / "fig1_overall_heatmap.png", task_rows["models"], list(KEY_METRICS))
    _register(figures_dir / "fig1_overall_heatmap.png", "overall heatmap")
    plots.build_f1_bars(figures_dir / "fig2_f1_comparison.png", task_rows)
    _register(figures_dir / "fig2_f1_comparison.png", "F1 comparison")
    plots.build_map_auroc_bars(figures_dir / "fig3_map_auroc_comparison.png", task_rows)
    _register(figures_dir / "fig3_map_auroc_comparison.png", "mAP/AUROC comparison")
    plots.build_label_heatmap(figures_dir / "fig4_per_label_f1_heatmap.png", task_rows, labels_by_task, kind="f1")
    _register(figures_dir / "fig4_per_label_f1_heatmap.png", "per-label F1 heatmap")
    plots.build_label_heatmap(figures_dir / "fig5_per_label_ap_heatmap.png", task_rows, labels_by_task, kind="ap")
    _register(figures_dir / "fig5_per_label_ap_heatmap.png", "per-label AP heatmap")
    plots.build_seed_stability(figures_dir / "fig6_seed_stability.png", task_rows)
    _register(figures_dir / "fig6_seed_stability.png", "seed stability")
    plots.build_task_comparison(figures_dir / "fig7_task_comparison.png", task_rows)
    _register(figures_dir / "fig7_task_comparison.png", "task comparison")
    return figures


def _build_report(freeze: dict, pin: str, models: list[dict], task_rows: dict, labels_by_task: dict,
                  figures: list[dict], access_log: dict, inputs: dict) -> str:
    lines: list[str] = []
    lines.append("# Line B B4.2 — Test Performance Evaluation & Modality Comparison")
    lines.append("")
    lines.append("> **Classification: exploratory / post-hoc.** The Test split was opened before B4; "
                 "results are not a blind test and must not be presented as confirmatory.")
    lines.append("")
    lines.append(f"- FREEZE: `{inputs['freeze_path']}` (SHA256 `{pin}`), protocol `B0 v0.2` `{freeze['protocol_sha256']}`")
    lines.append(f"- Test predictions: `{inputs['predictions_root']}` (18 models, 12,981 rows each)")
    lines.append(f"- Test access log: `{access_log['path']}` (SHA256 `{access_log['sha256']}`, {access_log['n_entries']} entries)")
    lines.append("- Threshold policies: **primary F1 = fixed 0.5** (B0 §6.2); **secondary F1 = frozen per-label Validation-F1**; AP/AUROC are threshold-free.")
    lines.append("- All metrics are recomputed from the frozen FP32 Test probability arrays. No Test threshold fitting, no calibration refit, no Bootstrap.")
    lines.append("")
    lines.append("## 1. Overall Test metrics (per model)")
    lines.append("")
    lines.append("### 1.1 Primary F1 policy (fixed 0.5)")
    lines.append("")
    headers = ["Task", "Modality", "Seed", "Micro-F1", "Macro-F1", "mAP", "Macro-AUROC", "n_AP"]
    rows = []
    for model in models:
        overall = model["overall"]["fixed_0.5"]
        rows.append([model["task"], model["modality"].upper(), model["seed"],
                     _fmt(overall["micro_f1"]), _fmt(overall["macro_f1"]),
                     _fmt(overall["macro_ap"]), _fmt(overall["macro_auroc"]), overall["n_defined_ap"]])
    lines.append(_md_table(headers, rows))
    lines.append("")
    lines.append("### 1.2 Secondary F1 policy (frozen per-label Validation-F1 thresholds)")
    lines.append("")
    rows = []
    for model in models:
        overall = model["overall"]["validation_f1"]
        rows.append([model["task"], model["modality"].upper(), model["seed"],
                     _fmt(overall["micro_f1"]), _fmt(overall["macro_f1"]),
                     _fmt(overall["macro_ap"]), _fmt(overall["macro_auroc"]), overall["n_defined_ap"]])
    lines.append(_md_table(headers, rows))
    lines.append("")
    lines.append("## 2. Modality comparison (3-seed mean ± sample SD)")
    lines.append("")
    for policy, title in (("fixed_0.5", "Primary policy: fixed 0.5"), ("validation_f1", "Secondary policy: frozen Validation-F1")):
        lines.append(f"### {title}")
        lines.append("")
        rows = []
        for task in ["Shared14", "UVSPEC15"]:
            for modality in MODALITIES:
                cell = [task, modality.upper()]
                for metric in KEY_METRICS:
                    record = task_rows["modality_mean_sd"][(task, policy, modality, metric)]
                    cell.append(f'{_fmt(record["mean"])} ± {_fmt(record["sd"])}')
                rows.append(cell)
        lines.append(_md_table(["Task", "Modality", "Micro-F1", "Macro-F1", "mAP", "Macro-AUROC"], rows))
        lines.append("")
    lines.append("## 3. Seed stability")
    lines.append("")
    lines.append("Per-seed values are in `seed_comparison.csv`; means/SDs in `overall_mean_sd.csv`. "
                 "SD is the sample SD across the three seeds (n=3, descriptive only).")
    lines.append("")
    rows = []
    for task in ["Shared14", "UVSPEC15"]:
        for modality in MODALITIES:
            cell = [task, modality.upper()]
            for metric in KEY_METRICS:
                policy = "validation_f1" if metric in ("micro_f1", "macro_f1") else "fixed_0.5"
                seeds = task_rows["seed_values"][(task, policy, modality, metric)]
                values = [item["value"] for item in seeds]
                cell.append(f"{_fmt(min(values))}–{_fmt(max(values))}")
            rows.append(cell)
    lines.append(_md_table(["Task", "Modality", "Micro-F1 range", "Macro-F1 range", "mAP range", "Macro-AUROC range"], rows))
    lines.append("")
    lines.append("## 4. Per-label results (seed mean)")
    lines.append("")
    for task in ["Shared14", "UVSPEC15"]:
        lines.append(f"### {task} ({len(labels_by_task[task])} labels)")
        lines.append("")
        rows = []
        for index, label in enumerate(labels_by_task[task]):
            cell = [index, label]
            for modality in MODALITIES:
                record = task_rows["per_label_mean_sd"][(task, "validation_f1", modality, label)]
                ap_record = task_rows["per_label_ap_mean_sd"][(task, modality, label)]
                cell.append(f'{_fmt(record["f1_mean"], 3)} / {_fmt(ap_record["average_precision_mean"], 3)}')
            tier_record = task_rows["per_label_ap_mean_sd"][(task, MODALITIES[0], label)]
            cell.append(f'{tier_record["test_positives"]} ({tier_record["support_tier"]})')
            rows.append(cell)
        lines.append(_md_table(["#", "Label", "IR F1/AP", "Raman F1/AP", "UV-Vis F1/AP", "Test positives (tier)"], rows))
        lines.append("")
    lines.append("## 5. Modality strengths and weaknesses (AP ranking)")
    lines.append("")
    for task in ["Shared14", "UVSPEC15"]:
        wins = {modality: 0 for modality in MODALITIES}
        for label in labels_by_task[task]:
            best = max(MODALITIES, key=lambda m: task_rows["per_label_ap_mean_sd"][(task, m, label)]["average_precision_mean"])
            wins[best] += 1
        lines.append(f"- **{task}**: AP-best modality counts — " + ", ".join(f"{m.upper()} {wins[m]}" for m in MODALITIES))
        for modality in MODALITIES:
            ranked = sorted(
                labels_by_task[task],
                key=lambda label: task_rows["per_label_ap_mean_sd"][(task, modality, label)]["average_precision_mean"],
                reverse=True,
            )
            normal = [label for label in ranked
                      if task_rows["per_label_ap_mean_sd"][(task, modality, label)]["support_tier"] != "very_low"]
            top = normal[:3]
            bottom = normal[-3:]
            lines.append(f"- {modality.upper()} strongest AP: " + ", ".join(f"{label} ({_fmt(task_rows['per_label_ap_mean_sd'][(task, modality, label)]['average_precision_mean'], 3)})" for label in top))
            lines.append(f"- {modality.upper()} weakest AP: " + ", ".join(f"{label} ({_fmt(task_rows['per_label_ap_mean_sd'][(task, modality, label)]['average_precision_mean'], 3)})" for label in bottom))
    lines.append("")
    lines.append("## 6. Figures")
    lines.append("")
    for figure in figures:
        lines.append(f"### {figure['category']}")
        lines.append("")
        lines.append(f"![{figure['category']}]({figure['path']})")
        lines.append("")
    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("- **exploratory / post-hoc**: historical Test information was read in B0; this is not a blind test.")
    lines.append("- n=3 seeds: Mean ± SD is descriptive; no significance claims, no modality ranking claims beyond descriptive ordering.")
    lines.append("- Very-low-support labels are reported but must not drive modality conclusions (B0 §6.3).")
    lines.append("- F1 at frozen Validation-F1 thresholds is out-of-sample on Test but the thresholds were fitted in-sample on Validation.")
    lines.append("- No Bootstrap confidence intervals in B4.2; the formal 2,000-replicate scaffold Bootstrap belongs to B4.3.")
    lines.append("")
    lines.append("## 8. Provenance")
    lines.append("")
    lines.append("- `analysis_manifest.json` records every input and output SHA256.")
    lines.append("- `verification.json` is produced by the independent verifier `training/line_b/verify_b4_2_analysis.py`.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    freeze = _load_freeze(args.freeze, args.freeze_sha256)
    access_log = _verify_test_access_log(freeze, args.freeze_sha256)
    if args.out_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.out_dir}")

    models: list[dict] = []
    for record in freeze["models"]:
        task = record["task"]
        label_set = record["label_set"]
        label_names = list(freeze["labels_and_data"]["label_sets"][label_set]["label_names"]["names"])
        model = _load_model(record, freeze, args.freeze_sha256, args.predictions_root, label_names)
        models.append(model)

    reference_row_index = models[0]["row_index"]
    for model in models[1:]:
        if not np.array_equal(model["row_index"], reference_row_index):
            raise SystemExit("Test row_index differs across models")

    # Evaluate every model under both frozen threshold policies.
    for model in models:
        label_names = list(freeze["labels_and_data"]["label_sets"][model["label_set"]]["label_names"]["names"])
        model["overall"] = {}
        model["per_label"] = {}
        for policy in POLICIES:
            if policy == "fixed_0.5":
                result = compute_multilabel_metrics(model["labels"], model["probabilities"], label_names, threshold=0.5)
            else:
                result = _metrics_with_per_label_thresholds(
                    model["labels"], model["probabilities"], label_names, model["thresholds"]
                )
            model["overall"][policy] = result
            model["per_label"][policy] = result["per_label"]

    # Assemble CSVs.
    overall_rows: list[dict] = []
    per_label_rows: list[dict] = []
    for model in models:
        label_names = list(freeze["labels_and_data"]["label_sets"][model["label_set"]]["label_names"]["names"])
        val_positives = np.asarray(model["validation_labels"].sum(axis=0)).astype(int)
        test_positives = np.asarray(model["labels"].sum(axis=0)).astype(int)
        macro_prevalence = float(np.mean(test_positives / TEST_N))
        for policy in POLICIES:
            overall = model["overall"][policy]
            overall_rows.append({
                "task": model["task"],
                "label_set": model["label_set"],
                "policy": policy,
                "run_name": model["run_name"],
                "modality": model["modality"],
                "seed": model["seed"],
                "n_samples": overall["n_samples"],
                "n_labels": overall["n_labels"],
                "threshold": "0.5" if policy == "fixed_0.5" else "per_label_frozen",
                "micro_precision": _round(overall["micro_precision"]),
                "micro_recall": _round(overall["micro_recall"]),
                "micro_f1": _round(overall["micro_f1"]),
                "macro_precision": _round(overall["macro_precision"]),
                "macro_recall": _round(overall["macro_recall"]),
                "macro_f1": _round(overall["macro_f1"]),
                "macro_ap": _round(overall["macro_ap"]),
                "macro_auroc": _round(overall["macro_auroc"]),
                "macro_prevalence": _round(macro_prevalence),
                "n_defined_ap": overall["n_defined_ap"],
                "n_defined_auroc": overall["n_defined_auroc"],
            })
            for index, (label, row) in enumerate(zip(label_names, model["per_label"][policy])):
                ap = row["average_precision"]
                per_label_rows.append({
                    "task": model["task"],
                    "label_set": model["label_set"],
                    "policy": policy,
                    "run_name": model["run_name"],
                    "modality": model["modality"],
                    "seed": model["seed"],
                    "label_index": index,
                    "label": label,
                    "threshold": _round(row.get("threshold", 0.5)),
                    "val_positives": int(val_positives[index]),
                    "test_positives": int(test_positives[index]),
                    "support": row["support"],
                    "prevalence": _round(row["prevalence"]),
                    "support_tier": _support_tier(val_positives[index], test_positives[index]),
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "tn": row["tn"],
                    "precision": _round(row["precision"]),
                    "recall": _round(row["recall"]),
                    "f1": _round(row["f1"]),
                    "average_precision": _round(ap),
                    "auroc": _round(row["auroc"]),
                    "ap_defined": bool(row["ap_defined"]),
                    "auroc_defined": bool(row["auroc_defined"]),
                    "ap_minus_prevalence": "" if ap is None else _round(ap - float(row["prevalence"])),
                })

    # Aggregations.
    def _group_keys(task: str, modality: str, policy: str):
        return [row for row in overall_rows if row["task"] == task and row["modality"] == modality and row["policy"] == policy]

    seed_values: dict = {}
    modality_mean_sd: dict = {}
    seed_rows: list[dict] = []
    for task, _label_set in TASKS:
        for policy in POLICIES:
            for modality in MODALITIES:
                group = _group_keys(task, modality, policy)
                for metric in OVERALL_METRICS:
                    values = [{"seed": row["seed"], "value": float(row[metric])} for row in group]
                    seed_values[(task, policy, modality, metric)] = values
                    mean, sd = _mean_sd([item["value"] for item in values])
                    modality_mean_sd[(task, policy, modality, metric)] = {"mean": mean, "sd": sd}
                    for item in values:
                        seed_rows.append({
                            "task": task,
                            "policy": policy,
                            "modality": modality,
                            "metric": metric,
                            "seed": item["seed"],
                            "value": _round(item["value"]),
                        })

    mean_sd_rows: list[dict] = []
    for task, _label_set in TASKS:
        for policy in POLICIES:
            for modality in MODALITIES:
                for metric in OVERALL_METRICS:
                    record = modality_mean_sd[(task, policy, modality, metric)]
                    mean_sd_rows.append({
                        "task": task,
                        "policy": policy,
                        "modality": modality,
                        "metric": metric,
                        "mean": _round(record["mean"]),
                        "sd": _round(record["sd"]),
                        "n_seeds": 3,
                    })

    modality_comparison_rows: list[dict] = []
    for task, _label_set in TASKS:
        for policy in POLICIES:
            for metric in OVERALL_METRICS:
                ranked = sorted(
                    MODALITIES,
                    key=lambda modality: modality_mean_sd[(task, policy, modality, metric)]["mean"],
                    reverse=True,
                )
                best = modality_mean_sd[(task, policy, ranked[0], metric)]["mean"]
                for rank, modality in enumerate(ranked, start=1):
                    record = modality_mean_sd[(task, policy, modality, metric)]
                    modality_comparison_rows.append({
                        "task": task,
                        "policy": policy,
                        "metric": metric,
                        "modality": modality,
                        "mean": _round(record["mean"]),
                        "sd": _round(record["sd"]),
                        "n_seeds": 3,
                        "rank": rank,
                        "delta_vs_best": _round(record["mean"] - best),
                    })

    per_label_mean_sd_rows: list[dict] = []
    per_label_mean_sd: dict = {}
    per_label_ap_mean_sd: dict = {}
    labels_by_task: dict[str, list[str]] = {}
    for task, _label_set in TASKS:
        label_names = labels_by_task[task] = list(freeze["labels_and_data"]["label_sets"][dict(TASKS)[task]]["label_names"]["names"])
        for modality in MODALITIES:
            for index, label in enumerate(label_names):
                rows_f1 = [row for row in per_label_rows if row["task"] == task and row["policy"] == "validation_f1" and row["modality"] == modality and row["label"] == label]
                rows_ap = [row for row in per_label_rows if row["task"] == task and row["policy"] == "fixed_0.5" and row["modality"] == modality and row["label"] == label]
                f1_mean, f1_sd = _mean_sd([row["f1"] for row in rows_f1])
                ap_mean, ap_sd = _mean_sd([row["average_precision"] for row in rows_ap])
                auroc_mean, auroc_sd = _mean_sd([row["auroc"] for row in rows_ap])
                precision_mean, precision_sd = _mean_sd([row["precision"] for row in rows_f1])
                recall_mean, recall_sd = _mean_sd([row["recall"] for row in rows_f1])
                test_positives = int(rows_ap[0]["test_positives"])
                val_positives = int(rows_ap[0]["val_positives"])
                tier = _support_tier(val_positives, test_positives)
                per_label_mean_sd[(task, "validation_f1", modality, label)] = {
                    "f1_mean": f1_mean, "f1_sd": f1_sd,
                    "precision_mean": precision_mean, "precision_sd": precision_sd,
                    "recall_mean": recall_mean, "recall_sd": recall_sd,
                }
                per_label_ap_mean_sd[(task, modality, label)] = {
                    "average_precision_mean": ap_mean, "average_precision_sd": ap_sd,
                    "auroc_mean": auroc_mean, "auroc_sd": auroc_sd,
                    "test_positives": test_positives, "val_positives": val_positives,
                    "support_tier": tier,
                }
                per_label_mean_sd_rows.append({
                    "task": task,
                    "policy": "validation_f1",
                    "modality": modality,
                    "label_index": index,
                    "label": label,
                    "val_positives": val_positives,
                    "test_positives": test_positives,
                    "support_tier": tier,
                    "f1_mean": _round(f1_mean),
                    "f1_sd": _round(f1_sd),
                    "precision_mean": _round(precision_mean),
                    "precision_sd": _round(precision_sd),
                    "recall_mean": _round(recall_mean),
                    "recall_sd": _round(recall_sd),
                    "average_precision_mean": _round(ap_mean),
                    "average_precision_sd": _round(ap_sd),
                    "auroc_mean": _round(auroc_mean),
                    "auroc_sd": _round(auroc_sd),
                    "n_seeds": 3,
                })

    per_label_comparison_rows: list[dict] = []
    for task, _label_set in TASKS:
        for label in labels_by_task[task]:
            row = {"task": task, "label": label}
            ap_values = {modality: per_label_ap_mean_sd[(task, modality, label)]["average_precision_mean"] for modality in MODALITIES}
            f1_values = {modality: per_label_mean_sd[(task, "validation_f1", modality, label)]["f1_mean"] for modality in MODALITIES}
            for modality in MODALITIES:
                row[f"ap_{modality}"] = _round(ap_values[modality])
                row[f"f1_{modality}"] = _round(f1_values[modality])
            row["best_ap_modality"] = max(MODALITIES, key=lambda m: ap_values[m])
            row["worst_ap_modality"] = min(MODALITIES, key=lambda m: ap_values[m])
            row["best_f1_modality"] = max(MODALITIES, key=lambda m: f1_values[m])
            row["ap_gap_best_worst"] = _round(max(ap_values.values()) - min(ap_values.values()))
            per_label_comparison_rows.append(row)

    figures: list[dict] = []
    task_rows = {
        "models": models,
        "modality_mean_sd": modality_mean_sd,
        "seed_values": seed_values,
        "per_label_mean_sd": per_label_mean_sd,
        "per_label_ap_mean_sd": per_label_ap_mean_sd,
    }
    inputs = {
        "freeze_path": relative_to_root(args.freeze),
        "predictions_root": relative_to_root(args.predictions_root),
    }

    outputs: dict = {}
    with atomic_output_dir(args.out_dir) as temporary:
        figures_dir = temporary / "figures"
        figures_dir.mkdir()
        outputs["figures_dir"] = figures_dir

        _write_csv(temporary / "overall_metrics.csv", overall_rows,
                   list(overall_rows[0].keys()))
        _write_csv(temporary / "per_label_metrics.csv", per_label_rows,
                   list(per_label_rows[0].keys()))
        _write_csv(temporary / "seed_comparison.csv", seed_rows,
                   ["task", "policy", "modality", "metric", "seed", "value"])
        _write_csv(temporary / "overall_mean_sd.csv", mean_sd_rows,
                   ["task", "policy", "modality", "metric", "mean", "sd", "n_seeds"])
        _write_csv(temporary / "modality_comparison.csv", modality_comparison_rows,
                   ["task", "policy", "metric", "modality", "mean", "sd", "n_seeds", "rank", "delta_vs_best"])
        _write_csv(temporary / "per_label_modality_mean_sd.csv", per_label_mean_sd_rows,
                   list(per_label_mean_sd_rows[0].keys()))
        _write_csv(temporary / "per_label_modality_comparison.csv", per_label_comparison_rows,
                   list(per_label_comparison_rows[0].keys()))

        figures = _build_figures(outputs, task_rows, labels_by_task)
        report_text = _build_report(freeze, args.freeze_sha256, models, task_rows, labels_by_task,
                                    figures, access_log, inputs)
        (temporary / "report.md").write_text(report_text, encoding="utf-8")

        # Manifest is written last so it can hash every analysis output.
        analysis_outputs = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "analysis_manifest.json":
                analysis_outputs.append({
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                })
        manifest = {
            "phase": "B4.2",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "classification": dict(CLASSIFICATION),
            "freeze": {
                "path": relative_to_root(args.freeze),
                "sha256": args.freeze_sha256,
                "status": freeze["status"],
                "protocol_version": freeze["protocol_version"],
                "protocol_sha256": freeze["protocol_sha256"],
            },
            "inputs": {
                "test_predictions": [
                    {
                        "task": model["task"],
                        "label_set": model["label_set"],
                        "modality": model["modality"],
                        "run_name": model["run_name"],
                        "seed": model["seed"],
                        "manifest_sha256": model["manifest_sha256"],
                        "probabilities_sha256": model["manifest"]["arrays"]["probabilities"]["sha256"],
                        "row_index_sha256": model["manifest"]["arrays"]["row_index"]["sha256"],
                        "labels_sha256": model["manifest"]["arrays"]["labels"]["sha256"],
                        "checkpoint_sha256": model["manifest"]["checkpoint"]["sha256"],
                        "split_path": model["split_path"],
                        "split_sha256": model["split_sha256"],
                        "threshold_path": model["threshold_path"],
                        "threshold_sha256": model["threshold_sha256"],
                    }
                    for model in models
                ],
                "test_access_log": access_log,
                "validation_label_matrix": {
                    "path": freeze["labels_and_data"]["label_sets"]["shared14"]["label_matrix"]["path"],
                    "sha256": freeze["labels_and_data"]["label_sets"]["shared14"]["label_matrix"]["sha256"],
                },
            },
            "counts": {
                "n_models": len(models),
                "n_tasks": 2,
                "n_labels_shared14": len(labels_by_task["Shared14"]),
                "n_labels_uvspec15": len(labels_by_task["UVSPEC15"]),
                "test_samples": TEST_N,
                "n_overall_rows": len(overall_rows),
                "n_per_label_rows": len(per_label_rows),
                "n_seed_rows": len(seed_rows),
            },
            "metrics": {
                "definitions": "training/line_b/metrics.py",
                "overall_metrics": list(OVERALL_METRICS),
                "policies": list(POLICIES),
                "primary_policy": "fixed_0.5",
                "secondary_policy": "validation_f1 (frozen per-label thresholds)",
            },
            "code": {
                "analysis_script": "training/line_b/evaluate_b4_test_metrics.py",
                "analysis_script_sha256": sha256_file(Path(__file__)),
                "metrics_sha256": sha256_file(ROOT / "training/line_b/metrics.py"),
                "plots_sha256": sha256_file(ROOT / "training/line_b/b4_2_plots.py"),
                "common_sha256": sha256_file(ROOT / "training/line_b/common.py"),
                "verifier": "training/line_b/verify_b4_2_analysis.py",
                "verifier_sha256": sha256_file(ROOT / "training/line_b/verify_b4_2_analysis.py")
                if (ROOT / "training/line_b/verify_b4_2_analysis.py").is_file() else None,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "pillow": PIL.__version__,
                "platform": platform.platform(),
            },
            "outputs": analysis_outputs,
        }
        write_json(temporary / "analysis_manifest.json", manifest)

    print(json.dumps({
        "out_dir": relative_to_root(args.out_dir),
        "n_models": len(models),
        "overall_rows": len(overall_rows),
        "per_label_rows": len(per_label_rows),
        "figures": len(figures),
        "n_outputs": len(analysis_outputs) + 1,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
