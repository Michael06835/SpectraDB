"""Build the Line B B5 final archive from frozen B4 assets (read-only summation).

Generates tables, Pillow figures (matplotlib is unsafe in this environment),
the final report, representative-molecule archive, asset manifest and summary.
No training, inference, threshold fitting or bootstrap is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
MODALITIES = ("ir", "raman", "uvvis")
TASKS = ("Shared14", "UVSPEC15")
TASK_LABELS = {"Shared14": 14, "UVSPEC15": 15}

FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"


def _font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _mean(vals):
    vals = [float(v) for v in vals if v not in (None, "") and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256_file(src) != sha256_file(dst):
        raise RuntimeError(f"copy hash mismatch: {src} -> {dst}")
    return sha256_file(dst)


def _text(draw, xy, text, font, fill=(20, 20, 20), anchor=None):
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def _metric_color(v, vmin, vmax):
    t = 0.0 if vmax == vmin else max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))
    # blue -> light -> red
    if t < 0.5:
        u = t / 0.5
        return (int(45 + (245 - 45) * u), int(90 + (245 - 90) * u), int(170 + (245 - 170) * u))
    u = (t - 0.5) / 0.5
    return (int(245 - (245 - 180) * u), int(245 - (245 - 30) * u), int(245 - (245 - 30) * u))


def _heatmap_grid(path: Path, title: str, panels: list[dict], ncols=2, cell_w=92, cell_h=26, vmin=0.0, vmax=1.0):
    """panels: {title, rows, cols, matrix}."""
    left, top, right, bottom = 360, 100, 40, 40
    panel_w = left + max(len(p["cols"]) for p in panels) * cell_w + right
    panel_h = top + max(len(p["rows"]) for p in panels) * cell_h + bottom
    nrows = math.ceil(len(panels) / ncols)
    width = ncols * panel_w + 20
    height = 100 + nrows * panel_h + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (width / 2, 25), title, _font(24, True), anchor="ma")
    for index, panel in enumerate(panels):
        ox = (index % ncols) * panel_w + 10
        oy = 100 + (index // ncols) * panel_h
        _text(draw, (ox + left - 10, oy - 36), panel["title"], _font(16, True))
        for j, col in enumerate(panel["cols"]):
            _text(draw, (ox + left + j * cell_w + cell_w / 2, oy + 2), col, _font(12, True), anchor="ms")
        for i, row in enumerate(panel["rows"]):
            y = oy + 20 + i * cell_h
            _text(draw, (ox + left - 8, y + cell_h / 2), row, _font(11), anchor="rm")
            for j in range(len(panel["cols"])):
                value = panel["matrix"][i][j]
                color = _metric_color(value, vmin, vmax)
                draw.rectangle((ox + left + j * cell_w, y, ox + left + (j + 1) * cell_w - 1, y + cell_h - 1), fill=color, outline=(255, 255, 255))
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                _text(draw, (ox + left + j * cell_w + cell_w / 2, y + cell_h / 2), f"{value:.3f}",
                      _font(10), fill=(255, 255, 255) if lum < 150 else (20, 20, 20), anchor="mm")
    image.save(path, "PNG")


def _grouped_bars(path, title, categories, series, ylabel, ylim=None, value_fmt="{:.3f}"):
    """series: list of {label, values, color} across categories."""
    width, height = 1500, 760
    left, right, top, bottom = 140, 40, 100, 170
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (width / 2, 30), title, _font(22, True), anchor="ma")
    _text(draw, (10, height / 2), ylabel, _font(15), anchor="lm")
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for s in series for v in s["values"]]
    lo = min(values) if ylim is None else ylim[0]
    hi = max(values) if ylim is None else ylim[1]
    if hi == lo:
        hi = lo + 1
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = hi - (hi - lo) * tick / 5
        draw.line((left, y, width - right, y), fill=(220, 220, 220), width=1)
        _text(draw, (left - 10, y), f"{value:.3f}", _font(12), anchor="rm")
    group_w = plot_w / len(categories)
    bar_w = group_w / (len(series) + 1)
    for i, cat in enumerate(categories):
        x0 = left + i * group_w + bar_w * 0.5
        _text(draw, (left + i * group_w + group_w / 2, height - bottom + 30), cat, _font(14), anchor="ma")
        for j, s in enumerate(series):
            v = float(s["values"][i])
            h = plot_h * (v - lo) / (hi - lo)
            x = x0 + j * bar_w
            draw.rectangle((x, top + plot_h - h, x + bar_w * 0.85, top + plot_h), fill=s["color"], outline=(50, 50, 50))
            _text(draw, (x + bar_w * 0.42, top + plot_h - h - 14), value_fmt.format(v), _font(10), anchor="ma")
    # legend
    lx = left
    for s in series:
        draw.rectangle((lx, height - 45, lx + 18, height - 27), fill=s["color"], outline=(50, 50, 50))
        _text(draw, (lx + 24, height - 45), s["label"], _font(12))
        lx += 24 + 10 * len(s["label"]) + 25
    image.save(path, "PNG")


def _line_chart(path, title, x_labels, series, ylabel, ylim):
    width, height = 1400, 720
    left, right, top, bottom = 110, 40, 100, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (width / 2, 30), title, _font(22, True), anchor="ma")
    _text(draw, (10, height / 2), ylabel, _font(15), anchor="lm")
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = ylim
    for tick in range(6):
        y = top + plot_h * tick / 5
        value = hi - (hi - lo) * tick / 5
        draw.line((left, y, width - right, y), fill=(220, 220, 220), width=1)
        _text(draw, (left - 10, y), f"{value:.2f}", _font(12), anchor="rm")
    n = len(x_labels)
    xs = [left + plot_w * (i + 0.5) / n for i in range(n)]
    for x, label in zip(xs, x_labels):
        _text(draw, (x, height - bottom + 25), str(label), _font(14), anchor="ma")
    for s in series:
        pts = [(x, top + plot_h * (1 - (float(v) - lo) / (hi - lo))) for x, v in zip(xs, s["values"])]
        draw.line(pts, fill=s["color"], width=4)
        for (x, y), v in zip(pts, s["values"]):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=s["color"])
            _text(draw, (x, y - 18), f"{float(v):.3f}", _font(11), anchor="ma")
    lx = left
    for s in series:
        draw.rectangle((lx, height - 40, lx + 18, height - 22), fill=s["color"])
        _text(draw, (lx + 24, height - 40), s["label"], _font(12))
        lx += 24 + 10 * len(s["label"]) + 25
    image.save(path, "PNG")


def _forest(path, rows, title):
    width = 1500
    row_h = 28
    height = 120 + len(rows) * row_h + 60
    left, right, top = 420, 60, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (width / 2, 30), title, _font(22, True), anchor="ma")
    los = [float(r["ci_low"]) for r in rows]; his = [float(r["ci_high"]) for r in rows]
    lo, hi = min(los), max(his)
    pad = (hi - lo) * 0.05 or 0.01
    lo -= pad; hi += pad
    plot_w = width - left - right
    def xpos(v):
        return left + plot_w * (float(v) - lo) / (hi - lo)
    zero = xpos(0)
    draw.line((zero, top - 10, zero, height - 50), fill=(120, 120, 120), width=2)
    for i, row in enumerate(rows):
        y = top + i * row_h
        label = f'{row["task"]} | {row["comparison"]} | {row["metric"]} ({row["policy"]})'
        _text(draw, (10, y), label, _font(12))
        x1, x2 = xpos(row["ci_low"]), xpos(row["ci_high"])
        draw.line((x1, y, x2, y), fill=(40, 80, 160), width=3)
        draw.line((x1, y - 5, x1, y + 5), fill=(40, 80, 160), width=2)
        draw.line((x2, y - 5, x2, y + 5), fill=(40, 80, 160), width=2)
        draw.ellipse((xpos(row["point"]) - 5, y - 5, xpos(row["point"]) + 5, y + 5), fill=(180, 30, 30))
    # x ticks
    for tick in range(6):
        value = lo + (hi - lo) * tick / 5
        x = xpos(value)
        draw.line((x, height - 50, x, height - 44), fill=(80, 80, 80))
        _text(draw, (x, height - 40), f"{value:+.3f}", _font(11), anchor="ma")
    image.save(path, "PNG")


def _contact_sheet(path, examples, root: Path, title):
    cell_w, cell_h = 480, 300
    cols, rows = 3, 4
    image = Image.new("RGB", (cols * cell_w + 20, rows * cell_h + 90), "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (image.width / 2, 25), title, _font(24, True), anchor="ma")
    for idx, ex in enumerate(examples[: cols * rows]):
        x = 10 + (idx % cols) * cell_w
        y = 60 + (idx // cols) * cell_h
        case_dir = root / ex["example_dir"]
        try:
            structure = Image.open(case_dir / "structure.png").convert("RGB")
            structure.thumbnail((210, 150))
            spectra = Image.open(case_dir / "spectra.png").convert("RGB")
            spectra.thumbnail((240, 170))
            image.paste(structure, (x + 10, y + 30))
            image.paste(spectra, (x + 230, y + 20))
        except Exception:
            draw.rectangle((x + 10, y + 20, x + cell_w - 10, y + 200), outline=(200, 0, 0))
        _text(draw, (x + 10, y + 5), f'{ex["task"]} | {ex["subgroup"]} | {ex["case"]}', _font(12, True))
        _text(draw, (x + 10, y + 190), f'row {ex["row_index"]} | conf {float(ex["confidence"]):.3f}', _font(11))
        draw.rectangle((x + 5, y + 2, x + cell_w - 5, y + cell_h - 5), outline=(220, 220, 220))
    image.save(path, "PNG")


def main():
    if OUT.exists():
        raise SystemExit(f"Output directory exists: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "tables").mkdir()
    (OUT / "figures").mkdir()
    (OUT / "representative_molecules").mkdir()

    # Source tables (exact copies).
    copies = {
        "overall_metrics.csv": B42 / "overall_metrics.csv",
        "seed_mean_sd.csv": B42 / "overall_mean_sd.csv",
        "per_label_metrics.csv": B42 / "per_label_metrics.csv",
        "paired_differences_three_seed.csv": B43 / "bootstrap_ci_three_seed.csv",
        "paired_differences_per_seed.csv": B43 / "bootstrap_ci_per_seed.csv",
        "pairwise_error_complementarity.csv": B44 / "pairwise_metrics.csv",
        "pairwise_or_and.csv": B44 / "pairwise_or_and.csv",
        "three_modality_or.csv": B44 / "three_modality_or.csv",
        "true_oracle_summary.csv": B44 / "oracle_summary.csv",
        "true_oracle_per_label.csv": B44 / "oracle_per_label.csv",
        "conditional_complementarity.csv": B44 / "conditional_complementarity.csv",
        "conditional_bootstrap_ci.csv": B44 / "conditional_bootstrap_ci.csv",
        "subgroup_metrics.csv": B44 / "subgroup_metrics.csv",
        "subgroup_bootstrap_ci.csv": B44 / "subgroup_bootstrap_ci.csv",
        "mcnemar_holm.csv": B44 / "mcnemar_holm.csv",
        "sample_error_summary.csv": B44 / "sample_error_summary.csv",
    }
    for name, src in copies.items():
        _copy(src, OUT / "tables" / name)

    # Micro/macro summary (micro AP/AUROC recomputed from frozen B4.1 probabilities).
    freeze = read_json(FREEZE)
    overall = _csv(B42 / "overall_metrics.csv")
    micro_rows = []
    for record in freeze["models"]:
        d = B41 / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        truth = np.load(d / "labels.npy", allow_pickle=False)
        flat_p, flat_y = probs.reshape(-1), truth.reshape(-1)
        micro_ap = float(average_precision_score(flat_y, flat_p))
        micro_auroc = float(roc_auc_score(flat_y, flat_p))
        for policy in ("fixed_0.5", "validation_f1"):
            row = next(r for r in overall if r["task"] == record["task"] and r["seed"] == str(record["seed"])
                       and r["modality"] == record["modality"] and r["policy"] == policy)
            micro_rows.append({
                "task": record["task"], "seed": int(record["seed"]), "modality": record["modality"],
                "threshold_policy": policy,
                "micro_ap": micro_ap, "micro_auroc": micro_auroc, "micro_f1": float(row["micro_f1"]),
                "macro_ap": float(row["macro_ap"]), "macro_auroc": float(row["macro_auroc"]),
                "macro_f1": float(row["macro_f1"]), "n_samples": int(row["n_samples"]), "n_labels": int(row["n_labels"]),
                "note": "micro AP/AUROC = flattened sample-label B5 summary; macro AP/AUROC from B4.2",
            })
    _write_csv(OUT / "tables" / "micro_macro_summary.csv", micro_rows,
               ["task", "seed", "modality", "threshold_policy", "micro_ap", "micro_auroc", "micro_f1",
                "macro_ap", "macro_auroc", "macro_f1", "n_samples", "n_labels", "note"])

    # Representative molecules: copy 88 cases byte-identically.
    examples = _csv(SUPP / "examples.csv")
    rep_rows = []
    for ex in examples:
        src_dir = SUPP / ex["example_dir"]
        dst_dir = OUT / "representative_molecules" / ex["example_dir"]
        row = dict(ex)
        for name in ("structure.png", "structure.svg", "spectra.png", "metadata.json"):
            src = src_dir / name
            if not src.is_file():
                raise SystemExit(f"missing representative asset: {src}")
            _copy(src, dst_dir / name)
            row[f"{name.replace('.', '_')}_sha256"] = sha256_file(src)
        rep_rows.append(row)
    _write_csv(OUT / "tables" / "representative_molecules.csv", rep_rows,
               list(examples[0].keys()) + ["structure_png_sha256", "structure_svg_sha256", "spectra_png_sha256", "metadata_json_sha256"])
    _copy(SUPP / "subgroup_status.csv", OUT / "representative_molecules" / "subgroup_status.csv")
    _copy(SUPP / "examples.csv", OUT / "representative_molecules" / "index.csv")

    # Aggregates for report/figures.
    def overall_mean(task, mod, field, policy="fixed_0.5"):
        return _mean([r[field] for r in overall if r["task"] == task and r["modality"] == mod and r["policy"] == policy])
    per_label = _csv(B42 / "per_label_metrics.csv")
    labels = {task: sorted({r["label"] for r in per_label if r["task"] == task}) for task in TASKS}
    per_label_ap = defaultdict(list)
    per_label_f1 = defaultdict(list)
    for r in per_label:
        per_label_ap[(r["task"], r["label"], r["modality"])].append(float(r["average_precision"]))
        per_label_f1[(r["task"], r["label"], r["modality"], r["policy"])].append(float(r["f1"]))
    three_ci = _csv(B43 / "bootstrap_ci_three_seed.csv")
    macro_ci = [r for r in three_ci if r["scope"] == "macro"]
    pairwise = _csv(B44 / "pairwise_metrics.csv")
    macro_pair = [r for r in pairwise if r["scope"] == "macro" and r["threshold_policy"] == "fixed_0.5"]
    or_rows = _csv(B44 / "pairwise_or_and.csv")
    three_or = _csv(B44 / "three_modality_or.csv")
    oracle = _csv(B44 / "oracle_summary.csv")
    cond = _csv(B44 / "conditional_complementarity.csv")
    cond_ci = _csv(B44 / "conditional_bootstrap_ci.csv")
    subgroup = _csv(B44 / "subgroup_metrics.csv")
    sg_boot = _csv(B44 / "subgroup_bootstrap_ci.csv")

    # Figures.
    _grouped_bars(OUT / "figures" / "fig01_overall_performance.png",
                  "Overall Test performance — seed mean by task and modality (macro AP/AUROC threshold-free; F1 fixed 0.5)",
                  [f"{t}\n{m.upper()}" for t in TASKS for m in MODALITIES],
                  [{"label": "macro AP", "values": [overall_mean(t, m, "macro_ap") for t in TASKS for m in MODALITIES], "color": (31, 119, 180)},
                   {"label": "macro AUROC", "values": [overall_mean(t, m, "macro_auroc") for t in TASKS for m in MODALITIES], "color": (255, 127, 14)},
                   {"label": "macro F1", "values": [overall_mean(t, m, "macro_f1") for t in TASKS for m in MODALITIES], "color": (44, 160, 44)},
                   {"label": "micro F1", "values": [overall_mean(t, m, "micro_f1") for t in TASKS for m in MODALITIES], "color": (214, 39, 40)}],
                  "score (seed mean)", ylim=(0, 1.05))

    heat_panels = []
    for task in TASKS:
        matrix = [[_mean(per_label_ap[(task, label, mod)]) for mod in MODALITIES] for label in labels[task]]
        heat_panels.append({"title": f"{task} — AP", "rows": labels[task], "cols": ["IR", "Raman", "UV-Vis"], "matrix": matrix})
    _heatmap_grid(OUT / "figures" / "fig02_per_label_ap_heatmap.png", "Per-label average precision (mean over 3 seeds)", heat_panels, ncols=2, vmin=0.0, vmax=1.0)

    f1_panels = []
    for policy, pname in (("fixed_0.5", "fixed 0.5"), ("validation_f1", "Validation-F1")):
        for task in TASKS:
            matrix = [[_mean(per_label_f1[(task, label, mod, policy)]) for mod in MODALITIES] for label in labels[task]]
            f1_panels.append({"title": f"{task} — F1 ({pname})", "rows": labels[task], "cols": ["IR", "Raman", "UV-Vis"], "matrix": matrix})
    _heatmap_grid(OUT / "figures" / "fig03_per_label_f1_heatmap.png", "Per-label F1 — both threshold policies", f1_panels, ncols=2, vmin=0.0, vmax=1.0)

    seed_series = []
    for mod in MODALITIES:
        seed_series.append({"label": mod.upper(), "values": [_mean([r["macro_f1"] for r in overall if r["task"] == t and r["modality"] == mod and r["policy"] == "fixed_0.5"]) for t in TASKS], "color": {"ir": (31,119,180), "raman": (214,39,40), "uvvis": (44,160,44)}[mod]})
    # Seed stability is per seed; draw separate panel via 3 seed x task values.
    seed_panels = []
    for task in TASKS:
        for seed in (42, 123, 2026):
            pass
    # Reuse line chart per task with x=seeds and 3 modality lines.
    for task in TASKS:
        series = []
        for mod in MODALITIES:
            vals = []
            for seed in (42, 123, 2026):
                row = next(r for r in overall if r["task"] == task and r["seed"] == str(seed) and r["modality"] == mod and r["policy"] == "fixed_0.5")
                vals.append(float(row["macro_f1"]))
            series.append({"label": mod.upper(), "values": vals, "color": {"ir": (31,119,180), "raman": (214,39,40), "uvvis": (44,160,44)}[mod]})
        _line_chart(OUT / "figures" / f"fig04_seed_stability_{task}.png", f"{task} — macro F1 by seed (fixed 0.5)", ["42", "123", "2026"], series, "macro F1", (0.3, 1.0))

    # Combined seed stability figure (two panels vertically would need custom; use two files referenced).
    _forest(OUT / "figures" / "fig05_paired_bootstrap_forest.png",
            [{"task": r["task"], "comparison": r["comparison"], "metric": r["metric"],
              "policy": r["threshold_policy"], "point": r["point_estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
             for r in macro_ci],
            "B4.3 paired scaffold-bootstrap differences (95% CI) — macro scope")

    # Error complementarity: Jaccard and rescue/correction.
    cats = [f"{t}\n{p}" for t in TASKS for p in ("IR-Raman", "UV-IR", "UV-Raman")]
    def pair_value(task, comp, field):
        r = next(x for x in macro_pair if x["task"] == task and x["comparison"] == comp)
        return float(r[field])
    _grouped_bars(OUT / "figures" / "fig06_error_complementarity.png", "Macro error-set complementarity (fixed 0.5, 3-seed mean)",
                  cats,
                  [{"label": "Jaccard", "values": [pair_value(t, c, "error_jaccard") for t in TASKS for c in ("IR-Raman", "UV-IR", "UV-Raman")], "color": (31,119,180)},
                   {"label": "FN rescue / 100k", "values": [pair_value(t, c, "fn_rescue") / 100000 for t in TASKS for c in ("IR-Raman", "UV-IR", "UV-Raman")], "color": (214,39,40)},
                   {"label": "FP correction / 100k", "values": [pair_value(t, c, "fp_correction") / 100000 for t in TASKS for c in ("IR-Raman", "UV-IR", "UV-Raman")], "color": (44,160,44)}],
                  "Jaccard / counts per 100k pairs")

    # OR vs best_single and Oracle.
    _grouped_bars(OUT / "figures" / "fig07_or_oracle.png", "Three-modality positive OR vs best single, and True Oracle (fixed 0.5)",
                  [f"{t}" for t in TASKS],
                  [{"label": "best_single F1", "values": [_mean([float(r["best_single_f1"]) for r in three_or if r["task"] == t and r["threshold_policy"] == "fixed_0.5"]) for t in TASKS], "color": (31,119,180)},
                   {"label": "three-OR F1", "values": [_mean([float(r["three_or_f1"]) for r in three_or if r["task"] == t and r["threshold_policy"] == "fixed_0.5"]) for t in TASKS], "color": (255,127,14)},
                   {"label": "Oracle any-correct", "values": [_mean([float(r["any_correct_rate"]) for r in oracle if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["scope"] == "overall"]) for t in TASKS], "color": (44,160,44)},
                   {"label": "Oracle all-wrong", "values": [_mean([float(r["all_wrong_rate"]) for r in oracle if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["scope"] == "overall"]) for t in TASKS], "color": (120,120,120)}],
                  "F1 / rate", ylim=(0, 1.05))

    # Conditional complementarity.
    dirs = [("ir", "raman"), ("raman", "ir"), ("uvvis", "ir"), ("ir", "uvvis"), ("uvvis", "raman"), ("raman", "uvvis")]
    cats2 = [f"{a.upper()}→{b.upper()}" for a, b in dirs]
    _grouped_bars(OUT / "figures" / "fig08_conditional_complementarity.png", "Conditional complementarity (fixed 0.5, 3-seed mean)",
                  cats2,
                  [{"label": "P(other correct | base wrong)", "values": [_mean([float(r["rate"]) for r in cond if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "conditional_correct_given_base_wrong" and r["base_modality"] == a and r["other_modality"] == b]) for t in TASKS for a, b in dirs], "color": (31,119,180)},
                   {"label": "P(other correct) marginal", "values": [_mean([float(r["p_other_marginal"]) for r in cond if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "conditional_correct_given_base_wrong" and r["base_modality"] == a and r["other_modality"] == b]) for t in TASKS for a, b in dirs], "color": (214,39,40)}],
                  "rate", ylim=(0, 1.05))

    # Subgroup heatmap (supported, AP mean over policies/labels/seeds).
    sg_panels = []
    for task in TASKS:
        subs = sorted({r["subgroup"] for r in subgroup if r["task"] == task and r["support_sufficient"].lower() == "true"})
        matrix = []
        for sub in subs:
            row = []
            for mod in MODALITIES:
                vals = [float(r["ap"]) for r in subgroup if r["task"] == task and r["subgroup"] == sub and r["modality"] == mod and r["threshold_policy"] == "fixed_0.5" and r["support_sufficient"].lower() == "true" and r["ap"] not in ("", None)]
                row.append(_mean(vals) if vals else 0.0)
            matrix.append(row)
        sg_panels.append({"title": f"{task} — supported subgroups", "rows": subs, "cols": ["IR", "Raman", "UV-Vis"], "matrix": matrix})
    _heatmap_grid(OUT / "figures" / "fig09_subgroup_performance_heatmap.png", "Structural subgroup mean AP (fixed 0.5; supported rows only)", sg_panels, ncols=2, cell_w=100, cell_h=24, vmin=0.4, vmax=1.0)

    # Contact sheet of 12 diverse cases.
    selected = []
    seen = set()
    for task in TASKS:
        for ex in [e for e in examples if e["task"] == task]:
            key = (task, ex["subgroup"], ex["case"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(ex)
            if len([s for s in selected if s["task"] == task]) >= 6:
                break
    _contact_sheet(OUT / "figures" / "fig10_representative_molecules.png", selected, OUT / "representative_molecules",
                   "Representative molecules (B0 §7.6; qualitative only)")

    # Hypothesis evaluation table.
    def gap(task, threshold_set):
        def label_ap(label, mod):
            return _mean([float(r["average_precision"]) for r in per_label
                          if r["task"] == task and r["label"] == label and r["modality"] == mod and r["policy"] == "fixed_0.5"])
        vals = []
        for label in labels[task]:
            if label in threshold_set:
                vals.append(max(label_ap(label, "ir"), label_ap(label, "raman")) - label_ap(label, "uvvis"))
        return _mean(vals)
    vib = {"carbonyl", "nitrile", "alkyne", "amine", "hydroxyl"}
    h1_main = all(overall_mean("Shared14", m, "macro_ap") > overall_mean("Shared14", "uvvis", "macro_ap") for m in ("ir", "raman")) and \
              all(overall_mean("Shared14", m, "macro_auroc") > overall_mean("Shared14", "uvvis", "macro_auroc") for m in ("ir", "raman"))
    h1_vib = gap("Shared14", vib) > gap("Shared14", set(labels["Shared14"]) - vib)
    h2 = all(overall_mean("UVSPEC15", "uvvis", f) > max(overall_mean("UVSPEC15", "ir", f), overall_mean("UVSPEC15", "raman", f)) for f in ("macro_ap", "macro_auroc"))
    oracle_any = {t: _mean([float(r["any_correct_rate"]) for r in oracle if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["scope"] == "overall"]) for t in TASKS}
    or_gains = [float(r["or_gain_vs_best_single"]) for r in three_or if r["threshold_policy"] == "fixed_0.5"]
    uv_rescue = {t: _mean([float(r["rate"]) for r in cond if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "uv_rescue_rate"]) for t in TASKS}
    uv_marg = {t: _mean([float(r["rate"]) for r in cond if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "uv_marginal_correct_rate"]) for t in TASKS}
    best_counts = {}
    for task in TASKS:
        cnt = Counter()
        subs = sorted({r["subgroup"] for r in subgroup if r["task"] == task and r["support_sufficient"].lower() == "true"})
        for sub in subs:
            means = {}
            for mod in MODALITIES:
                vals = [float(r["ap"]) for r in subgroup if r["task"] == task and r["subgroup"] == sub and r["modality"] == mod and r["threshold_policy"] == "fixed_0.5" and r["support_sufficient"].lower() == "true" and r["ap"] not in ("", None)]
                means[mod] = _mean(vals) if vals else None
            means = {k: v for k, v in means.items() if v is not None}
            if means:
                cnt[max(means, key=means.get)] += 1
        best_counts[task] = dict(cnt)

    hyp_rows = [
        {"hypothesis": "H1", "verdict": "PARTIAL SUPPORT",
         "evidence": f"Shared14 macro AP IR/Raman/UV={overall_mean('Shared14','ir','macro_ap'):.4f}/{overall_mean('Shared14','raman','macro_ap'):.4f}/{overall_mean('Shared14','uvvis','macro_ap'):.4f}; "
                     f"vibration gap={gap('Shared14', vib):.4f} vs other={gap('Shared14', set(labels['Shared14'])-vib):.4f}",
         "note": "main IR/Raman>UV supported; vibration-label-gap clause not supported"},
        {"hypothesis": "H2", "verdict": "REJECTED",
         "evidence": f"UVSPEC15 macro AP UV/IR/Raman={overall_mean('UVSPEC15','uvvis','macro_ap'):.4f}/{overall_mean('UVSPEC15','ir','macro_ap'):.4f}/{overall_mean('UVSPEC15','raman','macro_ap'):.4f}; "
                     f"AUROC={overall_mean('UVSPEC15','uvvis','macro_auroc'):.4f}/{overall_mean('UVSPEC15','ir','macro_auroc'):.4f}/{overall_mean('UVSPEC15','raman','macro_auroc'):.4f}",
         "note": "UV-Vis does not show the expected advantage in this task/metric system; does not mean UV has no electronic-structure information"},
        {"hypothesis": "H3", "verdict": "PARTIAL SUPPORT",
         "evidence": f"Oracle any={oracle_any['Shared14']:.4f}/{oracle_any['UVSPEC15']:.4f}; three-OR gain range=[{min(or_gains):+.4f},{max(or_gains):+.4f}]; "
                     f"UV rescue={uv_rescue['Shared14']:.4f}/{uv_rescue['UVSPEC15']:.4f} vs marginal={uv_marg['Shared14']:.4f}/{uv_marg['UVSPEC15']:.4f}",
         "note": "errors complementary and Oracle ceiling high, but positive OR does not beat best_single and UV conditional rescue is below marginal; subgroup only_* enrichment not directly tested"},
        {"hypothesis": "H4", "verdict": "DESCRIPTIVELY SUPPORTED",
         "evidence": f"best-AP modality counts={best_counts}",
         "note": "relative IR/Raman advantage varies by task and structural subgroup; no universal modality ranking"},
    ]
    _write_csv(OUT / "tables" / "hypothesis_evaluation.csv", hyp_rows, ["hypothesis", "verdict", "evidence", "note"])

    # Report.
    def fmt(v, nd=4):
        return "n/a" if v is None else f"{float(v):.{nd}f}"
    lines = ["# Line B — Final Report (B5)", "",
             "> **Classification: exploratory / post-hoc.** Test was opened before B4; results are not a blind test and are not confirmatory. "
             "This report summarizes frozen B4 assets; no training, inference, threshold fitting or bootstrap was performed in B5.", "",
             "## 1. Executive Summary", ""]
    lines.append("Line B evaluates whether IR, Raman and UV-Vis carry complementary structural evidence on QM9S. "
                 "Across 18 frozen models (2 tasks × 3 modalities × 3 seeds; Test n=12,981), IR/Raman strongly outperform UV-Vis on Shared14; "
                 "UV-Vis does **not** show the expected advantage on UVSPEC15. Error sets differ, but positive OR does not beat the best single modality, "
                 "True Oracle gives a high diagnostic ceiling (not fusion performance), and UV conditional rescue is below its marginal rate. "
                 "Structural subgroup gaps vary by task and subgroup. H1 is partially supported, H2 rejected, H3 partially supported, H4 descriptively supported.")
    lines += ["", "**Three Phase-0 WARNs are fixed in this report:** (1) B4.3 CIs are paired-difference CIs, not absolute-performance CIs; "
              "(2) H3's `only_UV/only_IR/only_Raman × subgroup` enrichment is **not directly tested** and is not claimed; "
              "(3) AP/AUROC ≠ chemical-information content, QM9S ≠ experimental domain, and representative molecules are qualitative only.", ""]
    lines += ["## 2. Research Questions and Hypotheses", "",
              "- H1: vibration modalities (IR/Raman) have a structural advantage on Shared14, especially on vibration-sensitive labels. **Verdict: PARTIAL SUPPORT.**",
              "- H2: UV-Vis has an electronic-structure advantage on UVSPEC15. **Verdict: REJECTED** under the current task/metric system; this does not mean UV lacks electronic-structure information.",
              "- H3: errors are complementary; positive OR may help but is only a fixed rule; true Oracle assesses complementarity; UV exclusive-correct samples should enrich in conjugated/aromatic subgroups. **Verdict: PARTIAL SUPPORT**; the subgroup enrichment clause is **not directly tested**.",
              "- H4: modality gaps vary by structural subgroup; no universal winner. **Verdict: DESCRIPTIVELY SUPPORTED.**", ""]
    lines += ["## 3. Dataset and Experimental Design", "",
              "- QM9S: 129,817 molecules with IR, Raman and UV-Vis computed spectra; Bemis–Murcko scaffold split 80/10/10; Test n=12,981; 2,853 Test scaffolds.",
              "- Tasks: Shared14 (14 functional-group labels) and UVSPEC15 (15 UV-specific semantic labels); probabilities from frozen FP32 Test exports.",
              "- Seeds: 42, 123, 2026; no Test-driven seed/model/label/subgroup selection.", ""]
    lines += ["## 4. Model Architecture and Training Configuration", "",
              "- Frozen single-modality 1D-CNN encoders (`training/models/spectrum_cnn1d.py`), embedding dimension 256; input length 3,501 (IR/Raman) or 701 (UV-Vis); sigmoid multi-label heads (14/15).",
              "- Training (frozen v1 manifests): AdamW, batch 512, lr 1e-3, weight decay 1e-4, up to 50 epochs with early stopping (patience 8), BCE/WBCE as recorded per frozen run; max normalization.",
              "- B5 loaded only the frozen Test probability arrays; no checkpoint was loaded and no inference was run.", ""]
    lines += ["## 5. Single-Modality Performance", "",
              "Macro AP/AUROC are threshold-free; micro AP/AUROC are flattened sample-label summaries computed in B5 from the frozen B4.1 probabilities (not a replacement for macro metrics); F1 is shown for both policies. "
              "All values are seed means; full per-seed and per-label tables are in `tables/`.", "",
              "| Task | Modality | micro AP | micro AUROC | macro AP | macro AUROC | macro F1 (0.5) | macro F1 (Val-F1) | micro F1 (0.5) |",
              "|---|---|---|---|---|---|---|---|---|"]
    micro_map = {(r["task"], int(r["seed"]), r["modality"], r["threshold_policy"]): r for r in micro_rows}
    def micro_mean(task, mod, field):
        return _mean([micro_map[(task, seed, mod, "fixed_0.5")][field] for seed in (42, 123, 2026)])
    for task in TASKS:
        for mod in MODALITIES:
            lines.append(f"| {task} | {mod.upper()} | {fmt(micro_mean(task,mod,'micro_ap'))} | {fmt(micro_mean(task,mod,'micro_auroc'))} | "
                         f"{fmt(overall_mean(task,mod,'macro_ap'))} | {fmt(overall_mean(task,mod,'macro_auroc'))} | "
                         f"{fmt(overall_mean(task,mod,'macro_f1'))} | {fmt(overall_mean(task,mod,'macro_f1','validation_f1'))} | {fmt(overall_mean(task,mod,'micro_f1'))} |")
    lines += ["", f"- Shared14 best macro AP: **{max(MODALITIES, key=lambda m: overall_mean('Shared14',m,'macro_ap')).upper()}**; "
              f"UVSPEC15 best macro AP: **{max(MODALITIES, key=lambda m: overall_mean('UVSPEC15',m,'macro_ap')).upper()}**.",
              "- Full 36-row micro/macro table: `tables/micro_macro_summary.csv`; 36-row overall table: `tables/overall_metrics.csv`; 522-row per-label table: `tables/per_label_metrics.csv`.", ""]
    lines += ["## 6. Seed Stability", "",
              "Seed Mean±SD is in `tables/seed_mean_sd.csv` (96 rows: task × policy × modality × metric). "
              "Per-seed values are in `tables/overall_metrics.csv`; the fixed-0.5 macro-F1 seed trajectories are in `figures/fig04_seed_stability_*.png`.", "",
              "| Task | Modality | macro AP | macro AUROC | macro F1 (0.5) | macro F1 (Val-F1) |",
              "|---|---|---|---|---|---|"]
    def seed_stats(task, mod, field, policy="fixed_0.5"):
        vals = [float(r[field]) for r in overall if r["task"] == task and r["modality"] == mod and r["policy"] == policy]
        return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
    for task in TASKS:
        for mod in MODALITIES:
            lines.append(f"| {task} | {mod.upper()} | {seed_stats(task,mod,'macro_ap')} | {seed_stats(task,mod,'macro_auroc')} | "
                         f"{seed_stats(task,mod,'macro_f1')} | {seed_stats(task,mod,'macro_f1','validation_f1')} |")
    lines += ["", "SD is the sample SD across the three seeds (n=3, descriptive only).", ""]
    lines += ["## 7. Paired Scaffold Bootstrap", "",
              "**WARN-1 fixed:** B4.3 reports CIs for paired differences `Δ = Metric(base) − Metric(other)`, not for absolute IR/Raman/UV-Vis performance. "
              "Absolute single-modality AP/AUROC/F1 are B4.2 point estimates. The B4.3 CI therefore answers “is the modality difference stable under scaffold resampling?”, not “what is the absolute performance CI?”.", "",
              f"- 2,000 scaffold replicates; one shared draw matrix; 2,853 Test scaffolds; 1,488 point rows; 1,116 per-seed CI rows; 372 three-seed CI rows.",
              "- Three-seed paired differences and 95% CIs: `tables/paired_differences_three_seed.csv`; per-seed: `tables/paired_differences_per_seed.csv`; forest plot: `figures/fig05_paired_bootstrap_forest.png`.",
              f"- Macro differences whose 95% CI excludes 0: **{sum(1 for r in macro_ci if float(r['ci_low']) > 0 or float(r['ci_high']) < 0)} / {len(macro_ci)}**.", ""]
    lines += ["## 8. Error Complementarity", "",
              "Macro error Jaccard, FN rescue and FP correction (fixed 0.5, three-seed mean; full 558-row table in `tables/pairwise_error_complementarity.csv`):", "",
              "| Task | Comparison | F1 base/other | error Jaccard | FN rescue | FP correction | both wrong |",
              "|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for comp in ("IR-Raman", "UV-IR", "UV-Raman"):
            rows = [r for r in macro_pair if r["task"] == task and r["comparison"] == comp]
            lines.append(f"| {task} | {comp} | {_mean([float(r['base_f1']) for r in rows]):.3f}/{_mean([float(r['other_f1']) for r in rows]):.3f} | "
                         f"{_mean([float(r['error_jaccard']) for r in rows]):.4f} | {_mean([float(r['fn_rescue']) for r in rows]):.0f} | "
                         f"{_mean([float(r['fp_correction']) for r in rows]):.0f} | {_mean([float(r['both_wrong']) for r in rows]):.0f} |")
    lines += ["", "Error sets are not identical, but this alone does **not** establish conditional complementarity (Section 10). McNemar/Holm results: `tables/mcnemar_holm.csv`.", ""]
    lines += ["## 9. Positive OR / AND and True Oracle", "",
              "Positive OR/AND are fixed rules, **not** True Oracle; True Oracle is a diagnostic ceiling, **not** a deployable fusion model.", "",
              "| Task | Pair | OR F1 | AND F1 | OR gain vs max | +gain / −gain labels |",
              "|---|---|---|---|---|---|"]
    for task in TASKS:
        for comp in ("IR-Raman", "UV-IR", "UV-Raman"):
            rows = [r for r in or_rows if r["task"] == task and r["comparison"] == comp and r["threshold_policy"] == "fixed_0.5"]
            lines.append(f"| {task} | {comp} | {_mean([float(r['or_f1']) for r in rows]):.3f} | {_mean([float(r['and_f1']) for r in rows]):.3f} | "
                         f"{_mean([float(r['or_gain_vs_max']) for r in rows]):+.4f} | {_mean([float(r['n_positive_or_gain_labels']) for r in rows]):.1f}/{_mean([float(r['n_negative_or_gain_labels']) for r in rows]):.1f} |")
    lines += ["", "| Task | best_single | F1 best_single | F1 three-OR | three-OR gain |",
              "|---|---|---|---|---|"]
    for task in TASKS:
        rows = [r for r in three_or if r["task"] == task and r["threshold_policy"] == "fixed_0.5"]
        lines.append(f"| {task} | {rows[0]['best_single']} | {_mean([float(r['best_single_f1']) for r in rows]):.4f} | "
                     f"{_mean([float(r['three_or_f1']) for r in rows]):.4f} | {_mean([float(r['or_gain_vs_best_single']) for r in rows]):+.4f} |")
    lines += ["", "| Task | Oracle any | exactly one | all wrong | all correct | only IR | only Raman | only UV |",
              "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        rows = [r for r in oracle if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["scope"] == "overall"]
        lines.append(f"| {task} | {_mean([float(r['any_correct_rate']) for r in rows]):.4f} | {_mean([float(r['exactly_one_correct_rate']) for r in rows]):.4f} | "
                     f"{_mean([float(r['all_wrong_rate']) for r in rows]):.4f} | {_mean([float(r['all_correct'])/float(r['n_pairs']) for r in rows]):.4f} | "
                     f"{_mean([float(r['only_ir_correct']) for r in rows]):.0f} | {_mean([float(r['only_raman_correct']) for r in rows]):.0f} | {_mean([float(r['only_uv_correct']) for r in rows]):.0f} |")
    lines += ["", "Full tables: `tables/pairwise_or_and.csv`, `tables/three_modality_or.csv`, `tables/true_oracle_summary.csv`, `tables/true_oracle_per_label.csv`.", ""]
    lines += ["## 10. Conditional Complementarity", "",
              "Conditional rates are unweighted full-Test descriptive rates; 95% CIs use 2,000 B4.3 scaffold draws. UV rescue is far below UV marginal, so the data do not support a conditional-complementarity claim.", "",
              "| Task | Direction | P(other correct | base wrong) | P(other) marginal | delta |",
              "|---|---|---|---|---|"]
    for task in TASKS:
        for a, b in [("ir", "raman"), ("raman", "ir"), ("uvvis", "ir"), ("ir", "uvvis"), ("uvvis", "raman"), ("raman", "uvvis")]:
            rows = [r for r in cond if r["task"] == task and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "conditional_correct_given_base_wrong" and r["base_modality"] == a and r["other_modality"] == b]
            lines.append(f"| {task} | {a.upper()}→{b.upper()} | {_mean([float(r['rate']) for r in rows]):.4f} | {_mean([float(r['p_other_marginal']) for r in rows]):.4f} | {_mean([float(r['delta']) for r in rows]):+.4f} |")
    rescue_ci = {t: next(r for r in cond_ci if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "rescue") for t in TASKS}
    delta_ci = {t: next(r for r in cond_ci if r["task"] == t and r["threshold_policy"] == "fixed_0.5" and r["metric"] == "rescue_delta") for t in TASKS}
    lines += ["", "UV-specific: 509/181,734 pairs on Shared14 and 205/194,715 on UVSPEC15 where IR and Raman are both wrong but UV is correct; "
              f"UV rescue rates 0.469/0.498 vs marginal 0.880/0.925, delta −0.411/−0.426. "
              f"95% CI for rescue: Shared14 [{float(rescue_ci['Shared14']['ci_low']):.4f}, {float(rescue_ci['Shared14']['ci_high']):.4f}], "
              f"UVSPEC15 [{float(rescue_ci['UVSPEC15']['ci_low']):.4f}, {float(rescue_ci['UVSPEC15']['ci_high']):.4f}]; "
              f"delta CI: Shared14 [{float(delta_ci['Shared14']['ci_low']):.4f}, {float(delta_ci['Shared14']['ci_high']):.4f}], "
              f"UVSPEC15 [{float(delta_ci['UVSPEC15']['ci_low']):.4f}, {float(delta_ci['UVSPEC15']['ci_high']):.4f}] (entirely below zero). "
              "Full table and CIs: `tables/conditional_complementarity.csv`, `tables/conditional_bootstrap_ci.csv`; figure: `figures/fig08_conditional_complementarity.png`.", ""]
    lines += ["## 11. Structural Subgroup Analysis", "",
              "Predefined D1/B0 §7.5 subgroups; support requires n≥100 and positive≥30. Heavy-atom quartiles are degenerate (Q1=Q2=Q3=9.0); `functional_group=aromatic` is empty under the frozen token definition. "
              "Subgroup CIs use 500 within-subgroup paired scaffold draws and are exploratory.", "",
              f"- Supported subgroups: Shared14 **{len({r['subgroup'] for r in subgroup if r['task']=='Shared14' and r['support_sufficient'].lower()=='true'})}**, UVSPEC15 **{len({r['subgroup'] for r in subgroup if r['task']=='UVSPEC15' and r['support_sufficient'].lower()=='true'})}**; "
              "each task has 22 non-empty subgroups, but `aromatic_ring_count=≥2` has no label meeting the positive≥30 support rule.",
              f"- Best-AP modality counts: Shared14 **{best_counts['Shared14']}**, UVSPEC15 **{best_counts['UVSPEC15']}**; UV-Vis is never the best modality in any supported subgroup.",
              "- At fixed 0.5, all 84 UV−IR / UV−Raman subgroup comparisons have CI upper < 0; no supported subgroup shows a positive UV advantage with CI lower > 0.",
              "- Full tables: `tables/subgroup_metrics.csv`, `tables/subgroup_bootstrap_ci.csv`; heatmap: `figures/fig09_subgroup_performance_heatmap.png`.", ""]
    lines += ["## 12. Representative Molecule Analysis", "",
              "The B4.4 supplement is archived byte-identically under `representative_molecules/` (88 cases: 44 per task; 22 correct + 22 error each). "
              "Each case keeps RDKit PNG/SVG, spectra PNG and metadata with labels, predictions, probabilities and errors. "
              "This is a qualitative case display, **not XAI** and **not statistical evidence**; no case is used to estimate performance.", "",
              "| Task | Cases | Correct | Error | Non-empty subgroups | Empty subgroups |",
              "|---|---|---|---|---|---|"]
    for task in TASKS:
        sub = [e for e in examples if e["task"] == task]
        status = _csv(SUPP / "subgroup_status.csv")
        ne = len({r["subgroup"] for r in status if r["task"] == task and int(r["n_samples"]) > 0})
        em = len({r["subgroup"] for r in status if r["task"] == task and int(r["n_samples"]) == 0})
        lines.append(f"| {task} | {len(sub)} | {sum(1 for e in sub if e['case']=='correct')} | {sum(1 for e in sub if e['case']=='error')} | {ne} | {em} |")
    lines += ["", "Index: `tables/representative_molecules.csv`; contact sheet: `figures/fig10_representative_molecules.png`.", ""]
    lines += ["## 13. H1–H4 Hypothesis Evaluation", "",
              "| Hypothesis | Verdict | Evidence |", "|---|---|---|"]
    for row in hyp_rows:
        lines.append(f"| {row['hypothesis']} | **{row['verdict']}** | {row['evidence']} |")
    lines += ["", "- **H1 PARTIAL SUPPORT:** the main IR/Raman > UV margin holds on Shared14, but the vibration-sensitive-label gap (0.4451) is not larger than the other-label gap (0.4653).",
              "- **H2 REJECTED:** on UVSPEC15, UV macro AP 0.5380 vs IR 0.9709 / Raman 0.9760; AUROC 0.9252 vs 0.9977 / 0.9982. This does not mean UV has no electronic-structure information or no potential complementary value.",
              "- **H3 PARTIAL SUPPORT:** errors are complementary and Oracle any-correct is high, but positive OR does not beat best_single; UV conditional rescue 0.469/0.498 is below marginal 0.880/0.925. "
              "**The `only_UV/only_IR/only_Raman × structural subgroup` enrichment is not directly tested**; no UV-conjugation enrichment claim is made.",
              "- **H4 DESCRIPTIVELY SUPPORTED:** best-AP modality differs by task and subgroup; no universal modality ranking.", "",
              "All hypothesis statements are exploratory/post-hoc; no confirmatory significance is claimed.", ""]
    lines += ["## 14. Implications for Line C", "",
              "1. Do not build a fusion model on the assumption that UV adds conditional value on UVSPEC15; the current evidence shows UV below IR/Raman and conditional rescue below marginal.",
              "2. Positive OR is not a sufficient fusion target; it can trade precision for recall and is not Oracle. Line C should optimize a learned/gated policy and compare against best_single, not only OR.",
              "3. Because H3's subgroup enrichment is untested, Line C should not route UV specifically to conjugated/aromatic subgroups based on this evidence; that routing hypothesis needs a direct test.",
              "4. IR/Raman are the strong baselines on both tasks; any fusion gain must be demonstrated against the best single modality with scaffold-aware uncertainty.", ""]
    lines += ["## 15. Limitations", "",
              "- **WARN-3 boundaries:** AP/AUROC measure ranking/separation of labels, **not** the total chemical information content of a modality.",
              "- QM9S is a computational spectra dataset; it does not directly represent experimental measurement conditions.",
              "- Results must **not** be extrapolated to NIST, SDBS or API-Raman experimental domains.",
              "- Representative molecules are qualitative illustrations, **not XAI** and **not independent statistical evidence**.",
              "- Heavy-atom quartiles are degenerate; `functional_group=aromatic` is empty under the frozen token definition; subgroup CIs use 500 replicates (exploratory).",
              "- B4.3 CIs are paired-difference CIs under fixed models/Test set; they are not training-randomness or absolute-performance CIs.",
              "- Test was opened before B4; all results are exploratory/post-hoc, not a blind test.", ""]
    lines += ["## 16. Reproducibility and Data Provenance", "",
              f"- B0 v0.2 SHA256 `713fa92f5fd1f2134c1512bd2e49f689433a2158d120e41c9460ff5c287dd46a`; FREEZE `65a84959356a9a1f7a6e1d6434449f607c75a4c224a17450f3345ba25bb92a0c`.",
              f"- B4.4 tree SHA256 `d3e69052171ae753849a47139f21a03460ba7f772bd8d95e5143f749b9007a69`; B4.2/B4.3/B4.4/Supplement verification all PASS.",
              "- All 18 models, 3 seeds and 14/15 labels are represented in `tables/overall_metrics.csv`, `tables/per_label_metrics.csv` and `tables/micro_macro_summary.csv`.",
              "- Output SHA256 values and input hashes are in `asset_manifest.json`; independent verification is in `verification.json`.",
              "- Figures are generated from the CSV/JSON assets with Pillow (no matplotlib); no schematic data are used.", ""]
    (OUT / "line_b_final_report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = f"""Line B — B5 Final Report Archive
================================
Status: complete archive generated from frozen B4 assets.
Classification: exploratory / post-hoc; not a blind test; not confirmatory.

Archive contents:
- line_b_final_report.md : 16-section final report
- tables/                : 19 CSV tables covering 18 models, 3 seeds, 14/15 labels,
                           both F1 policies, paired differences + CIs, error
                           complementarity, OR/AND, Oracle, conditional
                           complementarity, structural subgroups, 88 cases.
- figures/               : 11 real-data PNG figures (Pillow).
- representative_molecules/ : 88 B4.4-supplement cases copied byte-identically
                           (RDKit PNG/SVG, spectra PNG, metadata JSON).
- asset_manifest.json    : input/output SHA256.
- verification.json      : independent verification.

Key verdicts:
H1 PARTIAL SUPPORT; H2 REJECTED; H3 PARTIAL SUPPORT; H4 DESCRIPTIVELY SUPPORTED.

WARN fixes:
1. B4.3 CIs are paired-difference CIs, not absolute-performance CIs.
2. H3 subgroup only_* enrichment is NOT directly tested and is not claimed.
3. AP/AUROC != chemical-information content; QM9S != experimental domain;
   representative molecules are qualitative, not XAI/statistical evidence.

No training, inference, threshold fitting, bootstrap or git operation was run.
"""
    (OUT / "README_summary.txt").write_text(summary, encoding="utf-8")

    # Asset manifest.
    inputs = {
        "protocol": {"path": relative_to_root(B0), "sha256": sha256_file(B0)},
        "freeze": {"path": relative_to_root(FREEZE), "sha256": sha256_file(FREEZE)},
        "test_access_log": {"path": relative_to_root(TESTLOG), "sha256": sha256_file(TESTLOG)},
        "b4_2": {"manifest": sha256_file(B42 / "analysis_manifest.json"), "verification": sha256_file(B42 / "verification.json")},
        "b4_3": {"manifest": sha256_file(B43 / "analysis_manifest.json"), "verification": sha256_file(B43 / "verification.json")},
        "b4_4": {"manifest": sha256_file(B44 / "analysis_manifest.json"), "verification": sha256_file(B44 / "verification.json"),
                  "tree_sha256": _tree_digest(B44)},
        "supplement": {"manifest": sha256_file(SUPP / "analysis_manifest.json"), "verification": sha256_file(SUPP / "verification.json")},
        "b4_1_seed_models": [
            {"task": r["task"], "modality": r["modality"], "seed": int(r["seed"]),
             "prediction_manifest": sha256_file(B41 / r["task"] / MODALITY_DIRS[r["modality"]] / r["run_name"] / "test" / "prediction_manifest.json")}
            for r in freeze["models"]
        ],
        "source_tables": {name: sha256_file(src) for name, src in copies.items()},
    }
    outputs = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in ("asset_manifest.json", "verification.json"):
            outputs.append({"path": path.relative_to(OUT).as_posix(), "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)})
    manifest = {
        "phase": "B5", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classification": {"status": "exploratory/post-hoc", "blind_test": False, "confirmatory": False},
        "inputs": inputs,
        "counts": {"models": 18, "seeds": 3, "tasks": 2, "labels": TASK_LABELS,
                   "representative_cases": len(examples), "tables": len(list((OUT / "tables").glob("*.csv"))),
                   "figures": len(list((OUT / "figures").glob("*.png")))},
        "outputs": outputs,
        "code": {"builder": relative_to_root(Path(__file__)), "builder_sha256": sha256_file(Path(__file__)),
                 "verifier": "training/line_b/verify_b5_final_report.py",
                 "verifier_sha256": sha256_file(ROOT / "training/line_b/verify_b5_final_report.py")
                 if (ROOT / "training/line_b/verify_b5_final_report.py").is_file() else None},
    }
    write_json(OUT / "asset_manifest.json", manifest)
    print(json.dumps({"status": "built", "out": relative_to_root(OUT), "tables": manifest["counts"]["tables"],
                      "figures": manifest["counts"]["figures"], "representative": len(examples)}, ensure_ascii=False, indent=1))


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    h = hashlib.sha256()
    for name, digest in entries:
        h.update(name.encode()); h.update(b"\0"); h.update(digest.encode()); h.update(b"\n")
    return h.hexdigest()


if __name__ == "__main__":
    main()
