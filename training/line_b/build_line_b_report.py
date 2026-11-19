"""B3.5-D2: complete Validation-development report builder.

Read-only integration of accepted B3-A/C2/D1 Validation products.  This module
does not train, infer, refit thresholds, select models, access Test data, or run
the formal 2,000-rep B4 bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
import rdkit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    atomic_output_dir,
    load_label_names,
    relative_to_root,
    sha256_file,
    write_json,
)
from training.line_b.analyze_line_b_complementarity import _load_task_inputs
from training.line_b.bootstrap_line_b_scaffold import _check_accepted_inputs

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
POLICY = "fixed_0.5"
SEED = 42
RENDER_PYTHON = Path(sys.executable)
D2_BASE = ROOT / "runs/LINE-B-ANALYSIS/validation_development/B3.5-D2"
D1_ROOT = ROOT / "runs/LINE-B-ANALYSIS/validation_development/B3.5-D1/final_22subgroups_20reps"
C2_ROOT = ROOT / "runs/LINE-B-ANALYSIS/validation_development/B3.5-C2/final_fix"
B3A_ROOT = ROOT / "runs/LINE-B-ANALYSIS"
STRUCTURE_PATH = ROOT / "processed/qm9s/structures/qm9s_structure_library.csv"
SPECTRA = {
    "ir": ROOT / "processed/qm9s/prepared/ir_float32.npy",
    "raman": ROOT / "processed/qm9s/prepared/raman_float32.npy",
    "uvvis": ROOT / "processed/qm9s/prepared/uvvis_float32.npy",
}
PROTOCOL_PATH = ROOT / "docs/experiments/line_b_protocol_draft.md"
REQUIRED_D1_FILES = (
    "summary.json", "training_cooccurrence_support.csv", "training_cooccurrence_counts.csv",
    "training_cooccurrence_phi.csv", "training_cooccurrence_cramers_v.csv",
    "training_hierarchy_pairs.csv", "training_semantic_groups.csv",
    "subgroup_definitions.csv", "subgroup_metrics.csv", "subgroup_metrics_main.csv",
    "subgroup_conditional_three_seed.csv", "subgroup_oracle_three_seed.csv",
    "subgroup_or_gain_three_seed.csv", "subgroup_molecule_exact_three_seed.csv",
    "subgroup_bootstrap_sanity.csv", "subgroup_bootstrap_macro_counts.csv",
)
REQUIRED_C2_FILES = (
    "summary.json", "mcnemar_holm.csv", "spearman.csv", "sample_error_summary.csv",
    "hard_samples.csv", "or_gain_summary.csv", "high_confidence_errors.csv",
)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _verify_summary(summary_path: Path, required: tuple[str, ...], task: str) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("task") != task:
        raise ValueError(f"Task mismatch: {summary_path}")
    if summary.get("validation_development") is not True or summary.get("no_test_access") is not True:
        raise ValueError(f"Invalid development/no-Test flags: {summary_path}")
    outputs = summary.get("outputs") or {}
    if not outputs:
        raise ValueError(f"Empty outputs in {summary_path}")
    for name in required:
        path = summary_path.parent / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty required analysis output: {path}")
    for name, expected in outputs.items():
        path = summary_path.parent / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty output: {path}")
        if _sha(path) != expected:
            raise ValueError(f"Output SHA mismatch: {path}")
    return summary


def _structure_map() -> dict[int, dict]:
    mapping = {}
    with STRUCTURE_PATH.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["row_index"])] = row
    if len(mapping) != 129817:
        raise ValueError("Structure library is not complete")
    return mapping


def _subgroup_masks(d1: dict, d1_dir: Path, val_rows: np.ndarray, structure: dict[int, dict], selected_only: bool = True) -> list[dict]:
    q = d1["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
    q1, q2, q3 = float(q["q1"]), float(q["q2"]), float(q["q3"])
    selected = set(d1["subgroup_bootstrap"]["selected_subgroups"])
    definitions = [r for r in _csv(d1_dir / "subgroup_definitions.csv") if (not selected_only or r["subgroup"] in selected)]
    out = []
    for row in definitions:
        name = row["subgroup"]
        dimension, value = name.split("=", 1)
        if dimension == "functional_group":
            dimension = "functional_group_types"
        if dimension == "ring_count":
            vals = np.asarray([int(structure[int(i)]["ring_count"]) for i in val_rows])
            mask = vals >= 3 if value == ">=3" else vals == int(value)
        elif dimension == "aromatic_ring_count":
            vals = np.asarray([int(structure[int(i)]["aromatic_ring_count"]) for i in val_rows])
            mask = vals >= 2 if value == ">=2" else vals == int(value)
        elif dimension == "n_o_f":
            vals = np.asarray([int(structure[int(i)]["nitrogen_count"]) + int(structure[int(i)]["oxygen_count"]) + int(structure[int(i)]["fluorine_count"]) for i in val_rows])
            mask = vals >= 3 if value == ">=3" else vals == int(value)
        elif dimension == "heavy_atom_count":
            vals = np.asarray([int(structure[int(i)]["heavy_atom_count"]) for i in val_rows])
            mask = np.where(vals <= q1, "Q1", np.where(vals <= q2, "Q2", np.where(vals <= q3, "Q3", "Q4"))) == value
        elif dimension == "rotatable_bond_count":
            vals = np.asarray([int(structure[int(i)]["rotatable_bond_count"]) for i in val_rows])
            mask = vals >= 2 if value == ">=2" else vals == int(value)
        elif dimension == "fraction_csp3":
            mask = np.asarray([(float(structure[int(i)]["fraction_csp3"]) < 0.5) == (value == "<0.5") for i in val_rows])
        elif dimension == "functional_group_types":
            def has(i):
                tokens = {t.strip() for t in (structure[int(i)].get("functional_group_types") or "").split(";") if t.strip()}
                if value == "aromatic":
                    return any("aromatic" in t or "aryl" in t for t in tokens)
                return value in tokens
            mask = np.asarray([has(i) for i in val_rows])
        else:
            raise ValueError(f"Unknown subgroup: {name}")
        out.append({"subgroup": name, "mask": mask, "n_samples": int(mask.sum())})
    if selected_only and {d["subgroup"] for d in out} != selected:
        raise AssertionError("D1 selected subgroup set does not match definitions")
    if not selected_only and {d["subgroup"] for d in out} != {r["subgroup"] for r in _csv(d1_dir / "subgroup_definitions.csv")}:
        raise AssertionError("D1 full subgroup set does not match definitions")
    return out


def _subgroup_membership_audit(d1: dict, d1_dir: Path, all_masks: list[dict], val_rows: np.ndarray, labels: list[str], truth: np.ndarray) -> tuple[list[dict], dict]:
    definitions = {r["subgroup"]: r for r in _csv(d1_dir / "subgroup_definitions.csv")}
    main = _csv(d1_dir / "subgroup_metrics.csv")
    rows = []
    for definition in all_masks:
        name = definition["subgroup"]
        mask = definition["mask"]
        d1_count = int(definitions[name]["n_samples"])
        d2_count = int(mask.sum())
        d1_pos = {}
        for r in main:
            if r["subgroup"] != name:
                continue
            values = {int(x["n_positive"]) for x in main if x["subgroup"] == name and x["label"] == r["label"]}
            if len(values) != 1:
                raise AssertionError(f"Inconsistent D1 positive support for {name}/{r['label']}")
            d1_pos[r["label"]] = int(values.pop())
        d2_pos = {label: int(truth[mask, i].sum()) for i, label in enumerate(labels)}
        d1_sig = ";".join(f"{label}:{d1_pos.get(label, 0)}" for label in labels)
        d2_sig = ";".join(f"{label}:{d2_pos[label]}" for label in labels)
        row_hash = hashlib.sha256(",".join(str(int(x)) for x in sorted(val_rows[mask])).encode()).hexdigest()
        if d2_count != d1_count:
            raise AssertionError(f"D1 membership count mismatch for {name}: D1={d1_count}, D2={d2_count}")
        if d1_sig != d2_sig:
            raise AssertionError(f"D1 positive signature mismatch for {name}")
        rows.append({
            "subgroup": name, "n_samples_d1": d1_count, "n_samples_d2": d2_count,
            "positive_signature_d1": d1_sig, "positive_signature_d2": d2_sig,
            "row_index_set_sha256_d2": row_hash, "row_index_set_d1_persisted": False,
            "status": "PASS_COUNT_AND_POSITIVE_SIGNATURE",
        })
    # Six mutually exclusive dimensions must cover every Validation molecule exactly once.
    dimensions = ("ring_count", "aromatic_ring_count", "n_o_f", "heavy_atom_count", "rotatable_bond_count", "fraction_csp3")
    coverage = {}
    for dim in dimensions:
        members = [m for m in all_masks if m["subgroup"].split("=", 1)[0] == dim]
        if not members:
            raise AssertionError(f"Missing dimension: {dim}")
        stacked = np.sum([m["mask"].astype(int) for m in members], axis=0)
        if not np.all(stacked == 1):
            raise AssertionError(f"Subgroup coverage failure for dimension {dim}")
        coverage[dim] = {"n_bins": len(members), "covered": int(stacked.sum()), "exactly_one": True}
    return rows, coverage


def _select_examples(task_info: dict, masks: list[dict]) -> list[dict]:
    names = task_info["expected_names"]
    row_index = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    by = {(m["modality"], int(m["seed"])): m for m in task_info["models"]}
    exact = {}
    conf = {}
    for modality in MODALITIES:
        model = by[(modality, SEED)]
        pred = model["probabilities"] >= 0.5
        exact[modality] = np.all(pred == (model["labels"] == 1), axis=1)
        conf[modality] = np.maximum(model["probabilities"], 1.0 - model["probabilities"]).mean(axis=1)
    correct = exact["ir"] & exact["raman"] & exact["uvvis"]
    confidence = (conf["ir"] + conf["raman"] + conf["uvvis"]) / 3.0
    out = []
    for definition in masks:
        positions = np.flatnonzero(definition["mask"])
        correct_positions = [int(i) for i in positions if correct[i]]
        error_positions = [int(i) for i in positions if not correct[i]]
        if not correct_positions:
            out.append({"subgroup": definition["subgroup"], "status": "no_correct_sample", "row_index": None, "confidence": None, "selection": ""})
        else:
            median = float(np.median(confidence[correct_positions]))
            chosen = min(correct_positions, key=lambda i: (abs(float(confidence[i]) - median), int(row_index[i])))
            out.append({"subgroup": definition["subgroup"], "status": "correct", "row_index": int(row_index[chosen]), "confidence": float(confidence[chosen]), "median_group_confidence": median, "selection": "all three modalities exact on all labels; confidence nearest group median; row_index ascending tie-break"})
        if not error_positions:
            out.append({"subgroup": definition["subgroup"], "status": "no_error_sample", "row_index": None, "confidence": None, "selection": ""})
        else:
            chosen = min(error_positions, key=lambda i: (-float(confidence[i]), int(row_index[i])))
            out.append({"subgroup": definition["subgroup"], "status": "error", "row_index": int(row_index[chosen]), "confidence": float(confidence[chosen]), "selection": "any label in any modality wrong; highest confidence; row_index ascending tie-break"})
    return out


def _render_native(row_index: int, render_root: Path) -> tuple[Path, Path]:
    if not RENDER_PYTHON.is_file():
        raise FileNotFoundError(RENDER_PYTHON)
    render_root.mkdir(parents=True, exist_ok=True)
    library = render_root / "qm9s_structure_library.csv"
    if not library.is_file():
        shutil.copyfile(STRUCTURE_PATH, library)
    wrapper = render_root / "_native_render_wrapper.py"
    wrapper.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "import rdkit.Chem as Chem\n"
        "def _mol_to_file(mol, path):\n"
        "    Path(path).write_text(Chem.MolToMolBlock(mol), encoding='utf-8')\n"
        "Chem.MolToMolFile = _mol_to_file\n"
        "class _SDWriter:\n"
        "    def __init__(self, path): self.path = Path(path)\n"
        "    def write(self, mol): self.path.open('a', encoding='utf-8').write(Chem.MolToMolBlock(mol) + '$$$$\\n')\n"
        "    def close(self): pass\n"
        "Chem.SDWriter = _SDWriter\n"
        "script, row, output = sys.argv[1:4]\n"
        "sys.argv = [script, 'render', '--row-index', row, '--output-dir', output]\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('qm9s_structure_tools', script)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.main()\n",
        encoding='utf-8',
    )
    result = subprocess.run([
        str(RENDER_PYTHON), str(wrapper), str(ROOT / "scripts/qm9s_structure_tools.py"),
        str(row_index), str(render_root),
    ], capture_output=True, text=True)
    wrapper.unlink(missing_ok=True)
    base = render_root / "rendered" / f"row_{row_index:06d}"
    png = base / "structure_numbered.png"
    svg = base / "structure_numbered.svg"
    # RDKit's MolToMolFile can fail on non-ASCII paths after PNG/SVG are written.
    # PNG/SVG are the required native-rendering outputs for D2; no schematic fallback.
    if not png.is_file() or not svg.is_file():
        raise RuntimeError(f"Native RDKit render failed for row {row_index}: {result.stderr[-1000:]}")
    with Image.open(png) as im:
        im.verify()
    if "<svg" not in svg.read_text(encoding="utf-8"):
        raise ValueError(f"Invalid SVG: {svg}")
    return png, svg


def _plot_spectra(row_index: int, out_path: Path) -> None:
    width, panel_h, pad = 900, 220, 35
    height = panel_h * 3 + pad * 4
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for panel, modality in enumerate(MODALITIES):
        arr = np.load(SPECTRA[modality], mmap_mode="r", allow_pickle=False)
        if row_index >= arr.shape[0]:
            raise IndexError(f"row_index outside {modality} spectra: {row_index}")
        spectrum = np.asarray(arr[row_index], dtype=np.float32)
        top = pad + panel * (panel_h + pad)
        x0, x1, y0, y1 = pad, width - pad, top + panel_h, top
        vals = np.asarray(spectrum, dtype=float)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax == vmin:
            vmax = vmin + 1.0
        xs = x0 + np.arange(len(vals)) * (x1 - x0) / max(len(vals) - 1, 1)
        ys = y0 - (vals - vmin) * (y0 - y1) / (vmax - vmin)
        if len(xs) > 1:
            draw.line(list(zip(xs.tolist(), ys.tolist())), fill=(31, 78, 121), width=1)
        draw.rectangle([x0, y1, x1, y0], outline=(90, 90, 90), width=1)
        draw.text((x0, top), f"{modality.upper()} - prepared spectrum (source array; no experimental unit claimed)", fill="black", font=font)
        draw.text((x0, y1 - 16), "spectral index", fill="black", font=font)
        draw.text((x0 + 4, (y0 + y1) / 2), "prepared intensity (a.u.; source array)", fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _draw_heatmap(matrix: np.ndarray, labels: list[str], title: str, out: Path) -> None:
    n = len(labels)
    cell = max(18, min(42, 700 // max(n, 1)))
    margin = 150
    bar_w = 24
    w = margin + n * cell + 80 + bar_w
    h = margin + n * cell + 50
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if np.isnan(value):
                color = (230, 230, 230)
            else:
                t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
                color = (int(255 * (1 - t)), int(255 * (1 - abs(t - 0.5) * 2)), int(255 * t))
            x = margin + j * cell; y = margin + i * cell
            draw.rectangle([x, y, x + cell, y + cell], fill=color)
    for i, label in enumerate(labels):
        draw.text((8, margin + i * cell + 4), label[:22], fill="black", font=font)
        draw.text((margin + i * cell + 2, 8), label[:12], fill="black", font=font)
    bar_x = margin + n * cell + 22
    for k in range(120):
        t = 1.0 - k / 119.0
        color = (int(255 * (1 - t)), int(255 * (1 - abs(t - 0.5) * 2)), int(255 * t))
        draw.rectangle([bar_x, margin + k, bar_x + bar_w, margin + k + 1], fill=color)
    draw.rectangle([bar_x, margin, bar_x + bar_w, margin + 120], outline=(60, 60, 60))
    draw.text((bar_x - 5, margin - 16), f"{vmax:.2f}", fill="black", font=font)
    draw.text((bar_x - 5, margin + 122), f"{vmin:.2f}", fill="black", font=font)
    draw.text((margin, h - 24), title + f"  (min={vmin:.3f}, max={vmax:.3f})", fill="black", font=font)
    draw.text((margin, margin - 16), "columns", fill="black", font=font)
    draw.text((8, margin - 16), "rows", fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def _draw_bars(rows: list[dict], value_fields: list[str], title: str, out: Path) -> None:
    labels = [r["modality"].upper() for r in rows]
    width, height, pad = 720, 420, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    maxv = max(float(r[f]) for r in rows for f in value_fields) or 1.0
    group_w = (width - 2 * pad) / max(len(rows), 1)
    colors = [(31, 78, 121), (196, 89, 17), (46, 139, 87)]
    for i, row in enumerate(rows):
        for j, field in enumerate(value_fields):
            v = float(row[field]) / maxv
            x0 = pad + i * group_w + j * (group_w / (len(value_fields) + 1))
            bar_w = group_w / (len(value_fields) + 2)
            y0 = height - pad - v * (height - 2 * pad)
            draw.rectangle([x0, y0, x0 + bar_w, height - pad], fill=colors[j % len(colors)])
        draw.text((pad + i * group_w, height - 35), labels[i], fill="black", font=font)
    draw.text((pad, 15), title, fill="black", font=font)
    for j, field in enumerate(value_fields):
        draw.text((pad + j * 160, 35), field, fill=colors[j % len(colors)], font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def _draw_ci(rows: list[dict], title: str, out: Path) -> None:
    labels = [r["subgroup"] for r in rows]
    width = 1000; height = max(420, 26 * len(rows) + 80); pad = 190
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    vals = [v for r in rows for v in (r["point_estimate"], r["ci_low"], r["ci_high"]) if v not in ("", None)]
    vals = [float(v) for v in vals if v is not None]
    lo, hi = min(vals or [-1]), max(vals or [1])
    span = max(hi - lo, 1e-9)
    draw.text((10, 8), title, fill="black", font=font)
    for i, r in enumerate(rows):
        y = 35 + i * 24
        draw.text((8, y), r["subgroup"][:28], fill="black", font=font)
        x = lambda v: pad + (float(v) - lo) / span * (width - pad - 25)
        draw.line([x(r["ci_low"]), y + 7, x(r["ci_high"]), y + 7], fill=(31, 78, 121), width=2)
        draw.ellipse([x(r["point_estimate"]) - 3, y + 4, x(r["point_estimate"]) + 3, y + 10], fill=(196, 89, 17))
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def _md_table(rows: list[dict], fields: list[str], limit: int = 20) -> str:
    if not rows:
        return "_No rows._"
    head = "| " + " | ".join(fields) + " |\n|" + "|".join(["---"] * len(fields)) + "|\n"
    body = ""
    for row in rows[:limit]:
        body += "| " + " | ".join(str(row.get(f, "")) for f in fields) + " |\n"
    return head + body


def _audit_record(path: Path, use: str, source: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative_to_root(path), "size": path.stat().st_size, "sha256": _sha(path), "use": use, "source": source, "status": "PASS"}


def _build_task(task: str, label_set: str, analysis_root: Path, c2_root: Path,
                b3a_root: Path, out_root: Path, max_subgroups: int | None) -> dict:
    d1_dir = analysis_root / task / "structure_analysis"
    c2_dir = c2_root / task / "complementarity_extended"
    b3a_dir = b3a_root / task / "complementarity"
    d1 = _verify_summary(d1_dir / "summary.json", REQUIRED_D1_FILES, task)
    c2 = _verify_summary(c2_dir / "summary.json", REQUIRED_C2_FILES, task)
    task_info = _load_task_inputs(label_set, ROOT / "runs/LINE-B-PREDICTIONS", ROOT / "runs/LINE-B-THRESHOLDS", ROOT / "runs/LINE-B-METRICS", False)
    binding = _check_accepted_inputs(task_info, label_set)
    structure = _structure_map()
    val_rows = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    all_masks = _subgroup_masks(d1, d1_dir, val_rows, structure, selected_only=False)
    selected_names = set(d1["subgroup_bootstrap"]["selected_subgroups"])
    masks = [m for m in all_masks if m["subgroup"] in selected_names]
    truth = task_info["models"][0]["labels"]
    names = task_info["expected_names"]
    membership_audit, coverage_audit = _subgroup_membership_audit(d1, d1_dir, all_masks, val_rows, names, truth)
    membership_audit = [r for r in membership_audit if r["subgroup"] in selected_names]
    if max_subgroups:
        masks = masks[:max_subgroups]
    membership_audit = [r for r in membership_audit if r["subgroup"] in {m["subgroup"] for m in masks}]
    selections = _select_examples(task_info, masks)
    expected_records = 2 * len(masks)
    if len(selections) != expected_records:
        raise AssertionError(f"Expected {expected_records} selection records, got {len(selections)}")

    # Input audit.
    audit = []
    for summary_path, summary, source in ((d1_dir / "summary.json", d1, "D1"), (c2_dir / "summary.json", c2, "C2")):
        audit.append(_audit_record(summary_path, f"{source} summary", source))
        for name, expected in summary["outputs"].items():
            audit.append(_audit_record(summary_path.parent / name, f"{source} output {name}", source))
    for p in (
        ROOT / "runs/LINE-B-METRICS" / task / "model_metrics.csv",
        ROOT / "runs/LINE-B-METRICS" / task / "per_label_mean_sd.csv",
        ROOT / "runs/LINE-B-METRICS" / task / "summary.json",
        ROOT / "runs/LINE-B-THRESHOLDS" / task / "summary.json",
        STRUCTURE_PATH, SPECTRA["ir"], SPECTRA["raman"], SPECTRA["uvvis"],
        PROTOCOL_PATH,
    ):
        audit.append(_audit_record(p, "report input", "accepted Validation asset"))
    for model in task_info["models"]:
        audit.append(_audit_record(Path(model["manifest_path"]), "prediction manifest", "B3 Validation"))
        audit.append(_audit_record(Path(model["threshold_path"]), "threshold file", "B3 Validation"))
    for p in (b3a_dir / "summary.json", *(b3a_dir / n for n in ("model_mean_sd.csv", "pairwise_mean_sd.csv", "three_modality_mean_sd.csv"))):
        audit.append(_audit_record(p, "B3-A input", "B3-A"))

    task_out = Path(out_root) / task / "validation_report"
    with atomic_output_dir(task_out) as temporary:
        tables = temporary / "tables"
        figures = temporary / "figures"
        examples_root = temporary / "examples"
        render_root = temporary / "rendered_structures"
        tables.mkdir(); figures.mkdir(); examples_root.mkdir()
        _write_csv(tables / "input_audit.csv", audit, ["path", "size", "sha256", "use", "source", "status"])
        _write_csv(tables / "subgroup_membership_audit.csv", membership_audit, list(membership_audit[0].keys()))

        # Copy selected analysis tables.
        table_sources = {
            "global_metrics.csv": ROOT / "runs/LINE-B-METRICS" / task / "model_metrics.csv",
            "per_label_mean_sd.csv": ROOT / "runs/LINE-B-METRICS" / task / "per_label_mean_sd.csv",
            "cooccurrence_counts.csv": d1_dir / "training_cooccurrence_counts.csv",
            "cooccurrence_phi.csv": d1_dir / "training_cooccurrence_phi.csv",
            "cooccurrence_cramers_v.csv": d1_dir / "training_cooccurrence_cramers_v.csv",
            "cooccurrence_support.csv": d1_dir / "training_cooccurrence_support.csv",
            "hierarchy_pairs.csv": d1_dir / "training_hierarchy_pairs.csv",
            "semantic_groups.csv": d1_dir / "training_semantic_groups.csv",
            "subgroup_definitions.csv": d1_dir / "subgroup_definitions.csv",
            "subgroup_metrics_main.csv": d1_dir / "subgroup_metrics_main.csv",
            "subgroup_conditional_by_label.csv": d1_dir / "subgroup_conditional_by_label.csv",
            "subgroup_conditional_three_seed.csv": d1_dir / "subgroup_conditional_three_seed.csv",
            "subgroup_oracle_by_label.csv": d1_dir / "subgroup_oracle_by_label.csv",
            "subgroup_oracle_by_label_three_seed.csv": d1_dir / "subgroup_oracle_by_label_three_seed.csv",
            "subgroup_oracle_three_seed.csv": d1_dir / "subgroup_oracle_three_seed.csv",
            "subgroup_or_gain_three_seed.csv": d1_dir / "subgroup_or_gain_three_seed.csv",
            "subgroup_molecule_exact_three_seed.csv": d1_dir / "subgroup_molecule_exact_three_seed.csv",
            "subgroup_bootstrap_sanity.csv": d1_dir / "subgroup_bootstrap_sanity.csv",
            "subgroup_bootstrap_macro_counts.csv": d1_dir / "subgroup_bootstrap_macro_counts.csv",
            "mcnemar_holm.csv": c2_dir / "mcnemar_holm.csv",
            "spearman.csv": c2_dir / "spearman.csv",
            "sample_error_summary.csv": c2_dir / "sample_error_summary.csv",
            "hard_samples.csv": c2_dir / "hard_samples.csv",
            "or_gain_summary.csv": c2_dir / "or_gain_summary.csv",
            "high_confidence_errors.csv": c2_dir / "high_confidence_errors.csv",
            "b3a_model_mean_sd.csv": b3a_dir / "model_mean_sd.csv",
            "b3a_pairwise_mean_sd.csv": b3a_dir / "pairwise_mean_sd.csv",
        }
        for name, source in table_sources.items():
            if source is not None:
                shutil.copyfile(source, tables / name)

        # Global modality mean/SD table for the report.
        metrics = _csv(ROOT / "runs/LINE-B-METRICS" / task / "model_metrics.csv")
        fields = ("macro_ap", "macro_auroc", "macro_f1", "micro_f1")
        global_rows = []
        for modality in MODALITIES:
            rows = [r for r in metrics if r["modality"] == modality]
            row = {"modality": modality, "n_models": len(rows)}
            for field in fields:
                vals = [float(r[field]) for r in rows]
                row[field + "_mean"] = float(np.mean(vals))
                row[field + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            global_rows.append(row)
        _write_csv(tables / "global_performance.csv", global_rows, list(global_rows[0].keys()))
        _draw_bars(global_rows, ["macro_ap_mean", "macro_auroc_mean", "macro_f1_mean"], f"{task}: modality mean performance", figures / "global_performance.png")

        cooc = _csv(d1_dir / "training_cooccurrence_counts.csv")
        labels = [r["label"] for r in _csv(d1_dir / "training_cooccurrence_support.csv")]
        n = len(labels)
        cm = np.zeros((n, n))
        for r in cooc:
            cm[labels.index(r["label_a"]), labels.index(r["label_b"])] = float(r["both"])
        _draw_heatmap(cm, labels, f"{task}: co-occurrence counts", figures / "cooccurrence_heatmap.png")
        phi = np.zeros((n, n))
        for r in _csv(d1_dir / "training_cooccurrence_phi.csv"):
            phi[labels.index(r["label_a"]), labels.index(r["label_b"])] = float(r["phi"]) if r["phi"] not in ("", "None") else np.nan
        _draw_heatmap(phi, labels, f"{task}: phi correlation", figures / "phi_heatmap.png")
        cramers = np.zeros((n, n))
        for r in _csv(d1_dir / "training_cooccurrence_cramers_v.csv"):
            cramers[labels.index(r["label_a"]), labels.index(r["label_b"])] = float(r["cramers_v"]) if r["cramers_v"] not in ("", "None") else np.nan
        _draw_heatmap(cramers, labels, f"{task}: Cramer's V", figures / "cramers_v_heatmap.png")

        support = _csv(d1_dir / "subgroup_definitions.csv")
        support_main = _csv(d1_dir / "subgroup_metrics_main.csv")
        support_rows = []
        for s in support:
            if s["subgroup"] not in {d["subgroup"] for d in masks}:
                continue
            support_rows.append({"subgroup": s["subgroup"], "n_samples": int(s["n_samples"]), "n_main_rows": sum(1 for r in support_main if r["subgroup"] == s["subgroup"])})
        _draw_bars([{**r, "modality": r["subgroup"]} for r in support_rows], ["n_samples"], f"{task}: subgroup support", figures / "subgroup_support.png")

        # Selection, native render, per-molecule assets.
        example_records = []
        metadata_by_row = {}
        for item in selections:
            if item["row_index"] is None:
                example_records.append({**item, "structure_png": "", "structure_svg": "", "spectra_png": "", "metadata": ""})
                continue
            row = int(item["row_index"])
            example_dir = examples_root / f"row_{row:06d}"
            example_dir.mkdir(parents=True, exist_ok=True)
            if row not in metadata_by_row:
                png, svg = _render_native(row, render_root)
                shutil.copyfile(png, example_dir / "structure.png")
                shutil.copyfile(svg, example_dir / "structure.svg")
                _plot_spectra(row, example_dir / "spectra.png")
                pos = int(np.where(val_rows == row)[0][0])
                models = {(m["modality"], int(m["seed"])): m for m in task_info["models"]}
                truth = models[("ir", 42)]["labels"][pos]
                probs = {}; preds = {}; descriptive = {}
                for modality in MODALITIES:
                    model = models[(modality, 42)]
                    probs[modality] = model["probabilities"][pos]
                    preds[modality] = (probs[modality] >= 0.5).astype(int)
                    descriptive[modality] = float(np.maximum(probs[modality], 1.0 - probs[modality]).mean())
                structure_row = structure[row]
                metadata = {
                    "row_index": row, "canonical_smiles": structure_row["canonical_smiles"],
                    "molecular_formula": structure_row["molecular_formula"],
                    "seed": 42, "threshold_policy": POLICY,
                    "true_labels": {name: int(truth[i]) for i, name in enumerate(labels)},
                    "probabilities": {m: {name: float(probs[m][i]) for i, name in enumerate(labels)} for m in MODALITIES},
                    "hard_predictions": {m: {name: int(preds[m][i]) for i, name in enumerate(labels)} for m in MODALITIES},
                    "wrong_labels": {m: [name for i, name in enumerate(labels) if preds[m][i] != int(truth[i])] for m in MODALITIES},
                    "confidence": float(np.mean([descriptive[m] for m in MODALITIES])),
                    "descriptive_confidence": descriptive,
                    "renderer": "rdkit_native", "rdkit_version": rdkit.__version__,
                    "structure_png": f"examples/row_{row:06d}/structure.png",
                    "structure_svg": f"examples/row_{row:06d}/structure.svg",
                    "spectra_png": f"examples/row_{row:06d}/spectra.png",
                    "selection": [],
                    "data_sources": {"structure": relative_to_root(STRUCTURE_PATH), "spectra": {m: relative_to_root(SPECTRA[m]) for m in MODALITIES}},
                    "input_sha256": {
                        "structure": _sha(STRUCTURE_PATH),
                        **{m: _sha(SPECTRA[m]) for m in MODALITIES},
                        **{m: task_info_model["probabilities_sha256"] for m, task_info_model in [(x["modality"], x) for x in task_info["models"] if int(x["seed"]) == 42]},
                    },
                }
                metadata_by_row[row] = metadata
            metadata_by_row[row]["selection"].append({"subgroup": item["subgroup"], "status": item["status"], "confidence": item["confidence"], "selection": item["selection"]})
            metadata_by_row[row]["output_sha256"] = {
                "structure.png": _sha(example_dir / "structure.png"),
                "structure.svg": _sha(example_dir / "structure.svg"),
                "spectra.png": _sha(example_dir / "spectra.png"),
            }
            write_json(example_dir / "metadata.json", metadata_by_row[row])
            prefix = f"examples/row_{row:06d}"
            example_records.append({**item, "structure_png": f"{prefix}/structure.png", "structure_svg": f"{prefix}/structure.svg", "spectra_png": f"{prefix}/spectra.png", "metadata": f"{prefix}/metadata.json"})
        _write_csv(tables / "representative_molecules.csv", example_records, ["subgroup", "status", "row_index", "confidence", "median_group_confidence", "selection", "structure_png", "structure_svg", "spectra_png", "metadata"])

        # Bootstrap and complementarity figures.
        boot = _csv(d1_dir / "subgroup_bootstrap_sanity.csv")
        boot_ap = [r for r in boot if r["metric"] == "AP" and r["scope"] == "macro" and r["comparison"] == "UV-IR" and r["seed"] == "three_seed_mean" and r["threshold_policy"] == "threshold_free"]
        if boot_ap:
            _draw_ci(boot_ap, f"{task}: UV-IR Macro AP difference, Validation 20-rep CI (not B4)", figures / "bootstrap_uv_ir_ap.png")
        cond = [r for r in _csv(d1_dir / "subgroup_conditional_three_seed.csv") if r["metric"] == "conditional_delta" and r["threshold_policy"] == "fixed_0.5"]
        cond_rows = []
        for r in cond:
            value = float(r["mean"]) if r["mean"] not in ("", "None") else np.nan
            cond_rows.append({"subgroup": r["subgroup"] + ":" + r["comparison"], "comparison": r["comparison"], "mean": value})
        if cond_rows:
            _draw_bars([{**r, "modality": r["subgroup"][:28]} for r in cond_rows[:22]], ["mean"], f"{task}: conditional delta (Validation, fixed 0.5)", figures / "conditional_delta.png")

        # Markdown report.
        global_md = _md_table(global_rows, ["modality", "macro_ap_mean", "macro_auroc_mean", "macro_f1_mean", "n_models"])
        per_label = _csv(tables / "per_label_mean_sd.csv")
        sub_main = _csv(tables / "subgroup_metrics_main.csv")
        cond_three = _csv(tables / "subgroup_conditional_three_seed.csv")
        oracle_three = _csv(tables / "subgroup_oracle_three_seed.csv")
        or_three = _csv(tables / "subgroup_or_gain_three_seed.csv")
        examples_rows = [r for r in example_records if r["row_index"] is not None]
        refs = []
        for r in examples_rows:
            if r["row_index"] is None:
                continue
            refs.append(f"- `{r['subgroup']}` / {r['status']} / row {r['row_index']}: ![]({r['structure_png']}) ![]({r['spectra_png']})")
        protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8")
        def _protocol_h(label: str) -> str:
            match = re.search(rf"\*\*{label}（[^）]+）\*\*\s*\n\n(.+?)(?=\n\n\*\*H|\n\n>)", protocol_text, re.S)
            if not match:
                raise ValueError(f"Missing protocol hypothesis {label}")
            return match.group(1).strip()
        h1_text = _protocol_h("H1")
        h2_text = _protocol_h("H2")
        h3_text = _protocol_h("H3")
        h4_text = _protocol_h("H4")
        ir_ap = float(global_rows[0]["macro_ap_mean"]); ir_auc = float(global_rows[0]["macro_auroc_mean"])
        ra_ap = float(global_rows[1]["macro_ap_mean"]); ra_auc = float(global_rows[1]["macro_auroc_mean"])
        uv_ap = float(global_rows[2]["macro_ap_mean"]); uv_auc = float(global_rows[2]["macro_auroc_mean"])
        if task == "Shared14":
            h1_evidence = f"Shared14 macro AP/AUROC: IR {ir_ap:.4f}/{ir_auc:.4f}; Raman {ra_ap:.4f}/{ra_auc:.4f}; UV-Vis {uv_ap:.4f}/{uv_auc:.4f}. Macro direction is consistent."
            h1_assessment = "Validation-consistent, **insufficient evidence** for formal confirmation."
            h2_evidence = "Not evaluated in this Shared14 report; see the UVSPEC15 report."
            h2_assessment = "**Not evaluated here.**"
        else:
            h1_evidence = "Not evaluated in this UVSPEC15 report; see the Shared14 report."
            h1_assessment = "**Not evaluated here.**"
            h2_evidence = f"UVSPEC15 macro AP: UV-Vis {uv_ap:.4f}; IR {ir_ap:.4f}; Raman {ra_ap:.4f}. UV-Vis does not exceed both vibration modalities in the observed macro values."
            h2_assessment = "**Not supported in Validation**; formal Test evidence absent."
        hypothesis_md = f"""## Chapter 8 — Hypothesis Evaluation

The original B0 v0.2 definitions are quoted verbatim; no hypothesis was redefined.

| Hypothesis | Original definition / criterion | Supporting Validation evidence | Counter-evidence / limitation | Current assessment |
|---|---|---|---|---|
| H1 | {h1_text} | {h1_evidence} | Label-specific criterion is not formally tested; Validation CIs are descriptive; no Test access. | {h1_assessment} |
| H2 | {h2_text} | {h2_evidence} | Observed macro direction is not favorable to UV in UVSPEC15; no Test access. | {h2_assessment} |
| H3 | {h3_text} | Oracle and conditional tables are present (`subgroup_oracle_*`, `subgroup_conditional_*`). | The criterion explicitly references the same Test sample; only Validation development was run. Oracle is an idealized upper bound; OR is a rule-based union. | **Insufficient evidence** for formal H3. |
| H4 | {h4_text} | D1 reports all 22 non-empty structural subgroups and six-direction conditional results. | Subgroup support is uneven; heavy-atom partition is degenerate; no Test or 2,000-rep confirmation. | **Insufficient evidence**; descriptive only. |

No formal H1–H4 confirmation and no Test-level claim are made.
"""
        report = f"""# Line B {task} — Validation Development Report

**Status:** B3.5-D2 temporary Validation-development report. Not a B4/B5 formal report.  
**Test access:** none. **Formal 2,000 Bootstrap:** not run.

## Chapter 1 — Scope and Experimental Configuration

- Task: {task}; label set: {label_set}; labels: {len(labels)}.
- Modalities: IR, Raman, UV-Vis; Seeds: 42/123/2026; 9 Validation models per task.
- Validation split: 12,981 molecules; threshold policy primary fixed 0.5.
- Best Single: `{task_info['best_single']}` (accepted Validation metadata).
- Data source: accepted B3-A, C2 and D1 Validation assets; no Test access.

## Chapter 2 — Global Model Performance

{global_md}

Metrics are copied from `LINE-B-METRICS`; no recomputation or overwrite.

**Figure:** ![](figures/global_performance.png)

## Chapter 3 — Label-level Performance

{_md_table(per_label, ["modality", "label", "support_mean", "average_precision_mean", "auroc_mean", "f1_mean", "n_seeds"], 24)}

## Chapter 4 — Structural Co-occurrence

- Counts: `tables/cooccurrence_counts.csv`
- Phi: `tables/cooccurrence_phi.csv`
- Cramer's V: `tables/cooccurrence_cramers_v.csv`
- Support: `tables/cooccurrence_support.csv`
- Hierarchy: `tables/hierarchy_pairs.csv`
- Semantic groups: `tables/semantic_groups.csv`

**Figures:** ![](figures/cooccurrence_heatmap.png) ![](figures/phi_heatmap.png) ![](figures/cramers_v_heatmap.png)

## Chapter 5 — Structural Subgroup Analysis

- D1 definitions: {d1['subgroups']['n_definitions']}; non-empty: {d1['subgroups']['n_nonempty']}; main-support: {d1['subgroups']['n_main_support']} (at least one label meets n>=100 and positive>=30; not all labels).
- Heavy-atom quartiles: {json.dumps(d1['subgroups']['heavy_atom']['heavy_atom_quartiles'], sort_keys=True)}; degenerate={d1['subgroups']['heavy_atom']['degenerate_partition']}.
- Six-dimension coverage audit: {json.dumps(coverage_audit, sort_keys=True)}.

`tables/subgroup_membership_audit.csv` compares D1 counts and per-label positive signatures with D2 masks. D1 did not persist raw member row_index sets; the D2 row-set hashes are recorded, and D1 positive signatures must match exactly.

{_md_table(membership_audit, ["subgroup", "n_samples_d1", "n_samples_d2", "status"], 26)}

**Figure:** ![](figures/subgroup_support.png)

{_md_table(sub_main, ["subgroup", "n_samples", "n_positive", "modality", "threshold_policy", "label", "ap", "auroc", "f1", "support_sufficient"], 20)}

## Chapter 6 — Modality Complementarity

- C2 McNemar/Holm: `tables/mcnemar_holm.csv`
- C2 Spearman/error distributions: `tables/spearman.csv`, `tables/sample_error_summary.csv`
- Conditional by label: `tables/subgroup_conditional_by_label.csv`
- Conditional three-seed: `tables/subgroup_conditional_three_seed.csv`
- Oracle by label: `tables/subgroup_oracle_by_label.csv`
- Oracle by label three-seed: `tables/subgroup_oracle_by_label_three_seed.csv`
- Oracle summary: `tables/subgroup_oracle_three_seed.csv`
- OR gain: `tables/subgroup_or_gain_three_seed.csv`
- Bootstrap macro replicate counts: `tables/subgroup_bootstrap_macro_counts.csv`

**Figures:** ![](figures/bootstrap_uv_ir_ap.png) ![](figures/conditional_delta.png)

### Conditional complementarity (three-seed descriptive)

{_md_table(cond_three, ["subgroup", "comparison", "base_modality", "correcting_modality", "threshold_policy", "metric", "mean", "sd", "n_valid"], 20)}

### Oracle (three-seed descriptive)

{_md_table(oracle_three, ["subgroup", "threshold_policy", "metric", "mean", "sd", "n_valid"], 20)}

### OR gain (three-seed descriptive)

{_md_table(or_three, ["subgroup", "comparison", "threshold_policy", "metric", "mean", "sd", "n_valid"], 20)}

Oracle is an idealized correctness upper bound, not a trained fusion model. OR is a rule-based union, not a fusion model.

## Chapter 7 — Representative Molecules

Selection: Seed 42, fixed 0.5; correct = all labels exact in all three modalities; error = at least one wrong label; confidence = mean max(p,1-p) across all modalities/labels. Correct = confidence closest to subgroup median; error = highest confidence; row_index ascending tie-break. QM9S calculated spectra; spectral index is not an experimental unit.

{_md_table(example_records, ["subgroup", "status", "row_index", "confidence", "selection"], 44)}

{os.linesep.join(refs)}

{hypothesis_md}

## Chapter 9 — Limitations and Next Steps

- Validation-only; no Test inference.
- QM9S spectra are calculated, not experimental.
- Subgroup support is uneven; heavy-atom partition is degenerate.
- Three seeds do not replace formal Test evidence.
- Oracle is an idealized diagnostic; OR is a rule, not a learned fusion.
- No formal 2,000-rep B4 bootstrap was run.

Next: B4 Test access, formal bootstrap, and confirmatory evaluation under the B0 protocol; no such result is claimed here.
"""
        (temporary / "report.md").write_text(report, encoding="utf-8")
        _write_csv(temporary / "examples.csv", example_records, ["subgroup", "status", "row_index", "confidence", "median_group_confidence", "selection", "structure_png", "structure_svg", "spectra_png", "metadata"])

        # Remove the copied structure library from final assets.
        try:
            (render_root / "qm9s_structure_library.csv").unlink(missing_ok=True)
        except OSError:
            pass

        # Manifest: every input and every output.
        input_records = audit
        output_files = []
        for root in (temporary,):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name != "report_manifest.json":
                    output_files.append({"path": p.relative_to(temporary).as_posix(), "size": p.stat().st_size, "sha256": _sha(p)})
        manifest = {
            "phase": "B3.5-D2", "task": task, "label_set": label_set,
            "validation_development": True, "no_test_access": True,
            "formal_b4_b5": False, "bootstrap_2000": False,
            "protocol": {"path": relative_to_root(PROTOCOL_PATH), "sha256": _sha(PROTOCOL_PATH)},
            "environment": {
                "python": sys.version, "sys_executable": sys.executable,
                "rdkit": rdkit.__version__, "numpy": np.__version__,
                "pandas": __import__("pandas").__version__,
            },
            "inputs": input_records,
            "analysis_summaries": {
                "d1": {"path": relative_to_root(d1_dir / "summary.json"), "sha256": _sha(d1_dir / "summary.json")},
                "c2": {"path": relative_to_root(c2_dir / "summary.json"), "sha256": _sha(c2_dir / "summary.json")},
                "b3a": {"path": relative_to_root(b3a_dir / "summary.json"), "sha256": _sha(b3a_dir / "summary.json")},
            },
            "selection_rule": {
                "seed": 42, "threshold_policy": POLICY,
                "correct": "all labels exact in all three modalities",
                "error": "at least one wrong label in at least one modality",
                "confidence": "mean max(p,1-p) over all labels and three modalities",
                "correct_choice": "confidence nearest subgroup median; row_index ascending",
                "error_choice": "highest confidence; row_index ascending",
            },
            "subgroups": {
                "selected": [m["subgroup"] for m in masks], "n_selected": len(masks),
                "n_selection_records": len(example_records),
                "coverage_audit": coverage_audit,
                "membership_audit_table": "tables/subgroup_membership_audit.csv",
                "row_index_set_d1_persisted": False,
            },
            "code_sha256": {
                "training/line_b/build_line_b_report.py": _sha(Path(__file__)),
                "training/line_b/analyze_line_b_structure.py": _sha(ROOT / "training/line_b/analyze_line_b_structure.py"),
                "training/line_b/analyze_line_b_complementarity_extended.py": _sha(ROOT / "training/line_b/analyze_line_b_complementarity_extended.py"),
                "training/line_b/analyze_line_b_complementarity.py": _sha(ROOT / "training/line_b/analyze_line_b_complementarity.py"),
            },
            "outputs": output_files,
            "output_count": len(output_files),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "caveats": [
                "Validation-development only; not B4/B5 formal evidence.",
                "No Test data or predictions accessed.",
                "Formal 2,000-rep B4 bootstrap not executed.",
                "Native RDKit renderer is required and used; no Pillow structure fallback exists.",
                "QM9S calculated spectra have spectral-index x-axes; no experimental units are claimed.",
                "Oracle is idealized; OR is a rule-based union, not a trained fusion model.",
            ],
        }
        write_json(temporary / "report_manifest.json", manifest)
        manifest_sha = hashlib.sha256((temporary / "report_manifest.json").read_bytes()).hexdigest()
        (temporary / "report_manifest.sha256").write_text(manifest_sha + "\n", encoding="utf-8")
        return {"task": task, "manifest": manifest, "n_selected": len(masks), "n_records": len(example_records), "examples": len(examples_rows), "unique_examples": len(metadata_by_row)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "Shared14", "UVSPEC15"], default="all")
    parser.add_argument("--analysis-root", type=Path, default=D1_ROOT)
    parser.add_argument("--c2-root", type=Path, default=C2_ROOT)
    parser.add_argument("--b3a-root", type=Path, default=B3A_ROOT)
    parser.add_argument("--out-root", type=Path, default=D2_BASE / "r2_final")
    parser.add_argument("--max-subgroups", type=int, default=None)
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def _workspace_path(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise SystemExit(f"{label} must be inside the workspace: {resolved}")
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test:
        raise SystemExit("Test is not authorized in B3.5-D2")
    out_root = _workspace_path(args.out_root, "D2 output")
    d2_base = D2_BASE.resolve()
    if not out_root.is_relative_to(d2_base):
        raise SystemExit(f"D2 output must be under {d2_base}: {out_root}")
    if out_root.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {out_root}")
    analysis_root = _workspace_path(args.analysis_root, "D1 input")
    c2_root = _workspace_path(args.c2_root, "C2 input")
    b3a_root = _workspace_path(args.b3a_root, "B3-A input")
    if not RENDER_PYTHON.is_file():
        raise FileNotFoundError(RENDER_PYTHON)
    tasks = TASKS if args.task == "all" else [p for p in TASKS if p[0] == args.task]
    summaries = [_build_task(task, label_set, analysis_root, c2_root, b3a_root, out_root, args.max_subgroups) for task, label_set in tasks]
    print(json.dumps({
        "python": sys.executable,
        "rdkit": rdkit.__version__,
        "tasks": [
            {"task": s["task"], "selected_subgroups": s["n_selected"], "selection_records": s["n_records"], "unique_examples": s["unique_examples"], "outputs": s["manifest"]["output_count"]}
            for s in summaries
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
