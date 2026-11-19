"""B4.4-SUPPLEMENT: B0 §7.6 representative-molecule analysis (Test, Seed 42).

Read-only with respect to every existing asset.  Selects, per predefined
non-empty structural subgroup and task, one correct sample whose confidence is
closest to the subgroup median and one highest-confidence error sample.
Renders structure (RDKit), three spectra, and full prediction metadata.

Post-hoc descriptive supplement only; not XAI and not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import read_json, relative_to_root, sha256_array, sha256_file, write_json

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
MODALITIES = ("ir", "raman", "uvvis")
MODALITY_DIRS = {"ir": "IR", "raman": "Raman", "uvvis": "UVVis"}
SEED = 42
THRESHOLD = 0.5
FREEZE = ROOT / "runs" / "LINE-B-MANIFESTS" / "FREEZE.json"
PREDICTIONS = ROOT / "runs" / "LINE-B-TEST-PREDICTIONS"
B44 = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4"
STRUCTURE = ROOT / "processed" / "qm9s" / "structures" / "qm9s_structure_library.csv"
PREPARED = ROOT / "processed" / "qm9s" / "prepared"
D1 = ROOT / "runs" / "LINE-B-ANALYSIS" / "validation_development" / "B3.5-D1" / "final_22subgroups_20reps"
DEFAULT_OUT = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4-SUPPLEMENT-REPRESENTATIVE-MOLECULES"
FONT = Path("C:/Windows/Fonts/arial.ttf")


class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, ok, detail=""):
        self.items.append({"id": cid, "ok": bool(ok), "detail": str(detail)})

    def failed(self):
        return [i for i in self.items if not i["ok"]]


def _sha(path: Path) -> str:
    return sha256_file(path)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _tree_digest(root: Path) -> str:
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha(path)
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(files[name].encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def _slug(subgroup: str) -> str:
    return (subgroup.replace(">=", "ge").replace("<=", "le").replace("=", "_")
            .replace("<", "lt").replace(" ", "_").replace("/", "_"))


def _load_preflight(freeze_path: Path, pin: str) -> dict:
    protocol = ROOT / "docs" / "experiments" / "line_b_protocol_draft.md"
    checks = {}
    if _sha(freeze_path) != pin:
        raise SystemExit("FREEZE SHA256 mismatch")
    freeze = read_json(freeze_path)
    if freeze["status"] != "FROZEN" or freeze["phase"] != "B3":
        raise SystemExit("FREEZE is not frozen B3")
    if freeze["protocol_sha256"] != _sha(protocol):
        raise SystemExit("Protocol SHA mismatch")
    b44_manifest = read_json(B44 / "analysis_manifest.json")
    b44_verification = read_json(B44 / "verification.json")
    if b44_verification.get("status") != "PASS":
        raise SystemExit("B4.4 verification is not PASS")
    if b44_verification["analysis_manifest"]["sha256"] != _sha(B44 / "analysis_manifest.json"):
        raise SystemExit("B4.4 verification does not reference the current manifest")
    bad = []
    for item in b44_manifest["outputs"]:
        path = B44 / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            bad.append(item["path"])
    if bad:
        raise SystemExit(f"B4.4 output hash mismatch: {bad[:5]}")

    models = []
    for record in freeze["models"]:
        if int(record["seed"]) != SEED:
            continue
        out_dir = PREDICTIONS / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        manifest = read_json(out_dir / "prediction_manifest.json")
        probs = np.load(out_dir / "probabilities.npy", allow_pickle=False)
        row_index = np.load(out_dir / "row_index.npy", allow_pickle=False)
        labels = np.load(out_dir / "labels.npy", allow_pickle=False)
        if sha256_array(probs) != manifest["arrays"]["probabilities"]["sha256"]:
            raise SystemExit(f"probability hash mismatch: {out_dir}")
        if sha256_array(row_index) != manifest["arrays"]["row_index"]["sha256"]:
            raise SystemExit(f"row_index hash mismatch: {out_dir}")
        if sha256_array(labels) != manifest["arrays"]["labels"]["sha256"]:
            raise SystemExit(f"labels hash mismatch: {out_dir}")
        if (manifest.get("freeze") or {}).get("sha256") != pin:
            raise SystemExit(f"FREEZE pin mismatch: {out_dir}")
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        models.append({
            "task": record["task"], "label_set": record["label_set"], "modality": record["modality"],
            "run_name": record["run_name"], "seed": int(record["seed"]),
            "probabilities": probs, "labels": labels, "row_index": row_index, "names": names,
            "manifest_sha256": _sha(out_dir / "prediction_manifest.json"),
            "probabilities_sha256": manifest["arrays"]["probabilities"]["sha256"],
            "row_index_sha256": manifest["arrays"]["row_index"]["sha256"],
            "labels_sha256": manifest["arrays"]["labels"]["sha256"],
        })
    if len(models) != 6:
        raise SystemExit(f"Expected 6 Seed-42 models, got {len(models)}")

    structure = {}
    wanted = set(int(i) for i in models[0]["row_index"])
    with STRUCTURE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["row_index"])
            if index in wanted:
                structure[index] = row
    if len(structure) != len(wanted):
        raise SystemExit("Structure library missing Test rows")

    axes = {name: np.load(PREPARED / f"{name}_axis.npy") for name in MODALITIES}
    spectra_paths = {name: PREPARED / f"{name}_float32.npy" for name in MODALITIES}
    return {
        "freeze": freeze, "freeze_path": freeze_path, "freeze_sha256": pin,
        "protocol_sha256": _sha(protocol), "protocol_path": relative_to_root(protocol),
        "b44_manifest": b44_manifest, "b44_verification": b44_verification,
        "models": models, "structure": structure, "axes": axes, "spectra_paths": spectra_paths,
        "b44_tree": _tree_digest(B44),
        "test_access_log": {"path": "runs/LINE-B-MANIFESTS/test_access_log.json",
                            "sha256": _sha(ROOT / "runs/LINE-B-MANIFESTS/test_access_log.json")},
        "d1": {task: {"summary": _sha(D1 / task / "structure_analysis" / "summary.json"),
                      "definitions": _sha(D1 / task / "structure_analysis" / "subgroup_definitions.csv")}
               for task, _ in TASKS},
    }


def _subgroup_definitions(row_index: np.ndarray, structure: dict[int, dict], quartiles: dict):
    q1, q2, q3 = quartiles["q1"], quartiles["q2"], quartiles["q3"]
    def heavy_bin(v):
        if v <= q1: return "Q1"
        if v <= q2: return "Q2"
        if v <= q3: return "Q3"
        return "Q4"
    def toks(value):
        return {t.strip() for t in (value or "").split(";") if t.strip()}
    def ival(r, f): return int(float(r[f]))
    rows = [structure[int(i)] for i in row_index]
    definitions = []
    specs = [
        ("ring_count", ("0", "1", "2", ">=3"), lambda r: "0" if ival(r, "ring_count") == 0 else "1" if ival(r, "ring_count") == 1 else "2" if ival(r, "ring_count") == 2 else ">=3"),
        ("aromatic_ring_count", ("0", "1", ">=2"), lambda r: "0" if ival(r, "aromatic_ring_count") == 0 else "1" if ival(r, "aromatic_ring_count") == 1 else ">=2"),
        ("n_o_f", ("0", "1", "2", ">=3"), lambda r: str(v) if (v := ival(r, "nitrogen_count") + ival(r, "oxygen_count") + ival(r, "fluorine_count")) < 3 else ">=3"),
        ("heavy_atom_count", ("Q1", "Q2", "Q3", "Q4"), lambda r: heavy_bin(ival(r, "heavy_atom_count"))),
        ("rotatable_bond_count", ("0", "1", ">=2"), lambda r: "0" if ival(r, "rotatable_bond_count") == 0 else "1" if ival(r, "rotatable_bond_count") == 1 else ">=2"),
        ("fraction_csp3", ("<0.5", ">=0.5"), lambda r: "<0.5" if float(r["fraction_csp3"]) < 0.5 else ">=0.5"),
    ]
    for dimension, bins, getter in specs:
        for label in bins:
            mask = np.asarray([getter(r) == label for r in rows], dtype=bool)
            definitions.append({"subgroup": f"{dimension}={label}", "dimension": dimension, "bin": label,
                                "mask": mask, "n_samples": int(mask.sum())})
    for group in ("carbonyl", "aromatic", "nitrile", "alkyne", "amine", "hydroxyl"):
        if group == "carbonyl":
            mask = np.asarray(["carbonyl" in toks(r.get("functional_group_types", "")) for r in rows], dtype=bool)
        elif group == "aromatic":
            mask = np.asarray([any("aromatic" in t or "aryl" in t for t in toks(r.get("functional_group_types", ""))) for r in rows], dtype=bool)
        else:
            mask = np.asarray([group in toks(r.get("functional_group_types", "")) for r in rows], dtype=bool)
        definitions.append({"subgroup": f"functional_group={group}", "dimension": "functional_group_types",
                            "bin": group, "mask": mask, "n_samples": int(mask.sum())})
    return definitions


def _confidence(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack([probabilities[mod] for mod in MODALITIES], axis=0)  # 3 x n x L
    return np.maximum(stacked, 1.0 - stacked).mean(axis=(0, 2))


def _select(task: str, models: list[dict], definitions: list[dict]) -> tuple[list[dict], list[dict], dict]:
    by_mod = {m["modality"]: m for m in models}
    names = models[0]["names"]
    y = models[0]["labels"]
    probs = {mod: by_mod[mod]["probabilities"] for mod in MODALITIES}
    row_index = np.asarray(models[0]["row_index"], dtype=np.int64)
    pred = {mod: probs[mod] >= THRESHOLD for mod in MODALITIES}
    exact = {mod: np.all(pred[mod] == y, axis=1) for mod in MODALITIES}
    correct = exact["ir"] & exact["raman"] & exact["uvvis"]
    error = ~correct
    confidence = _confidence(probs)
    examples = []
    status_rows = []
    for definition in definitions:
        mask = definition["mask"]
        n = int(mask.sum())
        if n == 0:
            status_rows.append({
                "task": task, "subgroup": definition["subgroup"], "dimension": definition["dimension"],
                "bin": definition["bin"], "n_samples": 0, "positive_counts": "{}",
                "n_correct_candidates": 0, "n_error_candidates": 0,
                "correct_status": "empty_subgroup", "error_status": "empty_subgroup",
                "selected_correct_row": "", "selected_error_row": "",
            })
            continue
        idx = np.flatnonzero(mask)
        positive = {name: int(y[idx, j].sum()) for j, name in enumerate(names)}
        median_conf = float(np.median(confidence[idx]))
        correct_candidates = idx[correct[idx]]
        error_candidates = idx[error[idx]]
        selected = {}
        if len(correct_candidates):
            diff = np.abs(confidence[correct_candidates] - median_conf)
            order = np.lexsort((row_index[correct_candidates], diff))
            selected["correct"] = int(correct_candidates[order[0]])
        if len(error_candidates):
            order = np.lexsort((row_index[error_candidates], -confidence[error_candidates]))
            selected["error"] = int(error_candidates[order[0]])
        for case in ("correct", "error"):
            if case not in selected:
                continue
            pos = selected[case]
            row = int(row_index[pos])
            examples.append({
                "task": task, "subgroup": definition["subgroup"], "dimension": definition["dimension"],
                "bin": definition["bin"], "case": case, "row_index": row,
                "confidence": float(confidence[pos]), "median_confidence": median_conf,
                "n_samples": n, "n_correct_candidates": int(len(correct_candidates)),
                "n_error_candidates": int(len(error_candidates)),
                "positive_counts": positive,
                "pred": {mod: pred[mod][pos].astype(int) for mod in MODALITIES},
                "exact": {mod: bool(exact[mod][pos]) for mod in MODALITIES},
                "probabilities": {mod: probs[mod][pos] for mod in MODALITIES},
                "labels": y[pos].astype(int),
                "position": pos,
            })
        status_rows.append({
            "task": task, "subgroup": definition["subgroup"], "dimension": definition["dimension"],
            "bin": definition["bin"], "n_samples": n, "positive_counts": json.dumps(positive, ensure_ascii=False),
            "n_correct_candidates": int(len(correct_candidates)), "n_error_candidates": int(len(error_candidates)),
            "correct_status": "ok" if "correct" in selected else "no_correct_candidate",
            "error_status": "ok" if "error" in selected else "no_error_candidate",
            "selected_correct_row": int(row_index[selected["correct"]]) if "correct" in selected else "",
            "selected_error_row": int(row_index[selected["error"]]) if "error" in selected else "",
        })
    return examples, status_rows, {"correct": correct, "error": error, "confidence": confidence}


def _render_structure(smiles: str, png: Path, svg: Path) -> dict:
    from scripts.qm9s_structure_tools import canonicalize_and_renumber, make_atom_labels, draw_structure
    from rdkit.Chem import rdDepictor
    mol, canonical = canonicalize_and_renumber(smiles)
    labels = make_atom_labels(mol)
    rdDepictor.Compute2DCoords(mol)
    draw_structure(mol, labels, png, "png")
    draw_structure(mol, labels, svg, "svg")
    return {"canonical_smiles": canonical, "atom_labels": {str(k + 1): v for k, v in labels.items()}}


def _render_spectra(spectra: dict, axes: dict, labels: list[str], png: Path, row_index: int) -> None:
    width, height = 1500, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 20)
    small = ImageFont.truetype(str(FONT), 14)
    panel_h = 250
    tops = (120, 460, 800)
    for panel, mod in enumerate(MODALITIES):
        x0, y0, x1, y1 = 140, tops[panel], 1420, tops[panel] + panel_h
        values = spectra[mod]
        axis = axes[mod]
        draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmax == vmin:
            vmax = vmin + 1.0
        for tick in range(6):
            x = x0 + (x1 - x0) * tick / 5
            draw.line((x, y1, x, y1 + 6), fill=(60, 60, 60), width=1)
            draw.text((x, y1 + 10), f"{axis[0] + (axis[-1]-axis[0])*tick/5:.2f}", font=small, fill=(30,30,30), anchor="ma")
        points = [(x0 + (x1-x0) * i/(len(values)-1), y1 - (y1-y0) * (float(v)-vmin)/(vmax-vmin)) for i, v in enumerate(values)]
        draw.line(points, fill={"ir": (31,119,180), "raman": (214,39,40), "uvvis": (44,160,44)}[mod], width=2)
        draw.text((x0, y0 - 30), f"{mod.upper()} spectrum (row {row_index})", font=font, fill=(20,20,20))
        draw.text((x0, y0 + 8), f"axis {axis[0]:.2f}..{axis[-1]:.2f} (prepared axis value; units not recorded)", font=small, fill=(80,80,80))
        draw.text((x0 - 12, (y0+y1)/2), "intensity", font=small, fill=(30,30,30), anchor="rm")
    draw.text((width/2, 30), f"Test spectra — row_index {row_index} ({len(labels)} labels)", font=font, fill=(20,20,20), anchor="ma")
    image.save(png, "PNG")


def _build_report(examples, status_rows, task_counts, selection_notes) -> str:
    lines = ["# B4.4-SUPPLEMENT — Test Representative Molecules (B0 §7.6)", ""]
    lines.append("> **Post-hoc descriptive supplement.** Selection uses the pre-specified B0 §7.6 rule with Seed 42 and threshold 0.5. "
                 "It is qualitative only, is not XAI, and does not constitute confirmatory evidence.")
    lines.append("")
    lines.append("## Selection rule")
    lines.append("")
    lines.append("- Correct sample: all three modalities exactly match every label; choose the sample whose confidence is closest to the subgroup's median confidence (ties: row_index ascending).")
    lines.append("- Error sample: at least one modality/label is wrong; choose the highest-confidence error sample (ties: row_index ascending).")
    lines.append("- Confidence per sample = mean over 3 modalities × all task labels of max(p, 1−p).")
    lines.append("- Subgroup definitions are the frozen D1/B0 §7.5 definitions; support tables are separate from case selection.")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Task | Non-empty subgroups | Correct cases | Error cases | Missing statuses |")
    lines.append("|---|---|---|---|---|")
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in status_rows if r["task"] == task]
        nonempty = [r for r in rows if int(r["n_samples"]) > 0]
        missing = [r for r in rows if r["correct_status"] != "ok" or r["error_status"] != "ok"]
        lines.append(f"| {task} | {len(nonempty)} | {task_counts[task]['correct']} | {task_counts[task]['error']} | {len(missing)} |")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    lines.append("| Task | Subgroup | Case | row_index | Confidence | Subgroup median | Exact (IR/Raman/UV) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in examples:
        exact = "/".join("Y" if row["exact"][m] else "N" for m in MODALITIES)
        lines.append(f'| {row["task"]} | {row["subgroup"]} | {row["case"]} | {row["row_index"]} | {row["confidence"]:.4f} | '
                     f'{row["median_confidence"]:.4f} | {exact} |')
    lines.append("")
    lines.append("## Status and caveats")
    lines.append("")
    for task in ("Shared14", "UVSPEC15"):
        rows = [r for r in status_rows if r["task"] == task]
        for r in rows:
            if r["correct_status"] != "ok" or r["error_status"] != "ok":
                lines.append(f'- {task} `{r["subgroup"]}`: n={r["n_samples"]}, correct={r["correct_status"]}, error={r["error_status"]}.')
    lines.append("- Spectra use the prepared axes (IR/Raman 500–4000; UV-Vis 1–15); the prepared metadata records axis source but not units, so panels label the axis value rather than asserting units.")
    lines.append("- Structure images are RDKit renders of the structure-library canonical SMILES; no Pillow schematic fallback is used.")
    lines.append("- Overlapping subgroups may select the same molecule; no molecule is used as both correct and error case within one subgroup.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    if args.out_dir.exists():
        raise SystemExit(f"Output directory exists: {args.out_dir}")
    preflight = _load_preflight(args.freeze, args.freeze_sha256)

    outputs = []
    examples_all = []
    status_all = []
    task_counts = {}
    structure_sha = _sha(STRUCTURE)
    spectra_sha = {mod: _sha(preflight["spectra_paths"][mod]) for mod in MODALITIES}
    spectra_meta = {}
    for mod in MODALITIES:
        array = np.load(preflight["spectra_paths"][mod], mmap_mode="r")
        spectra_meta[mod] = {"file": relative_to_root(preflight["spectra_paths"][mod]),
                             "sha256": spectra_sha[mod], "shape": [int(x) for x in array.shape],
                             "dtype": str(array.dtype)}
    with __import__("training.line_b.common", fromlist=["atomic_output_dir"]).atomic_output_dir(args.out_dir) as temporary:
        for task, label_set in TASKS:
            task_dir = temporary / task
            models = [m for m in preflight["models"] if m["task"] == task]
            if len(models) != 3:
                raise SystemExit(f"{task}: expected 3 Seed-42 models")
            quartiles = read_json(D1 / task / "structure_analysis" / "summary.json")["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
            definitions = _subgroup_definitions(np.asarray(models[0]["row_index"]), preflight["structure"], quartiles)
            examples, status_rows, _ = _select(task, models, definitions)
            examples_all += examples
            status_all += status_rows
            task_counts[task] = {"correct": sum(1 for e in examples if e["case"] == "correct"),
                                 "error": sum(1 for e in examples if e["case"] == "error")}
            for example in examples:
                case_dir = task_dir / "examples" / _slug(example["subgroup"]) / f"row_{example['row_index']:06d}"
                case_dir.mkdir(parents=True, exist_ok=False)
                structure = preflight["structure"][example["row_index"]]
                structure_info = _render_structure(structure["canonical_smiles"],
                                                  case_dir / "structure.png", case_dir / "structure.svg")
                spectra = {mod: np.load(preflight["spectra_paths"][mod], mmap_mode="r")[example["row_index"]].astype(np.float32)
                           for mod in MODALITIES}
                _render_spectra(spectra, preflight["axes"], models[0]["names"], case_dir / "spectra.png", example["row_index"])
                metadata = {
                    "task": task, "label_set": label_set, "subgroup": example["subgroup"],
                    "dimension": example["dimension"], "bin": example["bin"], "case": example["case"],
                    "row_index": int(example["row_index"]), "seed": SEED, "threshold": THRESHOLD,
                    "selection_rule": ("correct: min |confidence - subgroup median|; tie row_index asc"
                                       if example["case"] == "correct" else
                                       "error: max confidence; tie row_index asc"),
                    "confidence": example["confidence"], "subgroup_median_confidence": example["median_confidence"],
                    "n_subgroup_samples": example["n_samples"],
                    "n_correct_candidates": example["n_correct_candidates"],
                    "n_error_candidates": example["n_error_candidates"],
                    "positive_counts": example["positive_counts"],
                    "true_labels": {name: int(example["labels"][i]) for i, name in enumerate(models[0]["names"])},
                    "predictions": {mod: {name: int(example["pred"][mod][i]) for i, name in enumerate(models[0]["names"])} for mod in MODALITIES},
                    "probabilities": {mod: {name: float(example["probabilities"][mod][i]) for i, name in enumerate(models[0]["names"])} for mod in MODALITIES},
                    "errors": {mod: [name for i, name in enumerate(models[0]["names"]) if example["pred"][mod][i] != example["labels"][i]] for mod in MODALITIES},
                    "exact_match": example["exact"],
                    "canonical_smiles": structure_info["canonical_smiles"],
                    "structure_library_smiles": structure["canonical_smiles"],
                    "molecular_formula": structure.get("molecular_formula"),
                    "qm9_number": structure.get("qm9_number"),
                    "atom_labels": structure_info["atom_labels"],
                    "structure_files": {"png": "structure.png", "svg": "structure.svg", "renderer": "RDKit rdMolDraw2D"},
                    "spectra_files": {"png": "spectra.png"},
                    "spectra_axes": {mod: {"file": relative_to_root(PREPARED / f"{mod}_axis.npy"),
                                            "n_points": int(len(preflight["axes"][mod])),
                                            "min": float(preflight["axes"][mod][0]),
                                            "max": float(preflight["axes"][mod][-1]),
                                            "units": "not recorded in prepared metadata"}
                                     for mod in MODALITIES},
                    "spectra_matrices": spectra_meta,
                    "source_hashes": {
                        "structure_library": structure_sha,
                        "structure_library_file": relative_to_root(STRUCTURE),
                        "d1_definitions": preflight["d1"][task]["definitions"],
                        "d1_summary": preflight["d1"][task]["summary"],
                        "b4_1_prediction_manifest": {m["modality"]: m["manifest_sha256"] for m in models},
                        "b4_1_probabilities": {m["modality"]: m["probabilities_sha256"] for m in models},
                    },
                }
                write_json(case_dir / "metadata.json", metadata)

        for row in examples_all:
            row["example_dir"] = (Path(row["task"]) / "examples" / _slug(row["subgroup"]) / f"row_{row['row_index']:06d}").as_posix()
        _write_csv(temporary / "examples.csv", examples_all,
                   ["task", "subgroup", "dimension", "bin", "case", "row_index", "confidence",
                    "median_confidence", "n_samples", "n_correct_candidates", "n_error_candidates",
                    "example_dir"])
        _write_csv(temporary / "subgroup_status.csv", status_all,
                   ["task", "subgroup", "dimension", "bin", "n_samples", "positive_counts",
                    "n_correct_candidates", "n_error_candidates", "correct_status", "error_status",
                    "selected_correct_row", "selected_error_row"])
        (temporary / "report.md").write_text(
            _build_report(examples_all, status_all, task_counts, {}), encoding="utf-8")

        summary = {
            "phase": "B4.4-SUPPLEMENT",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "classification": {"status": "post-hoc descriptive supplement", "xai": False,
                               "confirmatory": False, "selection_rule": "B0 v0.2 §7.6"},
            "config": {"seed": SEED, "threshold": THRESHOLD, "tasks": [t for t, _ in TASKS],
                       "b44_tree_sha256_before": preflight["b44_tree"]},
            "counts": {"examples": len(examples_all),
                       "correct": sum(1 for e in examples_all if e["case"] == "correct"),
                       "error": sum(1 for e in examples_all if e["case"] == "error"),
                       "subgroups_nonempty": sum(1 for s in status_all if int(s["n_samples"]) > 0),
                       "tasks": task_counts},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "rdkit": __import__("rdkit").__version__},
            "elapsed_started_at": started,
        }
        write_json(temporary / "summary.json", summary)
        files = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "analysis_manifest.json":
                files.append({"path": path.relative_to(temporary).as_posix(),
                              "sha256": _sha(path), "size_bytes": int(path.stat().st_size)})
        manifest = {
            "phase": "B4.4-SUPPLEMENT",
            "generated_at": summary["generated_at"],
            "classification": summary["classification"],
            "config": summary["config"],
            "inputs": {
                "protocol": {"path": preflight["protocol_path"], "sha256": preflight["protocol_sha256"]},
                "freeze": {"path": relative_to_root(args.freeze), "sha256": args.freeze_sha256},
                "test_access_log": preflight["test_access_log"],
                "b4_4": {"path": relative_to_root(B44),
                          "analysis_manifest_sha256": _sha(B44 / "analysis_manifest.json"),
                          "verification_sha256": _sha(B44 / "verification.json"),
                          "tree_sha256": preflight["b44_tree"]},
                "b4_1_seed42": [
                    {"task": m["task"], "modality": m["modality"], "run_name": m["run_name"],
                     "prediction_manifest_sha256": m["manifest_sha256"],
                     "probabilities_sha256": m["probabilities_sha256"],
                     "labels_sha256": m["labels_sha256"], "row_index_sha256": m["row_index_sha256"]}
                    for m in preflight["models"]
                ],
                "structure_library": {"path": relative_to_root(STRUCTURE), "sha256": _sha(STRUCTURE)},
                "spectra": {mod: {"path": relative_to_root(preflight["spectra_paths"][mod]),
                                   "sha256": _sha(preflight["spectra_paths"][mod])} for mod in MODALITIES},
                "axes": {mod: {"path": relative_to_root(PREPARED / f"{mod}_axis.npy"),
                                "sha256": _sha(PREPARED / f"{mod}_axis.npy")} for mod in MODALITIES},
                "d1": preflight["d1"],
            },
            "outputs": files,
            "code": {
                "analysis_script": relative_to_root(Path(__file__)),
                "analysis_script_sha256": _sha(Path(__file__)),
                "verifier": "training/line_b/verify_b4_representative_molecules.py",
                "verifier_sha256": _sha(ROOT / "training/line_b/verify_b4_representative_molecules.py")
                if (ROOT / "training/line_b/verify_b4_representative_molecules.py").is_file() else None,
                "structure_tools_sha256": _sha(ROOT / "scripts" / "qm9s_structure_tools.py"),
            },
            "environment": summary["environment"],
        }
        write_json(temporary / "analysis_manifest.json", manifest)
        b44_after = _tree_digest(B44)
        if b44_after != preflight["b44_tree"]:
            raise SystemExit("B4.4 tree changed during supplement")

    print(json.dumps({"out_dir": relative_to_root(args.out_dir), "counts": summary["counts"],
                      "b44_tree_unchanged": b44_after == preflight["b44_tree"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
