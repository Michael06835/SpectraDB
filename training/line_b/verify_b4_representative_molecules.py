"""Independent verification for B4.4-SUPPLEMENT representative molecules.

Recomputes the B0 §7.6 selection rule from the frozen Seed-42 Test predictions
and structure library, then checks every generated case, image, metadata and
coverage entry.  Confirms B4.4 is byte-unchanged.

Writes verification.json only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem

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
DEFAULT_ANALYSIS = ROOT / "runs" / "LINE-B-TEST-ANALYSIS" / "B4.4-SUPPLEMENT-REPRESENTATIVE-MOLECULES"


class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, ok, detail=""):
        self.items.append({"id": cid, "ok": bool(ok), "detail": str(detail)})

    def failed(self):
        return [i for i in self.items if not i["ok"]]


def _tree_digest(root: Path) -> str:
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(files[name].encode()); digest.update(b"\n")
    return digest.hexdigest()


def _slug(subgroup: str) -> str:
    return (subgroup.replace(">=", "ge").replace("<=", "le").replace("=", "_")
            .replace("<", "lt").replace(" ", "_").replace("/", "_"))


def _definitions(row_index, structure, quartiles):
    q1, q2, q3 = quartiles["q1"], quartiles["q2"], quartiles["q3"]
    def heavy(v):
        if v <= q1: return "Q1"
        if v <= q2: return "Q2"
        if v <= q3: return "Q3"
        return "Q4"
    def toks(v): return {t.strip() for t in (v or "").split(";") if t.strip()}
    def iv(r, f): return int(float(r[f]))
    rows = [structure[int(i)] for i in row_index]
    defs = []
    specs = [
        ("ring_count", ("0", "1", "2", ">=3"), lambda r: "0" if iv(r, "ring_count") == 0 else "1" if iv(r, "ring_count") == 1 else "2" if iv(r, "ring_count") == 2 else ">=3"),
        ("aromatic_ring_count", ("0", "1", ">=2"), lambda r: "0" if iv(r, "aromatic_ring_count") == 0 else "1" if iv(r, "aromatic_ring_count") == 1 else ">=2"),
        ("n_o_f", ("0", "1", "2", ">=3"), lambda r: str(v) if (v := iv(r, "nitrogen_count") + iv(r, "oxygen_count") + iv(r, "fluorine_count")) < 3 else ">=3"),
        ("heavy_atom_count", ("Q1", "Q2", "Q3", "Q4"), lambda r: heavy(iv(r, "heavy_atom_count"))),
        ("rotatable_bond_count", ("0", "1", ">=2"), lambda r: "0" if iv(r, "rotatable_bond_count") == 0 else "1" if iv(r, "rotatable_bond_count") == 1 else ">=2"),
        ("fraction_csp3", ("<0.5", ">=0.5"), lambda r: "<0.5" if float(r["fraction_csp3"]) < 0.5 else ">=0.5"),
    ]
    for dim, bins, getter in specs:
        for label in bins:
            mask = np.asarray([getter(r) == label for r in rows], dtype=bool)
            defs.append({"subgroup": f"{dim}={label}", "dimension": dim, "bin": label, "mask": mask, "n_samples": int(mask.sum())})
    for group in ("carbonyl", "aromatic", "nitrile", "alkyne", "amine", "hydroxyl"):
        if group == "carbonyl":
            mask = np.asarray(["carbonyl" in toks(r.get("functional_group_types", "")) for r in rows], bool)
        elif group == "aromatic":
            mask = np.asarray([any("aromatic" in t or "aryl" in t for t in toks(r.get("functional_group_types", ""))) for r in rows], bool)
        else:
            mask = np.asarray([group in toks(r.get("functional_group_types", "")) for r in rows], bool)
        defs.append({"subgroup": f"functional_group={group}", "dimension": "functional_group_types", "bin": group, "mask": mask, "n_samples": int(mask.sum())})
    return defs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    analysis = args.analysis_dir.resolve()
    checks = Checks()

    freeze = read_json(args.freeze)
    checks.add("freeze", sha256_file(args.freeze) == args.freeze_sha256 and freeze["status"] == "FROZEN")
    checks.add("protocol", freeze["protocol_sha256"] == sha256_file(ROOT / "docs/experiments/line_b_protocol_draft.md"))
    b44_manifest = read_json(B44 / "analysis_manifest.json")
    b44_verification = read_json(B44 / "verification.json")
    checks.add("b44.verification", b44_verification["status"] == "PASS")
    checks.add("b44.manifest_ref", b44_verification["analysis_manifest"]["sha256"] == sha256_file(B44 / "analysis_manifest.json"))
    b44_bad = []
    for item in b44_manifest["outputs"]:
        path = B44 / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            b44_bad.append(item["path"])
    checks.add("b44.outputs", not b44_bad, ",".join(b44_bad[:3]))
    b44_tree = _tree_digest(B44)

    manifest = read_json(analysis / "analysis_manifest.json")
    summary = read_json(analysis / "summary.json")
    checks.add("manifest.phase", manifest.get("phase") == "B4.4-SUPPLEMENT")
    checks.add("manifest.b44_tree", manifest["inputs"]["b4_4"]["tree_sha256"] == b44_tree)
    bad = []
    for item in manifest["outputs"]:
        path = analysis / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            bad.append(item["path"])
    checks.add("manifest.outputs", not bad, ",".join(bad[:5]))
    checks.add("manifest.input_hashes", all(
        (ROOT / manifest["inputs"][key]["path"]).is_file() and sha256_file(ROOT / manifest["inputs"][key]["path"]) == manifest["inputs"][key]["sha256"]
        for key in ("protocol", "freeze", "structure_library")
    ))
    checks.add("manifest.axes", all(sha256_file(ROOT / manifest["inputs"]["axes"][m]["path"]) == manifest["inputs"]["axes"][m]["sha256"] for m in MODALITIES))
    checks.add("manifest.spectra", all(sha256_file(ROOT / manifest["inputs"]["spectra"][m]["path"]) == manifest["inputs"]["spectra"][m]["sha256"] for m in MODALITIES))

    # Load Seed-42 models and structure.
    models = []
    for record in freeze["models"]:
        if int(record["seed"]) != SEED:
            continue
        d = PREDICTIONS / record["task"] / MODALITY_DIRS[record["modality"]] / record["run_name"] / "test"
        m = read_json(d / "prediction_manifest.json")
        probs = np.load(d / "probabilities.npy", allow_pickle=False)
        row = np.load(d / "row_index.npy", allow_pickle=False)
        labels = np.load(d / "labels.npy", allow_pickle=False)
        checks.add(f"model.hash.{record['task']}.{record['modality']}",
                   sha256_array(probs) == m["arrays"]["probabilities"]["sha256"]
                   and sha256_array(row) == m["arrays"]["row_index"]["sha256"]
                   and sha256_array(labels) == m["arrays"]["labels"]["sha256"])
        names = list(freeze["labels_and_data"]["label_sets"][record["label_set"]]["label_names"]["names"])
        models.append({"task": record["task"], "modality": record["modality"], "row_index": row,
                       "probabilities": probs, "labels": labels, "names": names})
    by_task = {task: {m["modality"]: m for m in models if m["task"] == task} for task, _ in TASKS}
    checks.add("models.counts", all(len(v) == 3 for v in by_task.values()))

    structure = {}
    with STRUCTURE.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            structure[int(row["row_index"])] = row

    examples = list(csv.DictReader((analysis / "examples.csv").open(encoding="utf-8-sig")))
    status = list(csv.DictReader((analysis / "subgroup_status.csv").open(encoding="utf-8-sig")))
    checks.add("examples.nonempty", len(examples) > 0, len(examples))
    checks.add("status.nonempty", len(status) > 0, len(status))

    max_conf_diff = 0.0
    selection_mismatch = 0
    metadata_mismatch = 0
    structure_mismatch = 0
    image_bad = 0
    case_dirs = []
    for task, _label_set in TASKS:
        m = by_task[task]
        row_index = np.asarray(m["ir"]["row_index"], dtype=np.int64)
        y = m["ir"]["labels"]
        probs = {mod: m[mod]["probabilities"] for mod in MODALITIES}
        pred = {mod: probs[mod] >= THRESHOLD for mod in MODALITIES}
        exact = {mod: np.all(pred[mod] == y, axis=1) for mod in MODALITIES}
        correct = exact["ir"] & exact["raman"] & exact["uvvis"]
        confidence = np.stack([probs[mod] for mod in MODALITIES], 0)
        confidence = np.maximum(confidence, 1 - confidence).mean(axis=(0, 2))
        names = m["ir"]["names"]
        quartiles = read_json(D1 / task / "structure_analysis" / "summary.json")["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
        defs = _definitions(row_index, structure, quartiles)
        status_by_group = {r["subgroup"]: r for r in status if r["task"] == task}
        for definition in defs:
            mask = definition["mask"]
            row = status_by_group.get(definition["subgroup"])
            checks.add(f"status.present.{task}.{definition['subgroup']}", row is not None)
            if row is None:
                continue
            n = int(mask.sum())
            if n != int(row["n_samples"]):
                metadata_mismatch += 1
            if n == 0:
                checks.add(f"status.empty.{task}.{definition['subgroup']}", row["correct_status"] == "empty_subgroup")
                continue
            idx = np.flatnonzero(mask)
            median_conf = float(np.median(confidence[idx]))
            cc = idx[correct[idx]]
            ec = idx[~correct[idx]]
            selected = {}
            if len(cc):
                diff = np.abs(confidence[cc] - median_conf)
                order = np.lexsort((row_index[cc], diff))
                selected["correct"] = int(cc[order[0]])
            if len(ec):
                order = np.lexsort((row_index[ec], -confidence[ec]))
                selected["error"] = int(ec[order[0]])
            checks.add(f"status.counts.{task}.{definition['subgroup']}",
                       int(row["n_correct_candidates"]) == len(cc) and int(row["n_error_candidates"]) == len(ec))
            checks.add(f"status.rows.{task}.{definition['subgroup']}",
                       str(row["selected_correct_row"]) == str(int(row_index[selected["correct"]]) if "correct" in selected else "")
                       and str(row["selected_error_row"]) == str(int(row_index[selected["error"]]) if "error" in selected else ""))
            for case in ("correct", "error"):
                expected_pos = selected.get(case)
                match = [r for r in examples if r["task"] == task and r["subgroup"] == definition["subgroup"] and r["case"] == case]
                if expected_pos is None:
                    checks.add(f"example.absent.{task}.{definition['subgroup']}.{case}", not match)
                    continue
                checks.add(f"example.present.{task}.{definition['subgroup']}.{case}", len(match) == 1)
                if len(match) != 1:
                    selection_mismatch += 1
                    continue
                expected_row = int(row_index[expected_pos])
                ex = match[0]
                if int(ex["row_index"]) != expected_row:
                    selection_mismatch += 1
                max_conf_diff = max(max_conf_diff, abs(float(ex["confidence"]) - float(confidence[expected_pos])),
                                    abs(float(ex["median_confidence"]) - median_conf))
                pos = expected_pos
                case_dir = analysis / ex["example_dir"]
                case_dirs.append(case_dir)
                metadata_path = case_dir / "metadata.json"
                if not metadata_path.is_file():
                    metadata_mismatch += 1
                    continue
                meta = read_json(metadata_path)
                if int(meta["row_index"]) != expected_row or meta["case"] != case or meta["subgroup"] != definition["subgroup"]:
                    metadata_mismatch += 1
                if meta["true_labels"] != {name: int(y[pos, i]) for i, name in enumerate(names)}:
                    metadata_mismatch += 1
                for mod in MODALITIES:
                    if meta["predictions"][mod] != {name: int(pred[mod][pos, i]) for i, name in enumerate(names)}:
                        metadata_mismatch += 1
                    if any(abs(meta["probabilities"][mod][name] - float(probs[mod][pos, i])) > 1e-9 for i, name in enumerate(names)):
                        metadata_mismatch += 1
                smiles = meta["structure_library_smiles"]
                if structure[expected_row].get("canonical_smiles") != smiles:
                    structure_mismatch += 1
                mol = Chem.MolFromSmiles(smiles)
                canonical = Chem.MolToSmiles(mol) if mol is not None else None
                if canonical != meta["canonical_smiles"]:
                    structure_mismatch += 1
                for name in ("structure.png", "spectra.png"):
                    p = case_dir / name
                    try:
                        with Image.open(p) as im:
                            if p.stat().st_size < 5000 or im.size[0] < 100:
                                image_bad += 1
                    except Exception:
                        image_bad += 1
                svg = case_dir / "structure.svg"
                if not svg.is_file() or "<svg" not in svg.read_text(encoding="utf-8") or svg.stat().st_size < 500:
                    image_bad += 1
    checks.add("examples.selection", selection_mismatch == 0, selection_mismatch)
    checks.add("examples.metadata", metadata_mismatch == 0, metadata_mismatch)
    checks.add("examples.structure", structure_mismatch == 0, structure_mismatch)
    checks.add("examples.images", image_bad == 0, image_bad)
    checks.add("examples.unique_dirs", len(case_dirs) == len(set(case_dirs)))
    checks.add("examples.confidence", max_conf_diff <= 1e-9, f"{max_conf_diff:.3e}")
    checks.add("b44.tree_unchanged", _tree_digest(B44) == b44_tree)

    failures = checks.failed()
    status_str = "PASS" if not failures else "FAILED"
    report = {
        "phase": "B4.4-SUPPLEMENT-verification",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status_str,
        "analysis_manifest": {"path": relative_to_root(analysis / "analysis_manifest.json"),
                              "sha256": sha256_file(analysis / "analysis_manifest.json")},
        "b4_4_tree_sha256": b44_tree,
        "summary": {"n_checks": len(checks.items), "n_failed": len(failures),
                    "n_examples": len(examples), "n_status_rows": len(status),
                    "selection_mismatch": selection_mismatch, "metadata_mismatch": metadata_mismatch,
                    "structure_mismatch": structure_mismatch, "image_bad": image_bad},
        "checks": checks.items,
        "failures": failures,
    }
    out = args.json_out or (analysis / "verification.json")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite: {out}")
    write_json(out, report)
    print(json.dumps({"status": status_str, "n_checks": len(checks.items), "n_failed": len(failures),
                      "failed_ids": [i["id"] for i in failures], "verification": relative_to_root(out)},
                     ensure_ascii=False, indent=1))
    raise SystemExit(0 if status_str == "PASS" else 1)


if __name__ == "__main__":
    main()
