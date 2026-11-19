"""Independent verifier for B3.5-D2 Validation-development reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.line_b.common import sha256_file
from training.line_b.analyze_line_b_complementarity import _load_task_inputs

TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
REQUIRED_TABLES = (
    "input_audit.csv", "global_performance.csv", "per_label_mean_sd.csv",
    "cooccurrence_counts.csv", "cooccurrence_phi.csv",
    "cooccurrence_cramers_v.csv", "cooccurrence_support.csv",
    "hierarchy_pairs.csv", "semantic_groups.csv", "subgroup_definitions.csv",
    "subgroup_membership_audit.csv", "subgroup_metrics_main.csv",
    "subgroup_conditional_by_label.csv", "subgroup_conditional_three_seed.csv",
    "subgroup_oracle_by_label.csv", "subgroup_oracle_by_label_three_seed.csv",
    "subgroup_oracle_three_seed.csv", "subgroup_or_gain_three_seed.csv",
    "subgroup_molecule_exact_three_seed.csv", "subgroup_bootstrap_sanity.csv",
    "subgroup_bootstrap_macro_counts.csv", "representative_molecules.csv",
    "mcnemar_holm.csv", "spearman.csv", "sample_error_summary.csv",
    "hard_samples.csv", "or_gain_summary.csv", "high_confidence_errors.csv",
    "b3a_model_mean_sd.csv", "b3a_pairwise_mean_sd.csv",
)
REQUIRED_FIGURES = (
    "global_performance.png", "cooccurrence_heatmap.png", "phi_heatmap.png",
    "cramers_v_heatmap.png", "subgroup_support.png", "bootstrap_uv_ir_ap.png",
    "conditional_delta.png",
)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _relocation_records() -> list[dict]:
    manifest = ROOT / "metadata/workspace_asset_relocation_manifest.json"
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return data.get("records", [])


def _workspace_path_text(path_text: str) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    return p if p.is_relative_to(ROOT.resolve()) else None


def _resolve_historical(path_text: str, expected_sha: str | None, expected_size: int | None, records: list[dict]) -> Path | None:
    """Resolve an old external path only through exact relocation records.

    All candidate fields are collected and validated before a unique result is
    accepted. A direct workspace path is checked with the same size/SHA rules;
    a mismatch never falls back to another relocation target.
    """
    if not path_text:
        return None
    def valid(path: Path | None) -> bool:
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return False
        if expected_size is not None and path.stat().st_size != int(expected_size):
            return False
        if expected_sha is not None and _sha(path) != expected_sha:
            return False
        return True
    direct = _workspace_path_text(path_text)
    if direct is not None and direct.exists():
        return direct if valid(direct) else None
    normalized = os.path.normcase(str(Path(path_text)))
    candidates = []
    for rec in records:
        if os.path.normcase(rec.get("source_path", "")) != normalized:
            continue
        for key in ("proposed_destination", "archive_destination", "previous_destination"):
            p = _workspace_path_text(rec.get(key, ""))
            if valid(p):
                candidates.append(p.resolve())
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def _independent_masks(d1: dict, d1_dir: Path, val_rows: np.ndarray, structure: dict[int, dict], selected: set[str]) -> dict[str, np.ndarray]:
    q = d1["subgroups"]["heavy_atom"]["heavy_atom_quartiles"]
    q1, q2, q3 = float(q["q1"]), float(q["q2"]), float(q["q3"])
    masks = {}
    for row in _csv(d1_dir / "subgroup_definitions.csv"):
        name = row["subgroup"]
        if name not in selected:
            continue
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
            raise ValueError(name)
        masks[name] = mask
    return masks


def _independent_selection(task_info: dict, masks: dict[str, np.ndarray]) -> dict[tuple[str, str], tuple[int, float]]:
    row_index = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    models = {(m["modality"], int(m["seed"])): m for m in task_info["models"]}
    exact = {}; confidence = {}
    per_model_confidence = {}
    for modality in ("ir", "raman", "uvvis"):
        model = models[(modality, 42)]
        pred = model["probabilities"] >= 0.5
        exact[modality] = np.all(pred == (model["labels"] == 1), axis=1)
        per_model_confidence[modality] = np.maximum(model["probabilities"], 1.0 - model["probabilities"]).mean(axis=1)
    correct = exact["ir"] & exact["raman"] & exact["uvvis"]
    confidence = (per_model_confidence["ir"] + per_model_confidence["raman"] + per_model_confidence["uvvis"]) / 3.0
    result = {}
    for subgroup, mask in masks.items():
        positions = np.flatnonzero(mask)
        cp = [int(i) for i in positions if correct[i]]
        ep = [int(i) for i in positions if not correct[i]]
        if cp:
            median = float(np.median(confidence[cp]))
            chosen = min(cp, key=lambda i: (abs(float(confidence[i]) - median), int(row_index[i])))
            result[(subgroup, "correct")] = (int(row_index[chosen]), float(confidence[chosen]))
        if ep:
            chosen = min(ep, key=lambda i: (-float(confidence[i]), int(row_index[i])))
            result[(subgroup, "error")] = (int(row_index[chosen]), float(confidence[chosen]))
    return result


def _structure_map() -> dict[int, dict]:
    mapping = {}
    with (ROOT / "processed/qm9s/structures/qm9s_structure_library.csv").open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["row_index"])] = row
    if len(mapping) != 129817:
        raise ValueError("Structure library incomplete")
    return mapping


def _asset_path(row: dict, field: str) -> str:
    return str(row.get(field, "") or "")


def _safe_task_path(task_root: Path, path_text: str) -> Path | None:
    if not path_text:
        return None
    rel = Path(path_text)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    try:
        resolved = (task_root / rel).resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(task_root.resolve()):
        return None
    if not resolved.is_file() or resolved.stat().st_size == 0:
        return None
    return resolved


def _check_relative_asset(task: str, task_root: Path, row: dict, field: str, expected_name: str, add) -> Path | None:
    text = _asset_path(row, field)
    if not text:
        add(f"{task}.asset.{field}.nonempty", False, "empty path")
        return None
    rel = Path(text)
    ok_relative = not rel.is_absolute() and ".." not in rel.parts
    add(f"{task}.asset.{field}.relative", ok_relative, text)
    if not ok_relative:
        return None
    try:
        resolved = (task_root / rel).resolve()
    except OSError as exc:
        add(f"{task}.asset.{field}.resolve", False, str(exc))
        return None
    inside = resolved.is_relative_to(task_root.resolve())
    add(f"{task}.asset.{field}.inside", inside, text)
    if not inside:
        return None
    exists = resolved.is_file() and resolved.stat().st_size > 0
    add(f"{task}.asset.{field}.exists", exists, text)
    if not exists:
        return None
    name_ok = rel.as_posix() == expected_name
    add(f"{task}.asset.{field}.expected_name", name_ok, f"{rel.as_posix()} != {expected_name}")
    if not name_ok:
        return None
    return resolved


def _strip_atom_maps(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _validate_mol_sdf(task: str, task_root: Path, rows: list[dict], add) -> dict:
    selected = {int(r["row_index"]) for r in rows if str(r.get("row_index", "")).strip() not in ("", "None")}
    stats = {"mol_files": 0, "sdf_files": 0, "mol_parsed": 0, "sdf_parsed": 0, "mol_failures": 0, "sdf_failures": 0, "failures": 0}
    for row in sorted(selected):
        mol_path = _safe_task_path(task_root, f"rendered_structures/rendered/row_{row:06d}/structure_atom_mapped.mol")
        sdf_path = _safe_task_path(task_root, f"rendered_structures/rendered/row_{row:06d}/structure_atom_mapped.sdf")
        metadata_path = _safe_task_path(task_root, f"examples/row_{row:06d}/metadata.json")
        if metadata_path is None:
            add(f"{task}.row_{row}.mol_sdf_metadata_safe", False, str(task_root / "examples" / f"row_{row:06d}" / "metadata.json"))
            stats["mol_failures"] += 1; stats["sdf_failures"] += 1
            stats["failures"] += 2
            continue
        add(f"{task}.row_{row}.mol_sdf_metadata_safe", True, str(metadata_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        canonical = Chem.MolFromSmiles(metadata["canonical_smiles"])
        expected_formula = metadata["molecular_formula"]
        mol_ok = False
        if mol_path is not None:
            stats["mol_files"] += 1
            mol = Chem.MolFromMolBlock(mol_path.read_text(encoding="utf-8", errors="replace"), sanitize=True, removeHs=False)
            parse_ok = mol is not None
            add(f"{task}.row_{row}.mol_parse", parse_ok, str(mol_path))
            if parse_ok:
                stats["mol_parsed"] += 1
                formula_ok = rdMolDescriptors.CalcMolFormula(mol) == expected_formula
                canonical_ok = Chem.MolToSmiles(_strip_atom_maps(mol), canonical=True, isomericSmiles=True) == Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
                add(f"{task}.row_{row}.mol_formula", formula_ok)
                add(f"{task}.row_{row}.mol_canonical", canonical_ok)
                mol_ok = formula_ok and canonical_ok
        else:
            add(f"{task}.row_{row}.mol_present", False, str(task_root / "rendered_structures/rendered" / f"row_{row:06d}" / "structure_atom_mapped.mol"))
        if not mol_ok:
            stats["mol_failures"] += 1
        sdf_ok = False
        if sdf_path is not None:
            stats["sdf_files"] += 1
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(sdf_path.read_bytes()), sanitize=True, removeHs=False)
            mols = [m for m in supplier]
            parse_ok = len(mols) == 1 and mols[0] is not None
            add(f"{task}.row_{row}.sdf_parse", parse_ok, f"{sdf_path} molecules={len(mols)}")
            if parse_ok:
                stats["sdf_parsed"] += 1
                formula_ok = rdMolDescriptors.CalcMolFormula(mols[0]) == expected_formula
                canonical_ok = Chem.MolToSmiles(_strip_atom_maps(mols[0]), canonical=True, isomericSmiles=True) == Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
                add(f"{task}.row_{row}.sdf_formula", formula_ok)
                add(f"{task}.row_{row}.sdf_canonical", canonical_ok)
                sdf_ok = formula_ok and canonical_ok
        else:
            add(f"{task}.row_{row}.sdf_present", False, str(task_root / "rendered_structures/rendered" / f"row_{row:06d}" / "structure_atom_mapped.sdf"))
        if not sdf_ok:
            stats["sdf_failures"] += 1
    stats["failures"] = stats["mol_failures"] + stats["sdf_failures"]
    counts_ok = (
        stats["mol_files"] == len(selected) and stats["sdf_files"] == len(selected)
        and stats["mol_parsed"] == len(selected) and stats["sdf_parsed"] == len(selected)
        and stats["failures"] == 0
    )
    add(f"{task}.mol_sdf_counts", counts_ok, json.dumps(stats))
    return stats


def verify(report_root: Path, max_subgroups: int | None = None, legacy_r1: bool = False) -> dict:
    checks = []
    failures = []
    warnings = []
    expected_groups = max_subgroups if max_subgroups else 22
    expected_records = 2 * expected_groups

    def add(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAILED"
        checks.append({"check": name, "status": status, "detail": detail})
        if not ok:
            failures.append(f"{name}: {detail}")

    report_root = Path(report_root).resolve()
    add("report_root", report_root.is_dir(), str(report_root))
    add("report_root_inside_workspace", report_root.is_relative_to(ROOT.resolve()), str(ROOT))
    if not report_root.is_dir() or not report_root.is_relative_to(ROOT.resolve()):
        return {"status": "FAILED", "checks": checks, "failures": failures, "warnings": warnings}
    relocation = _relocation_records()

    for task, label_set in TASKS:
        task_root = (report_root / task / "validation_report").resolve()
        structure_ok = (
            task_root.is_dir()
            and task_root.name == "validation_report"
            and task_root.parent.name == task
            and task_root.parent.parent == report_root
        )
        add(f"{task}.task_root_exists", task_root.is_dir(), str(task_root))
        add(f"{task}.task_root_inside_workspace", task_root.is_relative_to(ROOT.resolve()), str(task_root))
        add(f"{task}.task_root_inside_report", task_root.is_relative_to(report_root), str(task_root))
        add(f"{task}.task_root_structure", structure_ok, str(task_root))
        if not structure_ok or not task_root.is_relative_to(ROOT.resolve()) or not task_root.is_relative_to(report_root):
            continue
        manifest_path = _safe_task_path(task_root, "report_manifest.json")
        add(f"{task}.manifest_exists", manifest_path is not None, str(task_root / "report_manifest.json"))
        if manifest_path is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add(f"{task}.validation_development", manifest.get("validation_development") is True)
        add(f"{task}.no_test_access", manifest.get("no_test_access") is True)
        add(f"{task}.formal_b4_b5_false", manifest.get("formal_b4_b5") is False)
        add(f"{task}.bootstrap_2000_false", manifest.get("bootstrap_2000") is False)
        add(f"{task}.protocol_present", (manifest.get("protocol") or {}).get("sha256") is not None)
        add(f"{task}.rdkit_native", "rdkit" in (manifest.get("environment") or {}))

        # Manifest inputs and outputs: all nonempty and hash-valid.
        inputs = manifest.get("inputs") or []
        add(f"{task}.inputs_nonempty", bool(inputs))
        for item in inputs:
            expected_size = item.get("size")
            path = _resolve_historical(item["path"], item.get("sha256"), expected_size, relocation)
            if path is None:
                add(f"{task}.input_resolved", False, item["path"]); continue
            add(f"{task}.input_resolved", True, str(path))
            if not path.is_file():
                add(f"{task}.input_exists", False, item["path"]); continue
            if path.stat().st_size == 0:
                add(f"{task}.input_nonempty", False, item["path"]); continue
            add(f"{task}.input_sha", _sha(path) == item["sha256"], item["path"])
        outputs = manifest.get("outputs") or []
        add(f"{task}.outputs_nonempty", bool(outputs))
        for item in outputs:
            rel = Path(item["path"])
            relative_ok = not rel.is_absolute() and ".." not in rel.parts
            add(f"{task}.output_relative.{item['path']}", relative_ok, item["path"])
            if not relative_ok:
                add(f"{task}.output_exists.{item['path']}", False, "unsafe output path")
                continue
            try:
                path = (task_root / rel).resolve()
            except OSError as exc:
                add(f"{task}.output_resolve.{item['path']}", False, str(exc))
                add(f"{task}.output_exists.{item['path']}", False, "resolve failed")
                continue
            inside = path.is_relative_to(task_root.resolve())
            add(f"{task}.output_inside.{item['path']}", inside, item["path"])
            if not inside:
                add(f"{task}.output_exists.{item['path']}", False, "output escapes task_root")
                continue
            if not path.is_file():
                add(f"{task}.output_exists.{item['path']}", False, item["path"]); continue
            if path.stat().st_size == 0:
                add(f"{task}.output_nonempty.{item['path']}", False, item["path"]); continue
            add(f"{task}.output_sha.{item['path']}", _sha(path) == item["sha256"], item["path"])
        declared = {item["path"].replace("\\", "/") for item in outputs}
        actual = {
            p.relative_to(task_root).as_posix()
            for p in task_root.rglob("*")
            if p.is_file() and p.name not in ("report_manifest.json", "report_manifest.sha256")
        }
        add(f"{task}.manifest_output_set", declared == actual, f"declared={len(declared)} actual={len(actual)}")
        manifest_sha_file = _safe_task_path(task_root, "report_manifest.sha256")
        add(f"{task}.manifest_sha_file", manifest_sha_file is not None, str(task_root / "report_manifest.sha256"))
        if manifest_sha_file is not None:
            add(f"{task}.manifest_sha_value", manifest_sha_file.read_text(encoding="utf-8").strip() == _sha(manifest_path))

        # Required tables/figures.
        for name in REQUIRED_TABLES:
            p = _safe_task_path(task_root, f"tables/{name}")
            add(f"{task}.table.{name}", p is not None, str(task_root / "tables" / name))
        for name in REQUIRED_FIGURES:
            p = _safe_task_path(task_root, f"figures/{name}")
            ok = p is not None
            if ok:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception:
                    ok = False
            add(f"{task}.figure.{name}", ok, str(task_root / "figures" / name))

        report = _safe_task_path(task_root, "report.md")
        add(f"{task}.report_exists", report is not None, str(task_root / "report.md"))
        if report is not None:
            text = report.read_text(encoding="utf-8")
            for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
                rel_path = Path(rel)
                relative_ok = not rel_path.is_absolute() and ".." not in rel_path.parts
                add(f"{task}.report_image_relative.{rel}", relative_ok, rel)
                if not relative_ok:
                    continue
                p = (task_root / rel).resolve()
                inside = p.is_relative_to(task_root.resolve())
                add(f"{task}.report_image_inside.{rel}", inside, rel)
                if not inside:
                    continue
                add(f"{task}.report_image.{rel}", p.is_file() and p.stat().st_size > 0, str(p))

        # Selection records: 22 subgroups x 2 positions.
        examples = _safe_task_path(task_root, "examples.csv")
        add(f"{task}.examples_exists", examples is not None, str(task_root / "examples.csv"))
        if examples is not None:
            rows = _csv(examples)
            add(f"{task}.examples_records_{expected_records}", len(rows) == expected_records, str(len(rows)))
            groups = {}
            for row in rows:
                groups.setdefault(row["subgroup"], []).append(row)
            add(f"{task}.examples_subgroups_{expected_groups}", len(groups) == expected_groups, str(len(groups)))
            add(f"{task}.examples_two_positions", all(len(v) == 2 for v in groups.values()))
            add(f"{task}.examples_exact_statuses", all({r["status"] for r in v} == {"correct", "error"} for v in groups.values()))
            add(f"{task}.examples_unique_row_or_status", all(len({(r.get("row_index"), r.get("status")) for r in v}) == len(v) for v in groups.values()))

            rep_path = _safe_task_path(task_root, "tables/representative_molecules.csv")
            add(f"{task}.representative_exists", rep_path is not None, str(task_root / "tables/representative_molecules.csv"))
            rep_rows = _csv(rep_path) if rep_path is not None else []
            fields = ("structure_png", "structure_svg", "spectra_png", "metadata")
            names = ("structure.png", "structure.svg", "spectra.png", "metadata.json")
            for label, asset_rows in (("examples", rows), ("representative", rep_rows)):
                for row in asset_rows:
                    idx = str(row.get("row_index", "")).strip()
                    if idx in ("", "None"):
                        add(f"{task}.{label}.asset.empty_row_paths", all(not _asset_path(row, f) for f in fields))
                        continue
                    expected = f"examples/row_{int(idx):06d}"
                    if not legacy_r1:
                        meta_path = None
                        for field, name in zip(fields, names):
                            checked = _check_relative_asset(task, task_root, row, field, f"{expected}/{name}", add)
                            if field == "metadata":
                                meta_path = checked
                        if meta_path is not None:
                            md = json.loads(meta_path.read_text(encoding="utf-8"))
                            for field in ("structure_png", "structure_svg", "spectra_png"):
                                add(f"{task}.{label}.metadata_path.{field}", md.get(field) == _asset_path(row, field))
                    else:
                        for name in names:
                            p = task_root / "examples" / f"row_{int(idx):06d}" / name
                            add(f"{task}.{label}.legacy_asset.{name}", p.is_file() and p.stat().st_size > 0, str(p))
            add(f"{task}.legacy_r1_mode", True, f"legacy_r1={legacy_r1}")
            if legacy_r1:
                add(f"{task}.legacy_csv_paths_skipped", True, "R1 is immutable; its historical CSV absolute paths are not rewritten")
            key = lambda r: (r.get("subgroup"), r.get("status"))
            ex_map = {key(r): r for r in rows}
            rep_map = {key(r): r for r in rep_rows}
            add(f"{task}.csv_records_match", ex_map == rep_map)
            add(f"{task}.csv_paths_match", all({f: ex_map[k][f] for f in fields} == {f: rep_map[k][f] for f in fields} for k in ex_map if k in rep_map))


        # Native per-molecule assets + metadata.
        selected_rows = {r["row_index"] for r in _csv(examples) if r["row_index"] not in ("", None)} if examples is not None else set()
        for row in sorted(selected_rows):
            base_rel = f"examples/row_{int(row):06d}"
            safe_png = _safe_task_path(task_root, f"{base_rel}/structure.png")
            safe_svg = _safe_task_path(task_root, f"{base_rel}/structure.svg")
            safe_spec = _safe_task_path(task_root, f"{base_rel}/spectra.png")
            safe_meta = _safe_task_path(task_root, f"{base_rel}/metadata.json")
            add(f"{task}.row_{row}.structure_png", safe_png is not None, str(task_root / base_rel / "structure.png"))
            if safe_png is not None:
                try:
                    with Image.open(safe_png) as im: im.verify()
                    add(f"{task}.row_{row}.png_decode", True)
                except Exception as exc:
                    add(f"{task}.row_{row}.png_decode", False, str(exc))
            add(f"{task}.row_{row}.structure_svg", safe_svg is not None, str(task_root / base_rel / "structure.svg"))
            if safe_svg is not None:
                add(f"{task}.row_{row}.svg_content", "<svg" in safe_svg.read_text(encoding="utf-8"))
            add(f"{task}.row_{row}.spectra", safe_spec is not None, str(task_root / base_rel / "spectra.png"))
            add(f"{task}.row_{row}.metadata_safe", safe_meta is not None, str(task_root / base_rel / "metadata.json"))
            add(f"{task}.row_{row}.metadata", safe_meta is not None, str(task_root / base_rel / "metadata.json"))
            if safe_meta is None:
                continue
            md = json.loads(safe_meta.read_text(encoding="utf-8"))
            required = ("row_index", "canonical_smiles", "molecular_formula", "seed", "threshold_policy", "true_labels", "probabilities", "hard_predictions", "wrong_labels", "confidence", "descriptive_confidence", "renderer", "structure_png", "structure_svg", "spectra_png", "selection", "data_sources", "input_sha256", "output_sha256")
            add(f"{task}.row_{row}.metadata_fields", all(k in md for k in required))
            add(f"{task}.row_{row}.renderer_native", md.get("renderer") == "rdkit_native")
            for key, safe in (("structure.png", safe_png), ("structure.svg", safe_svg), ("spectra.png", safe_spec)):
                if "output_sha256" in md:
                    add(f"{task}.row_{row}.metadata_sha.{key}", safe is not None and md["output_sha256"].get(key) == _sha(safe))
            add(f"{task}.row_{row}.index", int(md.get("row_index", -1)) == int(row))
            add(f"{task}.row_{row}.selection", bool(md.get("selection")))

        if examples is not None:
            mol_sdf_stats = _validate_mol_sdf(task, task_root, _csv(examples), add)
        else:
            mol_sdf_stats = {"mol_files": 0, "sdf_files": 0, "mol_parsed": 0, "sdf_parsed": 0, "failures": 0}

        # Cross-check row_index against authoritative Validation.
        info = _load_task_inputs(label_set, ROOT / "runs/LINE-B-PREDICTIONS", ROOT / "runs/LINE-B-THRESHOLDS", ROOT / "runs/LINE-B-METRICS", False)
        authoritative = set(int(x) for x in np.asarray(info["models"][0]["row_index"]).tolist())
        add(f"{task}.row_index_authoritative", set(int(x) for x in selected_rows).issubset(authoritative))

        d1_rec = (manifest.get("analysis_summaries") or {}).get("d1") or {}
        d1_path = _resolve_historical(d1_rec.get("path", ""), d1_rec.get("sha256"), None, relocation)
        add(f"{task}.d1_summary_resolved", d1_path is not None, d1_rec.get("path", ""))

        # Membership audit against D1 counts and independently rebuilt masks.
        membership_path = _safe_task_path(task_root, "tables/subgroup_membership_audit.csv")
        add(f"{task}.membership_file_safe", membership_path is not None, str(task_root / "tables/subgroup_membership_audit.csv"))
        if membership_path is not None:
            membership = _csv(membership_path)
            add(f"{task}.membership_rows_{expected_groups}", len(membership) == expected_groups, str(len(membership)))
            add(f"{task}.membership_counts", all(r["n_samples_d1"] == r["n_samples_d2"] for r in membership))
            add(f"{task}.membership_status", all(r["status"] == "PASS_COUNT_AND_POSITIVE_SIGNATURE" for r in membership))
            add(f"{task}.membership_status_string", all(r["status"] == "PASS_COUNT_AND_POSITIVE_SIGNATURE" for r in membership))
            independent_ok = False
            if d1_path is not None and d1_path.is_file() and examples is not None:
                d1 = json.loads(d1_path.read_text(encoding="utf-8"))
                selected = set(d1["subgroup_bootstrap"]["selected_subgroups"])
                if max_subgroups:
                    order = [r["subgroup"] for r in _csv(d1_path.parent / "subgroup_definitions.csv") if r["subgroup"] in selected]
                    selected = set(order[:max_subgroups])
                val_rows = np.asarray(info["models"][0]["row_index"], dtype=np.int64)
                masks = _independent_masks(d1, d1_path.parent, val_rows, _structure_map(), selected)
                labels = np.asarray(info["models"][0]["labels"])
                names = list(info["expected_names"])
                def parse_sig(text):
                    out = {}
                    for part in str(text or "").split(";"):
                        if not part:
                            continue
                        k, v = part.split(":", 1); out[k] = int(v)
                    return out
                all_ok = True
                for r in membership:
                    subgroup = r["subgroup"]
                    mask = masks.get(subgroup)
                    if mask is None:
                        add(f"{task}.positive_signature.{subgroup}", False, "missing independent mask")
                        all_ok = False
                        continue
                    independent_sig = ";".join(f"{name}:{int(labels[mask, i].sum())}" for i, name in enumerate(names))
                    d1_sig = parse_sig(r["positive_signature_d1"])
                    d2_sig = parse_sig(r["positive_signature_d2"])
                    independent = parse_sig(independent_sig)
                    same = d1_sig == d2_sig == independent
                    add(f"{task}.positive_signature.{subgroup}", same, "" if same else f"d1={d1_sig} d2={d2_sig} independent={independent}")
                    all_ok = all_ok and same
                independent_ok = all_ok
                add(f"{task}.positive_signature_independent", independent_ok, "D1/D2/independent signatures")
            else:
                add(f"{task}.positive_signature_independent", False, "missing D1 summary or examples")
            add(f"{task}.row_index_set_d1_persisted", all(r.get("row_index_set_d1_persisted") == "False" for r in membership))
        else:
            add(f"{task}.positive_signature_independent", False, "missing membership audit")

        if d1_path is not None and d1_path.is_file() and examples is not None:
            d1 = json.loads(d1_path.read_text(encoding="utf-8"))
            selected = set(d1["subgroup_bootstrap"]["selected_subgroups"])
            if max_subgroups:
                order = [r["subgroup"] for r in _csv(d1_path.parent / "subgroup_definitions.csv") if r["subgroup"] in selected]
                selected = set(order[:max_subgroups])
            val_rows = np.asarray(info["models"][0]["row_index"], dtype=np.int64)
            masks = _independent_masks(d1, d1_path.parent, val_rows, _structure_map(), selected)
            expected = _independent_selection(info, masks)
            actual = {(r["subgroup"], r["status"]): (int(r["row_index"]), float(r["confidence"])) for r in _csv(examples) if r["row_index"] not in ("", None)}
            same_rows = {k: v[0] for k, v in expected.items()} == {k: v[0] for k, v in actual.items()}
            same_conf = all(abs(expected[k][1] - actual[k][1]) <= 1e-12 for k in expected if k in actual)
            add(f"{task}.selection_reproducible_rows", same_rows)
            add(f"{task}.selection_reproducible_confidence", same_conf)

        # D1/C2 analysis summaries must be intact and non-Test.
        for key in ("d1", "c2", "b3a"):
            rec = (manifest.get("analysis_summaries") or {}).get(key) or {}
            p = _resolve_historical(rec.get("path", ""), rec.get("sha256"), None, relocation)
            add(f"{task}.{key}.summary_resolved", p is not None, rec.get("path", ""))
            if p is None:
                continue
            add(f"{task}.{key}.summary_exists", p.is_file())
            if p.is_file():
                add(f"{task}.{key}.summary_sha", _sha(p) == rec.get("sha256"))
                s = json.loads(p.read_text(encoding="utf-8"))
                if key != "b3a":
                    add(f"{task}.{key}.no_test", s.get("no_test_access") is True)
                    add(f"{task}.{key}.validation", s.get("validation_development") is True)

        # D1 subgroup count.
        if d1_path is not None and d1_path.is_file():
            d1 = json.loads(d1_path.read_text(encoding="utf-8"))
            add(f"{task}.d1_selected_22", d1["subgroup_bootstrap"]["n_selected"] == 22, str(d1["subgroup_bootstrap"]["n_selected"]))
            add(f"{task}.d1_nonempty_22", d1["subgroup_bootstrap"]["n_nonempty"] == 22, str(d1["subgroup_bootstrap"]["n_nonempty"]))
            if max_subgroups:
                add(f"{task}.d1_smoke_selected", d1["subgroup_bootstrap"]["n_selected"] == 22)
            add(f"{task}.d1_reps_20", d1["bootstrap_config"]["reps"] == 20)
            add(f"{task}.d1_formal_false", d1.get("formal_b4") is False)

        # No explicit Test or formal 2000 execution.
        report_text = report.read_text(encoding="utf-8") if report is not None else ""
        normalized_report = re.sub(r"\*", "", report_text).lower()
        add(f"{task}.no_formal_2000_in_report", "formal 2,000 bootstrap: not run" in normalized_report)
        add(f"{task}.no_test_access_claim", "test access: none" in normalized_report)

    status = "PASS" if not failures else "FAILED"
    return {"status": status, "checks": checks, "failures": failures, "warnings": warnings,
            "n_pass": sum(1 for c in checks if c["status"] == "PASS"),
            "n_failed": sum(1 for c in checks if c["status"] == "FAILED")}


def _json_output_path(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise SystemExit(f"Verifier JSON output must be inside the workspace: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not resolved.parent.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit(f"Verifier JSON parent escapes workspace: {resolved.parent}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--max-subgroups", type=int, default=None)
    parser.add_argument("--legacy-r1", action="store_true", help="Validate immutable R1 historical CSV paths without rewriting them")
    parser.add_argument("--json-out", type=Path, default=ROOT / "runs/LINE-B-ANALYSIS/validation_development/B3.5-D2/r2_workspace_validation/final_verification/verification.json")
    args = parser.parse_args()
    result = verify(args.report_root.resolve(), args.max_subgroups, args.legacy_r1)
    print(json.dumps({k: result[k] for k in ("status", "n_pass", "n_failed", "failures")}, ensure_ascii=False, indent=2))
    json_out = _json_output_path(args.json_out)
    json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
