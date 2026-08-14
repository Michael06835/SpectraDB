from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdDepictor, rdMolDescriptors

ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
PREPARED = ROOT / "processed" / "qm9s" / "prepared"
DEFAULT_MANIFEST = PREPARED / "qm9s_manifest_with_modality_validity.csv"
DEFAULT_OUTPUT = ROOT / "processed" / "qm9s" / "structures"

NUMBERING_SCHEME = (
    "RDKit canonical graph order; element-specific labels "
    "(C1,C2,...; N1,N2,...; O1,O2,...; F1,F2,...)"
)

FUNCTIONAL_GROUP_SMARTS = {
    # Alcohol/phenol-type hydroxyl. Excludes carboxylic-acid OH.
    "hydroxyl": "[OX2H][#6;!$(C=O)]",
    # Ether oxygen between two non-carbonyl carbons. Excludes ester O-C(=O).
    "ether": "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine": "[NX3;!$([N][C](=O));!$([N+](=O)[O-])]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[NX3+](=O)[O-]",
    "imine": "[CX3]=[NX2]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "c_f_bond": "[#6]-[F]",
    "peroxide": "[OX2]-[OX2]",
}

PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()}
if any(pattern is None for pattern in PATTERNS.values()):
    bad = [name for name, pattern in PATTERNS.items() if pattern is None]
    raise RuntimeError(f"SMARTS 编译失败：{bad}")


def canonicalize_and_renumber(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效 SMILES：{smiles}")
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"规范 SMILES 无法解析：{canonical_smiles}")
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    order = sorted(range(mol.GetNumAtoms()), key=lambda atom_index: (ranks[atom_index], atom_index))
    mol = Chem.RenumberAtoms(mol, order)
    Chem.SanitizeMol(mol)
    return mol, canonical_smiles


def make_atom_labels(mol: Chem.Mol) -> dict[int, str]:
    counters: defaultdict[str, int] = defaultdict(int)
    labels: dict[int, str] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counters[symbol] += 1
        labels[atom.GetIdx()] = f"{symbol}{counters[symbol]}"
    return labels


def atom_ring_sizes(mol: Chem.Mol) -> dict[int, list[int]]:
    result: defaultdict[int, list[int]] = defaultdict(list)
    for ring in mol.GetRingInfo().AtomRings():
        for atom_index in ring:
            result[atom_index].append(len(ring))
    return {atom_index: sorted(set(sizes)) for atom_index, sizes in result.items()}


def build_atom_record(row_index: int, qm9_number: int | None, mol: Chem.Mol, labels: dict[int, str]) -> dict:
    ring_sizes = atom_ring_sizes(mol)
    atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atoms.append({
            "atom_index": idx + 1,
            "atom_label": labels[idx],
            "element": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic": atom.GetIsAromatic(),
            "hybridization": str(atom.GetHybridization()),
            "degree": atom.GetDegree(),
            "total_valence": atom.GetTotalValence(),
            "total_hydrogens": atom.GetTotalNumHs(),
            "is_in_ring": atom.IsInRing(),
            "ring_sizes": ring_sizes.get(idx, []),
            "chiral_tag": str(atom.GetChiralTag()),
            "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            "neighbors": [labels[n.GetIdx()] for n in atom.GetNeighbors()],
        })
    return {"row_index": row_index, "qm9_number": qm9_number, "numbering_scheme": NUMBERING_SCHEME, "atoms": atoms}


def build_bond_record(row_index: int, qm9_number: int | None, mol: Chem.Mol, labels: dict[int, str]) -> dict:
    bonds = []
    for bond in mol.GetBonds():
        bonds.append({
            "bond_index": bond.GetIdx() + 1,
            "begin_atom": labels[bond.GetBeginAtomIdx()],
            "end_atom": labels[bond.GetEndAtomIdx()],
            "bond_type": str(bond.GetBondType()),
            "bond_order": bond.GetBondTypeAsDouble(),
            "is_aromatic": bond.GetIsAromatic(),
            "is_conjugated": bond.GetIsConjugated(),
            "is_in_ring": bond.IsInRing(),
            "stereo": str(bond.GetStereo()),
        })
    return {"row_index": row_index, "qm9_number": qm9_number, "bonds": bonds}


def functional_group_matches(mol: Chem.Mol, labels: dict[int, str]) -> list[dict]:
    groups: list[dict] = []
    for name, pattern in PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern, uniquify=True)
        for match_number, match in enumerate(matches, start=1):
            atom_set = set(match)
            internal_bonds = [
                bond.GetIdx() + 1
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
            ]
            groups.append({
                "group_name": name,
                "instance": match_number,
                "smarts": FUNCTIONAL_GROUP_SMARTS[name],
                "atom_indices": [idx + 1 for idx in match],
                "atom_labels": [labels[idx] for idx in match],
                "bond_indices": internal_bonds,
            })
    return groups


def ring_annotations(mol: Chem.Mol, labels: dict[int, str]) -> list[dict]:
    rings = []
    for ring_number, ring in enumerate(mol.GetRingInfo().AtomRings(), start=1):
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        rings.append({
            "ring_index": ring_number,
            "size": len(ring),
            "atom_indices": [idx + 1 for idx in ring],
            "atom_labels": [labels[idx] for idx in ring],
            "is_aromatic": all(atom.GetIsAromatic() for atom in atoms),
            "heteroatom_count": sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms),
            "element_composition": dict(Counter(atom.GetSymbol() for atom in atoms)),
        })
    return rings


def conjugated_components(mol: Chem.Mol, labels: dict[int, str]) -> list[dict]:
    adjacency: defaultdict[int, set[int]] = defaultdict(set)
    eligible_bonds = []
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() or bond.GetIsAromatic():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[a].add(b)
            adjacency[b].add(a)
            eligible_bonds.append(bond)
    visited: set[int] = set()
    components: list[dict] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_atoms: set[int] = set()
        while queue:
            current = queue.popleft()
            component_atoms.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append({
            "component_index": len(components) + 1,
            "atom_indices": [idx + 1 for idx in sorted(component_atoms)],
            "atom_labels": [labels[idx] for idx in sorted(component_atoms)],
            "bond_indices": [
                bond.GetIdx() + 1
                for bond in eligible_bonds
                if bond.GetBeginAtomIdx() in component_atoms and bond.GetEndAtomIdx() in component_atoms
            ],
        })
    return components


def build_functional_record(row_index: int, qm9_number: int | None, mol: Chem.Mol, labels: dict[int, str]) -> dict:
    return {
        "row_index": row_index,
        "qm9_number": qm9_number,
        "functional_groups": functional_group_matches(mol, labels),
        "rings": ring_annotations(mol, labels),
        "conjugated_components": conjugated_components(mol, labels),
    }


def optional_bool(source_row: dict, name: str):
    value = source_row.get(name)
    if value is None or pd.isna(value):
        return ""
    return bool(value)


def library_row(source_row: dict, mol: Chem.Mol, canonical_smiles: str, groups: list[dict]) -> dict:
    element_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    rings = mol.GetRingInfo().AtomRings()
    aromatic_ring_count = sum(all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring) for ring in rings)
    return {
        "row_index": int(source_row["row_index"]),
        "qm9_number": int(source_row["qm9_number"]) if "qm9_number" in source_row and not pd.isna(source_row["qm9_number"]) else "",
        "canonical_smiles": canonical_smiles,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mol_wt": round(Descriptors.ExactMolWt(mol), 6),
        "heavy_atom_count": mol.GetNumAtoms(),
        "bond_count": mol.GetNumBonds(),
        "carbon_count": element_counts.get("C", 0),
        "nitrogen_count": element_counts.get("N", 0),
        "oxygen_count": element_counts.get("O", 0),
        "fluorine_count": element_counts.get("F", 0),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "ring_count": len(rings),
        "aromatic_ring_count": aromatic_ring_count,
        "rotatable_bond_count": Lipinski.NumRotatableBonds(mol),
        "h_bond_donor_count": Lipinski.NumHDonors(mol),
        "h_bond_acceptor_count": Lipinski.NumHAcceptors(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 6),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 6),
        "functional_group_types": ";".join(sorted({group["group_name"] for group in groups})),
        "functional_group_instance_count": len(groups),
        "numbering_scheme": NUMBERING_SCHEME,
        "valid_ir": optional_bool(source_row, "valid_ir"),
        "valid_raman": optional_bool(source_row, "valid_raman"),
        "valid_uvvis": optional_bool(source_row, "valid_uvvis"),
        "valid_multimodal_complete": optional_bool(source_row, "valid_multimodal_complete"),
    }


LIBRARY_COLUMNS = [
    "row_index", "qm9_number", "canonical_smiles", "molecular_formula", "exact_mol_wt",
    "heavy_atom_count", "bond_count", "carbon_count", "nitrogen_count", "oxygen_count",
    "fluorine_count", "formal_charge", "ring_count", "aromatic_ring_count",
    "rotatable_bond_count", "h_bond_donor_count", "h_bond_acceptor_count", "tpsa",
    "fraction_csp3", "functional_group_types", "functional_group_instance_count",
    "numbering_scheme", "valid_ir", "valid_raman", "valid_uvvis", "valid_multimodal_complete",
]


def command_build(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    paths = {
        "library": output_dir / "qm9s_structure_library.csv",
        "atoms": output_dir / "qm9s_atom_annotations.jsonl.gz",
        "bonds": output_dir / "qm9s_bond_annotations.jsonl.gz",
        "groups": output_dir / "qm9s_functional_groups.jsonl.gz",
        "invalid": output_dir / "qm9s_invalid_structures.csv",
        "definitions": output_dir / "functional_group_definitions.json",
        "report": output_dir / "build_structure_library_report.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出文件已存在。确认需要重建时加 --overwrite：\n" + "\n".join(str(path) for path in existing))
    manifest = pd.read_csv(manifest_path)
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()
    smiles_column = "canonical_smiles" if "canonical_smiles" in manifest.columns else "smiles"
    if smiles_column not in manifest.columns:
        raise KeyError("manifest 中没有 canonical_smiles 或 smiles 列。")
    temp_paths = {name: path.with_name(path.name + ".tmp") for name, path in paths.items() if name not in {"definitions", "report"}}
    for path in [*paths.values(), *temp_paths.values()]:
        if path.exists():
            path.unlink()
    group_counts: Counter[str] = Counter()
    invalid_rows = []
    valid_count = 0
    with (
        temp_paths["library"].open("w", newline="", encoding="utf-8-sig") as library_file,
        gzip.open(temp_paths["atoms"], "wt", encoding="utf-8") as atom_file,
        gzip.open(temp_paths["bonds"], "wt", encoding="utf-8") as bond_file,
        gzip.open(temp_paths["groups"], "wt", encoding="utf-8") as group_file,
    ):
        writer = csv.DictWriter(library_file, fieldnames=LIBRARY_COLUMNS)
        writer.writeheader()
        total = len(manifest)
        for processed, source_row in enumerate(manifest.to_dict(orient="records"), start=1):
            row_index = int(source_row["row_index"])
            qm9_number = int(source_row["qm9_number"]) if "qm9_number" in source_row and not pd.isna(source_row["qm9_number"]) else None
            smiles = str(source_row[smiles_column])
            try:
                mol, canonical_smiles = canonicalize_and_renumber(smiles)
                labels = make_atom_labels(mol)
                atom_record = build_atom_record(row_index, qm9_number, mol, labels)
                bond_record = build_bond_record(row_index, qm9_number, mol, labels)
                functional_record = build_functional_record(row_index, qm9_number, mol, labels)
                for group in functional_record["functional_groups"]:
                    group_counts[group["group_name"]] += 1
                writer.writerow(library_row(source_row, mol, canonical_smiles, functional_record["functional_groups"]))
                atom_file.write(json.dumps(atom_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                bond_file.write(json.dumps(bond_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                group_file.write(json.dumps(functional_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                valid_count += 1
            except Exception as exc:
                invalid_rows.append({"row_index": row_index, "qm9_number": qm9_number, "smiles": smiles, "error": repr(exc)})
            if processed % 5000 == 0 or processed == total:
                print(f"已处理 {processed:,}/{total:,}；有效 {valid_count:,}；失败 {len(invalid_rows):,}")
    pd.DataFrame(invalid_rows, columns=["row_index", "qm9_number", "smiles", "error"]).to_csv(temp_paths["invalid"], index=False, encoding="utf-8-sig")
    for name in ("library", "atoms", "bonds", "groups", "invalid"):
        temp_paths[name].replace(paths[name])
    paths["definitions"].write_text(json.dumps({
        "numbering_scheme": NUMBERING_SCHEME,
        "note": "官能团允许重叠。例如酯基实例也会同时计入羰基。",
        "patterns": FUNCTIONAL_GROUP_SMARTS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "requested_rows": int(len(manifest)),
        "valid_rows": int(valid_count),
        "invalid_rows": int(len(invalid_rows)),
        "smiles_column": smiles_column,
        "numbering_scheme": NUMBERING_SCHEME,
        "functional_group_instance_counts": dict(group_counts.most_common()),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print("QM9S 结构库建立完成")
    print("=" * 72)
    print(f"有效结构：{valid_count:,}")
    print(f"无效结构：{len(invalid_rows):,}")
    print(f"输出目录：{output_dir}")


def load_library_row(library_path: Path, row_index: int) -> dict:
    library = pd.read_csv(library_path)
    selected = library.loc[library["row_index"] == row_index]
    if selected.empty:
        raise KeyError(f"结构库中不存在 row_index={row_index}")
    return selected.iloc[0].to_dict()


def draw_structure(mol: Chem.Mol, labels: dict[int, str], output_path: Path, image_format: str, highlight_atoms: list[int] | None = None) -> None:
    # Lazy import: building the structure library does not require RDKit drawing DLLs.
    from rdkit.Chem.Draw import rdMolDraw2D

    width, height = 1200, 800
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height) if image_format == "svg" else rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    for atom_index, label in labels.items():
        options.atomLabels[atom_index] = label
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms or [])
    drawer.FinishDrawing()
    if image_format == "svg":
        output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")
    else:
        output_path.write_bytes(drawer.GetDrawingText())


def command_render(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    library_path = output_dir / "qm9s_structure_library.csv"
    if not library_path.exists():
        raise FileNotFoundError(f"请先建立结构库：{library_path}")
    record = load_library_row(library_path, args.row_index)
    mol, canonical_smiles = canonicalize_and_renumber(str(record["canonical_smiles"]))
    labels = make_atom_labels(mol)
    rdDepictor.Compute2DCoords(mol)
    highlight_atoms: list[int] = []
    highlighted_instances = []
    if args.highlight_group:
        matches = mol.GetSubstructMatches(PATTERNS[args.highlight_group], uniquify=True)
        highlight_atoms = sorted({idx for match in matches for idx in match})
        highlighted_instances = [[labels[idx] for idx in match] for match in matches]
    molecule_dir = output_dir / "rendered" / f"row_{args.row_index:06d}"
    molecule_dir.mkdir(parents=True, exist_ok=True)
    svg_path = molecule_dir / "structure_numbered.svg"
    png_path = molecule_dir / "structure_numbered.png"
    mol_path = molecule_dir / "structure_atom_mapped.mol"
    sdf_path = molecule_dir / "structure_atom_mapped.sdf"
    json_path = molecule_dir / "structure_metadata.json"
    draw_structure(mol, labels, svg_path, "svg", highlight_atoms)
    draw_structure(mol, labels, png_path, "png", highlight_atoms)
    mapped_mol = Chem.Mol(mol)
    for atom in mapped_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mapped_mol.SetProp("row_index", str(args.row_index))
    mapped_mol.SetProp("qm9_number", str(record.get("qm9_number", "")))
    mapped_mol.SetProp("canonical_smiles", canonical_smiles)
    mapped_mol.SetProp("atom_label_map", json.dumps({str(idx + 1): label for idx, label in labels.items()}, ensure_ascii=False))
    Chem.MolToMolFile(mapped_mol, str(mol_path))
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mapped_mol)
    writer.close()
    metadata = {
        "row_index": args.row_index,
        "qm9_number": record.get("qm9_number"),
        "canonical_smiles": canonical_smiles,
        "molecular_formula": record.get("molecular_formula"),
        "numbering_scheme": NUMBERING_SCHEME,
        "atom_labels": {str(idx + 1): label for idx, label in labels.items()},
        "highlight_group": args.highlight_group,
        "highlighted_instances": highlighted_instances,
        "files": {"svg": str(svg_path), "png": str(png_path), "mol": str(mol_path), "sdf": str(sdf_path)},
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print("结构渲染完成")
    print("=" * 72)
    print(f"SMILES：{canonical_smiles}")
    print(f"分子式：{record.get('molecular_formula')}")
    print(f"输出目录：{molecule_dir}")
    if args.highlight_group:
        print(f"高亮 {args.highlight_group}：{highlighted_instances}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QM9S 结构库建立与候选结构绘图工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="建立结构库和原子/键/官能团标注")
    build_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    build_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--overwrite", action="store_true")
    build_parser.set_defaults(func=command_build)
    render_parser = subparsers.add_parser("render", help="按 row_index 渲染带编号结构")
    render_parser.add_argument("--row-index", type=int, required=True)
    render_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    render_parser.add_argument("--highlight-group", choices=sorted(PATTERNS), default=None)
    render_parser.set_defaults(func=command_render)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
