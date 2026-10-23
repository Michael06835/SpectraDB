#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build QM9S UV-Vis-specific molecular-structure annotations from the frozen
UV-Vis Chemical Semantic Annotation Schema v1.0.

This script implements the CURRENT builder scope only:
    A_raw_pi_conjugation_descriptors
    B1_conjugation_semantics
    B2_aromatic_semantics
    B3_carbonyl_chromophore_semantics
    B4_nitrogen_chromophore_semantics
    B5_auxochrome_and_electronic_context

Reserved-for-later fields (Woodward/Fieser, UV spectral descriptors, and
reserved donor/acceptor path distances) are intentionally NOT generated.

Important invariants
--------------------
1. Input row order is preserved exactly. No sorting, deduplication, or filtering.
2. Invalid annotations are NOT silently converted to all-zero labels:
   binary rows use -1 and numeric rows use NaN when annotation_valid == 0.
3. RDKit's GetIsConjugated() perception is stored separately from the custom
   UV pi-system semantics.
4. The frozen v1.0 schema is treated as authoritative. This builder does not
   invent new v1.0 labels.

Examples
--------
Sanity check only:
    python scripts/build_qm9s_uvvis_annotations.py --self-test

Build from a CSV containing a SMILES column:
    python scripts/build_qm9s_uvvis_annotations.py ^
        --input master/qm9s_master.csv ^
        --smiles-col smiles ^
        --schema metadata/uvvis_annotation_schema_v1.0.json ^
        --output-dir processed/qm9s/uvvis_annotations_v1

First 1000 rows only:
    python scripts/build_qm9s_uvvis_annotations.py ^
        --input master/qm9s_master.csv ^
        --smiles-col smiles ^
        --limit 1000 ^
        --allow-count-mismatch ^
        --output-dir processed/qm9s/uvvis_annotations_v1_smoke

Dependencies
------------
    numpy
    RDKit

Python standard library is used for CSV/JSON/statistics output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem


BUILDER_VERSION = "1.0.1"
DEFAULT_EXPECTED_QM9S_COUNT = 129_817
CURRENT_SCOPE_DEFAULT = [
    "A_raw_pi_conjugation_descriptors",
    "B1_conjugation_semantics",
    "B2_aromatic_semantics",
    "B3_carbonyl_chromophore_semantics",
    "B4_nitrogen_chromophore_semantics",
    "B5_auxochrome_and_electronic_context",
]

ALLOWED_PI_ENDPOINT_HYBRIDIZATIONS = {
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PiNode:
    node_id: int
    kind: str
    atoms: frozenset[int]
    bond_idx: Optional[int] = None
    aromatic_system_id: Optional[int] = None

    @property
    def is_aromatic(self) -> bool:
        return self.kind == "AROMATIC_SYSTEM"

    @property
    def is_nonaromatic(self) -> bool:
        return not self.is_aromatic


@dataclass
class AromaticTopology:
    aromatic_ring_atom_sets: List[Set[int]]
    aromatic_ring_bond_sets: List[Set[int]]
    system_atom_sets: List[Set[int]]
    system_ring_indices: List[List[int]]
    atom_to_systems: Dict[int, Set[int]]
    fused_system_count: int
    biaryl_single_bond_link_count: int


@dataclass
class UVGraph:
    nodes: Dict[int, PiNode]
    adjacency: Dict[int, Set[int]]
    included_nodes: Set[int]
    components: List[Set[int]]
    node_to_component: Dict[int, int]
    atom_to_nodes: Dict[int, Set[int]]
    pi_framework_atoms: Set[int]
    selected_unit_bond_to_node: Dict[int, int]


@dataclass
class AnnotationResult:
    values: Dict[str, Any]
    valid: bool
    warning_count: int
    error_code: str


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_commit_or_unknown(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def csv_write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_single_bond(bond: Chem.Bond) -> bool:
    return bond.GetBondType() == rdchem.BondType.SINGLE and not bond.GetIsAromatic()


def is_nonaromatic_multiple_bond(bond: Chem.Bond) -> bool:
    return (not bond.GetIsAromatic()) and bond.GetBondType() in {
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
    }


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def connected_components(nodes: Iterable[int], adjacency: Dict[int, Set[int]]) -> List[Set[int]]:
    remaining = set(nodes)
    comps: List[Set[int]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp: Set[int] = set()
        remaining.remove(start)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adjacency.get(u, set()):
                if v in remaining:
                    remaining.remove(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def longest_simple_path_node_count(nodes: Set[int], adjacency: Dict[int, Set[int]]) -> int:
    """Exact DFS; safe here because QM9/QM9S molecules are very small."""
    if not nodes:
        return 0
    best = 1

    def dfs(u: int, visited: Set[int]) -> None:
        nonlocal best
        best = max(best, len(visited))
        for v in adjacency.get(u, set()):
            if v in nodes and v not in visited:
                visited.add(v)
                dfs(v, visited)
                visited.remove(v)

    for start in nodes:
        dfs(start, {start})
    return best


def longest_path_in_induced_kind_graph(
    node_ids: Set[int],
    adjacency: Dict[int, Set[int]],
    nodes: Dict[int, PiNode],
    allowed_kinds: Set[str],
) -> int:
    chosen = {n for n in node_ids if nodes[n].kind in allowed_kinds}
    if not chosen:
        return 0
    # Only nodes that actually belong to the UV graph are passed here, so a
    # singleton aromatic-independent nonaromatic unit cannot appear by accident.
    return longest_simple_path_node_count(chosen, adjacency)


def ring_bond_set_from_ring_atoms(mol: Chem.Mol, ring_atoms: Sequence[int]) -> Set[int]:
    bonds: Set[int] = set()
    n = len(ring_atoms)
    for i in range(n):
        a = int(ring_atoms[i])
        b = int(ring_atoms[(i + 1) % n])
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            bonds.add(bond.GetIdx())
    return bonds


# -----------------------------------------------------------------------------
# Schema handling
# -----------------------------------------------------------------------------

def load_schema(path: Path, allow_unfrozen: bool = False) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    if schema.get("schema_version") != "1.0":
        raise ValueError(f"Expected schema_version=1.0, got {schema.get('schema_version')!r}")

    if not allow_unfrozen and schema.get("schema_status") != "frozen":
        raise ValueError(
            "Schema is not marked frozen. Use the canonical FINAL v1.0 schema, "
            "or pass --allow-unfrozen-schema only for deliberate development tests."
        )

    return schema


def current_scope_from_schema(schema: Dict[str, Any]) -> List[str]:
    policy = schema.get("implementation_policy_v1", {})
    scope = policy.get("current_builder_scope")
    if scope:
        return list(scope)
    return list(CURRENT_SCOPE_DEFAULT)


def current_features(schema: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    binary: List[Dict[str, Any]] = []
    numeric: List[Dict[str, Any]] = []
    categorical: List[Dict[str, Any]] = []

    scope = current_scope_from_schema(schema)
    layers = schema["layers"]

    for layer_name in scope:
        if layer_name not in layers:
            raise KeyError(f"Schema builder scope references missing layer: {layer_name}")
        for feat in layers[layer_name].get("features", []):
            if feat.get("implementation_status") == "reserved_for_later":
                continue
            if feat.get("derivation") == "reserved":
                continue
            typ = feat["type"]
            if typ == "binary":
                binary.append(feat)
            elif typ in {"integer", "float"}:
                numeric.append(feat)
            elif typ == "categorical":
                categorical.append(feat)
            else:
                raise ValueError(
                    f"Unsupported current feature type {typ!r} for {feat.get('name')!r}."
                )

    return binary, numeric, categorical


# -----------------------------------------------------------------------------
# Aromatic topology
# -----------------------------------------------------------------------------

def build_aromatic_topology(mol: Chem.Mol) -> AromaticTopology:
    rings_raw = list(Chem.GetSymmSSSR(mol))
    aromatic_ring_atom_sets: List[Set[int]] = []
    aromatic_ring_bond_sets: List[Set[int]] = []

    for ring in rings_raw:
        atoms = [int(x) for x in ring]
        if atoms and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            aromatic_ring_atom_sets.append(set(atoms))
            aromatic_ring_bond_sets.append(ring_bond_set_from_ring_atoms(mol, atoms))

    nr = len(aromatic_ring_atom_sets)
    parent = list(range(nr))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Fused aromatic rings belong to the same system only when they share a ring bond.
    for i in range(nr):
        for j in range(i + 1, nr):
            if aromatic_ring_bond_sets[i] & aromatic_ring_bond_sets[j]:
                union(i, j)

    grouped: Dict[int, List[int]] = defaultdict(list)
    for i in range(nr):
        grouped[find(i)].append(i)

    system_ring_indices = list(grouped.values())
    system_atom_sets: List[Set[int]] = []
    atom_to_systems: Dict[int, Set[int]] = defaultdict(set)

    for sid, ring_indices in enumerate(system_ring_indices):
        atoms: Set[int] = set()
        for ridx in ring_indices:
            atoms |= aromatic_ring_atom_sets[ridx]
        system_atom_sets.append(atoms)
        for atom_idx in atoms:
            atom_to_systems[atom_idx].add(sid)

    fused_system_count = sum(1 for ring_indices in system_ring_indices if len(ring_indices) >= 2)

    biaryl_links = 0
    for bond in mol.GetBonds():
        if not is_single_bond(bond) or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sa = atom_to_systems.get(a, set())
        sb = atom_to_systems.get(b, set())
        if any(x != y for x in sa for y in sb):
            biaryl_links += 1

    return AromaticTopology(
        aromatic_ring_atom_sets=aromatic_ring_atom_sets,
        aromatic_ring_bond_sets=aromatic_ring_bond_sets,
        system_atom_sets=system_atom_sets,
        system_ring_indices=system_ring_indices,
        atom_to_systems=dict(atom_to_systems),
        fused_system_count=fused_system_count,
        biaryl_single_bond_link_count=biaryl_links,
    )


# -----------------------------------------------------------------------------
# RDKit conjugated-subgraph descriptors
# -----------------------------------------------------------------------------

def rdkit_conjugated_subgraph_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    conj_bonds = [bond for bond in mol.GetBonds() if bond.GetIsConjugated()]
    conj_bond_ids = {b.GetIdx() for b in conj_bonds}
    conj_atoms: Set[int] = set()
    atom_adj: Dict[int, Set[int]] = defaultdict(set)

    for bond in conj_bonds:
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        conj_atoms.update([a, b])
        atom_adj[a].add(b)
        atom_adj[b].add(a)

    comps = connected_components(conj_atoms, atom_adj) if conj_atoms else []

    max_atoms = 0
    max_bonds = 0
    max_pi_units = 0
    max_cc_double = 0
    max_cc_triple = 0
    longest_atoms = 0
    longest_bonds = 0
    cyclic_count = 0
    branched_count = 0
    max_degree = 0
    branch_points = 0

    for comp in comps:
        max_atoms = max(max_atoms, len(comp))
        comp_bonds = [
            bond
            for bond in conj_bonds
            if bond.GetBeginAtomIdx() in comp and bond.GetEndAtomIdx() in comp
        ]
        max_bonds = max(max_bonds, len(comp_bonds))

        pi_units = 0
        cc_double = 0
        cc_triple = 0
        for bond in comp_bonds:
            if bond.GetIsAromatic():
                continue
            if bond.GetBondType() not in {rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE}:
                continue
            a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
            b = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
            pair = tuple(sorted((a, b)))
            if pair in {(6, 6), (6, 7), (6, 8), (7, 7)}:
                pi_units += 1
            if pair == (6, 6) and bond.GetBondType() == rdchem.BondType.DOUBLE:
                cc_double += 1
            if pair == (6, 6) and bond.GetBondType() == rdchem.BondType.TRIPLE:
                cc_triple += 1

        max_pi_units = max(max_pi_units, pi_units)
        max_cc_double = max(max_cc_double, cc_double)
        max_cc_triple = max(max_cc_triple, cc_triple)

        lp_atoms = longest_simple_path_node_count(comp, atom_adj)
        longest_atoms = max(longest_atoms, lp_atoms)
        longest_bonds = max(longest_bonds, max(0, lp_atoms - 1))

        comp_ring = any(bond.IsInRing() for bond in comp_bonds)
        if comp_ring:
            cyclic_count += 1

        degrees = [len(atom_adj.get(a, set()) & comp) for a in comp]
        if any(d >= 3 for d in degrees):
            branched_count += 1
        if degrees:
            max_degree = max(max_degree, max(degrees))
        branch_points += sum(1 for d in degrees if d >= 3)

    heavy = heavy_atom_count(mol)
    frac = (len(conj_atoms) / heavy) if heavy else 0.0

    return {
        "rdkit_conjugated_bond_count": len(conj_bond_ids),
        "rdkit_conjugated_atom_count": len(conj_atoms),
        "rdkit_conjugated_component_count": len(comps),
        "rdkit_max_conjugated_component_atoms": max_atoms,
        "rdkit_max_conjugated_component_bonds": max_bonds,
        "rdkit_max_conjugated_pi_units": max_pi_units,
        "rdkit_max_conjugated_C=C_count": max_cc_double,
        "rdkit_max_conjugated_C#C_count": max_cc_triple,
        "rdkit_longest_conjugated_path_atoms": longest_atoms,
        "rdkit_longest_conjugated_path_bonds": longest_bonds,
        "rdkit_fraction_conjugated_heavy_atoms": float(frac),
        "rdkit_cyclic_conjugated_component_count": cyclic_count,
        "rdkit_branched_conjugated_component_count": branched_count,
        "rdkit_max_conjugated_graph_degree": max_degree,
        "rdkit_conjugated_branch_point_count": branch_points,
    }


# -----------------------------------------------------------------------------
# UV pi-system graph
# -----------------------------------------------------------------------------

def classify_pi_bond_kind(mol: Chem.Mol, bond: Chem.Bond) -> Optional[str]:
    if bond.GetIsAromatic():
        return None
    bt = bond.GetBondType()
    if bt not in {rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE}:
        return None

    a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
    b = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
    pair = tuple(sorted((a, b)))

    if bt == rdchem.BondType.DOUBLE:
        if pair == (6, 6):
            return "C=C"
        if pair == (6, 8):
            return "C=O"
        if pair == (6, 7):
            return "C=N"
        if pair == (7, 7):
            return "N=N"
    elif bt == rdchem.BondType.TRIPLE:
        if pair == (6, 6):
            return "C#C"
        if pair == (6, 7):
            return "C#N"
    return None


def endpoint_is_pi_capable(atom: Chem.Atom) -> bool:
    return atom.GetIsAromatic() or atom.GetHybridization() in ALLOWED_PI_ENDPOINT_HYBRIDIZATIONS


def build_uv_pi_graph(mol: Chem.Mol, arom: AromaticTopology) -> UVGraph:
    nodes: Dict[int, PiNode] = {}
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    selected_unit_bond_to_node: Dict[int, int] = {}
    atom_to_nodes: Dict[int, Set[int]] = defaultdict(set)

    next_id = 0

    # Aromatic ring systems are intrinsically delocalized and each forms one node.
    aromatic_node_ids: List[int] = []
    for sid, atoms in enumerate(arom.system_atom_sets):
        node = PiNode(
            node_id=next_id,
            kind="AROMATIC_SYSTEM",
            atoms=frozenset(atoms),
            aromatic_system_id=sid,
        )
        nodes[next_id] = node
        aromatic_node_ids.append(next_id)
        for a in atoms:
            atom_to_nodes[a].add(next_id)
        next_id += 1

    # Explicit non-aromatic pi units listed by the frozen v1.0 schema.
    nonaromatic_node_ids: List[int] = []
    for bond in mol.GetBonds():
        kind = classify_pi_bond_kind(mol, bond)
        if kind is None:
            continue
        atoms = frozenset([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        node = PiNode(
            node_id=next_id,
            kind=kind,
            atoms=atoms,
            bond_idx=bond.GetIdx(),
        )
        nodes[next_id] = node
        selected_unit_bond_to_node[bond.GetIdx()] = next_id
        nonaromatic_node_ids.append(next_id)
        for a in atoms:
            atom_to_nodes[a].add(next_id)
        next_id += 1

    # Non-aromatic unit <-> non-aromatic unit coupling.
    # Shared-center multiple bonds are cumulated motifs and do not receive ordinary
    # conjugation edges solely because of the shared atom.
    for i, a_id in enumerate(nonaromatic_node_ids):
        a_node = nodes[a_id]
        for b_id in nonaromatic_node_ids[i + 1:]:
            b_node = nodes[b_id]
            if a_node.atoms & b_node.atoms:
                continue

            coupled = False
            for ai in a_node.atoms:
                for bi in b_node.atoms:
                    bond = mol.GetBondBetweenAtoms(ai, bi)
                    if bond is None or not is_single_bond(bond):
                        continue
                    aa = mol.GetAtomWithIdx(ai)
                    bb = mol.GetAtomWithIdx(bi)
                    if endpoint_is_pi_capable(aa) and endpoint_is_pi_capable(bb):
                        coupled = True
                        break
                if coupled:
                    break

            if coupled:
                adjacency[a_id].add(b_id)
                adjacency[b_id].add(a_id)

    # Aromatic-system <-> exocyclic non-aromatic pi-unit coupling.
    for arom_id in aromatic_node_ids:
        arom_node = nodes[arom_id]
        for unit_id in nonaromatic_node_ids:
            unit = nodes[unit_id]
            coupled = False
            for ai in arom_node.atoms:
                for ui in unit.atoms:
                    bond = mol.GetBondBetweenAtoms(ai, ui)
                    if bond is None or not is_single_bond(bond):
                        continue
                    if endpoint_is_pi_capable(mol.GetAtomWithIdx(ui)):
                        coupled = True
                        break
                if coupled:
                    break
            # Some RDKit-aromatic heterocycles (e.g. alpha-pyrone and
            # conjugated cyclic carbonates) have an exocyclic C=O whose
            # carbonyl carbon is itself an aromatic ring atom. The aromatic
            # ring-system node and C=O node therefore OVERLAP at C; there is
            # no intervening SINGLE bond. Connect these two pi units without
            # promoting arbitrary cumulenes or fusing independent rings.
            if (
                not coupled
                and unit.kind == "C=O"
                and any(
                    mol.GetAtomWithIdx(ai).GetAtomicNum() == 6
                    and mol.GetAtomWithIdx(ai).GetIsAromatic()
                    for ai in arom_node.atoms & unit.atoms
                )
            ):
                coupled = True

            if coupled:
                adjacency[arom_id].add(unit_id)
                adjacency[unit_id].add(arom_id)

    # Frozen v1.0 rule:
    # - aromatic nodes always count as UV pi-system components;
    # - isolated non-aromatic pi units do NOT create singleton components.
    included_nodes: Set[int] = set(aromatic_node_ids)
    included_nodes |= {nid for nid in nonaromatic_node_ids if len(adjacency.get(nid, set())) > 0}

    components = connected_components(included_nodes, adjacency) if included_nodes else []
    node_to_component: Dict[int, int] = {}
    for cid, comp in enumerate(components):
        for nid in comp:
            node_to_component[nid] = cid

    pi_framework_atoms: Set[int] = set()
    # Attachment chemistry (B5) needs to know atoms belonging to explicit pi units
    # even when an isolated non-aromatic unit is not counted as a UV component.
    for node in nodes.values():
        pi_framework_atoms |= set(node.atoms)

    return UVGraph(
        nodes=nodes,
        adjacency=dict(adjacency),
        included_nodes=included_nodes,
        components=components,
        node_to_component=node_to_component,
        atom_to_nodes=dict(atom_to_nodes),
        pi_framework_atoms=pi_framework_atoms,
        selected_unit_bond_to_node=selected_unit_bond_to_node,
    )


def uv_graph_descriptors(mol: Chem.Mol, graph: UVGraph) -> Dict[str, Any]:
    uv_atoms: Set[int] = set()
    max_atoms = 0
    max_units = 0
    max_linear_pi_units = 0

    for comp in graph.components:
        atoms: Set[int] = set()
        for nid in comp:
            atoms |= set(graph.nodes[nid].atoms)
        uv_atoms |= atoms
        max_atoms = max(max_atoms, len(atoms))
        max_units = max(max_units, len(comp))
        max_linear_pi_units = max(
            max_linear_pi_units,
            longest_simple_path_node_count(comp, graph.adjacency),
        )

    max_cc = longest_path_in_induced_kind_graph(
        graph.included_nodes, graph.adjacency, graph.nodes, {"C=C"}
    )
    max_cct = longest_path_in_induced_kind_graph(
        graph.included_nodes, graph.adjacency, graph.nodes, {"C#C"}
    )

    heavy = heavy_atom_count(mol)
    frac = (len(uv_atoms) / heavy) if heavy else 0.0

    all_nonarom_multiple_bonds = [
        b for b in mol.GetBonds() if is_nonaromatic_multiple_bond(b)
    ]
    coupled_selected_bonds: Set[int] = set()
    for bond_idx, nid in graph.selected_unit_bond_to_node.items():
        if nid in graph.included_nodes:
            coupled_selected_bonds.add(bond_idx)

    isolated_unsat = sum(
        1 for bond in all_nonarom_multiple_bonds if bond.GetIdx() not in coupled_selected_bonds
    )

    isolated_alkene = 0
    isolated_alkyne = 0
    for bond in mol.GetBonds():
        kind = classify_pi_bond_kind(mol, bond)
        if kind not in {"C=C", "C#C"}:
            continue
        nid = graph.selected_unit_bond_to_node.get(bond.GetIdx())
        if nid is None or nid not in graph.included_nodes:
            if kind == "C=C":
                isolated_alkene += 1
            else:
                isolated_alkyne += 1

    cumulated_centers = 0
    for atom in mol.GetAtoms():
        multiple = 0
        for bond in atom.GetBonds():
            if is_nonaromatic_multiple_bond(bond):
                multiple += 1
        if multiple >= 2:
            cumulated_centers += 1

    return {
        "uv_pi_system_component_count": len(graph.components),
        "uv_pi_system_atom_count": len(uv_atoms),
        "max_uv_pi_system_atoms": max_atoms,
        "max_uv_pi_system_units": max_units,
        "fraction_uv_pi_system_heavy_atoms": float(frac),
        "max_linear_conjugated_pi_units": max_linear_pi_units,
        "max_linear_conjugated_C=C_units": max_cc,
        "max_linear_conjugated_C#C_units": max_cct,
        "isolated_unsaturated_unit_count": isolated_unsat,
        "isolated_alkene_count": isolated_alkene,
        "isolated_alkyne_count": isolated_alkyne,
        "cumulated_unsaturation_unit_count": cumulated_centers,
    }


# -----------------------------------------------------------------------------
# Ring/exocyclic descriptors
# -----------------------------------------------------------------------------

def exocyclic_descriptors(mol: Chem.Mol) -> Tuple[int, int]:
    rings = [set(int(x) for x in ring) for ring in Chem.GetSymmSSSR(mol)]
    unique_bonds = 0
    relations = 0

    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rel_for_bond = 0
        for ring in rings:
            ina = a in ring
            inb = b in ring
            if ina ^ inb:
                rel_for_bond += 1
        if rel_for_bond:
            unique_bonds += 1
            relations += rel_for_bond

    return unique_bonds, relations


# -----------------------------------------------------------------------------
# Carbonyl / N motif helpers
# -----------------------------------------------------------------------------

def carbonyl_units(mol: Chem.Mol, graph: UVGraph) -> List[Tuple[int, int, int]]:
    """Return (node_id, carbon_idx, oxygen_idx) for all C=O units."""
    out: List[Tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        b = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        if {a.GetAtomicNum(), b.GetAtomicNum()} != {6, 8}:
            continue
        c_idx = a.GetIdx() if a.GetAtomicNum() == 6 else b.GetIdx()
        o_idx = b.GetIdx() if a.GetAtomicNum() == 6 else a.GetIdx()
        nid = graph.selected_unit_bond_to_node[bond.GetIdx()]
        out.append((nid, c_idx, o_idx))
    return out


def imine_units(mol: Chem.Mol, graph: UVGraph) -> List[Tuple[int, int, int]]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        b = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        if {a.GetAtomicNum(), b.GetAtomicNum()} != {6, 7}:
            continue
        c_idx = a.GetIdx() if a.GetAtomicNum() == 6 else b.GetIdx()
        n_idx = b.GetIdx() if a.GetAtomicNum() == 6 else a.GetIdx()
        nid = graph.selected_unit_bond_to_node[bond.GetIdx()]
        out.append((nid, c_idx, n_idx))
    return out


def nitrile_units(mol: Chem.Mol, graph: UVGraph) -> List[Tuple[int, int, int]]:
    out = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() != rdchem.BondType.TRIPLE:
            continue
        a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        b = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        if {a.GetAtomicNum(), b.GetAtomicNum()} != {6, 7}:
            continue
        c_idx = a.GetIdx() if a.GetAtomicNum() == 6 else b.GetIdx()
        n_idx = b.GetIdx() if a.GetAtomicNum() == 6 else a.GetIdx()
        nid = graph.selected_unit_bond_to_node[bond.GetIdx()]
        out.append((nid, c_idx, n_idx))
    return out


def nitro_nitrogen_indices(mol: Chem.Mol) -> List[int]:
    result: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        o_neighbors: List[Tuple[Chem.Atom, Chem.Bond]] = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8:
                o_neighbors.append((other, bond))
        if len(o_neighbors) < 2:
            continue

        has_double_o = any(b.GetBondType() == rdchem.BondType.DOUBLE for _, b in o_neighbors)
        has_single_anionic_o = any(
            b.GetBondType() == rdchem.BondType.SINGLE and o.GetFormalCharge() < 0
            for o, b in o_neighbors
        )
        # Standard sanitized nitro is [N+](=O)[O-]. A neutral fallback is kept
        # for robustness if a source structure uses an equivalent valence form.
        standard = atom.GetFormalCharge() > 0 and has_double_o and has_single_anionic_o
        neutral_fallback = (
            atom.GetFormalCharge() == 0
            and sum(1 for _, b in o_neighbors if b.GetBondType() == rdchem.BondType.DOUBLE) >= 2
        )
        if standard or neutral_fallback:
            result.append(atom.GetIdx())
    return result


def nn_double_bond_count(mol: Chem.Mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() != rdchem.BondType.DOUBLE:
            continue
        a = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
        b = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
        if a == 7 and b == 7:
            count += 1
    return count


def carbonyl_subtype_flags(
    mol: Chem.Mol,
    carbonyl_c_idx: int,
    carbonyl_o_idx: int,
    carbonyl_carbon_set: Set[int],
) -> Dict[str, bool]:
    c_atom = mol.GetAtomWithIdx(carbonyl_c_idx)
    other_neighbors = [
        nbr for nbr in c_atom.GetNeighbors() if nbr.GetIdx() != carbonyl_o_idx
    ]

    has_n = any(n.GetAtomicNum() == 7 for n in other_neighbors)
    has_f = any(n.GetAtomicNum() == 9 for n in other_neighbors)
    single_o_neighbors = []
    for n in other_neighbors:
        bond = mol.GetBondBetweenAtoms(carbonyl_c_idx, n.GetIdx())
        # RDKit may represent endocyclic C-O bonds as AROMATIC in
        # unsaturated lactones/carbonate-like five- and six-membered rings.
        # They are still C(=O)-O connectivity for carbonyl-family auditing.
        if (
            n.GetAtomicNum() == 8
            and bond is not None
            and (
                is_single_bond(bond)
                or (c_atom.GetIsAromatic() and n.GetIsAromatic() and bond.GetIsAromatic())
            )
        ):
            single_o_neighbors.append(n)

    is_acid = any(o.GetTotalNumHs() > 0 for o in single_o_neighbors)

    is_anhydride = False
    ester_like_o = False
    for o in single_o_neighbors:
        for nbr in o.GetNeighbors():
            if nbr.GetIdx() == carbonyl_c_idx:
                continue
            if nbr.GetAtomicNum() != 6:
                continue
            if nbr.GetIdx() in carbonyl_carbon_set:
                is_anhydride = True
            else:
                ester_like_o = True

    # Avoid calling carbamates ordinary esters in v1.0's coarse subtype audit.
    # C(=O)(O)(O) is a carbonate-family carbonyl: a carbonic-acid ester.
    # As in the original v1.0 implementation for non-aromatic carbonates,
    # group this with the coarse 'ester' subclass for auditing purposes.
    carbonate_like = len(single_o_neighbors) >= 2
    is_ester = (ester_like_o or carbonate_like) and not has_n and not is_anhydride
    is_amide = has_n
    is_acyl_fluoride = has_f
    is_aldehyde = c_atom.GetTotalNumHs() > 0

    carbon_neighbors = sum(1 for n in other_neighbors if n.GetAtomicNum() == 6)
    hetero_neighbors = sum(1 for n in other_neighbors if n.GetAtomicNum() not in {1, 6})
    is_ketone = carbon_neighbors >= 2 and hetero_neighbors == 0 and not is_aldehyde

    return {
        "aldehyde": is_aldehyde,
        "ketone": is_ketone,
        "ester": is_ester,
        "amide": is_amide,
        "carboxylic_acid": is_acid,
        "acyl_fluoride": is_acyl_fluoride,
        "anhydride": is_anhydride,
    }


def max_consecutive_cc_from_carbonyl(graph: UVGraph, carbonyl_node_id: int) -> int:
    """Maximum consecutive C=C nodes starting immediately after the carbonyl node."""
    best = 0

    def dfs_cc(nid: int, visited: Set[int], count: int) -> None:
        nonlocal best
        best = max(best, count)
        for nxt in graph.adjacency.get(nid, set()):
            if nxt in visited or graph.nodes[nxt].kind != "C=C":
                continue
            visited.add(nxt)
            dfs_cc(nxt, visited, count + 1)
            visited.remove(nxt)

    for nbr in graph.adjacency.get(carbonyl_node_id, set()):
        if graph.nodes[nbr].kind == "C=C":
            dfs_cc(nbr, {carbonyl_node_id, nbr}, 1)
    return best


# -----------------------------------------------------------------------------
# B2 aromatic-N helper
# -----------------------------------------------------------------------------

def aromatic_n_subtypes(mol: Chem.Mol) -> Tuple[int, int, int]:
    """
    Return (pyridine_like, pyrrole_like, unclassified_aromatic_N).

    Operational v1.0 rule:
    - neutral aromatic N with degree 2 and zero H -> pyridine-like;
    - neutral aromatic N with at least one H OR degree >=3 -> pyrrole-like;
    - charged/otherwise unusual aromatic N -> unclassified warning.

    This is deliberately explicit and testable, matching the schema requirement
    that builder logic must not infer N class from element identity alone.
    """
    pyridine_like = 0
    pyrrole_like = 0
    unclassified = 0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or not atom.GetIsAromatic():
            continue
        if atom.GetFormalCharge() != 0:
            unclassified += 1
            continue
        if atom.GetTotalNumHs() > 0 or atom.GetDegree() >= 3:
            pyrrole_like += 1
        elif atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0:
            pyridine_like += 1
        else:
            unclassified += 1

    return pyridine_like, pyrrole_like, unclassified


# -----------------------------------------------------------------------------
# B5 attachment helpers
# -----------------------------------------------------------------------------

def is_acyl_oxygen(mol: Chem.Mol, o_idx: int, carbonyl_carbons: Set[int]) -> bool:
    atom = mol.GetAtomWithIdx(o_idx)
    if atom.GetAtomicNum() != 8:
        return False
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() not in carbonyl_carbons:
            continue
        bond = mol.GetBondBetweenAtoms(o_idx, nbr.GetIdx())
        if bond is not None and is_single_bond(bond):
            return True
    return False


def single_bond_pi_anchor_neighbors(
    mol: Chem.Mol,
    atom_idx: int,
    pi_framework_atoms: Set[int],
) -> Set[int]:
    anchors: Set[int] = set()
    atom = mol.GetAtomWithIdx(atom_idx)
    for bond in atom.GetBonds():
        if not is_single_bond(bond):
            continue
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() in pi_framework_atoms:
            anchors.add(other.GetIdx())
    return anchors


def atom_anchor_component_ids(anchor_atom: int, graph: UVGraph) -> Set[int]:
    cids: Set[int] = set()
    for nid in graph.atom_to_nodes.get(anchor_atom, set()):
        if nid in graph.node_to_component:
            cids.add(graph.node_to_component[nid])
    return cids


# -----------------------------------------------------------------------------
# Full molecule annotation
# -----------------------------------------------------------------------------

def annotate_mol(mol: Chem.Mol) -> Tuple[Dict[str, Any], int]:
    values: Dict[str, Any] = {}
    warnings = 0

    # ----- A: raw graph facts -----
    values["heavy_atom_count"] = heavy_atom_count(mol)
    values["nonaromatic_double_bond_count"] = sum(
        1
        for b in mol.GetBonds()
        if (not b.GetIsAromatic()) and b.GetBondType() == rdchem.BondType.DOUBLE
    )
    values["triple_bond_count"] = sum(
        1 for b in mol.GetBonds() if b.GetBondType() == rdchem.BondType.TRIPLE
    )
    values["aromatic_bond_count"] = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    values["explicit_unsaturated_unit_count"] = sum(
        1 for b in mol.GetBonds() if is_nonaromatic_multiple_bond(b)
    )

    values.update(rdkit_conjugated_subgraph_descriptors(mol))

    arom = build_aromatic_topology(mol)
    graph = build_uv_pi_graph(mol, arom)
    values.update(uv_graph_descriptors(mol, graph))

    exo_unique, exo_rel = exocyclic_descriptors(mol)
    values["exocyclic_double_bond_count"] = exo_unique
    values["exocyclic_ring_relation_count"] = exo_rel

    # ----- B1: conjugation semantics -----
    values["has_conjugation"] = int(values["uv_pi_system_component_count"] > 0)

    nonarom_edge_exists = False
    for u in graph.included_nodes:
        if graph.nodes[u].is_aromatic:
            continue
        for v in graph.adjacency.get(u, set()):
            if v <= u or v not in graph.included_nodes:
                continue
            if graph.nodes[v].is_nonaromatic:
                nonarom_edge_exists = True
                break
        if nonarom_edge_exists:
            break
    values["has_nonaromatic_conjugation"] = int(nonarom_edge_exists)

    values["isolated_unsaturation"] = int(values["isolated_unsaturated_unit_count"] > 0)
    values["isolated_alkene"] = int(values["isolated_alkene_count"] > 0)
    values["isolated_alkyne"] = int(values["isolated_alkyne_count"] > 0)
    values["conjugated_diene"] = int(values["max_linear_conjugated_C=C_units"] >= 2)
    values["conjugated_triene"] = int(values["max_linear_conjugated_C=C_units"] >= 3)
    values["extended_polyene"] = int(values["max_linear_conjugated_C=C_units"] >= 4)

    # Enyne / diyne are path-pattern semantics on the induced C=C/C#C graph.
    enyne = 0
    diyne = int(values["max_linear_conjugated_C#C_units"] >= 2)
    allowed = {nid for nid in graph.included_nodes if graph.nodes[nid].kind in {"C=C", "C#C"}}
    if allowed:
        comps = connected_components(allowed, graph.adjacency)
        for comp in comps:
            kinds = {graph.nodes[n].kind for n in comp}
            if "C=C" in kinds and "C#C" in kinds:
                enyne = 1
                break
    values["conjugated_enyne"] = enyne
    values["conjugated_diyne"] = diyne
    values["cumulated_unsaturation"] = int(values["cumulated_unsaturation_unit_count"] > 0)

    ring_atoms_all = {a for ring in Chem.GetSymmSSSR(mol) for a in map(int, ring)}
    acyclic = 0
    cyclic = 0
    branched = 0
    mixed = 0
    for comp in graph.components:
        comp_atoms: Set[int] = set()
        for nid in comp:
            comp_atoms |= set(graph.nodes[nid].atoms)
        has_ring = bool(comp_atoms & ring_atoms_all)
        if has_ring:
            cyclic = 1
        # For acyclic_conjugated_system, require actual coupling (>=2 nodes), not
        # an aromatic singleton. This follows the schema's wording.
        if len(comp) >= 2 and not has_ring:
            acyclic = 1
        if any(len(graph.adjacency.get(nid, set()) & comp) >= 3 for nid in comp):
            branched = 1
        kinds = {graph.nodes[nid].kind for nid in comp}
        if "AROMATIC_SYSTEM" in kinds and any(k != "AROMATIC_SYSTEM" for k in kinds):
            mixed = 1

    values["acyclic_conjugated_system"] = acyclic
    values["cyclic_conjugated_system"] = cyclic
    values["branched_conjugated_system"] = branched
    values["multiple_independent_conjugated_systems"] = int(
        values["uv_pi_system_component_count"] >= 2
    )
    values["mixed_aromatic_nonaromatic_conjugation"] = mixed

    nonarom_branch = 0
    for nid in graph.included_nodes:
        if graph.nodes[nid].is_nonaromatic:
            deg_nonarom = sum(
                1
                for v in graph.adjacency.get(nid, set())
                if v in graph.included_nodes and graph.nodes[v].is_nonaromatic
            )
            if deg_nonarom >= 3:
                nonarom_branch = 1
                break
    values["cross_conjugation_candidate"] = nonarom_branch

    # ----- B2: aromatic semantics -----
    values["aromatic_atom_count"] = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    values["aromatic_ring_count"] = len(arom.aromatic_ring_atom_sets)
    values["aromatic_system_count"] = len(arom.system_atom_sets)
    values["largest_aromatic_system_atoms"] = max(
        (len(s) for s in arom.system_atom_sets), default=0
    )
    values["largest_aromatic_system_rings"] = max(
        (len(r) for r in arom.system_ring_indices), default=0
    )
    values["fused_aromatic_ring_system_count"] = arom.fused_system_count
    values["biaryl_single_bond_link_count"] = arom.biaryl_single_bond_link_count

    py_like, pr_like, n_unclassified = aromatic_n_subtypes(mol)
    values["pyridine_like_aromatic_N_count"] = py_like
    values["pyrrole_like_aromatic_N_count"] = pr_like
    warnings += n_unclassified

    values["has_aromatic_system"] = int(values["aromatic_ring_count"] > 0)
    values["single_aromatic_ring"] = int(values["aromatic_ring_count"] == 1)
    values["multiple_aromatic_rings"] = int(values["aromatic_ring_count"] >= 2)
    values["multiple_independent_aromatic_systems"] = int(values["aromatic_system_count"] >= 2)
    values["fused_aromatic_system"] = int(values["fused_aromatic_ring_system_count"] > 0)
    values["has_biaryl_single_bond_link"] = int(values["biaryl_single_bond_link_count"] > 0)

    values["carbocyclic_aromatic"] = int(
        any(all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in s) for s in arom.system_atom_sets)
    )
    values["heteroaromatic"] = int(
        any(any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in s) for s in arom.system_atom_sets)
    )
    values["N_heteroaromatic"] = int(
        any(a.GetIsAromatic() and a.GetAtomicNum() == 7 for a in mol.GetAtoms())
    )
    values["O_heteroaromatic"] = int(
        any(a.GetIsAromatic() and a.GetAtomicNum() == 8 for a in mol.GetAtoms())
    )
    values["has_pyridine_like_aromatic_N"] = int(py_like > 0)
    values["has_pyrrole_like_aromatic_N"] = int(pr_like > 0)

    arom_nonarom_edge = 0
    for u in graph.included_nodes:
        for v in graph.adjacency.get(u, set()):
            if v <= u or v not in graph.included_nodes:
                continue
            if graph.nodes[u].is_aromatic != graph.nodes[v].is_aromatic:
                arom_nonarom_edge = 1
                break
        if arom_nonarom_edge:
            break
    values["aromatic_to_nonaromatic_pi_conjugation"] = arom_nonarom_edge

    # ----- B3: carbonyl semantics -----
    carbonyls = carbonyl_units(mol, graph)
    carbonyl_carbons = {c for _, c, _ in carbonyls}
    values["carbonyl_count"] = len(carbonyls)

    subtype_count = defaultdict(int)
    subtype_by_node: Dict[int, Dict[str, bool]] = {}
    isolated_carbonyl_count = 0
    pi_conj_carbonyl_count = 0
    max_carb_cc_path = 0

    for nid, c_idx, o_idx in carbonyls:
        flags = carbonyl_subtype_flags(mol, c_idx, o_idx, carbonyl_carbons)
        subtype_by_node[nid] = flags
        for key, flag in flags.items():
            if flag:
                subtype_count[key] += 1

        if len(graph.adjacency.get(nid, set())) > 0:
            pi_conj_carbonyl_count += 1
        else:
            isolated_carbonyl_count += 1

        max_carb_cc_path = max(max_carb_cc_path, max_consecutive_cc_from_carbonyl(graph, nid))

    values["isolated_carbonyl_count"] = isolated_carbonyl_count
    values["pi_conjugated_carbonyl_count"] = pi_conj_carbonyl_count
    values["aldehyde_count"] = subtype_count["aldehyde"]
    values["ketone_count"] = subtype_count["ketone"]
    values["ester_count"] = subtype_count["ester"]
    values["amide_count"] = subtype_count["amide"]
    values["carboxylic_acid_carbonyl_count"] = subtype_count["carboxylic_acid"]
    values["acyl_fluoride_carbonyl_count"] = subtype_count["acyl_fluoride"]
    values["anhydride_carbonyl_count"] = subtype_count["anhydride"]

    other_carbonyl = 0
    for nid, _, _ in carbonyls:
        if not any(subtype_by_node[nid].values()):
            other_carbonyl += 1
    values["other_carbonyl_count"] = other_carbonyl
    if other_carbonyl > 0:
        warnings += 1

    values["max_carbonyl_conjugated_C=C_path_count"] = max_carb_cc_path
    values["has_carbonyl"] = int(values["carbonyl_count"] > 0)
    values["isolated_carbonyl"] = int(values["isolated_carbonyl_count"] > 0)
    values["pi_conjugated_carbonyl"] = int(values["pi_conjugated_carbonyl_count"] > 0)
    values["alpha_beta_unsaturated_carbonyl"] = int(max_carb_cc_path >= 1)
    values["extended_unsaturated_carbonyl"] = int(max_carb_cc_path >= 2)

    aryl_carbonyl = 0
    conj_aldehyde = 0
    conj_ketone = 0
    pi_ext_ester = 0
    pi_ext_amide = 0
    for nid, c_idx, _ in carbonyls:
        nbrs = graph.adjacency.get(nid, set())
        # Embedded aromatic-ring carbonyls are not automatically aryl-carbonyl
        # substituents (in contrast to Ar-C(=O)-R).
        if (
            not mol.GetAtomWithIdx(c_idx).GetIsAromatic()
            and any(graph.nodes[v].is_aromatic for v in nbrs)
        ):
            aryl_carbonyl = 1
        if nbrs:
            flags = subtype_by_node[nid]
            conj_aldehyde |= int(flags["aldehyde"])
            conj_ketone |= int(flags["ketone"])
            pi_ext_ester |= int(flags["ester"])
            pi_ext_amide |= int(flags["amide"])

    values["aryl_carbonyl"] = int(aryl_carbonyl)
    values["conjugated_aldehyde"] = int(conj_aldehyde)
    values["conjugated_ketone"] = int(conj_ketone)
    values["pi_extended_ester"] = int(pi_ext_ester)
    values["pi_extended_amide"] = int(pi_ext_amide)

    # ----- B4: nitrogen chromophore semantics -----
    imines = imine_units(mol, graph)
    nitriles = nitrile_units(mol, graph)
    nitro_ns = nitro_nitrogen_indices(mol)

    values["imine_count"] = len(imines)
    values["isolated_imine_count"] = sum(1 for nid, _, _ in imines if not graph.adjacency.get(nid, set()))
    values["pi_conjugated_imine_count"] = values["imine_count"] - values["isolated_imine_count"]
    values["nitrile_count"] = len(nitriles)
    values["isolated_nitrile_count"] = sum(1 for nid, _, _ in nitriles if not graph.adjacency.get(nid, set()))
    values["pi_conjugated_nitrile_count"] = values["nitrile_count"] - values["isolated_nitrile_count"]
    values["nitro_count"] = len(nitro_ns)

    pi_conj_nitro_count = 0
    nitro_attachment_anchors: Dict[int, Set[int]] = {}
    for n_idx in nitro_ns:
        n_atom = mol.GetAtomWithIdx(n_idx)
        anchors: Set[int] = set()
        for nbr in n_atom.GetNeighbors():
            if nbr.GetAtomicNum() == 8:
                continue
            if nbr.GetIdx() in graph.pi_framework_atoms:
                anchors.add(nbr.GetIdx())
        nitro_attachment_anchors[n_idx] = anchors
        if anchors:
            pi_conj_nitro_count += 1

    values["pi_conjugated_nitro_count"] = pi_conj_nitro_count
    values["NN_double_bond_count"] = nn_double_bond_count(mol)
    values["has_imine"] = int(values["imine_count"] > 0)
    values["isolated_imine"] = int(values["isolated_imine_count"] > 0)
    values["pi_conjugated_imine"] = int(values["pi_conjugated_imine_count"] > 0)
    values["has_nitrile"] = int(values["nitrile_count"] > 0)
    values["isolated_nitrile"] = int(values["isolated_nitrile_count"] > 0)
    values["pi_conjugated_nitrile"] = int(values["pi_conjugated_nitrile_count"] > 0)
    values["has_nitro"] = int(values["nitro_count"] > 0)
    values["pi_conjugated_nitro"] = int(values["pi_conjugated_nitro_count"] > 0)
    values["has_NN_double_bond"] = int(values["NN_double_bond_count"] > 0)

    # ----- B5: auxochrome and donor/acceptor context -----
    nitro_n_set = set(nitro_ns)
    imine_n_set = {n for _, _, n in imines}
    nitrile_n_set = {n for _, _, n in nitriles}

    lone_pair_o: Set[int] = set()
    hydroxy_o: Set[int] = set()
    alkoxy_o: Set[int] = set()
    n_attached: Set[int] = set()
    donor_n: Set[int] = set()
    f_attached: Set[int] = set()

    donor_component_occurrences: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()

        if z == 8:
            if idx in graph.pi_framework_atoms:
                continue
            if is_acyl_oxygen(mol, idx, carbonyl_carbons):
                continue
            anchors = single_bond_pi_anchor_neighbors(mol, idx, graph.pi_framework_atoms)
            if not anchors:
                continue

            lone_pair_o.add(idx)
            if atom.GetTotalNumHs() > 0:
                hydroxy_o.add(idx)
                motif = "hydroxy"
            else:
                # Coarse v1.0 alkoxy/ether-like rule: neutral O, no H, at least two
                # heavy-atom neighbors, all local bonds single, and not acyl O.
                heavy_deg = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
                all_single = all(is_single_bond(b) for b in atom.GetBonds())
                if atom.GetFormalCharge() == 0 and heavy_deg >= 2 and all_single:
                    alkoxy_o.add(idx)
                    motif = "alkoxy"
                else:
                    motif = "other_O"

            if motif in {"hydroxy", "alkoxy"}:
                for anchor in anchors:
                    for cid in atom_anchor_component_ids(anchor, graph):
                        donor_component_occurrences[cid].append((motif, idx))

        elif z == 7:
            if idx in graph.pi_framework_atoms:
                continue
            anchors = single_bond_pi_anchor_neighbors(mol, idx, graph.pi_framework_atoms)
            if not anchors:
                continue
            n_attached.add(idx)

            amide_like = any(n.GetIdx() in carbonyl_carbons for n in atom.GetNeighbors())
            excluded = (
                idx in imine_n_set
                or idx in nitrile_n_set
                or idx in nitro_n_set
                or atom.GetFormalCharge() != 0
                or atom.GetDegree() >= 4
                or amide_like
                or any(b.GetBondType() in {rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE} for b in atom.GetBonds())
            )
            if not excluded:
                donor_n.add(idx)
                for anchor in anchors:
                    for cid in atom_anchor_component_ids(anchor, graph):
                        donor_component_occurrences[cid].append(("donor_N", idx))

        elif z == 9:
            anchors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in graph.pi_framework_atoms]
            if anchors:
                f_attached.add(idx)

    values["lone_pair_O_pi_attachment_count"] = len(lone_pair_o)
    values["hydroxy_pi_attachment_count"] = len(hydroxy_o)
    values["alkoxy_pi_attachment_count"] = len(alkoxy_o)
    values["N_pi_attachment_count"] = len(n_attached)
    values["lone_pair_donor_N_pi_attachment_count"] = len(donor_n)
    values["fluorine_pi_attachment_count"] = len(f_attached)
    values["donor_attachment_count"] = len(hydroxy_o) + len(alkoxy_o) + len(donor_n)

    acceptor_count = (
        values["pi_conjugated_carbonyl_count"]
        + values["pi_conjugated_nitrile_count"]
        + values["pi_conjugated_nitro_count"]
    )
    values["acceptor_attachment_count"] = acceptor_count

    acceptor_component_occurrences: Dict[int, List[Tuple[str, int]]] = defaultdict(list)

    for nid, _, _ in carbonyls:
        cid = graph.node_to_component.get(nid)
        if cid is not None and graph.adjacency.get(nid, set()):
            acceptor_component_occurrences[cid].append(("carbonyl", nid))

    for nid, _, _ in nitriles:
        cid = graph.node_to_component.get(nid)
        if cid is not None and graph.adjacency.get(nid, set()):
            acceptor_component_occurrences[cid].append(("nitrile", nid))

    for n_idx, anchors in nitro_attachment_anchors.items():
        cids: Set[int] = set()
        for anchor in anchors:
            cids |= atom_anchor_component_ids(anchor, graph)
        for cid in cids:
            acceptor_component_occurrences[cid].append(("nitro", n_idx))

    pair_count = 0
    for cid in set(donor_component_occurrences) | set(acceptor_component_occurrences):
        d_unique = set(donor_component_occurrences.get(cid, []))
        a_unique = set(acceptor_component_occurrences.get(cid, []))
        pair_count += len(d_unique) * len(a_unique)

    values["donor_acceptor_pair_count_same_system"] = pair_count
    # Reserved donor/acceptor positional path distances are intentionally omitted.
    values["lone_pair_O_attached_to_pi_system"] = int(values["lone_pair_O_pi_attachment_count"] > 0)
    values["hydroxy_attached_to_pi_system"] = int(values["hydroxy_pi_attachment_count"] > 0)
    values["alkoxy_attached_to_pi_system"] = int(values["alkoxy_pi_attachment_count"] > 0)
    values["N_attached_to_pi_system"] = int(values["N_pi_attachment_count"] > 0)
    values["lone_pair_donor_N_attached_to_pi_system"] = int(values["lone_pair_donor_N_pi_attachment_count"] > 0)
    values["fluorine_attached_to_pi_system"] = int(values["fluorine_pi_attachment_count"] > 0)
    values["has_donor_substituted_pi_system"] = int(values["donor_attachment_count"] > 0)
    values["has_acceptor_substituted_pi_system"] = int(values["acceptor_attachment_count"] > 0)
    values["donor_acceptor_same_conjugated_system"] = int(pair_count > 0)

    return values, warnings


# -----------------------------------------------------------------------------
# Consistency validation
# -----------------------------------------------------------------------------

def validate_annotation_values(v: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    def req(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    req(v["has_conjugation"] == int(v["uv_pi_system_component_count"] > 0), "has_conjugation/component mismatch")
    req(not v["conjugated_triene"] or v["conjugated_diene"], "triene !-> diene")
    req(not v["extended_polyene"] or v["conjugated_triene"], "polyene !-> triene")
    req(not v["conjugated_diene"] or v["has_nonaromatic_conjugation"], "diene !-> nonaromatic conjugation")
    req(not v["has_nonaromatic_conjugation"] or v["has_conjugation"], "nonaromatic conjugation !-> conjugation")
    req(not v["conjugated_diene"] or v["max_linear_conjugated_C=C_units"] >= 2, "diene path <2")
    req(not v["conjugated_triene"] or v["max_linear_conjugated_C=C_units"] >= 3, "triene path <3")
    req(not v["extended_polyene"] or v["max_linear_conjugated_C=C_units"] >= 4, "polyene path <4")
    req(not v["isolated_alkene"] or v["isolated_alkene_count"] >= 1, "isolated alkene count mismatch")
    req(not v["isolated_alkyne"] or v["isolated_alkyne_count"] >= 1, "isolated alkyne count mismatch")
    req(not v["multiple_independent_conjugated_systems"] or v["uv_pi_system_component_count"] >= 2, "multiple systems count mismatch")
    req(0.0 <= v["fraction_uv_pi_system_heavy_atoms"] <= 1.0 + 1e-8, "UV fraction outside [0,1]")
    req(0.0 <= v["rdkit_fraction_conjugated_heavy_atoms"] <= 1.0 + 1e-8, "RDKit fraction outside [0,1]")

    req(not v["single_aromatic_ring"] or v["aromatic_ring_count"] == 1, "single aromatic ring mismatch")
    req(not v["multiple_aromatic_rings"] or v["aromatic_ring_count"] >= 2, "multiple aromatic rings mismatch")
    req(not v["multiple_independent_aromatic_systems"] or v["aromatic_system_count"] >= 2, "aromatic systems mismatch")
    req(not v["fused_aromatic_system"] or v["fused_aromatic_ring_system_count"] >= 1, "fused count mismatch")
    req(not v["fused_aromatic_system"] or v["multiple_aromatic_rings"], "fused !-> multiple aromatic rings")
    req(not v["heteroaromatic"] or v["has_aromatic_system"], "heteroaromatic !-> aromatic")
    req(not v["N_heteroaromatic"] or v["heteroaromatic"], "N heteroaromatic !-> heteroaromatic")
    req(not v["O_heteroaromatic"] or v["heteroaromatic"], "O heteroaromatic !-> heteroaromatic")
    req(not v["has_pyridine_like_aromatic_N"] or v["N_heteroaromatic"], "pyridine-like N hierarchy mismatch")
    req(not v["has_pyrrole_like_aromatic_N"] or v["N_heteroaromatic"], "pyrrole-like N hierarchy mismatch")

    req(v["has_carbonyl"] == int(v["carbonyl_count"] >= 1), "has_carbonyl/count mismatch")
    req(not v["isolated_carbonyl"] or v["isolated_carbonyl_count"] >= 1, "isolated carbonyl mismatch")
    req(not v["pi_conjugated_carbonyl"] or v["pi_conjugated_carbonyl_count"] >= 1, "pi carbonyl mismatch")
    req(v["isolated_carbonyl_count"] + v["pi_conjugated_carbonyl_count"] == v["carbonyl_count"], "carbonyl partition mismatch")
    req(not v["alpha_beta_unsaturated_carbonyl"] or v["pi_conjugated_carbonyl"], "alpha-beta !-> pi carbonyl")
    req(not v["alpha_beta_unsaturated_carbonyl"] or v["max_carbonyl_conjugated_C=C_path_count"] >= 1, "alpha-beta path mismatch")
    req(not v["extended_unsaturated_carbonyl"] or v["alpha_beta_unsaturated_carbonyl"], "extended carbonyl !-> alpha-beta")
    req(not v["extended_unsaturated_carbonyl"] or v["max_carbonyl_conjugated_C=C_path_count"] >= 2, "extended carbonyl path mismatch")

    req(v["has_imine"] == int(v["imine_count"] >= 1), "has_imine/count mismatch")
    req(v["isolated_imine_count"] + v["pi_conjugated_imine_count"] == v["imine_count"], "imine partition mismatch")
    req(v["has_nitrile"] == int(v["nitrile_count"] >= 1), "has_nitrile/count mismatch")
    req(v["isolated_nitrile_count"] + v["pi_conjugated_nitrile_count"] == v["nitrile_count"], "nitrile partition mismatch")
    req(v["has_nitro"] == int(v["nitro_count"] >= 1), "has_nitro/count mismatch")
    req(not v["pi_conjugated_nitro"] or v["has_nitro"], "pi nitro !-> nitro")

    req(not v["hydroxy_attached_to_pi_system"] or v["lone_pair_O_attached_to_pi_system"], "hydroxy O hierarchy mismatch")
    req(not v["alkoxy_attached_to_pi_system"] or v["lone_pair_O_attached_to_pi_system"], "alkoxy O hierarchy mismatch")
    req(not v["lone_pair_donor_N_attached_to_pi_system"] or v["N_attached_to_pi_system"], "donor N hierarchy mismatch")
    req(not v["has_donor_substituted_pi_system"] or v["donor_attachment_count"] >= 1, "donor count mismatch")
    req(not v["has_acceptor_substituted_pi_system"] or v["acceptor_attachment_count"] >= 1, "acceptor count mismatch")
    req(not v["donor_acceptor_same_conjugated_system"] or v["donor_attachment_count"] >= 1, "D-A donor mismatch")
    req(not v["donor_acceptor_same_conjugated_system"] or v["acceptor_attachment_count"] >= 1, "D-A acceptor mismatch")
    req(not v["donor_acceptor_same_conjugated_system"] or v["donor_acceptor_pair_count_same_system"] >= 1, "D-A pair mismatch")

    return errors


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def detect_smiles_column(fieldnames: Sequence[str]) -> Optional[str]:
    candidates = [
        "smiles",
        "SMILES",
        "canonical_smiles",
        "canonicalSMILES",
        "mol_smiles",
        "smi",
    ]
    for c in candidates:
        if c in fieldnames:
            return c
    lower = {x.lower(): x for x in fieldnames}
    for c in ["smiles", "canonical_smiles", "mol_smiles", "smi"]:
        if c in lower:
            return lower[c]
    return None


def load_smiles_records(
    path: Path,
    smiles_col: Optional[str],
    id_col: Optional[str],
    limit: Optional[int],
) -> Tuple[List[str], List[str]]:
    suffix = path.suffix.lower()
    smiles: List[str] = []
    ids: List[str] = []

    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            if not reader.fieldnames:
                raise ValueError("Input table has no header.")
            chosen_smiles = smiles_col or detect_smiles_column(reader.fieldnames)
            if chosen_smiles is None or chosen_smiles not in reader.fieldnames:
                raise ValueError(
                    f"Could not determine SMILES column. Available columns: {reader.fieldnames}. "
                    "Pass --smiles-col explicitly."
                )
            if id_col is not None and id_col not in reader.fieldnames:
                raise ValueError(f"--id-col {id_col!r} not found in input columns.")

            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                smiles.append((row.get(chosen_smiles) or "").strip())
                ids.append((row.get(id_col) or str(i)).strip() if id_col else str(i))

    elif suffix in {".smi", ".smiles", ".txt"}:
        with path.open("r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                stripped = line.strip()
                if not stripped:
                    smiles.append("")
                    ids.append(str(i))
                    continue
                parts = stripped.split()
                smiles.append(parts[0])
                ids.append(parts[1] if len(parts) > 1 else str(i))

    elif suffix in {".sdf", ".mol"}:
        if suffix == ".mol":
            mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
            if mol is None:
                smiles = [""]
                ids = ["0"]
            else:
                smiles = [Chem.MolToSmiles(mol, isomericSmiles=True)]
                ids = [mol.GetProp("_Name") if mol.HasProp("_Name") else "0"]
        else:
            supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
            for i, mol in enumerate(supplier):
                if limit is not None and i >= limit:
                    break
                if mol is None:
                    smiles.append("")
                    ids.append(str(i))
                else:
                    smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                    ids.append(mol.GetProp("_Name") if mol.HasProp("_Name") else str(i))
    else:
        raise ValueError(
            f"Unsupported input extension {suffix!r}. Supported: .csv .tsv .smi .smiles .txt .sdf .mol"
        )

    return smiles, ids


def load_uv_valid_mask(path: Optional[Path], n: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"UV valid mask length {len(arr)} != source sample count {n}")
    return arr.astype(bool)


def load_split_labels(path: Optional[Path], n: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    suffix = path.suffix.lower()

    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        labels = np.full(n, "unassigned", dtype="U16")
        for key in ["train", "val", "test"]:
            if key in z:
                idx = np.asarray(z[key], dtype=np.int64).reshape(-1)
                if np.any(idx < 0) or np.any(idx >= n):
                    raise ValueError(f"Split {key} contains out-of-range indices.")
                labels[idx] = key
        return labels

    if suffix == ".npy":
        arr = np.asarray(np.load(path, allow_pickle=True)).reshape(-1)
        if len(arr) != n:
            raise ValueError(f"Split array length {len(arr)} != source sample count {n}")
        if np.issubdtype(arr.dtype, np.integer):
            mapping = {0: "train", 1: "val", 2: "test"}
            return np.asarray([mapping.get(int(x), "unassigned") for x in arr], dtype="U16")
        return arr.astype("U16")

    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        labels = np.full(n, "unassigned", dtype="U16")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            fields = reader.fieldnames or []
            idx_col = "index" if "index" in fields else ("qm9s_index" if "qm9s_index" in fields else None)
            split_col = "split" if "split" in fields else None
            if idx_col is None or split_col is None:
                raise ValueError("Split CSV/TSV must contain index/qm9s_index and split columns.")
            for row in reader:
                idx = int(row[idx_col])
                if idx < 0 or idx >= n:
                    raise ValueError(f"Split index {idx} out of range for n={n}")
                labels[idx] = row[split_col].strip()
        return labels

    raise ValueError("Unsupported split file. Use .npz, .npy, .csv, or .tsv")


# -----------------------------------------------------------------------------
# Statistics / output
# -----------------------------------------------------------------------------

def annotation_statistics_rows(
    binary: np.ndarray,
    numeric: np.ndarray,
    valid_mask: np.ndarray,
    binary_names: List[str],
    numeric_names: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid_n = int(valid_mask.sum())

    for j, name in enumerate(binary_names):
        vals = binary[valid_mask, j]
        pos = int(np.sum(vals == 1))
        neg = int(np.sum(vals == 0))
        rows.append({
            "feature": name,
            "type": "binary",
            "valid_n": valid_n,
            "positive_count": pos,
            "negative_count": neg,
            "prevalence": (pos / valid_n) if valid_n else math.nan,
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
        })

    for j, name in enumerate(numeric_names):
        vals = numeric[valid_mask, j].astype(float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "feature": name,
            "type": "numeric",
            "valid_n": int(len(vals)),
            "positive_count": "",
            "negative_count": "",
            "prevalence": "",
            "mean": float(np.mean(vals)) if len(vals) else math.nan,
            "std": float(np.std(vals)) if len(vals) else math.nan,
            "min": float(np.min(vals)) if len(vals) else math.nan,
            "max": float(np.max(vals)) if len(vals) else math.nan,
        })

    return rows


def split_statistics_rows(
    binary: np.ndarray,
    numeric: np.ndarray,
    valid_mask: np.ndarray,
    split_labels: Optional[np.ndarray],
    binary_names: List[str],
    numeric_names: List[str],
) -> List[Dict[str, Any]]:
    if split_labels is None:
        split_labels = np.full(len(valid_mask), "all", dtype="U16")

    rows: List[Dict[str, Any]] = []
    split_order = [s for s in ["train", "val", "test", "all", "unassigned"] if s in set(split_labels)]
    for s in split_order:
        mask = valid_mask & (split_labels == s)
        n = int(mask.sum())
        for j, name in enumerate(binary_names):
            vals = binary[mask, j]
            pos = int(np.sum(vals == 1))
            rows.append({
                "split": s,
                "feature": name,
                "type": "binary",
                "valid_n": n,
                "positive_count": pos,
                "prevalence_or_mean": (pos / n) if n else math.nan,
            })
        for j, name in enumerate(numeric_names):
            vals = numeric[mask, j].astype(float)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "split": s,
                "feature": name,
                "type": "numeric",
                "valid_n": int(len(vals)),
                "positive_count": "",
                "prevalence_or_mean": float(np.mean(vals)) if len(vals) else math.nan,
            })
    return rows


def write_cooccurrence_csv(
    path: Path,
    binary: np.ndarray,
    valid_mask: np.ndarray,
    names: List[str],
) -> None:
    if not names:
        csv_write_rows(path, ["feature"], [])
        return
    x = binary[valid_mask].astype(np.int64)
    co = x.T @ x
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature"] + names)
        for i, name in enumerate(names):
            writer.writerow([name] + [int(v) for v in co[i]])


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def run_self_test() -> None:
    cases = [
        ("ethene", "C=C", {"isolated_alkene": 1, "has_conjugation": 0, "conjugated_diene": 0}),
        ("1,3-butadiene", "C=CC=C", {"conjugated_diene": 1, "max_linear_conjugated_C=C_units": 2}),
        ("allene", "C=C=C", {"cumulated_unsaturation": 1, "conjugated_diene": 0}),
        ("benzene", "c1ccccc1", {"aromatic_ring_count": 1, "aromatic_system_count": 1, "has_conjugation": 1}),
        ("biphenyl", "c1ccccc1-c2ccccc2", {"aromatic_ring_count": 2, "aromatic_system_count": 2, "biaryl_single_bond_link_count": 1}),
        ("acrolein", "C=CC=O", {"alpha_beta_unsaturated_carbonyl": 1, "max_carbonyl_conjugated_C=C_path_count": 1}),
        ("phenol", "Oc1ccccc1", {"hydroxy_attached_to_pi_system": 1}),
        ("anisole", "COc1ccccc1", {"alkoxy_attached_to_pi_system": 1}),
        ("acetamide", "CC(=O)N", {"amide_count": 1, "isolated_carbonyl": 1, "lone_pair_donor_N_attached_to_pi_system": 0}),
        ("benzonitrile", "N#Cc1ccccc1", {"pi_conjugated_nitrile": 1}),
        ("nitrobenzene", "[O-][N+](=O)c1ccccc1", {"pi_conjugated_nitro": 1}),
        ("pyridine", "n1ccccc1", {"has_pyridine_like_aromatic_N": 1, "has_pyrrole_like_aromatic_N": 0}),
        ("pyrrole", "c1cc[nH]c1", {"has_pyrrole_like_aromatic_N": 1}),
        ("N-methylpyrrole", "Cn1cccc1", {"has_pyrrole_like_aromatic_N": 1}),
        ("2-pyrone_aromatic_lactone", "O=c1cccco1", {
            "carbonyl_count": 1, "ester_count": 1, "other_carbonyl_count": 0,
            "pi_conjugated_carbonyl_count": 1, "isolated_carbonyl_count": 0,
            "aryl_carbonyl": 0,
        }),
        ("aromatic_cyclic_carbonate", "O=c1occo1", {
            "carbonyl_count": 1, "ester_count": 1, "other_carbonyl_count": 0,
            "pi_conjugated_carbonyl_count": 1, "isolated_carbonyl_count": 0,
            "aryl_carbonyl": 0,
        }),
        ("heteroaromatic_carbonate", "O=c1onno1", {
            "carbonyl_count": 1, "ester_count": 1, "other_carbonyl_count": 0,
            "pi_conjugated_carbonyl_count": 1, "isolated_carbonyl_count": 0,
        }),
        ("saturated_cyclic_carbonate_regression", "O=C1OCO1", {
            "carbonyl_count": 1, "ester_count": 1, "other_carbonyl_count": 0,
            "pi_conjugated_carbonyl_count": 0, "isolated_carbonyl_count": 1,
        }),
        ("benzaldehyde_aryl_carbonyl", "O=Cc1ccccc1", {
            "carbonyl_count": 1, "aldehyde_count": 1, "aryl_carbonyl": 1,
        }),
    ]

    failed: List[str] = []
    for name, smi, expected in cases:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        if mol is None:
            failed.append(f"{name}: parse failed")
            continue
        vals, _ = annotate_mol(mol)
        consistency = validate_annotation_values(vals)
        if consistency:
            failed.append(f"{name}: consistency errors: {consistency}")
        for key, exp in expected.items():
            got = vals.get(key)
            if got != exp:
                failed.append(f"{name}: {key}: expected {exp!r}, got {got!r}")

    if failed:
        print("SELF-TEST: FAIL")
        for item in failed:
            print("  -", item)
        raise SystemExit(2)

    print(f"SELF-TEST: PASS ({len(cases)} molecules)")


# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    schema_path = Path(args.schema).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)

    schema = load_schema(schema_path, allow_unfrozen=args.allow_unfrozen_schema)
    binary_feats, numeric_feats, categorical_feats = current_features(schema)
    binary_names = [f["name"] for f in binary_feats]
    numeric_names = [f["name"] for f in numeric_feats]
    categorical_names = [f["name"] for f in categorical_feats]

    if categorical_names:
        raise NotImplementedError(
            "The frozen v1.0 current builder scope is expected to have no active categorical fields. "
            f"Found: {categorical_names}"
        )

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Pass --overwrite only if you intentionally want to replace its generated files."
        )
    ensure_dir(output_dir)

    print(f"[INFO] Loading source: {input_path}")
    smiles, source_ids = load_smiles_records(
        input_path,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        limit=args.limit,
    )
    n = len(smiles)
    if n == 0:
        raise ValueError("No source rows loaded.")

    if (
        not args.allow_count_mismatch
        and args.limit is None
        and args.expected_count is not None
        and n != args.expected_count
    ):
        raise ValueError(
            f"Loaded {n} rows but expected {args.expected_count}. "
            "This guard protects QM9S row alignment. If the mismatch is deliberate, "
            "inspect the source first and then pass --allow-count-mismatch."
        )

    uv_mask = load_uv_valid_mask(Path(args.uv_valid_mask).resolve() if args.uv_valid_mask else None, n)
    split_labels = load_split_labels(Path(args.split_file).resolve() if args.split_file else None, n)

    nb = len(binary_names)
    nn = len(numeric_names)
    binary = np.full((n, nb), -1, dtype=np.int8)
    numeric = np.full((n, nn), np.nan, dtype=np.float32)
    categorical = np.empty((n, 0), dtype=np.int16)
    valid_mask = np.zeros(n, dtype=np.uint8)
    warning_counts = np.zeros(n, dtype=np.int16)
    error_codes = [""] * n

    bindex = {name: j for j, name in enumerate(binary_names)}
    nindex = {name: j for j, name in enumerate(numeric_names)}
    expected_current = set(binary_names) | set(numeric_names)

    print(f"[INFO] Rows: {n}")
    print(f"[INFO] Current binary fields: {nb}")
    print(f"[INFO] Current numeric fields: {nn}")
    print("[INFO] Row order will be preserved exactly; no rows will be dropped.")

    start_time = time.time()
    success = 0
    failed = 0

    for i, smi in enumerate(smiles):
        try:
            if not smi:
                raise ValueError("PARSE_FAILED: empty SMILES")

            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is None:
                raise ValueError("PARSE_FAILED: RDKit MolFromSmiles returned None")

            vals, warnings = annotate_mol(mol)

            missing = expected_current - set(vals)
            if missing:
                raise RuntimeError("MISSING_FEATURES: " + ",".join(sorted(missing)))

            consistency_errors = validate_annotation_values(vals)
            if consistency_errors:
                raise RuntimeError("CONSISTENCY_FAILED: " + " | ".join(consistency_errors))

            for name, j in bindex.items():
                val = int(vals[name])
                if val not in (0, 1):
                    raise RuntimeError(f"NON_BINARY_VALUE: {name}={val!r}")
                binary[i, j] = val

            for name, j in nindex.items():
                numeric[i, j] = float(vals[name])

            valid_mask[i] = 1
            warning_counts[i] = int(warnings)
            success += 1

        except Exception as exc:
            failed += 1
            msg = str(exc)
            if msg.startswith("PARSE_FAILED"):
                code = "PARSE_FAILED"
            elif msg.startswith("MISSING_FEATURES"):
                code = "MISSING_FEATURES"
            elif msg.startswith("CONSISTENCY_FAILED"):
                code = "CONSISTENCY_FAILED"
            else:
                code = "ANNOTATION_EXCEPTION"
            error_codes[i] = f"{code}: {msg}"[:1000]
            if args.fail_fast:
                raise

        if (i + 1) % args.log_every == 0 or (i + 1) == n:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n - i - 1) / rate if rate > 0 else 0.0
            print(
                f"[INFO] {i+1}/{n} | valid={success} failed={failed} | "
                f"{rate:.1f} mol/s | ETA {eta/60:.1f} min"
            )

    # Save arrays.
    np.save(output_dir / "structure_binary_labels.npy", binary)
    np.save(output_dir / "structure_numeric_features.npy", numeric)
    np.save(output_dir / "structure_categorical_features.npy", categorical)
    np.save(output_dir / "annotation_valid_mask.npy", valid_mask)
    np.save(output_dir / "annotation_warning_count.npy", warning_counts)

    json_dump(binary_names, output_dir / "binary_label_names.json")
    json_dump(numeric_names, output_dir / "numeric_feature_names.json")
    json_dump(categorical_names, output_dir / "categorical_feature_names.json")
    json_dump(error_codes, output_dir / "annotation_error_codes.json")

    # Keep a copy of the exact schema used to build this annotation set.
    shutil.copy2(schema_path, output_dir / "uvvis_annotation_schema_v1.0.json")

    valid_bool = valid_mask.astype(bool)
    stats = annotation_statistics_rows(
        binary, numeric, valid_bool, binary_names, numeric_names
    )
    csv_write_rows(
        output_dir / "annotation_statistics.csv",
        ["feature", "type", "valid_n", "positive_count", "negative_count", "prevalence", "mean", "std", "min", "max"],
        stats,
    )

    split_stats = split_statistics_rows(
        binary, numeric, valid_bool, split_labels, binary_names, numeric_names
    )
    csv_write_rows(
        output_dir / "annotation_split_statistics.csv",
        ["split", "feature", "type", "valid_n", "positive_count", "prevalence_or_mean"],
        split_stats,
    )

    write_cooccurrence_csv(
        output_dir / "annotation_cooccurrence.csv",
        binary,
        valid_bool,
        binary_names,
    )

    build_info = {
        "schema_version": schema.get("schema_version"),
        "schema_revision": schema.get("schema_revision"),
        "schema_status": schema.get("schema_status"),
        "uv_pi_system_definition_version": schema.get("uv_pi_system_definition_version"),
        "annotation_builder_version": BUILDER_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "aromaticity_model": schema.get("molecular_processing_policy", {}).get("aromaticity_model"),
        "ring_basis": schema.get("molecular_processing_policy", {}).get("ring_basis"),
        "sanitization_policy": schema.get("molecular_processing_policy", {}).get("sanitization"),
        "tautomer_canonicalization": schema.get("molecular_processing_policy", {}).get("tautomer_canonicalization"),
        "charge_neutralization": schema.get("molecular_processing_policy", {}).get("charge_neutralization"),
        "protomer_normalization": schema.get("molecular_processing_policy", {}).get("protomer_normalization"),
        "source_structure_preserved": schema.get("molecular_processing_policy", {}).get("source_structure_preserved"),
        "source_structure_file": str(input_path),
        "source_structure_sha256": sha256_file(input_path),
        "schema_file": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "source_sample_count": n,
        "uv_valid_sample_count": int(uv_mask.sum()) if uv_mask is not None else None,
        "successful_annotation_count": int(valid_mask.sum()),
        "failed_annotation_count": int(n - valid_mask.sum()),
        "warning_annotation_count": int(np.sum(warning_counts > 0)),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_or_unknown(Path.cwd()),
        "row_alignment_policy": "input order preserved exactly; no sorting/deduplication/filtering",
        "invalid_row_binary_sentinel": -1,
        "invalid_row_numeric_sentinel": "NaN",
        "binary_feature_count": nb,
        "numeric_feature_count": nn,
        "categorical_feature_count": len(categorical_names),
        "uv_valid_mask_file": str(Path(args.uv_valid_mask).resolve()) if args.uv_valid_mask else None,
        "split_file": str(Path(args.split_file).resolve()) if args.split_file else None,
    }
    json_dump(build_info, output_dir / "annotation_build_info.json")

    # Compact failure table for human inspection.
    failure_rows = []
    for i, code in enumerate(error_codes):
        if code:
            failure_rows.append({
                "row_index": i,
                "source_id": source_ids[i],
                "smiles": smiles[i],
                "error": code,
            })
    csv_write_rows(
        output_dir / "annotation_failures.csv",
        ["row_index", "source_id", "smiles", "error"],
        failure_rows,
    )

    elapsed = time.time() - start_time
    print("[DONE] UV-Vis structure annotation build complete.")
    print(f"[DONE] Output: {output_dir}")
    print(f"[DONE] Valid: {int(valid_mask.sum())}/{n}")
    print(f"[DONE] Failures: {n - int(valid_mask.sum())}")
    print(f"[DONE] Runtime: {elapsed/60:.2f} min")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build frozen v1.0 UV-Vis chemical-semantic annotations for QM9S."
    )
    p.add_argument("--input", help="QM9S structure source (.csv/.tsv/.smi/.smiles/.txt/.sdf/.mol)")
    p.add_argument("--smiles-col", default=None, help="SMILES column for CSV/TSV; auto-detected if omitted")
    p.add_argument("--id-col", default=None, help="Optional stable source ID column for failure logs")
    p.add_argument(
        "--schema",
        default="metadata/uvvis_annotation_schema_v1.0.json",
        help="Frozen canonical schema JSON",
    )
    p.add_argument(
        "--output-dir",
        default="processed/qm9s/uvvis_annotations_v1",
        help="Output directory",
    )
    p.add_argument("--uv-valid-mask", default=None, help="Optional existing UV valid mask .npy; rows are NOT filtered")
    p.add_argument("--split-file", default=None, help="Optional split assignment (.npz/.npy/.csv/.tsv)")
    p.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_QM9S_COUNT)
    p.add_argument("--allow-count-mismatch", action="store_true", help="Disable the 129817-row alignment guard")
    p.add_argument("--allow-unfrozen-schema", action="store_true", help="Development only; canonical runs should not use this")
    p.add_argument("--limit", type=int, default=None, help="Process first N rows only (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--self-test", action="store_true", help="Run built-in chemistry sanity tests and exit")
    args = p.parse_args()

    if not args.self_test and not args.input:
        p.error("--input is required unless --self-test is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    build(args)


if __name__ == "__main__":
    main()
