"""B3.5-D1: training label co-occurrence and structural subgroup infrastructure.

This is a Validation-development module.  It reads authoritative Training
indices for co-occurrence and authoritative Validation indices/predictions for
subgroup analysis.  It never reads Test indices, refits thresholds, or writes
formal LINE-B-ANALYSIS/LINE-B-FINAL assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.line_b.common import (
    PREPARED,
    ROOT as PROJECT_ROOT,
    atomic_output_dir,
    load_label_names,
    relative_to_root,
    resolve_label_set,
    sha256_file,
    write_json,
)
from training.line_b.analyze_line_b_complementarity import (
    POLICIES,
    _aggregate,
    _confusion,
    _load_task_inputs,
    _pair_rows,
    _three_rows,
    _write_csv,
)
from training.line_b.analyze_line_b_complementarity_extended import (
    _molecule_exact_rows,
    _oracle_rows,
    _three_seed_summary,
)
from training.line_b.bootstrap_line_b_scaffold import _check_accepted_inputs
from training.line_b.bootstrap_line_b_scaffold import _classification_f1, _ranking_metrics, _load_scaffold_map

PHASE = "B3.5-D1"
MODALITIES = ("ir", "raman", "uvvis")
TASKS = (("Shared14", "shared14"), ("UVSPEC15", "uvspec15"))
STRUCTURE_PATH = PROJECT_ROOT / "processed/qm9s/structures/qm9s_structure_library.csv"
MANIFEST_PATH = PREPARED / "qm9s_manifest_prepared.csv"
SPLIT_PATHS = {
    modality: PREPARED / f"{modality}_scaffold_split_valid.npz"
    for modality in MODALITIES
}
LABEL_PATHS = {
    "Shared14": PREPARED / "functional_group_labels.npy",
    "UVSPEC15": PREPARED / "uvspec15_labels.npy",
}
LABEL_NAME_PATHS = {
    "Shared14": PREPARED / "functional_group_label_names.json",
    "UVSPEC15": PREPARED / "uvspec15_label_names.json",
}
HIERARCHY = {
    "Shared14": (
        ("carbonyl", "aldehyde"), ("carbonyl", "amide"),
        ("carbonyl", "ester"), ("carbonyl", "ketone"),
    ),
    "UVSPEC15": (
        ("has_conjugation", "has_nonaromatic_conjugation"),
        ("has_conjugation", "acyclic_conjugated_system"),
        ("has_conjugation", "cyclic_conjugated_system"),
        ("has_conjugation", "has_aromatic_system"),
        ("has_conjugation", "mixed_aromatic_nonaromatic_conjugation"),
        ("has_aromatic_system", "heteroaromatic"),
        ("has_aromatic_system", "mixed_aromatic_nonaromatic_conjugation"),
        ("pi_conjugated_carbonyl", "alpha_beta_unsaturated_carbonyl"),
        ("pi_conjugated_carbonyl", "aryl_carbonyl"),
    ),
}
SEMANTIC_GROUPS = {
    "Shared14": {
        "carbonyl": ("aldehyde", "amide", "carbonyl", "ester", "ketone"),
        "nitrogen": ("amide", "amine", "imine", "nitrile", "nitro"),
        "oxygen": ("aldehyde", "carbonyl", "ester", "ether", "hydroxyl", "ketone"),
        "unsaturation": ("alkene", "alkyne", "imine", "nitrile"),
        "heteroatom": ("c_f_bond", "nitro"),
    },
    "UVSPEC15": {
        "conjugation": ("has_conjugation", "has_nonaromatic_conjugation", "acyclic_conjugated_system", "cyclic_conjugated_system", "mixed_aromatic_nonaromatic_conjugation"),
        "aromatic": ("has_aromatic_system", "heteroaromatic", "mixed_aromatic_nonaromatic_conjugation"),
        "carbonyl": ("pi_conjugated_carbonyl", "alpha_beta_unsaturated_carbonyl", "aryl_carbonyl"),
        "nitrogen": ("pi_conjugated_imine", "pi_conjugated_nitrile"),
        "donor_acceptor": ("has_donor_substituted_pi_system", "has_acceptor_substituted_pi_system", "donor_acceptor_same_conjugated_system"),
    },
}
BOOTSTRAP_SUBGROUPS = None


def _sha(path: Path) -> str:
    return sha256_file(path)


def _read_validation_split(modality: str) -> tuple[np.ndarray, np.ndarray]:
    path = SPLIT_PATHS[modality]
    with np.load(path, allow_pickle=False) as data:
        train = np.asarray(data["train"], dtype=np.int64)
        val = np.asarray(data["val"], dtype=np.int64)
    return train, val


def _read_train_only(modality: str) -> np.ndarray:
    return _read_validation_split(modality)[0]


def _read_val_only(modality: str) -> np.ndarray:
    return _read_validation_split(modality)[1]


def _training_common(task: str) -> tuple[np.ndarray, dict]:
    trains = {modality: _read_train_only(modality) for modality in MODALITIES}
    common = np.intersect1d(np.intersect1d(trains["ir"], trains["raman"]), trains["uvvis"])
    common = np.sort(common.astype(np.int64))
    coverage = {
        "rule": "intersection of ir/raman/uvvis authoritative Training indices; Test indices are never read",
        "per_modality_train_n": {m: int(len(v)) for m, v in trains.items()},
        "common_train_n": int(len(common)),
        "excluded_from_common": {
            m: int(len(v) - len(common)) for m, v in trains.items()
        },
    }
    return common, coverage


def _bind_structure_inputs(task: str, task_info: dict) -> dict:
    accepted = _check_accepted_inputs(task_info, "shared14" if task == "Shared14" else "uvspec15")
    verification = json.loads((PROJECT_ROOT / accepted["verification_path"]).read_text(encoding="utf-8"))
    raw = {Path(r["path"]).name: r for r in verification.get("raw_assets") or []}
    checks = {}
    for path, name in [
        (MANIFEST_PATH, "qm9s_manifest_prepared.csv"),
        (STRUCTURE_PATH, "qm9s_structure_library.csv"),
        (LABEL_PATHS[task], LABEL_PATHS[task].name),
        (LABEL_NAME_PATHS[task], LABEL_NAME_PATHS[task].name),
        (SPLIT_PATHS["ir"], "ir_scaffold_split_valid.npz"),
        (SPLIT_PATHS["raman"], "raman_scaffold_split_valid.npz"),
        (SPLIT_PATHS["uvvis"], "uvvis_scaffold_split_valid.npz"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
        if name not in raw:
            raise ValueError(f"B3.5-B v2 does not register {name}")
        actual = _sha(path)
        if actual != raw[name]["sha256"]:
            raise ValueError(f"Input changed since B3.5-B v2: {name}")
        checks[name] = {"path": relative_to_root(path), "sha256": actual}
    return {"accepted": accepted, "files": checks}


def _read_structure() -> tuple[dict[int, dict], np.ndarray]:
    mapping = {}
    with STRUCTURE_PATH.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["row_index"])
            if idx in mapping:
                raise ValueError(f"Duplicate structure row_index: {idx}")
            mapping[idx] = row
    if set(mapping) != set(range(129817)):
        raise ValueError("Structure library row_index is not 0..129816")
    return mapping, np.asarray(sorted(mapping), dtype=np.int64)


def _tokens(value: str) -> set[str]:
    return {token.strip() for token in (value or "").split(";") if token.strip()}


def _parse_int(row: dict, field: str) -> int:
    value = row.get(field, "")
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing structure field {field}")
    return int(float(value))


def _parse_float(row: dict, field: str) -> float:
    value = row.get(field, "")
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing structure field {field}")
    return float(value)


def _n_o_f_bin(value: int) -> str:
    value = int(value)
    return str(value) if value < 3 else ">=3"


def _subgroup_definitions(val_rows: np.ndarray, val_struct: list[dict], train_heavy: np.ndarray) -> tuple[list[dict], dict]:
    q1, q2, q3 = [float(x) for x in np.percentile(train_heavy, [25, 50, 75], method="linear")]
    duplicates = {"q1_eq_q2": q1 == q2, "q2_eq_q3": q2 == q3, "all_equal": q1 == q2 == q3}

    def heavy_bin(value: int) -> str:
        if value <= q1:
            return "Q1"
        if value <= q2:
            return "Q2"
        if value <= q3:
            return "Q3"
        return "Q4"

    definitions = []
    for dimension, labels, getter in [
        ("ring_count", ("0", "1", "2", ">=3"), lambda r: "0" if _parse_int(r, "ring_count") == 0 else "1" if _parse_int(r, "ring_count") == 1 else "2" if _parse_int(r, "ring_count") == 2 else ">=3"),
        ("aromatic_ring_count", ("0", "1", ">=2"), lambda r: "0" if _parse_int(r, "aromatic_ring_count") == 0 else "1" if _parse_int(r, "aromatic_ring_count") == 1 else ">=2"),
        ("n_o_f", ("0", "1", "2", ">=3"), lambda r: _n_o_f_bin(_parse_int(r, "nitrogen_count") + _parse_int(r, "oxygen_count") + _parse_int(r, "fluorine_count"))),
        ("heavy_atom_count", ("Q1", "Q2", "Q3", "Q4"), lambda r: heavy_bin(_parse_int(r, "heavy_atom_count"))),
        ("rotatable_bond_count", ("0", "1", ">=2"), lambda r: "0" if _parse_int(r, "rotatable_bond_count") == 0 else "1" if _parse_int(r, "rotatable_bond_count") == 1 else ">=2"),
        ("fraction_csp3", ("<0.5", ">=0.5"), lambda r: "<0.5" if _parse_float(r, "fraction_csp3") < 0.5 else ">=0.5"),
    ]:
        for label in labels:
            mask = np.asarray([getter(r) == label for r in val_struct], dtype=bool)
            definitions.append({"subgroup": f"{dimension}={label}", "dimension": dimension, "bin": label, "n_samples": int(mask.sum()), "mask": mask})
    for group in ("carbonyl", "aromatic", "nitrile", "alkyne", "amine", "hydroxyl"):
        if group == "carbonyl":
            mask = np.asarray(["carbonyl" in _tokens(r.get("functional_group_types", "")) for r in val_struct], dtype=bool)
        elif group == "aromatic":
            mask = np.asarray([any("aromatic" in t or "aryl" in t for t in _tokens(r.get("functional_group_types", ""))) for r in val_struct], dtype=bool)
        else:
            mask = np.asarray([group in _tokens(r.get("functional_group_types", "")) for r in val_struct], dtype=bool)
        definitions.append({"subgroup": f"functional_group={group}", "dimension": "functional_group_types", "bin": group, "n_samples": int(mask.sum()), "mask": mask})
    coverage = {}
    for dimension in ("ring_count", "aromatic_ring_count", "n_o_f", "heavy_atom_count", "rotatable_bond_count", "fraction_csp3"):
        masks = [d["mask"] for d in definitions if d["dimension"] == dimension]
        if not masks:
            raise AssertionError(f"No masks for {dimension}")
        total = int(sum(mask.sum() for mask in masks))
        per_row = np.sum(np.vstack(masks), axis=0)
        if total != len(val_rows) or not np.all(per_row == 1):
            raise AssertionError(f"Mutually exclusive bin coverage failed for {dimension}: total={total}, rows={len(val_rows)}")
        coverage[dimension] = {"n_samples": int(len(val_rows)), "sum_bin_memberships": total, "all_rows_exactly_one": True}
    heavy_counts = {label: int(sum(heavy_bin(_parse_int(r, "heavy_atom_count")) == label for r in val_struct)) for label in ("Q1", "Q2", "Q3", "Q4")}
    meta = {
        "heavy_atom_quartiles": {"q1": q1, "q2": q2, "q3": q3},
        "duplicate_boundaries": duplicates,
        "degenerate_partition": bool(duplicates["all_equal"] or heavy_counts["Q2"] == 0 and heavy_counts["Q3"] == 0 and heavy_counts["Q4"] == 0),
        "heavy_atom_bin_counts": heavy_counts,
        "coverage": coverage,
        "rule": "Training common-set linear quartiles; <= boundaries, duplicate boundaries preserved; no equal-heavy-atom splitting",
        "val_n": int(len(val_rows)),
    }
    return definitions, meta


def _cooccurrence(task: str, common_train: np.ndarray, names: list[str]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    labels = np.load(LABEL_PATHS[task], mmap_mode="r", allow_pickle=False)[common_train].astype(np.int64)
    if labels.shape[1] != len(names):
        raise ValueError(f"Label width mismatch: {task}")
    support = labels.sum(axis=0).astype(np.int64)
    n = int(labels.shape[0])
    counts = labels.T @ labels
    support_rows = [{"task": task, "label_index": i, "label": name, "support": int(support[i]), "n_train": n, "prevalence": float(support[i] / n)} for i, name in enumerate(names)]
    count_rows = []
    conditional_rows = []
    phi_rows = []
    v_rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            n11 = int(counts[i, j]); n1a = int(support[i]); n1b = int(support[j])
            n10 = n1a - n11; n01 = n1b - n11; n00 = n - n11 - n10 - n01
            phi = None if min(n1a, n - n1a, n1b, n - n1b) == 0 else float((n11 * n00 - n10 * n01) / math.sqrt(n1a * (n - n1a) * n1b * (n - n1b)))
            conditional = None if n1a == 0 else float(n11 / n1a)
            count_rows.append({"task": task, "label_a": a, "label_b": b, "both": n11, "support_a": n1a, "support_b": n1b, "n_train": n})
            conditional_rows.append({"task": task, "label_a": a, "label_b": b, "p_b_given_a": conditional, "support_a": n1a})
            phi_rows.append({"task": task, "label_a": a, "label_b": b, "phi": phi, "n_train": n})
            v_rows.append({"task": task, "label_a": a, "label_b": b, "cramers_v": None if phi is None else abs(phi), "n_train": n})
    hierarchy = []
    for parent, child in HIERARCHY[task]:
        if parent not in names or child not in names:
            continue
        pi, ci = names.index(parent), names.index(child)
        both = int(counts[pi, ci]); ps = int(support[pi]); cs = int(support[ci])
        child_without_parent = int(((labels[:, ci] == 1) & (labels[:, pi] == 0)).sum())
        hierarchy.append({
            "task": task, "parent": parent, "child": child, "parent_support": ps, "child_support": cs,
            "both": both,
            "p_child_given_parent": (both / ps) if ps else None,
            "p_parent_given_child": (both / cs) if cs else None,
            "child_positive_parent_negative": child_without_parent,
            "strict_containment": bool(child_without_parent == 0),
            "definition_support": "predefined_semantic_hierarchy; pending human confirmation",
        })
    semantic = []
    for group, members in SEMANTIC_GROUPS[task].items():
        present = [m for m in members if m in names]
        semantic.append({"task": task, "semantic_group": group, "members": ";".join(present), "n_members": len(present), "union_support": int(labels[:, [names.index(m) for m in present]].max(axis=1).sum()) if present else 0, "definition_support": "semantic_group_only"})
    return support_rows, count_rows, conditional_rows, phi_rows, v_rows, hierarchy, semantic


def _evaluate_subgroups(task: str, task_info: dict, row_index: np.ndarray, definitions: list[dict], val_struct: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    names = task_info["expected_names"]
    metric_rows = []
    main_rows = []
    for definition in definitions:
        mask = definition["mask"]
        if int(mask.sum()) == 0:
            continue
        sub_info = {
            "task": task, "label_set": task_info["label_set"], "expected_names": names,
            "best_single": task_info["best_single"], "models": [],
        }
        for model in task_info["models"]:
            sub_info["models"].append({
                **model,
                "probabilities": model["probabilities"][mask],
                "labels": model["labels"][mask],
                "row_index": model["row_index"][mask],
            })
        for model in sub_info["models"]:
            for policy in POLICIES:
                for index, label in enumerate(names):
                    y = model["labels"][:, index].astype(np.int64)
                    p = model["probabilities"][:, index].astype(np.float64)
                    threshold = 0.5 if policy == "fixed_0.5" else float(model["thresholds"][index])
                    pred = (p >= threshold).astype(np.int64)
                    pos = int((y == 1).sum()); neg = int((y == 0).sum())
                    if pos > 0 and neg > 0:
                        ap = float(average_precision_score(y, p))
                        auroc = float(roc_auc_score(y, p))
                    else:
                        ap = None; auroc = None
                    conf = _confusion(y, pred)
                    enough = int(len(y)) >= 100 and pos >= 30
                    row = {
                        "task": task, "label_set": task_info["label_set"],
                        "subgroup": definition["subgroup"], "dimension": definition["dimension"], "bin": definition["bin"],
                        "n_samples": int(len(y)), "n_positive": pos, "n_negative": neg,
                        "prevalence": (pos / len(y)) if len(y) else None,
                        "seed": int(model["seed"]), "modality": model["modality"], "threshold_policy": policy,
                        "label_index": index, "label": label, "threshold": threshold,
                        "ap": ap, "auroc": auroc, "f1": conf["f1"],
                        "support_sufficient": enough,
                        "support_reason": "ok" if enough else ("n<100" if len(y) < 100 else "positive<30"),
                    }
                    metric_rows.append(row)
                    if enough:
                        main_rows.append(row)
    three_seed = _three_seed_summary(
        "subgroup_metrics", metric_rows,
        ("task", "label_set", "subgroup", "dimension", "bin", "modality", "threshold_policy", "label_index", "label"),
        ("n_samples", "n_positive", "prevalence", "ap", "auroc", "f1"),
    )
    return metric_rows, main_rows, three_seed


def _subgroup_conditional(task: str, task_info: dict, row_index: np.ndarray, definitions: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    names = task_info["expected_names"]
    cond_summary, cond_label, oracle_summary, oracle_label_all, or_gain, exact_summary = [], [], [], [], [], []
    for definition in definitions:
        mask = definition["mask"]
        if int(mask.sum()) == 0:
            continue
        sub_info = {
            "task": task, "label_set": task_info["label_set"], "expected_names": names,
            "best_single": task_info["best_single"], "models": [],
        }
        for model in task_info["models"]:
            sub_info["models"].append({
                **model,
                "probabilities": model["probabilities"][mask],
                "labels": model["labels"][mask],
                "row_index": model["row_index"][mask],
            })
        pair_rows = _pair_rows(sub_info)
        three_rows = _three_rows(sub_info)
        oracle, oracle_label, gains, pairwise, pairwise_label = _oracle_rows(sub_info, three_rows, pair_rows)
        # Preserve the already-defined per-label Oracle rows.
        for row in oracle_label:
            oracle_label_all.append({"subgroup": definition["subgroup"], **row})
        for row in pairwise:
            row = dict(row)
            row["base_modality"] = row.get("modality_a")
            row["correcting_modality"] = row.get("modality_b")
            row["comparison_direction"] = row.get("comparison")
            cond_summary.append({"subgroup": definition["subgroup"], **row})
        for row in pairwise_label:
            row = dict(row)
            row["base_modality"] = row.get("modality_a")
            row["correcting_modality"] = row.get("modality_b")
            row["comparison_direction"] = row.get("comparison")
            cond_label.append({"subgroup": definition["subgroup"], **row})
        for row in oracle:
            oracle_summary.append({"subgroup": definition["subgroup"], **row})
        for row in gains:
            or_gain.append({"subgroup": definition["subgroup"], **row})
        _, exact = _molecule_exact_rows(sub_info)
        for row in exact:
            exact_summary.append({"subgroup": definition["subgroup"], **row})
    return cond_summary, cond_label, oracle_summary, oracle_label_all, or_gain, exact_summary


def _subgroup_bootstrap_sanity(task: str, task_info: dict, definitions: list[dict], reps: int = 20, max_subgroups: int | None = None) -> list[dict]:
    """Scaffold-cluster bootstrap; all predefined subgroups are retained by default."""
    names = task_info["expected_names"]
    candidates = [d for d in definitions if int(d["mask"].sum()) >= 2]
    selected_defs = candidates if max_subgroups is None else candidates[:max_subgroups]
    scaffold_map, _ = _load_scaffold_map()
    rows = []
    macro_rows_all = []
    for definition in selected_defs:
        subgroup = definition["subgroup"]
        mask = definition["mask"]
        n_sub = int(mask.sum())
        row_index = np.asarray(task_info["models"][0]["row_index"][mask], dtype=np.int64)
        cluster_names = sorted({scaffold_map[int(i)] for i in row_index})
        cidx = {name: i for i, name in enumerate(cluster_names)}
        cluster_id = np.asarray([cidx[scaffold_map[int(i)]] for i in row_index], dtype=np.int64)
        n_clusters = len(cluster_names)
        submodels = {}
        for model in task_info["models"]:
            submodels[(int(model["seed"]), model["modality"])] = {
                "probabilities": model["probabilities"][mask],
                "labels": model["labels"][mask],
                "thresholds": model["thresholds"],
            }
        rng = np.random.default_rng(20261117)
        values = defaultdict(lambda: np.full(reps, np.nan, dtype=np.float64))
        macro_counts = defaultdict(lambda: np.zeros(reps, dtype=np.int64))
        pos_counts = {
            name: int(submodels[(42, "ir")]["labels"][:, i].sum())
            for i, name in enumerate(names)
        }
        def support_fields(scope: str) -> tuple:
            if not scope.startswith("label:"):
                return "", ""
            name = scope.split(":", 1)[1]
            return pos_counts[name], ("sufficient" if n_sub >= 100 and pos_counts[name] >= 30 else "insufficient")
        point_values = {}
        # Unweighted point estimates and conditional rates.
        point_arrays = {}
        for seed in (42, 123, 2026):
            for modality in MODALITIES:
                model = submodels[(seed, modality)]
                truth = model["labels"]
                prob = model["probabilities"]
                arr = {}
                for policy in ("threshold_free", "fixed_0.5", "validation_f1"):
                    ap = np.full(len(names), np.nan); au = np.full(len(names), np.nan); f1 = np.full(len(names), np.nan)
                    for i in range(len(names)):
                        if policy == "threshold_free":
                            r = _ranking_metrics(truth[:, i], prob[:, i], np.ones(len(truth)))
                            ap[i] = np.nan if r["ap"] is None else r["ap"]
                            au[i] = np.nan if r["auroc"] is None else r["auroc"]
                        else:
                            thr = 0.5 if policy == "fixed_0.5" else float(model["thresholds"][i])
                            f1[i] = _classification_f1(truth[:, i], prob[:, i], np.ones(len(truth)), thr)["f1"]
                    arr[policy] = {"ap": ap, "auroc": au, "f1": f1}
                point_arrays[(seed, modality)] = arr
        for seed in (42, 123, 2026):
            for base, other, comparison in (("uvvis", "ir", "UV-IR"), ("uvvis", "raman", "UV-Raman"), ("ir", "raman", "IR-Raman")):
                for policy, metric in (("threshold_free", "AP"), ("threshold_free", "AUROC"), ("fixed_0.5", "F1"), ("validation_f1", "F1")):
                    key = "ap" if metric == "AP" else ("auroc" if metric == "AUROC" else "f1")
                    va = point_arrays[(seed, base)][policy][key]
                    vb = point_arrays[(seed, other)][policy][key]
                    valid = np.isfinite(va) & np.isfinite(vb)
                    point_values[(comparison, metric, "macro", seed, policy)] = float(np.mean(va[valid] - vb[valid])) if valid.any() else None
                    for i, name in enumerate(names):
                        point_values[(comparison, metric, f"label:{name}", seed, policy)] = float(va[i] - vb[i]) if np.isfinite(va[i]) and np.isfinite(vb[i]) else None
                if base != "uvvis":
                    continue
                for policy in POLICIES:
                    y = submodels[(seed, "uvvis")]["labels"]
                    uv = submodels[(seed, "uvvis")]; base_model = submodels[(seed, other)]
                    uv_pred = uv["probabilities"] >= np.where(policy == "fixed_0.5", 0.5, uv["thresholds"]).reshape(1, -1)
                    base_pred = base_model["probabilities"] >= np.where(policy == "fixed_0.5", 0.5, base_model["thresholds"]).reshape(1, -1)
                    base_correct = base_pred == (y == 1); uv_correct = uv_pred == (y == 1)
                    base_wrong = ~base_correct
                    other_only = base_wrong & uv_correct
                    for i, name in enumerate(names):
                        cond = None if base_wrong[:, i].sum() == 0 else float(other_only[:, i].sum() / base_wrong[:, i].sum())
                        p_other = float(uv_correct[:, i].mean())
                        point_values[(comparison, "conditional_delta", f"label:{name}", seed, policy)] = None if cond is None else cond - p_other
                    cond_all = other_only.sum() / base_wrong.sum() if base_wrong.sum() else None
                    p_all = uv_correct.sum() / uv_correct.size
                    point_values[(comparison, "conditional_delta", "overall", seed, policy)] = None if cond_all is None else cond_all - p_all

        for rep in range(reps):
            draw_counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters).astype(np.float64)
            weight = draw_counts[cluster_id].astype(np.float64)
            per_seed = {}
            for seed in (42, 123, 2026):
                per_seed[seed] = {}
                for modality in MODALITIES:
                    model = submodels[(seed, modality)]
                    truth = model["labels"]; prob = model["probabilities"]
                    arr = {"ap": np.full(len(names), np.nan), "auroc": np.full(len(names), np.nan), "fixed_0.5": np.full(len(names), np.nan), "validation_f1": np.full(len(names), np.nan)}
                    for i in range(len(names)):
                        rank = _ranking_metrics(truth[:, i], prob[:, i], weight)
                        if rank["ap"] is not None:
                            arr["ap"][i] = rank["ap"]; arr["auroc"][i] = rank["auroc"]
                        arr["fixed_0.5"][i] = _classification_f1(truth[:, i], prob[:, i], weight, 0.5)["f1"]
                        arr["validation_f1"][i] = _classification_f1(truth[:, i], prob[:, i], weight, float(model["thresholds"][i]))["f1"]
                    per_seed[seed][modality] = arr
            for seed in (42, 123, 2026):
                for base, other, comparison in (("uvvis", "ir", "UV-IR"), ("uvvis", "raman", "UV-Raman"), ("ir", "raman", "IR-Raman")):
                    for policy, metric, key in (("threshold_free", "AP", "ap"), ("threshold_free", "AUROC", "auroc"), ("fixed_0.5", "F1", "fixed_0.5"), ("validation_f1", "F1", "validation_f1")):
                        va = per_seed[seed][base][key]; vb = per_seed[seed][other][key]
                        valid = np.isfinite(va) & np.isfinite(vb)
                        values[(comparison, metric, "macro", seed, policy)][rep] = float(np.mean(va[valid] - vb[valid])) if valid.any() else np.nan
                        macro_counts[(comparison, metric, seed, policy)][rep] = int(valid.sum())
                        for i, name in enumerate(names):
                            values[(comparison, metric, f"label:{name}", seed, policy)][rep] = float(va[i] - vb[i]) if np.isfinite(va[i]) and np.isfinite(vb[i]) else np.nan
                    if base != "uvvis":
                        continue
                    for policy in POLICIES:
                        y = submodels[(seed, "uvvis")]["labels"]
                        uv = submodels[(seed, "uvvis")]; base_model = submodels[(seed, other)]
                        uv_pred = uv["probabilities"] >= np.where(policy == "fixed_0.5", 0.5, uv["thresholds"]).reshape(1, -1)
                        base_pred = base_model["probabilities"] >= np.where(policy == "fixed_0.5", 0.5, base_model["thresholds"]).reshape(1, -1)
                        bc = (base_pred == (y == 1)).astype(np.float64); uc = (uv_pred == (y == 1)).astype(np.float64)
                        bw = 1.0 - bc; oo = bw * uc
                        num_l = (oo * weight[:, None]).sum(axis=0)
                        den_l = (bw * weight[:, None]).sum(axis=0)
                        for i, name in enumerate(names):
                            cond = np.nan if den_l[i] == 0 else num_l[i] / den_l[i]
                            p_other = float((uc[:, i] * weight).sum() / weight.sum())
                            values[(comparison, "conditional_delta", f"label:{name}", seed, policy)][rep] = cond - p_other if np.isfinite(cond) else np.nan
                        cond_all = (oo * weight[:, None]).sum() / (bw * weight[:, None]).sum() if (bw * weight[:, None]).sum() else np.nan
                        p_all = float((uc * weight[:, None]).sum() / (weight.sum() * uc.shape[1]))
                        values[(comparison, "conditional_delta", "overall", seed, policy)][rep] = cond_all - p_all if np.isfinite(cond_all) else np.nan

        for key, arr in values.items():
            label, metric, scope, seed, policy = key
            if policy == "":
                policy = "threshold_free"
            valid = arr[np.isfinite(arr)]
            ci_low = ci_high = None
            if len(valid) >= 2:
                ci = np.percentile(valid, [2.5, 97.5]); ci_low, ci_high = float(ci[0]), float(ci[1])
            rows.append({
                "task": task, "subgroup": subgroup, "comparison": label,
                "metric": metric, "scope": scope, "seed": seed, "threshold_policy": policy,
                "point_estimate": point_values.get((label, metric, scope, seed, policy)),
                "ci_low": ci_low, "ci_high": ci_high,
                "n_valid": int(len(valid)), "n_undefined": int(reps - len(valid)),
                "n_positive": support_fields(scope)[0], "support_status": support_fields(scope)[1],
                "validation_support_status": support_fields(scope)[1] or "not_applicable",
                "test_support_status": "pending",
                "metric_status": "defined" if len(valid) else "undefined",
                "ci_status": "available" if ci_low is not None else "unavailable",
                "criterion_status": "not_evaluable_validation_sanity",
                "n_labels_total": len(names) if scope == "macro" else 1,
                "n_labels_defined": (float(macro_counts.get((label, metric, seed, policy), np.zeros(reps, dtype=np.int64)).mean()) if scope == "macro" else 1.0),
                "n_labels_defined_min": int(macro_counts.get((label, metric, seed, policy), np.zeros(reps, dtype=np.int64)).min()) if scope == "macro" else 1,
                "n_labels_defined_max": int(macro_counts.get((label, metric, seed, policy), np.zeros(reps, dtype=np.int64)).max()) if scope == "macro" else 1,
                "n_labels_undefined_mean": (len(names) - float(macro_counts.get((label, metric, seed, policy), np.zeros(reps, dtype=np.int64)).mean())) if scope == "macro" else 0.0,
                "reps": reps, "rng_seed": 20261117,
                "b4_rule": "", "b4_rule_met": None,
                "note": "sanity only; not formal B4 2,000 bootstrap",
            })
        # Three-seed paired mean and CI.
        grouped = defaultdict(dict)
        for (label, metric, scope, seed, policy), arr in values.items():
            grouped[(label, metric, scope, policy)][seed] = arr
        for (label, metric, scope, policy), by_seed in grouped.items():
            matrix = np.vstack([by_seed[s] for s in (42, 123, 2026)])
            valid_rows = np.all(np.isfinite(matrix), axis=0)
            three = matrix[:, valid_rows].mean(axis=0) if valid_rows.any() else np.asarray([])
            if scope == "macro" and metric in ("AP", "AUROC"):
                label_arrays = []
                for name in names:
                    label_arrays.append(np.vstack([
                        values.get((label, metric, "label:" + name, s, policy), np.full(reps, np.nan))
                        for s in (42, 123, 2026)
                    ]))
                common_counts = np.zeros(reps, dtype=np.int64)
                for rep_index in range(reps):
                    common_counts[rep_index] = sum(bool(np.all(np.isfinite(arr[:, rep_index]))) for arr in label_arrays)
                macro_counts[(label, metric, "three_seed_mean", policy)] = common_counts
            elif scope == "macro":
                macro_counts[(label, metric, "three_seed_mean", policy)] = np.full(reps, len(names), dtype=np.int64)
            points = []
            for seed in (42, 123, 2026):
                points.append(point_values.get((label, metric, scope, seed, policy)))
            point_mean = None if any(p is None for p in points) else float(np.mean(points))
            ci_low = ci_high = None
            if len(three) >= 2:
                ci = np.percentile(three, [2.5, 97.5]); ci_low, ci_high = float(ci[0]), float(ci[1])
            rows.append({
                "task": task, "subgroup": subgroup, "comparison": label,
                "metric": metric, "scope": scope, "seed": "three_seed_mean", "threshold_policy": policy,
                "point_estimate": point_mean, "ci_low": ci_low, "ci_high": ci_high,
                "n_valid": int(len(three)), "n_undefined": int(reps - len(three)),
                "n_positive": support_fields(scope)[0], "support_status": support_fields(scope)[1],
                "validation_support_status": support_fields(scope)[1] or "not_applicable",
                "test_support_status": "pending",
                "metric_status": "defined" if len(three) else "undefined",
                "ci_status": "available" if ci_low is not None else "unavailable",
                "criterion_status": "not_evaluable_validation_sanity",
                "n_labels_total": len(names) if scope == "macro" else 1,
                "n_labels_defined": (float(macro_counts.get((label, metric, "three_seed_mean", policy), np.zeros(reps, dtype=np.int64)).mean()) if scope == "macro" else 1.0),
                "n_labels_defined_min": int(macro_counts.get((label, metric, "three_seed_mean", policy), np.zeros(reps, dtype=np.int64)).min()) if scope == "macro" else 1,
                "n_labels_defined_max": int(macro_counts.get((label, metric, "three_seed_mean", policy), np.zeros(reps, dtype=np.int64)).max()) if scope == "macro" else 1,
                "n_labels_undefined_mean": (len(names) - float(macro_counts.get((label, metric, "three_seed_mean", policy), np.zeros(reps, dtype=np.int64)).mean())) if scope == "macro" else 0.0,
                "reps": reps, "rng_seed": 20261117,
                "b4_rule": (
                    "UV marginal: delta_AP >= 0.02 and CI low > 0"
                    if metric == "AP" and scope == "macro" and label.startswith("UV-")
                    else ("conditional complementarity: delta > 0 and CI low > 0" if metric == "conditional_delta" and scope == "overall" and label.startswith("UV-") else "")
                ),
                # D1 is Validation development; never decide the B4 criterion here.
                "b4_rule_met": None,
                "note": "sanity only; not formal B4 2,000 bootstrap; seed SD not mixed into CI",
            })
        macro_rows_all.extend(_macro_count_rows(task, subgroup, names, macro_counts, reps))
    return rows, macro_rows_all


def _macro_count_rows(task: str, subgroup: str, names: list[str], macro_counts: dict, reps: int) -> list[dict]:
    rows = []
    for (comparison, metric, seed, policy), arr in macro_counts.items():
        total = len(names)
        for rep_index, count in enumerate(arr):
            defined = int(count)
            undefined = int(total - defined)
            if defined + undefined != total:
                raise AssertionError("Macro label count does not sum to total")
            rows.append({
                "task": task, "subgroup": subgroup, "comparison": comparison,
                "metric": metric, "seed": seed, "threshold_policy": policy,
                "replicate_index": int(rep_index), "n_labels_total": total,
                "n_labels_defined": defined, "n_labels_undefined": undefined,
            })
    return rows


def _analyze_task(task: str, label_set: str, out_root: Path, bootstrap_max_subgroups: int | None) -> dict:
    task_info = _load_task_inputs(
        label_set, ROOT / "runs/LINE-B-PREDICTIONS", ROOT / "runs/LINE-B-THRESHOLDS",
        ROOT / "runs/LINE-B-METRICS", False,
    )
    binding = _bind_structure_inputs(task, task_info)
    common_train, coverage = _training_common(task)
    names = load_label_names(LABEL_NAME_PATHS[task])
    structure, all_rows = _read_structure()
    val_rows = np.asarray(task_info["models"][0]["row_index"], dtype=np.int64)
    if len(val_rows) != 12981 or len(np.unique(val_rows)) != 12981:
        raise ValueError("Unexpected Validation row_index")
    if set(val_rows.tolist()) - set(structure):
        raise ValueError("Validation row_index missing from structure library")
    val_struct = [structure[int(i)] for i in val_rows]
    train_struct = [structure[int(i)] for i in common_train]
    train_heavy = np.asarray([_parse_int(r, "heavy_atom_count") for r in train_struct], dtype=np.int64)
    definitions, subgroup_meta = _subgroup_definitions(val_rows, val_struct, train_heavy)

    support, counts, conditional, phi, v, hierarchy, semantic = _cooccurrence(task, common_train, names)
    metric_rows, main_rows, three_seed = _evaluate_subgroups(task, task_info, val_rows, definitions, val_struct)
    cond_summary, cond_label, oracle_summary, oracle_label_all, or_gain, exact_summary = _subgroup_conditional(task, task_info, val_rows, definitions)
    # Consistency: per-label Oracle sums must reproduce overall Oracle counts.
    oracle_fields = (
        "any_correct", "exactly_one_correct", "all_wrong", "all_correct",
        "only_ir_correct", "only_raman_correct", "only_uv_correct",
        "ir_raman_any_correct", "three_any_correct",
        "uv_extra_oracle_correct", "ir_raman_both_wrong_uv_correct",
    )
    for overall in oracle_summary:
        subset = [r for r in oracle_label_all if r["subgroup"] == overall["subgroup"] and r["seed"] == overall["seed"] and r["threshold_policy"] == overall["threshold_policy"]]
        if not subset:
            raise AssertionError(f"Missing per-label Oracle rows for {overall['subgroup']}/{overall['seed']}/{overall['threshold_policy']}")
        if sum(int(r["n_pairs"]) for r in subset) != int(overall["n_pairs"]):
            raise AssertionError("Per-label Oracle n_pairs do not match overall")
        for field in oracle_fields:
            if sum(int(r[field]) for r in subset) != int(overall[field]):
                raise AssertionError(f"Per-label Oracle mismatch: {field}")
    cond_three = _three_seed_summary(
        "conditional_complementarity", cond_summary,
        ("task", "label_set", "subgroup", "comparison", "threshold_policy"),
        ("conditional_correct_given_base_wrong", "p_other_correct", "conditional_delta",
         "fn_rescue_rate", "fp_correction_rate", "new_fp_rate", "new_fn_rate", "error_jaccard"),
    )
    direction_fields = {
        row["comparison"]: (row["base_modality"], row["correcting_modality"], row["comparison_direction"])
        for row in cond_summary
    }
    if len(direction_fields) != 6:
        raise AssertionError(f"Expected six conditional directions, found {len(direction_fields)}")
    for row in cond_three:
        base, correcting, direction = direction_fields[row["comparison"]]
        row["base_modality"] = base
        row["correcting_modality"] = correcting
        row["comparison_direction"] = direction
    oracle_three = _three_seed_summary(
        "sample_label_oracle", oracle_summary,
        ("task", "label_set", "subgroup", "threshold_policy"),
        ("any_correct_rate", "exactly_one_correct_rate", "all_wrong_rate", "all_correct_rate",
         "only_ir_correct_rate", "only_raman_correct_rate", "only_uv_correct_rate",
         "ir_raman_any_correct_rate", "three_any_correct_rate",
         "uv_extra_oracle_correct_rate", "ir_raman_both_wrong_uv_correct_rate"),
    )
    oracle_label_three = _three_seed_summary(
        "sample_label_oracle_by_label", oracle_label_all,
        ("task", "label_set", "subgroup", "label_index", "label", "threshold_policy"),
        ("any_correct", "exactly_one_correct", "all_wrong", "all_correct",
         "only_ir_correct", "only_raman_correct", "only_uv_correct",
         "ir_raman_any_correct", "three_any_correct",
         "uv_extra_oracle_correct", "ir_raman_both_wrong_uv_correct"),
    )
    or_three = _three_seed_summary(
        "or_gain", or_gain,
        ("task", "label_set", "subgroup", "comparison", "threshold_policy"),
        ("positive_labels", "zero_labels", "negative_labels", "mean_or_gain"),
    )
    exact_three = _three_seed_summary(
        "molecule_exact_pattern", exact_summary,
        ("task", "label_set", "subgroup", "threshold_policy"),
        ("all_three_exact", "all_three_exact_rate", "none_exact", "none_exact_rate",
         "partial_exact", "partial_exact_rate", "only_ir_exact", "only_ir_exact_rate",
         "only_raman_exact", "only_raman_exact_rate", "only_uv_exact", "only_uv_exact_rate"),
    )
    bootstrap, bootstrap_macro_counts = _subgroup_bootstrap_sanity(task, task_info, definitions, reps=20, max_subgroups=bootstrap_max_subgroups)
    nonempty_subgroups = {d["subgroup"] for d in definitions if int(d["mask"].sum()) >= 2}
    executed_subgroups = {r["subgroup"] for r in bootstrap}
    if bootstrap_max_subgroups is None and executed_subgroups != nonempty_subgroups:
        raise AssertionError("Not all non-empty predefined subgroups were executed")
    if not executed_subgroups.issubset(nonempty_subgroups):
        raise AssertionError("Bootstrap executed an empty/unknown subgroup")

    task_out = Path(out_root) / task / "structure_analysis"
    with atomic_output_dir(task_out) as temporary:
        _write_csv(temporary / "training_cooccurrence_support.csv", support, list(support[0].keys()))
        _write_csv(temporary / "training_cooccurrence_counts.csv", counts, list(counts[0].keys()))
        _write_csv(temporary / "training_cooccurrence_conditional.csv", conditional, list(conditional[0].keys()))
        _write_csv(temporary / "training_cooccurrence_phi.csv", phi, list(phi[0].keys()))
        _write_csv(temporary / "training_cooccurrence_cramers_v.csv", v, list(v[0].keys()))
        _write_csv(temporary / "training_hierarchy_pairs.csv", hierarchy, list(hierarchy[0].keys()))
        _write_csv(temporary / "training_semantic_groups.csv", semantic, list(semantic[0].keys()))
        _write_csv(temporary / "training_coverage.csv", [{"task": task, **coverage}], list({"task": task, **coverage}.keys()))
        _write_csv(temporary / "subgroup_definitions.csv", [{k: v for k, v in d.items() if k != "mask"} for d in definitions], ["subgroup", "dimension", "bin", "n_samples"])
        _write_csv(temporary / "subgroup_metrics.csv", metric_rows, list(metric_rows[0].keys()))
        _write_csv(temporary / "subgroup_metrics_main.csv", main_rows, list(main_rows[0].keys()))
        _write_csv(temporary / "subgroup_metrics_three_seed.csv", three_seed, list(three_seed[0].keys()))
        _write_csv(temporary / "subgroup_conditional_summary.csv", cond_summary, list(cond_summary[0].keys()))
        _write_csv(temporary / "subgroup_conditional_by_label.csv", cond_label, list(cond_label[0].keys()))
        _write_csv(temporary / "subgroup_conditional_three_seed.csv", cond_three, list(cond_three[0].keys()))
        _write_csv(temporary / "subgroup_oracle_summary.csv", oracle_summary, list(oracle_summary[0].keys()))
        _write_csv(temporary / "subgroup_oracle_by_label.csv", oracle_label_all, list(oracle_label_all[0].keys()))
        _write_csv(temporary / "subgroup_oracle_by_label_three_seed.csv", oracle_label_three, list(oracle_label_three[0].keys()))
        _write_csv(temporary / "subgroup_oracle_three_seed.csv", oracle_three, list(oracle_three[0].keys()))
        _write_csv(temporary / "subgroup_or_gain.csv", or_gain, list(or_gain[0].keys()))
        _write_csv(temporary / "subgroup_or_gain_three_seed.csv", or_three, list(or_three[0].keys()))
        _write_csv(temporary / "subgroup_molecule_exact.csv", exact_summary, list(exact_summary[0].keys()))
        _write_csv(temporary / "subgroup_molecule_exact_three_seed.csv", exact_three, list(exact_three[0].keys()))
        _write_csv(temporary / "subgroup_bootstrap_sanity.csv", bootstrap, list(bootstrap[0].keys()))
        _write_csv(temporary / "subgroup_bootstrap_macro_counts.csv", bootstrap_macro_counts, list(bootstrap_macro_counts[0].keys()))
        output_names = [
            "training_cooccurrence_support.csv", "training_cooccurrence_counts.csv",
            "training_cooccurrence_conditional.csv", "training_cooccurrence_phi.csv",
            "training_cooccurrence_cramers_v.csv", "training_hierarchy_pairs.csv",
            "training_semantic_groups.csv", "training_coverage.csv", "subgroup_definitions.csv", "subgroup_metrics.csv",
            "subgroup_metrics_main.csv", "subgroup_metrics_three_seed.csv",
            "subgroup_conditional_summary.csv", "subgroup_conditional_by_label.csv",
            "subgroup_conditional_three_seed.csv", "subgroup_oracle_summary.csv",
            "subgroup_oracle_by_label.csv",
            "subgroup_oracle_by_label_three_seed.csv",
            "subgroup_oracle_three_seed.csv", "subgroup_or_gain.csv",
            "subgroup_or_gain_three_seed.csv", "subgroup_molecule_exact.csv",
            "subgroup_molecule_exact_three_seed.csv", "subgroup_bootstrap_sanity.csv",
            "subgroup_bootstrap_macro_counts.csv",
        ]
        summary = {
            "phase": PHASE, "task": task, "label_set": label_set,
            "validation_development": True, "no_test_access": True,
            "split": "validation", "formal_b4": False,
            "n_models": 9, "n_labels": len(names),
            "seeds": [42, 123, 2026], "threshold_policies": {"primary": "fixed_0.5", "secondary": "validation_f1"},
            "best_single": task_info["best_single"],
            "inputs": binding,
            "training": {"common_n": int(len(common_train)), "coverage": coverage},
            "subgroups": {
                "n_definitions": len(definitions),
                "n_nonempty": len([d for d in definitions if int(d["mask"].sum()) >= 2]),
                "n_main_support": len({r["subgroup"] for r in main_rows}),
                "n_main_support_definition": "subgroups with at least one label meeting n>=100 and positive>=30; not all labels",
                "heavy_atom": subgroup_meta,
            },
            "protocol": {
                "path": "docs/experiments/line_b_protocol_draft.md",
                "sha256": _sha(ROOT / "docs/experiments/line_b_protocol_draft.md"),
                "version": "B0 v0.2",
            },
            "bootstrap_config": {
                "rng": 20261117, "reps": 20, "ci": "95% percentile",
                "resampling_unit": "murcko_scaffold", "sampling": "with replacement",
                "paired_across_models_and_seeds": True, "weighting": "scaffold draw count",
                "split": "validation", "formal_b4": False, "no_test_access": True,
            },
            "subgroup_bootstrap": {
                "selected_subgroups": sorted({r["subgroup"] for r in bootstrap}),
                "all_predefined_subgroups": [d["subgroup"] for d in definitions if int(d["mask"].sum()) >= 2],
                "n_selected": len(executed_subgroups),
                "n_nonempty": len(nonempty_subgroups),
                "reps": 20, "status": "sanity_only_not_formal_2000", "rows": len(bootstrap),
                "macro_count_rows": len(bootstrap_macro_counts),
                "sanity_limited": bootstrap_max_subgroups is not None,
            },
            "outputs": {name: _sha(temporary / name) for name in output_names},
            "code_sha256": {
                rel: _sha(ROOT / rel) for rel in (
                    "training/line_b/analyze_line_b_structure.py",
                    "training/line_b/analyze_line_b_complementarity.py",
                    "training/line_b/analyze_line_b_complementarity_extended.py",
                    "training/line_b/bootstrap_line_b_scaffold.py",
                )
            },
            "environment": {"python": sys.version, "numpy": np.__version__, "cwd": str(ROOT)},
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "caveats": [
                "Co-occurrence uses the intersection of three authoritative Training modality-valid indices; rule is development-pending human confirmation.",
                "Subgroups use structure-library descriptors, never model predictions.",
                "Validation subgroup metrics are exploratory and do not select models, thresholds, seeds or labels.",
                "No Test index or Test prediction is accessed.",
            ],
        }
        write_json(temporary / "summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "Shared14", "UVSPEC15"], default="all")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bootstrap-max-subgroups", type=int, default=None, help="Development sanity limit; omit for all predefined subgroups.")
    parser.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test:
        raise SystemExit("Test is not authorized in B3.5-D1")
    out_root = args.out_root.resolve()
    if out_root.is_relative_to(ROOT.resolve()):
        raise SystemExit("D1 outputs must use a temporary directory outside the project; formal LINE-B-ANALYSIS/LINE-B-FINAL writes are not authorized.")
    tasks = TASKS
    if args.task != "all":
        tasks = [pair for pair in TASKS if pair[0] == args.task]
    summaries = [_analyze_task(task, label_set, out_root, args.bootstrap_max_subgroups) for task, label_set in tasks]
    terminal = []
    for summary in summaries:
        task = summary["task"]
        terminal.append({
            "task": task,
            "n_nonempty_subgroups": summary["subgroups"]["n_nonempty"],
            "n_executed_bootstrap_subgroups": summary["subgroup_bootstrap"]["n_selected"],
            "selected_subgroups": summary["subgroup_bootstrap"]["selected_subgroups"],
            "reps": summary["bootstrap_config"]["reps"],
            "output_file_count": len(summary["outputs"]) + 1,
            "summary_path": str(Path(out_root) / task / "structure_analysis" / "summary.json"),
        })
    print(json.dumps(terminal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
