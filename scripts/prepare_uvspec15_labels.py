"""Prepare the 15 UV-specific binary labels without altering the source annotations.

Usage from SpectraDB project root:
    python scripts/prepare_uvspec15_labels.py

Creates (under processed/qm9s/prepared):
    uvspec15_labels.npy            shape [129817, 15], float32
    uvspec15_label_names.json      ordered names
    uvspec15_label_manifest.json   source, schema and row-alignment provenance

No train/validation/test samples are reordered or dropped.  Invalid UV spectra are
excluded by the *existing* uvvis_scaffold_split_valid.npz at training time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "processed" / "qm9s" / "prepared"
ANNOTATION = ROOT / "processed" / "qm9s" / "uvvis_annotations_v1_0_1"

# Deliberately ordered: this is the exact supervised output-head ordering.
UVSPEC15_LABELS = [
    "has_conjugation",
    "has_nonaromatic_conjugation",
    "acyclic_conjugated_system",
    "cyclic_conjugated_system",
    "has_aromatic_system",
    "heteroaromatic",
    "mixed_aromatic_nonaromatic_conjugation",
    "pi_conjugated_carbonyl",
    "alpha_beta_unsaturated_carbonyl",
    "aryl_carbonyl",
    "pi_conjugated_imine",
    "pi_conjugated_nitrile",
    "has_donor_substituted_pi_system",
    "has_acceptor_substituted_pi_system",
    "donor_acceptor_same_conjugated_system",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare() -> None:
    names_path = ANNOTATION / "binary_label_names.json"
    matrix_path = ANNOTATION / "structure_binary_labels.npy"
    valid_path = ANNOTATION / "annotation_valid_mask.npy"
    uv_path = PREPARED / "uvvis_float32.npy"
    split_path = PREPARED / "uvvis_scaffold_split_valid.npz"
    masks_path = PREPARED / "modality_valid_masks.npz"
    info_path = ANNOTATION / "annotation_build_info.json"

    required = [names_path, matrix_path, valid_path, uv_path,
                split_path, masks_path, info_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    source_names = json.loads(names_path.read_text(encoding="utf-8-sig"))
    if not isinstance(source_names, list) or len(set(source_names)) != len(source_names):
        raise ValueError("binary_label_names.json must be a unique ordered list")
    if len(UVSPEC15_LABELS) != 15 or len(set(UVSPEC15_LABELS)) != 15:
        raise ValueError("Selected labels must be 15 unique names")
    absent = [s for s in UVSPEC15_LABELS if s not in source_names]
    if absent:
        raise ValueError("Missing source label(s): " + ", ".join(absent))

    columns = [source_names.index(s) for s in UVSPEC15_LABELS]
    raw = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    annotation_valid = np.load(valid_path, allow_pickle=False)
    spectra = np.load(uv_path, mmap_mode="r", allow_pickle=False)
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    if info.get("annotation_builder_version") != "1.0.1":
        raise ValueError("Expected annotation builder v1.0.1")
    if raw.ndim != 2 or raw.shape[1] != len(source_names):
        raise ValueError(f"Annotation shape mismatch: {raw.shape}")
    n = raw.shape[0]
    if spectra.shape != (n, 701):
        raise ValueError(f"Expected spectral matrix [{n}, 701]; found {spectra.shape}")
    if annotation_valid.shape != (n,) or not bool(annotation_valid.all()):
        raise ValueError("Not every source molecule was successfully annotated")

    with np.load(masks_path, allow_pickle=False) as masks:
        uv_valid = masks["valid_uvvis"]
    with np.load(split_path, allow_pickle=False) as splits:
        indices = {k: np.asarray(splits[k], dtype=np.int64) for k in ("train", "val", "test")}
    concat = np.concatenate(list(indices.values()))
    if uv_valid.shape != (n,) or not np.array_equal(np.sort(concat), np.flatnonzero(uv_valid)):
        raise ValueError("UV-valid split must contain precisely the valid original row indices")
    if len(np.unique(concat)) != len(concat):
        raise ValueError("Duplicate indices in split")

    # Original row order and row count retained; the old dataset can mmap this file unchanged.
    subset = np.asarray(raw[:, columns], dtype=np.float32)
    if subset.shape != (n, 15) or not np.all((subset == 0) | (subset == 1)):
        raise ValueError("Selected labels are not strictly binary")

    targets = [
        PREPARED / "uvspec15_labels.npy",
        PREPARED / "uvspec15_label_names.json",
        PREPARED / "uvspec15_label_manifest.json",
    ]
    if any(p.exists() for p in targets):
        raise FileExistsError(
            "UVSPEC15 prepared files already exist; inspect them rather than silently overwrite:\n"
            + "\n".join(str(p) for p in targets if p.exists())
        )
    print("UVSPEC15: source shape", raw.shape, "-> output", subset.shape)
    for i, name in enumerate(UVSPEC15_LABELS):
        t, v, te = (int(subset[indices[k], i].sum()) for k in ("train", "val", "test"))
        if not (t > 0 and v > 0 and te > 0):
            raise ValueError(f"Label {name}: train={t}, val={v}, test={te}; stop before preparation")
        print(f"  {i:02d} {name:45s} train={t:6d} val={v:5d} test={te:5d}")

    manifest = {
        "target_set": "UVSPEC15",
        "label_names": UVSPEC15_LABELS,
        "column_indices_in_source": columns,
        "source_annotation_dir": str(ANNOTATION.relative_to(ROOT)),
        "source_annotation_builder_version": info["annotation_builder_version"],
        "source_schema_sha256": info.get("schema_sha256"),
        "source_binary_labels_sha256": file_sha256(matrix_path),
        "source_row_count": n,
        "label_matrix_shape": [n, 15],
        "row_alignment": "same row positions as original QM9S spectra; no filtering or sorting",
        "uv_split": str(split_path.relative_to(ROOT)),
        "split_sizes": {k: len(v) for k, v in indices.items()},
        "num_workers_for_training": 4,
    }
    tmp = PREPARED / "uvspec15_labels.npy.tmp"
    try:
        with tmp.open("wb") as fh:
            np.save(fh, subset, allow_pickle=False)
        tmp.replace(targets[0])
        targets[1].write_text(json.dumps(UVSPEC15_LABELS, indent=2), encoding="utf-8")
        targets[2].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)
    print("PASS: split and row alignment; UVSPEC15 files created")
    for p in targets:
        print(" ", p.relative_to(ROOT))


if __name__ == "__main__":
    prepare()
