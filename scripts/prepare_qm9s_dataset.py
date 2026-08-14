from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")

RAW_DIR = ROOT / "raw" / "qm9s"
MANIFEST_PATH = ROOT / "processed" / "qm9s" / "qm9s_manifest.csv"
OUT_DIR = ROOT / "processed" / "qm9s" / "prepared"

MODALITIES = {
    "ir": RAW_DIR / "ir_boraden.csv",
    "raman": RAW_DIR / "raman_boraden.csv",
    "uvvis": RAW_DIR / "uv_boraden.csv",
}


def json_safe(value: Any) -> Any:
    """将 NumPy、Path 等对象转换成可写入 JSON 的普通类型。"""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def detect_layout(path: Path) -> dict[str, Any]:
    """
    检查 CSV 布局：
    - 判断第一列是否是行索引；
    - 提取光谱列；
    - 尝试从列名解析光谱横轴。
    """
    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()

    if len(columns) < 2:
        raise RuntimeError(f"{path.name} 列数不足：{len(columns)}")

    first_column = columns[0]

    first_values = pd.read_csv(
        path,
        nrows=32,
        usecols=[first_column],
    ).iloc[:, 0]

    name_is_index = (
        str(first_column).lower().startswith("unnamed")
        or str(first_column).lower()
        in {"index", "row", "row_index", "id"}
    )

    numeric = pd.to_numeric(first_values, errors="coerce")
    values_are_sequential = False

    if numeric.notna().all() and len(numeric) >= 3:
        values = numeric.to_numpy(dtype=np.int64)
        expected = np.arange(
            values[0],
            values[0] + len(values),
            dtype=np.int64,
        )
        values_are_sequential = np.array_equal(values, expected)

    index_column = (
        first_column
        if name_is_index or values_are_sequential
        else None
    )

    spectrum_columns = (
        columns[1:]
        if index_column is not None
        else columns
    )

    numeric_axis = pd.to_numeric(
        pd.Index(spectrum_columns),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if np.isfinite(numeric_axis).all():
        axis = numeric_axis.astype(np.float32)
        axis_source = "csv_column_names"
    else:
        axis = np.arange(
            len(spectrum_columns),
            dtype=np.float32,
        )
        axis_source = "sample_point_index"

    differences = np.diff(axis.astype(np.float64))

    if len(differences) == 0:
        monotonic = "single_point"
    elif np.all(differences > 0):
        monotonic = "increasing"
    elif np.all(differences < 0):
        monotonic = "decreasing"
    else:
        monotonic = "not_monotonic"

    return {
        "index_column": index_column,
        "spectrum_columns": spectrum_columns,
        "axis": axis,
        "axis_source": axis_source,
        "axis_monotonic": monotonic,
        "axis_duplicate_count": int(
            len(axis) - len(np.unique(axis))
        ),
        "axis_step_min": (
            float(differences.min())
            if len(differences)
            else None
        ),
        "axis_step_median": (
            float(np.median(differences))
            if len(differences)
            else None
        ),
        "axis_step_max": (
            float(differences.max())
            if len(differences)
            else None
        ),
    }


def robust_intensity_outliers(
    row_absmax: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    """
    对每条光谱的最大绝对强度取 log10，
    使用 median + MAD 检查异常强度。

    这里只标记，不删除、不裁剪。
    """
    flags = np.zeros(len(row_absmax), dtype=bool)

    valid = (
        np.isfinite(row_absmax)
        & (row_absmax > 0)
    )

    if valid.sum() < 10:
        return flags, {
            "log10_median": None,
            "log10_mad": None,
            "threshold": threshold,
        }

    log_values = np.log10(
        row_absmax[valid].astype(np.float64)
    )

    median = float(np.median(log_values))
    mad = float(
        np.median(np.abs(log_values - median))
    )

    if mad <= 0:
        return flags, {
            "log10_median": median,
            "log10_mad": mad,
            "threshold": threshold,
        }

    robust_z = np.zeros(
        len(row_absmax),
        dtype=np.float64,
    )

    robust_z[valid] = (
        np.abs(
            np.log10(row_absmax[valid]) - median
        )
        / (1.4826 * mad)
    )

    flags = robust_z > threshold

    return flags, {
        "log10_median": median,
        "log10_mad": mad,
        "threshold": threshold,
    }


def convert_modality(
    modality: str,
    csv_path: Path,
    sample_count: int,
    chunk_size: int,
    zero_tolerance: float,
    negative_tolerance: float,
    outlier_threshold: float,
) -> dict[str, Any]:
    """将一个光谱 CSV 转为 float32 NPY，并完成质量检查。"""

    print(f"\n{'=' * 70}")
    print(f"处理模态：{modality.upper()}")
    print(f"{'=' * 70}")

    layout = detect_layout(csv_path)

    spectrum_columns = layout["spectrum_columns"]
    axis = layout["axis"]
    point_count = len(spectrum_columns)

    output_path = OUT_DIR / f"{modality}_float32.npy"
    temporary_path = OUT_DIR / f"{modality}_float32.tmp.npy"
    axis_path = OUT_DIR / f"{modality}_axis.npy"
    qc_path = OUT_DIR / f"{modality}_qc_metrics.npz"

    np.save(axis_path, axis)

    matrix = open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, point_count),
    )

    row_min = np.empty(
        sample_count,
        dtype=np.float32,
    )
    row_max = np.empty(
        sample_count,
        dtype=np.float32,
    )
    row_absmax = np.empty(
        sample_count,
        dtype=np.float32,
    )
    row_mean_abs = np.empty(
        sample_count,
        dtype=np.float32,
    )

    nonfinite_per_row = np.zeros(
        sample_count,
        dtype=np.int32,
    )
    negative_per_row = np.zeros(
        sample_count,
        dtype=np.int32,
    )
    zero_flags = np.zeros(
        sample_count,
        dtype=bool,
    )

    total_nonfinite = 0
    total_negative = 0
    cursor = 0

    print(f"输入文件：{csv_path}")
    print(f"样本数：{sample_count:,}")
    print(f"每条光谱采样点：{point_count:,}")
    print(f"移除的索引列：{layout['index_column']!r}")
    print(
        f"光谱轴：{axis[0]} → {axis[-1]}，"
        f"{layout['axis_monotonic']}"
    )

    reader = pd.read_csv(
        csv_path,
        usecols=spectrum_columns,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        try:
            values = chunk.to_numpy(
                dtype=np.float32,
                copy=True,
            )
        except (TypeError, ValueError):
            values = (
                chunk.apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .to_numpy(
                    dtype=np.float32,
                    copy=True,
                )
            )

        row_end = cursor + len(values)

        if row_end > sample_count:
            raise RuntimeError(
                f"{csv_path.name} 行数超过 manifest："
                f"{row_end} > {sample_count}"
            )

        finite_mask = np.isfinite(values)

        current_nonfinite = (
            ~finite_mask
        ).sum(axis=1).astype(np.int32)

        nonfinite_per_row[
            cursor:row_end
        ] = current_nonfinite

        total_nonfinite += int(
            current_nonfinite.sum()
        )

        # 模型无法直接处理 NaN/Inf。
        # 这里替换为 0，同时完整记录数量。
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        current_negative = (
            values < -negative_tolerance
        ).sum(axis=1).astype(np.int32)

        negative_per_row[
            cursor:row_end
        ] = current_negative

        total_negative += int(
            current_negative.sum()
        )

        minimum = values.min(axis=1)
        maximum = values.max(axis=1)
        absmax = np.max(
            np.abs(values),
            axis=1,
        )
        mean_abs = np.mean(
            np.abs(values),
            axis=1,
        )

        row_min[cursor:row_end] = minimum
        row_max[cursor:row_end] = maximum
        row_absmax[cursor:row_end] = absmax
        row_mean_abs[cursor:row_end] = mean_abs

        zero_flags[cursor:row_end] = (
            absmax <= zero_tolerance
        )

        matrix[cursor:row_end] = values
        cursor = row_end

        if (
            chunk_number % 10 == 0
            or cursor == sample_count
        ):
            print(
                f"已转换 "
                f"{cursor:,} / {sample_count:,}"
            )

    matrix.flush()
    del matrix

    if cursor != sample_count:
        raise RuntimeError(
            f"{csv_path.name} 行数与 manifest 不一致："
            f"{cursor} != {sample_count}"
        )

    temporary_path.replace(output_path)

    outlier_flags, outlier_model = (
        robust_intensity_outliers(
            row_absmax=row_absmax,
            threshold=outlier_threshold,
        )
    )

    np.savez_compressed(
        qc_path,
        row_min=row_min,
        row_max=row_max,
        row_absmax=row_absmax,
        row_mean_abs=row_mean_abs,
        nonfinite_per_row=nonfinite_per_row,
        negative_per_row=negative_per_row,
        zero_flags=zero_flags,
        intensity_outlier_flags=outlier_flags,
    )

    check_array = np.load(
        output_path,
        mmap_mode="r",
    )

    expected_shape = (
        sample_count,
        point_count,
    )

    if check_array.shape != expected_shape:
        raise RuntimeError(
            f"{modality} NPY shape错误："
            f"{check_array.shape} != {expected_shape}"
        )

    if check_array.dtype != np.float32:
        raise RuntimeError(
            f"{modality} NPY dtype错误："
            f"{check_array.dtype}"
        )

    print(f"{modality.upper()} 转换完成。")
    print(f"输出：{output_path}")
    print(f"NaN/Inf 替换数量：{total_nonfinite}")
    print(
        "含明显负强度的光谱数："
        f"{int((negative_per_row > 0).sum())}"
    )
    print(
        f"全零光谱数：{int(zero_flags.sum())}"
    )
    print(
        "异常强度光谱数："
        f"{int(outlier_flags.sum())}"
    )

    return {
        "input_csv": csv_path,
        "output_npy": output_path,
        "axis_file": axis_path,
        "qc_metrics_file": qc_path,
        "shape": [
            sample_count,
            point_count,
        ],
        "dtype": "float32",
        "index_column_removed": layout[
            "index_column"
        ],
        "axis": {
            "source": layout["axis_source"],
            "points": point_count,
            "start": float(axis[0]),
            "end": float(axis[-1]),
            "monotonic": layout[
                "axis_monotonic"
            ],
            "duplicate_count": layout[
                "axis_duplicate_count"
            ],
            "step_min": layout[
                "axis_step_min"
            ],
            "step_median": layout[
                "axis_step_median"
            ],
            "step_max": layout[
                "axis_step_max"
            ],
        },
        "quality": {
            "nonfinite_values_replaced": (
                total_nonfinite
            ),
            "spectra_with_negative_values": int(
                (negative_per_row > 0).sum()
            ),
            "negative_value_count": (
                total_negative
            ),
            "zero_spectra": int(
                zero_flags.sum()
            ),
            "intensity_outlier_spectra": int(
                outlier_flags.sum()
            ),
            "outlier_model": outlier_model,
        },
        "intensity_percentiles": {
            "absmax_min": float(
                np.min(row_absmax)
            ),
            "absmax_p01": float(
                np.percentile(row_absmax, 1)
            ),
            "absmax_p50": float(
                np.percentile(row_absmax, 50)
            ),
            "absmax_p99": float(
                np.percentile(row_absmax, 99)
            ),
            "absmax_max": float(
                np.max(row_absmax)
            ),
        },
        "_flags": {
            "nonfinite": (
                nonfinite_per_row > 0
            ),
            "negative": (
                negative_per_row > 0
            ),
            "zero": zero_flags,
            "outlier": outlier_flags,
        },
    }


def canonicalize_and_scaffold(
    smiles: str,
    row_index: int,
) -> tuple[str, str, bool]:
    """生成 canonical SMILES 和 Bemis–Murcko scaffold。"""

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return (
            "",
            f"INVALID::{row_index}",
            False,
        )

    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )

    scaffold = (
        MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=False,
        )
    )

    # 纯链状分子会得到空 scaffold。
    # 这里用 canonical SMILES 回退，避免所有链状分子
    # 被错误分配到同一个数据集分区。
    if not scaffold:
        scaffold = f"ACYCLIC::{canonical}"

    return canonical, scaffold, True


def make_scaffold_split(
    scaffolds: list[str],
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """按 scaffold 分组，生成互不重叠的 train/val/test。"""

    test_fraction = (
        1.0
        - train_fraction
        - val_fraction
    )

    if min(
        train_fraction,
        val_fraction,
        test_fraction,
    ) <= 0:
        raise ValueError(
            "train、val、test 比例必须均大于 0。"
        )

    groups: dict[str, list[int]] = (
        defaultdict(list)
    )

    for index, scaffold in enumerate(scaffolds):
        groups[scaffold].append(index)

    rng = np.random.default_rng(seed)

    grouped_items = list(groups.items())

    # 同样大小的 scaffold group 随机打散，
    # 但整体结果由 seed 固定。
    rng.shuffle(grouped_items)

    # 大 group 优先分配。
    grouped_items.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    split_names = (
        "train",
        "val",
        "test",
    )

    targets = {
        "train": (
            len(scaffolds)
            * train_fraction
        ),
        "val": (
            len(scaffolds)
            * val_fraction
        ),
        "test": (
            len(scaffolds)
            * test_fraction
        ),
    }

    assigned: dict[str, list[int]] = {
        name: []
        for name in split_names
    }

    for _, indices in grouped_items:
        group_size = len(indices)

        # 选择加入该 group 后，
        # 相对目标填充比例最小的分区。
        selected_split = min(
            split_names,
            key=lambda name: (
                (
                    len(assigned[name])
                    + group_size
                )
                / targets[name],
                len(assigned[name]),
            ),
        )

        assigned[selected_split].extend(
            indices
        )

    arrays = {
        name: np.asarray(
            sorted(indices),
            dtype=np.int64,
        )
        for name, indices
        in assigned.items()
    }

    labels = np.empty(
        len(scaffolds),
        dtype="<U5",
    )

    for name, indices in arrays.items():
        labels[indices] = name

    return arrays, labels


def build_scaffold_split(
    manifest: pd.DataFrame,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    """生成 scaffold split 和新的 prepared manifest。"""

    print(f"\n{'=' * 70}")
    print("生成 Bemis–Murcko scaffold split")
    print(f"{'=' * 70}")

    canonical_smiles: list[str] = []
    scaffolds: list[str] = []
    valid_smiles: list[bool] = []

    for index, smiles in enumerate(
        manifest["smiles"].astype(str)
    ):
        canonical, scaffold, valid = (
            canonicalize_and_scaffold(
                smiles=smiles,
                row_index=index,
            )
        )

        canonical_smiles.append(canonical)
        scaffolds.append(scaffold)
        valid_smiles.append(valid)

        if (index + 1) % 10000 == 0:
            print(
                f"已处理 "
                f"{index + 1:,} / "
                f"{len(manifest):,}"
            )

    split_arrays, split_labels = (
        make_scaffold_split(
            scaffolds=scaffolds,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
    )

    split_path = (
        OUT_DIR
        / "scaffold_split_80_10_10.npz"
    )

    np.savez_compressed(
        split_path,
        train=split_arrays["train"],
        val=split_arrays["val"],
        test=split_arrays["test"],
    )

    prepared_manifest = manifest.copy()

    prepared_manifest[
        "canonical_smiles"
    ] = canonical_smiles

    prepared_manifest[
        "murcko_scaffold"
    ] = scaffolds

    prepared_manifest[
        "valid_smiles"
    ] = valid_smiles

    prepared_manifest[
        "split"
    ] = split_labels

    prepared_manifest_path = (
        OUT_DIR
        / "qm9s_manifest_prepared.csv"
    )

    prepared_manifest.to_csv(
        prepared_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    scaffold_sets = {
        name: set(
            prepared_manifest.loc[
                indices,
                "murcko_scaffold",
            ]
        )
        for name, indices
        in split_arrays.items()
    }

    overlap = {
        "train_val": len(
            scaffold_sets["train"]
            & scaffold_sets["val"]
        ),
        "train_test": len(
            scaffold_sets["train"]
            & scaffold_sets["test"]
        ),
        "val_test": len(
            scaffold_sets["val"]
            & scaffold_sets["test"]
        ),
    }

    counts = {
        name: int(len(indices))
        for name, indices
        in split_arrays.items()
    }

    print("Scaffold split完成：")
    print(f"train：{counts['train']:,}")
    print(f"val：  {counts['val']:,}")
    print(f"test： {counts['test']:,}")
    print(f"scaffold交集：{overlap}")

    return {
        "method": (
            "Bemis-Murcko scaffold split; "
            "acyclic molecules use canonical "
            "SMILES as fallback grouping key"
        ),
        "seed": seed,
        "split_file": split_path,
        "prepared_manifest": (
            prepared_manifest_path
        ),
        "invalid_smiles": int(
            (
                ~np.asarray(
                    valid_smiles,
                    dtype=bool,
                )
            ).sum()
        ),
        "unique_scaffolds": int(
            len(set(scaffolds))
        ),
        "counts": counts,
        "fractions": {
            name: (
                len(indices)
                / len(manifest)
            )
            for name, indices
            in split_arrays.items()
        },
        "scaffold_overlap": overlap,
    }


def write_qc_flags(
    manifest: pd.DataFrame,
    modality_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把所有异常样本索引写入一个 CSV。"""

    records: list[dict[str, Any]] = []

    for modality, result in modality_results.items():
        flags = result["_flags"]

        union = (
            flags["nonfinite"]
            | flags["negative"]
            | flags["zero"]
            | flags["outlier"]
        )

        for index in np.flatnonzero(union):
            row = manifest.iloc[int(index)]

            records.append(
                {
                    "row_index": int(
                        row["row_index"]
                    ),
                    "qm9_number": row[
                        "qm9_number"
                    ],
                    "smiles": row["smiles"],
                    "modality": modality,
                    "has_nonfinite_replaced": bool(
                        flags["nonfinite"][index]
                    ),
                    "has_negative_intensity": bool(
                        flags["negative"][index]
                    ),
                    "is_zero_spectrum": bool(
                        flags["zero"][index]
                    ),
                    "is_intensity_outlier": bool(
                        flags["outlier"][index]
                    ),
                }
            )

    output_path = (
        OUT_DIR
        / "qc_flagged_samples.csv"
    )

    columns = [
        "row_index",
        "qm9_number",
        "smiles",
        "modality",
        "has_nonfinite_replaced",
        "has_negative_intensity",
        "is_zero_spectrum",
        "is_intensity_outlier",
    ]

    pd.DataFrame(
        records,
        columns=columns,
    ).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "file": output_path,
        "flagged_records": len(records),
    }


def clear_outputs(overwrite: bool) -> None:
    """避免误把上次未完成的临时文件当作正式数据。"""

    expected_files = [
        OUT_DIR / "ir_float32.npy",
        OUT_DIR / "raman_float32.npy",
        OUT_DIR / "uvvis_float32.npy",
        OUT_DIR / "ir_float32.tmp.npy",
        OUT_DIR / "raman_float32.tmp.npy",
        OUT_DIR / "uvvis_float32.tmp.npy",
        OUT_DIR / "ir_axis.npy",
        OUT_DIR / "raman_axis.npy",
        OUT_DIR / "uvvis_axis.npy",
        OUT_DIR / "ir_qc_metrics.npz",
        OUT_DIR / "raman_qc_metrics.npz",
        OUT_DIR / "uvvis_qc_metrics.npz",
        OUT_DIR / "scaffold_split_80_10_10.npz",
        OUT_DIR / "qm9s_manifest_prepared.csv",
        OUT_DIR / "qc_flagged_samples.csv",
        OUT_DIR / "prepare_qm9s_report.json",
    ]

    existing = [
        path
        for path in expected_files
        if path.exists()
    ]

    if existing and not overwrite:
        names = "\n".join(
            f"  {path}"
            for path in existing
        )

        raise FileExistsError(
            "检测到已有输出文件：\n"
            f"{names}\n\n"
            "确认需要重建后，添加 --overwrite。"
        )

    if overwrite:
        for path in existing:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将 QM9S 的 IR、Raman、UV-Vis "
            "CSV 转为 float32 NPY，生成 "
            "scaffold split，并完成质量检查。"
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260801,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"缺少 manifest：{MANIFEST_PATH}"
        )

    for modality, path in MODALITIES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 {modality} CSV：{path}"
            )

    clear_outputs(
        overwrite=args.overwrite
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "row_index",
        "qm9_number",
        "smiles",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "manifest 缺少字段："
            f"{sorted(missing_columns)}"
        )

    manifest = (
        manifest
        .sort_values("row_index")
        .reset_index(drop=True)
    )

    expected_index = np.arange(
        len(manifest),
        dtype=np.int64,
    )

    actual_index = manifest[
        "row_index"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        actual_index,
        expected_index,
    ):
        raise RuntimeError(
            "manifest 的 row_index 不是 "
            "从0开始的连续整数。"
        )

    print(f"QM9S样本数：{len(manifest):,}")
    print(f"输出目录：{OUT_DIR}")
    print(
        "注意：本脚本不归一化、不删除异常谱。"
    )
    print(
        "NaN和Inf会替换为0，并记录在QC报告中。"
    )

    modality_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for modality, csv_path in MODALITIES.items():
        modality_results[modality] = (
            convert_modality(
                modality=modality,
                csv_path=csv_path,
                sample_count=len(manifest),
                chunk_size=args.chunk_size,
                zero_tolerance=(
                    args.zero_tolerance
                ),
                negative_tolerance=(
                    args.negative_tolerance
                ),
                outlier_threshold=(
                    args.outlier_threshold
                ),
            )
        )

    split_summary = build_scaffold_split(
        manifest=manifest,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    qc_summary = write_qc_flags(
        manifest=manifest,
        modality_results=modality_results,
    )

    # _flags 是内部数组，不写入 JSON。
    for result in modality_results.values():
        result.pop("_flags", None)

    report = {
        "source": "QM9S",
        "sample_count": len(manifest),
        "output_directory": OUT_DIR,
        "storage": {
            "dtype": "float32",
            "normalization": "none",
            "npy_memory_mapping_supported": True,
        },
        "cleaning_policy": {
            "nan_and_inf": (
                "replaced with 0 and logged"
            ),
            "negative_intensity": (
                "preserved and flagged"
            ),
            "zero_spectrum": (
                "preserved and flagged"
            ),
            "intensity_outlier": (
                "preserved and flagged"
            ),
        },
        "modalities": modality_results,
        "scaffold_split": split_summary,
        "quality_flags": qc_summary,
    }

    report_path = (
        OUT_DIR
        / "prepare_qm9s_report.json"
    )

    report_path.write_text(
        json.dumps(
            json_safe(report),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    print("QM9S数据准备全部完成")
    print(f"{'=' * 70}")

    for path in sorted(OUT_DIR.iterdir()):
        if path.is_file():
            size_mb = (
                path.stat().st_size
                / 1024**2
            )

            print(
                f"{path.name:38s}"
                f"{size_mb:12.2f} MB"
            )

    print(f"\n总报告：{report_path}")


if __name__ == "__main__":
    main()
