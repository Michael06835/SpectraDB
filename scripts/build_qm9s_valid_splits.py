from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
DATA = ROOT / "processed" / "qm9s" / "prepared"

BASE_SPLIT_PATH = DATA / "scaffold_split_80_10_10.npz"
MANIFEST_PATH = DATA / "qm9s_manifest_prepared.csv"

QC_PATHS = {
    "ir": DATA / "ir_qc_metrics.npz",
    "raman": DATA / "raman_qc_metrics.npz",
    "uvvis": DATA / "uvvis_qc_metrics.npz",
}

OUTPUT_SPLITS = {
    "ir": DATA / "ir_scaffold_split_valid.npz",
    "raman": DATA / "raman_scaffold_split_valid.npz",
    "uvvis": DATA / "uvvis_scaffold_split_valid.npz",
}

MASK_PATH = DATA / "modality_valid_masks.npz"

MULTIMODAL_COMPLETE_SPLIT_PATH = (
    DATA / "multimodal_complete_scaffold_split.npz"
)

OUTPUT_MANIFEST_PATH = (
    DATA / "qm9s_manifest_with_modality_validity.csv"
)

REPORT_PATH = DATA / "modality_validity_report.json"


def load_qc(path: Path) -> dict[str, np.ndarray]:
    qc = np.load(path)

    zero = qc["zero_flags"].astype(bool)
    nonfinite = qc["nonfinite_per_row"] > 0
    negative = qc["negative_per_row"] > 0
    outlier = qc["intensity_outlier_flags"].astype(bool)

    # 当前只过滤全零谱和含NaN/Inf的谱。
    # 负值和强度离群谱暂时保留。
    invalid = zero | nonfinite
    valid = ~invalid

    return {
        "valid": valid,
        "invalid": invalid,
        "zero": zero,
        "nonfinite": nonfinite,
        "negative": negative,
        "outlier": outlier,
    }


def filter_split(
    base_splits: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: indices[valid_mask[indices]]
        for name, indices in base_splits.items()
    }


def counts(
    split: dict[str, np.ndarray],
) -> dict[str, int]:
    return {
        name: int(len(indices))
        for name, indices in split.items()
    }


def main() -> None:
    required_files = [
        BASE_SPLIT_PATH,
        MANIFEST_PATH,
        *QC_PATHS.values(),
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"缺少文件：{path}")

    manifest = pd.read_csv(MANIFEST_PATH)
    sample_count = len(manifest)

    base_npz = np.load(BASE_SPLIT_PATH)

    base_splits = {
        name: base_npz[name].astype(np.int64)
        for name in ("train", "val", "test")
    }

    base_total = sum(
        len(indices)
        for indices in base_splits.values()
    )

    if base_total != sample_count:
        raise RuntimeError(
            f"基础split共{base_total}条，"
            f"manifest共{sample_count}条，数量不一致。"
        )

    qc_results = {
        modality: load_qc(path)
        for modality, path in QC_PATHS.items()
    }

    for modality, qc in qc_results.items():
        if len(qc["valid"]) != sample_count:
            raise RuntimeError(
                f"{modality} QC长度异常："
                f"{len(qc['valid'])} != {sample_count}"
            )

    valid_ir = qc_results["ir"]["valid"]
    valid_raman = qc_results["raman"]["valid"]
    valid_uvvis = qc_results["uvvis"]["valid"]

    valid_complete = (
        valid_ir
        & valid_raman
        & valid_uvvis
    )

    # 保存每个样本的模态有效性，不修改任何光谱矩阵。
    np.savez_compressed(
        MASK_PATH,
        valid_ir=valid_ir,
        valid_raman=valid_raman,
        valid_uvvis=valid_uvvis,
        valid_multimodal_complete=valid_complete,
    )

    modality_reports = {}

    for modality, output_path in OUTPUT_SPLITS.items():
        qc = qc_results[modality]

        filtered = filter_split(
            base_splits,
            qc["valid"],
        )

        np.savez_compressed(
            output_path,
            train=filtered["train"],
            val=filtered["val"],
            test=filtered["test"],
        )

        modality_reports[modality] = {
            "split_file": str(output_path),
            "valid_samples": int(qc["valid"].sum()),
            "invalid_samples": int(qc["invalid"].sum()),
            "zero_spectra_filtered": int(qc["zero"].sum()),
            "nonfinite_spectra_filtered": int(
                qc["nonfinite"].sum()
            ),
            "negative_spectra_preserved": int(
                qc["negative"].sum()
            ),
            "intensity_outliers_preserved": int(
                qc["outlier"].sum()
            ),
            "split_counts": counts(filtered),
        }

    # 三个模态都有效的完整配对集合。
    multimodal_complete_split = filter_split(
        base_splits,
        valid_complete,
    )

    np.savez_compressed(
        MULTIMODAL_COMPLETE_SPLIT_PATH,
        train=multimodal_complete_split["train"],
        val=multimodal_complete_split["val"],
        test=multimodal_complete_split["test"],
    )

    # 给manifest增加有效性字段，但不删除任何行。
    output_manifest = manifest.copy()

    output_manifest["valid_ir"] = valid_ir
    output_manifest["valid_raman"] = valid_raman
    output_manifest["valid_uvvis"] = valid_uvvis
    output_manifest[
        "valid_multimodal_complete"
    ] = valid_complete

    output_manifest["available_modality_count"] = (
        valid_ir.astype(np.uint8)
        + valid_raman.astype(np.uint8)
        + valid_uvvis.astype(np.uint8)
    )

    output_manifest.to_csv(
        OUTPUT_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "total_samples": sample_count,
        "policy": {
            "original_rows_deleted": False,
            "filter_zero_spectra": True,
            "filter_nonfinite_spectra": True,
            "filter_negative_values": False,
            "filter_intensity_outliers": False,
        },
        "base_scaffold_split": counts(base_splits),
        "modalities": modality_reports,
        "multimodal_complete": {
            "split_file": str(
                MULTIMODAL_COMPLETE_SPLIT_PATH
            ),
            "valid_samples": int(valid_complete.sum()),
            "excluded_samples": int(
                (~valid_complete).sum()
            ),
            "split_counts": counts(
                multimodal_complete_split
            ),
        },
        "invalid_overlap": {
            "ir_and_raman": int(
                (
                    ~valid_ir
                    & ~valid_raman
                ).sum()
            ),
            "ir_and_uvvis": int(
                (
                    ~valid_ir
                    & ~valid_uvvis
                ).sum()
            ),
            "raman_and_uvvis": int(
                (
                    ~valid_raman
                    & ~valid_uvvis
                ).sum()
            ),
            "all_three": int(
                (
                    ~valid_ir
                    & ~valid_raman
                    & ~valid_uvvis
                ).sum()
            ),
        },
        "mask_file": str(MASK_PATH),
        "manifest_file": str(
            OUTPUT_MANIFEST_PATH
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("模态有效样本索引生成完成")
    print("=" * 72)

    print(f"全局样本数：{sample_count:,}")

    for modality in ("ir", "raman", "uvvis"):
        item = modality_reports[modality]

        print(
            f"{modality.upper():7s}"
            f"有效={item['valid_samples']:,}，"
            f"过滤={item['invalid_samples']:,}，"
            f"split={item['split_counts']}"
        )

    print(
        "三模态全部有效："
        f"{int(valid_complete.sum()):,}"
    )

    print(f"\n报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
