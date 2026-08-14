from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
DATA = ROOT / "processed" / "qm9s" / "prepared"

MANIFEST_PATH = DATA / "qm9s_manifest_with_modality_validity.csv"
OUTPUT_PATH = DATA / "morgan_fp_2048.npy"
VALID_MASK_PATH = DATA / "morgan_fp_valid_mask.npy"
REPORT_PATH = DATA / "morgan_fp_2048_report.json"
INVALID_PATH = DATA / "morgan_fp_invalid_smiles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"缺少 manifest：{MANIFEST_PATH}")

    if OUTPUT_PATH.exists() and not args.overwrite:
        array = np.load(OUTPUT_PATH, mmap_mode="r")
        print(f"指纹文件已存在：{OUTPUT_PATH}")
        print(f"shape={array.shape}, dtype={array.dtype}")
        print("如需重新生成，请加 --overwrite")
        return

    manifest = pd.read_csv(MANIFEST_PATH)

    if "canonical_smiles" in manifest.columns:
        smiles_column = "canonical_smiles"
    elif "smiles" in manifest.columns:
        smiles_column = "smiles"
    else:
        raise KeyError("manifest 中没有 canonical_smiles 或 smiles 列。")

    sample_count = len(manifest)

    print(f"样本数：{sample_count:,}")
    print(f"SMILES列：{smiles_column}")
    print(f"Morgan：radius={args.radius}, nBits={args.nbits}")

    fingerprints = np.lib.format.open_memmap(
        OUTPUT_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, args.nbits),
    )

    valid_mask = np.ones(sample_count, dtype=bool)
    invalid_rows = []

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.nbits,
    )

    for row_index, smiles in enumerate(manifest[smiles_column].astype(str)):
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            valid_mask[row_index] = False
            fingerprints[row_index] = 0

            invalid_rows.append(
                {
                    "row_index": row_index,
                    "smiles": smiles,
                }
            )
        else:
            fingerprint = generator.GetFingerprint(molecule)
            array = np.zeros(args.nbits, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            fingerprints[row_index] = array

        if (row_index + 1) % 5000 == 0:
            print(
                f"已处理 {row_index + 1:,}/{sample_count:,}"
            )

    fingerprints.flush()
    np.save(VALID_MASK_PATH, valid_mask)

    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=["row_index", "smiles"],
    )
    invalid_df.to_csv(
        INVALID_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "manifest": str(MANIFEST_PATH),
        "output": str(OUTPUT_PATH),
        "valid_mask": str(VALID_MASK_PATH),
        "sample_count": sample_count,
        "valid_count": int(valid_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
        "radius": args.radius,
        "nbits": args.nbits,
        "dtype": "uint8",
        "shape": [sample_count, args.nbits],
        "smiles_column": smiles_column,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("Morgan 指纹生成完成")
    print("=" * 70)
    print(f"输出：{OUTPUT_PATH}")
    print(f"shape=({sample_count}, {args.nbits})")
    print(f"有效 SMILES：{int(valid_mask.sum()):,}")
    print(f"无效 SMILES：{int((~valid_mask).sum()):,}")

    if invalid_rows:
        raise RuntimeError(
            "检测到无效 SMILES，请先检查 "
            f"{INVALID_PATH}"
        )


if __name__ == "__main__":
    main()
