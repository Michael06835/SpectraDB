from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
RAW = ROOT / "raw" / "qm9s"
OUT_DIR = ROOT / "processed" / "qm9s"

PT_PATH = RAW / "qm9s.pt"

SPECTRA = {
    "ir": RAW / "ir_boraden.csv",
    "raman": RAW / "raman_boraden.csv",
    "uvvis": RAW / "uv_boraden.csv",
}

MANIFEST_PATH = OUT_DIR / "qm9s_manifest.csv"
REPORT_PATH = OUT_DIR / "qm9s_manifest_report.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_spectrum_csv(path: Path) -> dict:
    """
    检查大型光谱 CSV：
    1. 获取第一列名称；
    2. 分块统计行数；
    3. 检查第一列是否连续递增。
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少光谱文件：{path}")

    first_column = pd.read_csv(path, nrows=0).columns[0]

    total_rows = 0
    start_value = None
    sequential = True

    for chunk in pd.read_csv(
        path,
        usecols=[first_column],
        chunksize=20000,
    ):
        values = pd.to_numeric(
            chunk.iloc[:, 0],
            errors="coerce",
        )

        if values.isna().any():
            sequential = False
            total_rows += len(chunk)
            continue

        values_np = values.astype(np.int64).to_numpy()

        if start_value is None and len(values_np) > 0:
            start_value = int(values_np[0])

        if start_value is not None:
            expected = np.arange(
                start_value + total_rows,
                start_value + total_rows + len(values_np),
                dtype=np.int64,
            )

            if not np.array_equal(values_np, expected):
                sequential = False

        total_rows += len(chunk)

    return {
        "path": str(path),
        "first_column": first_column,
        "rows": total_rows,
        "index_start": start_value,
        "index_sequential": sequential,
    }


def to_python_scalar(value):
    """将 Tensor、NumPy标量或普通值转换为适合写入CSV的值。"""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return str(value.detach().cpu().tolist())

    if isinstance(value, np.generic):
        return value.item()

    return value


print("正在检查三个光谱文件……")

spectra_info = {
    name: inspect_spectrum_csv(path)
    for name, path in SPECTRA.items()
}

for name, info in spectra_info.items():
    print(
        f"{name}: rows={info['rows']}, "
        f"first_column={info['first_column']!r}, "
        f"index_start={info['index_start']}, "
        f"sequential={info['index_sequential']}"
    )

print("\n正在载入 qm9s.pt，文件较大，停顿一段时间属于正常现象……")

dataset = torch.load(
    PT_PATH,
    map_location="cpu",
    weights_only=False,
)

if not isinstance(dataset, list):
    raise TypeError(
        f"qm9s.pt 顶层对象不是 list，而是 {type(dataset)}"
    )

sample_count = len(dataset)
print(f"qm9s.pt 样本数：{sample_count}")

for name, info in spectra_info.items():
    if info["rows"] != sample_count:
        raise ValueError(
            f"{name} 行数 {info['rows']} 与 "
            f"qm9s.pt 样本数 {sample_count} 不一致"
        )

print("\n开始生成 manifest……")

seen_numbers = set()
duplicate_numbers = 0
missing_smiles = 0

with MANIFEST_PATH.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "row_index",
            "qm9_number",
            "smiles",
            "num_atoms",
            "ir_csv_row",
            "raman_csv_row",
            "uvvis_csv_row",
            "source",
            "data_domain",
        ],
    )

    writer.writeheader()

    for row_index, sample in enumerate(dataset):
        number = to_python_scalar(
            getattr(sample, "number", "")
        )

        smiles = getattr(sample, "smile", "")
        if smiles is None:
            smiles = ""
        smiles = str(smiles)

        if not smiles:
            missing_smiles += 1

        if number in seen_numbers:
            duplicate_numbers += 1
        else:
            seen_numbers.add(number)

        z = getattr(sample, "z", None)
        num_atoms = int(z.numel()) if torch.is_tensor(z) else ""

        writer.writerow(
            {
                "row_index": row_index,
                "qm9_number": number,
                "smiles": smiles,
                "num_atoms": num_atoms,
                "ir_csv_row": row_index,
                "raman_csv_row": row_index,
                "uvvis_csv_row": row_index,
                "source": "QM9S",
                "data_domain": "calculated",
            }
        )

        if (row_index + 1) % 10000 == 0:
            print(
                f"已写入 {row_index + 1:,} / "
                f"{sample_count:,}"
            )

report_lines = [
    "QM9S Manifest Build Report",
    "==========================",
    f"qm9s.pt samples: {sample_count}",
    f"manifest rows: {sample_count}",
    f"missing smiles: {missing_smiles}",
    f"duplicate qm9_number: {duplicate_numbers}",
    "",
]

for name, info in spectra_info.items():
    report_lines.extend(
        [
            f"[{name}]",
            f"path: {info['path']}",
            f"rows: {info['rows']}",
            f"first column: {info['first_column']}",
            f"index start: {info['index_start']}",
            f"index sequential: {info['index_sequential']}",
            "",
        ]
    )

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\n处理完成。")
print(f"Manifest：{MANIFEST_PATH}")
print(f"报告：{REPORT_PATH}")

print("\n前5个样本：")
for index in range(min(5, sample_count)):
    sample = dataset[index]
    print(
        index,
        to_python_scalar(getattr(sample, "number", "")),
        getattr(sample, "smile", ""),
    )
