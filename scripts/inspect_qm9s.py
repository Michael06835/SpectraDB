from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
QM9S = ROOT / "raw" / "qm9s"
CSV_DIR = QM9S / "qm9s_csv"
REPORT = ROOT / "metadata" / "qm9s_inspection.txt"

REPORT.parent.mkdir(parents=True, exist_ok=True)

spectra_files = [
    QM9S / "ir_boraden.csv",
    QM9S / "raman_boraden.csv",
    QM9S / "uv_boraden.csv",
]

def count_csv_rows(path: Path) -> int:
    """只读取第一列，分块统计行数，避免把大文件全部载入内存。"""
    total = 0
    try:
        for chunk in pd.read_csv(path, usecols=[0], chunksize=20000):
            total += len(chunk)
        return total
    except Exception:
        return -1

def inspect_csv(path: Path) -> list[str]:
    lines = []
    lines.append(f"文件：{path}")
    lines.append(f"大小：{path.stat().st_size / 1024**3:.3f} GB")

    try:
        sample = pd.read_csv(path, nrows=3)
        lines.append(f"列数：{len(sample.columns)}")
        lines.append(f"前20个列名：{sample.columns[:20].tolist()}")
        lines.append(f"样例形状：{sample.shape}")
        lines.append("前三行前10列：")
        lines.append(sample.iloc[:, :10].to_string(index=False))
    except Exception as exc:
        lines.append(f"读取失败：{type(exc).__name__}: {exc}")

    rows = count_csv_rows(path)
    lines.append(f"数据行数：{rows if rows >= 0 else '统计失败'}")
    return lines

output = []

output.append("========== QM9S 数据检查 ==========\n")

output.append("========== 顶层文件 ==========")
for path in sorted(QM9S.iterdir()):
    if path.is_file():
        output.append(
            f"{path.name}\t{path.stat().st_size / 1024**3:.3f} GB"
        )
    else:
        output.append(f"{path.name}\t<DIR>")

output.append("\n========== 展宽光谱 ==========")
spectra_row_counts = {}

for path in spectra_files:
    output.append("")
    if not path.exists():
        output.append(f"缺少文件：{path}")
        continue

    details = inspect_csv(path)
    output.extend(details)
    spectra_row_counts[path.name] = count_csv_rows(path)

output.append("\n========== qm9s_csv 内容 ==========")

if not CSV_DIR.exists():
    output.append(f"没有找到目录：{CSV_DIR}")
else:
    files = sorted(
        [p for p in CSV_DIR.rglob("*") if p.is_file()],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    output.append(f"文件总数：{len(files)}")

    for path in files:
        output.append(
            f"{path.relative_to(CSV_DIR)}\t"
            f"{path.stat().st_size / 1024**2:.2f} MB"
        )

    csv_files = [p for p in files if p.suffix.lower() == ".csv"]

    output.append("\n========== qm9s_csv 表格详情 ==========")

    for path in csv_files:
        output.append("")
        output.extend(inspect_csv(path))

output.append("\n========== 三模态行数比较 ==========")
for name, rows in spectra_row_counts.items():
    output.append(f"{name}: {rows}")

valid_counts = [v for v in spectra_row_counts.values() if v >= 0]
if len(valid_counts) == 3 and len(set(valid_counts)) == 1:
    output.append("结果：IR、Raman、UV-Vis 三个文件行数一致。")
else:
    output.append("结果：三个光谱文件行数不一致或统计失败，需要进一步检查。")

REPORT.write_text("\n".join(output), encoding="utf-8")

print(f"检查完成：{REPORT}")
print("\n".join(output[-10:]))
