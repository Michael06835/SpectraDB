from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Projects\曦源计划\数据\SpectraDB")
DATA_DIR = ROOT / "raw" / "raman_experimental"

SPECTRA_PATH = DATA_DIR / "raman_spectra_api_compounds.csv"
INFO_PATH = DATA_DIR / "API_Product_Information.xlsx"

REPORT_PATH = ROOT / "metadata" / "api_raman_inspection.txt"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

lines: list[str] = []

lines.append("Experimental API Raman Dataset Inspection")
lines.append("=" * 70)
lines.append(f"Data directory: {DATA_DIR}")
lines.append("")

for path in (SPECTRA_PATH, INFO_PATH):
    lines.append(
        f"{path.name}: "
        f"{path.stat().st_size / 1024**2:.2f} MB"
        if path.exists()
        else f"MISSING: {path}"
    )

lines.append("")
lines.append("=" * 70)
lines.append("RAMAN CSV")
lines.append("=" * 70)

if not SPECTRA_PATH.exists():
    lines.append(f"Missing file: {SPECTRA_PATH}")
else:
    try:
        sample = pd.read_csv(SPECTRA_PATH, nrows=5)
        separator_mode = "comma"
    except Exception:
        sample = pd.read_csv(
            SPECTRA_PATH,
            nrows=5,
            sep=None,
            engine="python",
        )
        separator_mode = "auto-detected"

    lines.append(f"Separator mode: {separator_mode}")
    lines.append(f"Number of columns: {len(sample.columns)}")
    lines.append(f"First 20 columns: {sample.columns[:20].tolist()}")
    lines.append(f"Last 20 columns: {sample.columns[-20:].tolist()}")
    lines.append("")
    lines.append("First 5 rows, first 12 columns:")
    lines.append(sample.iloc[:, :12].to_string(index=False))

    first_col = sample.columns[0]
    row_count = 0

    for chunk in pd.read_csv(
        SPECTRA_PATH,
        usecols=[first_col],
        chunksize=500,
    ):
        row_count += len(chunk)

    lines.append("")
    lines.append(f"CSV data rows: {row_count}")
    lines.append(f"First column: {first_col!r}")

lines.append("")
lines.append("=" * 70)
lines.append("PRODUCT INFORMATION XLSX")
lines.append("=" * 70)

if not INFO_PATH.exists():
    lines.append(f"Missing file: {INFO_PATH}")
else:
    workbook = pd.ExcelFile(INFO_PATH)
    lines.append(f"Sheet names: {workbook.sheet_names}")

    for sheet_name in workbook.sheet_names:
        lines.append("")
        lines.append(f"[Sheet: {sheet_name}]")

        frame = pd.read_excel(
            INFO_PATH,
            sheet_name=sheet_name,
            nrows=10,
        )

        lines.append(f"Columns: {frame.columns.tolist()}")
        lines.append(f"Preview shape: {frame.shape}")
        lines.append(frame.head(10).to_string(index=False))

REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(f"检查完成：{REPORT_PATH}")
print("\n".join(lines[-25:]))
