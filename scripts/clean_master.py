import pandas as pd
from pathlib import Path

RAW_PATH = Path("master/compound_master_100000_random_raw.csv")
CLEAN_PATH = Path("master/compound_master_100000_random_clean.csv")

df = pd.read_csv(RAW_PATH)

df["isotope_atom_count"] = pd.to_numeric(df["isotope_atom_count"], errors="coerce")
df["covalent_unit_count"] = pd.to_numeric(df["covalent_unit_count"], errors="coerce")

clean = df[
    (df["isotope_atom_count"] == 0)
    & (df["covalent_unit_count"] == 1)
].copy()

clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

print(f"Raw: {len(df)}")
print(f"Clean: {len(clean)}")
print(f"Removed: {len(df) - len(clean)}")
print(f"Saved to: {CLEAN_PATH}")