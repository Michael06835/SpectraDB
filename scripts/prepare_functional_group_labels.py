from pathlib import Path
import gzip
import json
import csv
import numpy as np

# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "processed" / "qm9s" / "structures" / "qm9s_functional_groups.jsonl.gz"
OUT_DIR = ROOT / "processed" / "qm9s" / "prepared"

OUT_LABELS = OUT_DIR / "functional_group_labels.npy"
OUT_NAMES = OUT_DIR / "functional_group_label_names.json"
OUT_COUNTS = OUT_DIR / "functional_group_label_counts.csv"

IR_PATH = OUT_DIR / "ir_float32.npy"

EXPECTED_N = 129817

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_group_names(obj):
    """
    Extract unique functional-group names from one molecule.
    Each group is binary at molecule level:
    present = 1, absent = 0.
    """
    result = set()

    groups = obj.get("functional_groups", [])

    if isinstance(groups, list):
        for g in groups:
            if isinstance(g, dict):
                name = g.get("group_name") or g.get("name")
                if name:
                    result.add(str(name))
            elif isinstance(g, str):
                result.add(g)

    elif isinstance(groups, dict):
        # Compatibility fallback
        if "group_name" in groups:
            result.add(str(groups["group_name"]))
        else:
            for name, value in groups.items():
                if value:
                    result.add(str(name))

    return result


# ============================================================
# Pass 1:
# discover all existing functional-group categories
# and verify row_index integrity
# ============================================================

print("Reading:", SRC)

label_names_set = set()
seen = np.zeros(EXPECTED_N, dtype=bool)
n_records = 0

with gzip.open(SRC, "rt", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        if not line.strip():
            continue

        obj = json.loads(line)

        if "row_index" not in obj:
            raise RuntimeError(f"Missing row_index at line {line_no}")

        idx = int(obj["row_index"])

        if idx < 0 or idx >= EXPECTED_N:
            raise RuntimeError(
                f"row_index out of range at line {line_no}: {idx}"
            )

        if seen[idx]:
            raise RuntimeError(
                f"Duplicate row_index detected: {idx}"
            )

        seen[idx] = True
        n_records += 1

        label_names_set.update(extract_group_names(obj))


if n_records != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} records, but found {n_records}"
    )

missing = np.where(~seen)[0]

if len(missing) != 0:
    raise RuntimeError(
        f"Missing row_index values: {missing[:20].tolist()} "
        f"(total={len(missing)})"
    )


# Fixed deterministic order.
# This does NOT alter the existing categories; it only assigns columns.
label_names = sorted(label_names_set)

print()
print("Detected functional-group labels:")
for i, name in enumerate(label_names):
    print(f"{i:2d}: {name}")

print()
print("Number of labels:", len(label_names))


# ============================================================
# Pass 2:
# build molecule x functional-group binary matrix
# ============================================================

name_to_col = {
    name: i
    for i, name in enumerate(label_names)
}

Y = np.zeros(
    (EXPECTED_N, len(label_names)),
    dtype=np.uint8
)

with gzip.open(SRC, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        obj = json.loads(line)
        idx = int(obj["row_index"])

        groups = extract_group_names(obj)

        for name in groups:
            Y[idx, name_to_col[name]] = 1


# ============================================================
# Validate against IR matrix
# ============================================================

ir = np.load(IR_PATH, mmap_mode="r")

if ir.shape[0] != Y.shape[0]:
    raise RuntimeError(
        f"IR/label row mismatch: IR={ir.shape[0]}, labels={Y.shape[0]}"
    )


# ============================================================
# Save
# ============================================================

np.save(OUT_LABELS, Y)

with open(OUT_NAMES, "w", encoding="utf-8") as f:
    json.dump(
        label_names,
        f,
        ensure_ascii=False,
        indent=2
    )

positive_counts = Y.sum(axis=0)

with open(OUT_COUNTS, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "label_index",
        "functional_group",
        "positive_molecules",
        "positive_fraction"
    ])

    for i, name in enumerate(label_names):
        count = int(positive_counts[i])

        writer.writerow([
            i,
            name,
            count,
            count / EXPECTED_N
        ])


# ============================================================
# Final report
# ============================================================

print()
print("=" * 60)
print("Functional-group label matrix created successfully")
print("=" * 60)

print("labels shape :", Y.shape)
print("labels dtype :", Y.dtype)
print("IR shape     :", ir.shape)

print()
print("Positive molecules per label:")

order = np.argsort(-positive_counts)

for i in order:
    count = int(positive_counts[i])
    frac = count / EXPECTED_N

    print(
        f"{label_names[i]:20s} "
        f"{count:8d} "
        f"{frac:8.3%}"
    )

print()
print("Example molecules:")

for idx in range(min(5, EXPECTED_N)):
    active = [
        label_names[j]
        for j in np.where(Y[idx] == 1)[0]
    ]
    print(f"row {idx}: {active}")

print()
print("Saved:")
print(" ", OUT_LABELS)
print(" ", OUT_NAMES)
print(" ", OUT_COUNTS)
