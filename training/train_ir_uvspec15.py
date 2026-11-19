"""B2 IR -> UVSPEC15, Seed 42 (FULL)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.line_b.b2_train import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            "ir",
            42,
            "EXP-IR-003-UVSPEC15-FULL",
            entrypoint=__file__,
        )
    )
