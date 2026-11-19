"""B2 Raman -> UVSPEC15, Seed 123."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.line_b.b2_train import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            "raman",
            123,
            "EXP-RAMAN-003-UVSPEC15-SEED123",
            entrypoint=__file__,
        )
    )
