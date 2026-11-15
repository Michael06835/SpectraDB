"""DEPRECATED — 本文件不再包含任何校准实现，请勿运行。

原 IR 阈值校准实现已完整迁入：

    archive/2026-11-15/calibrate_ir_thresholds.py

该历史脚本仅作参考保留，**不得重跑**。原因：

1. 它使用 `open(..., "w")` 写盘，会静默覆盖既有校准产物；
2. 它硬编码 BCE Seed 42 的 `RUN_DIR`，无法用于其余五个模型；
3. 它使用 `batch_size=1024 / num_workers=0` 推理，与统一的 512/4 口径不同，
   会产生约 2e-5 级的数值差异。

请改用两个参数化脚本：

    python training/calibrate_ir_original_thresholds.py     --loss {bce,wbce} --seed {42,123,2026}
    python training/calibrate_ir_recalibrated_thresholds.py --loss {bce,wbce} --seed {42,123,2026}

本文件不读取任何数据、不初始化 CUDA、不拟合阈值、不写入任何文件；
直接运行时立即以非零状态退出。
"""
from __future__ import annotations

import sys

_EXIT_CODE = 2

_MESSAGE = """\
[deprecated] training/calibrate_ir_thresholds.py is disabled.

The original implementation now lives at:
    archive/2026-11-15/calibrate_ir_thresholds.py
That copy is kept for historical reference only -- do NOT re-run it: it writes
results with mode "w" (silently overwriting existing files), hard-codes the BCE
Seed 42 run directory, and uses batch_size=1024 / num_workers=0 inference.

Use one of the parameterized replacements instead:
  python training/calibrate_ir_original_thresholds.py     --loss {bce,wbce} --seed {42,123,2026}
  python training/calibrate_ir_recalibrated_thresholds.py --loss {bce,wbce} --seed {42,123,2026}

This entry point loaded no data, started no GPU work, fitted no thresholds and
wrote no files. See docs/experiments/ir_repro_protocol.md section 7 and
docs/experiments/IR_v1_experimental_strategy.md for details.
"""


def main() -> int:
    """Print the deprecation notice and fail fast; performs no other work."""
    print(_MESSAGE, file=sys.stderr)
    return _EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
