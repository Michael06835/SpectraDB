"""DEPRECATED — 本文件不再包含任何校准实现，请勿运行。

原 Raman 单模型阈值校准实现已逐字节迁入：

    archive/2026-11-16/calibrate_raman_thresholds.py

该历史脚本仅作参考保留，**不得重跑**。原因：

1. 它把 `RUN_DIR` 硬编码为 `runs/EXP-RAMAN-001-FULL`，无法用于其余五个模型；
2. 它以 `open(..., "w")` 直接覆写该目录内的 4 个既有文件
   （optimal_thresholds.json / threshold_comparison.csv /
    test_per_label_metrics_calibrated.csv / threshold_calibration_summary.json）；
3. 它输出的是旧 schema（扁平阈值 dict、无 checkpoint_sha256、
   无 validation_threshold_comparison.csv、无 test_per_label_metrics_default.csv），
   与 24 条件汇总器不兼容。

请改用参数化脚本：

    python training/calibrate_raman_original_thresholds.py     --loss {bce,wbce} --seed {42,123,2026}
    python training/calibrate_raman_recalibrated_thresholds.py --loss {bce,wbce} --seed {42,123,2026}

两者均写入独立目录（runs/RAMAN-THRESHOLD-CALIBRATION/、
runs/RAMAN-BN-THRESHOLD-CALIBRATION/），并以"目录不存在"为写入前提。

本文件不读取数据、不初始化 CUDA、不拟合阈值、不写入任何文件；
直接运行时立即以非零状态退出。
"""
from __future__ import annotations

import sys

_EXIT_CODE = 2

_MESSAGE = """\
[deprecated] training/calibrate_raman_thresholds.py is disabled.

The original single-model calibration now lives at:
    archive/2026-11-16/calibrate_raman_thresholds.py
That copy is kept for historical reference only -- do NOT re-run it: it
hard-codes RUN_DIR=runs/EXP-RAMAN-001-FULL, overwrites 4 existing artifacts
there with mode "w", and emits the legacy schema (flat threshold dict, no
checkpoint_sha256, no validation per-label comparison).

Use one of the parameterized replacements instead:
  python training/calibrate_raman_original_thresholds.py     --loss {bce,wbce} --seed {42,123,2026}
  python training/calibrate_raman_recalibrated_thresholds.py --loss {bce,wbce} --seed {42,123,2026}

This entry point loaded no data, started no GPU work, fitted no thresholds and
wrote no files.
"""


def main() -> int:
    """Print the deprecation notice and fail fast; performs no other work."""
    print(_MESSAGE, file=sys.stderr)
    return _EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
