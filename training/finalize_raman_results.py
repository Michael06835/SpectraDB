"""DEPRECATED — 本文件不再包含任何汇总实现，请勿运行。

原 Raman-v1 汇总构建器已逐字节迁入：

    archive/2026-11-16/finalize_raman_results.py

该历史脚本仅作参考保留，**不得重跑**。原因：

1. 它把输出目录固定为 `runs/Raman-v1-FINAL`，并以 `exist_ok=True` 建目录；
2. 它以 10 处 `open(..., "w")` 覆写该目录内已冻结的
   `raman_seed_summary.*` / `raman_per_label_3seed_summary.csv` /
   `raman_ablation_summary.csv` / `raman_final_summary.json` /
   `RAMAN_FINAL_SUMMARY.md` 及 4 张图；
3. 它汇总的是 v1 历史口径（BCE 3 seed + WBCE seed 42 + 旧校准），
   不是 6 模型 × 2 BN × 2 阈值 = 24 条件矩阵。

Raman 的 24 条件汇总请使用（Phase 6 之后）：

    python training/summarize_raman_ablation.py

本文件不读取数据、不初始化 CUDA、不汇总、不写入任何文件；
直接运行时立即以非零状态退出。
"""
from __future__ import annotations

import sys

_EXIT_CODE = 2

_MESSAGE = """\
[deprecated] training/finalize_raman_results.py is disabled.

The original Raman-v1 summary builder now lives at:
    archive/2026-11-16/finalize_raman_results.py
That copy is kept for historical reference only -- do NOT re-run it: it rewrites
runs/Raman-v1-FINAL/ in place (10 files, mode "w") and summarizes the legacy v1
layout, not the 24-condition matrix.

For the 24-condition ablation use:
  python training/summarize_raman_ablation.py

This entry point loaded no data, started no GPU work, summarized nothing and
wrote no files.
"""


def main() -> int:
    """Print the deprecation notice and fail fast; performs no other work."""
    print(_MESSAGE, file=sys.stderr)
    return _EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
