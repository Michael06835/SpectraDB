# UVSPEC15 消融实验与原始文件对应关系（24 条件）

用途：将 `runs/UVSPEC15-ABLATION-SUMMARY/` 中的 24 个评估条件逐一映射回原始实验文件与 checkpoint，供审计与论文引用。**所有数值以原始 JSON/CSV 为唯一来源**；本文件不重算、不修改任何实验产物。

## 1. 实验矩阵

- 6 次独立训练 = 损失（普通 BCE / Weighted BCE）× seed（42/123/2026）；
- 每模型 2 种 BN 推理统计量：Original（原始 checkpoint）/ Recalibrated（train-only 重估 running stats）；
- 每模型 2 种阈值：0.5（默认）/ Validation-F1（Validation 上逐标签 F1 最优）；
- 合计每 split 24 条件（Test 24、Validation 24），逐标签结果 720 行（15 标签 × 24 条件 × 2 splits）。

## 2. 六个训练 run

| run_name | 损失 | seed | 训练产物目录 |
|---|---|---|---|
| EXP-UVVIS-003-UVSPEC15-FULL | BCE | 42 | `runs/EXP-UVVIS-003-UVSPEC15-FULL/` |
| EXP-UVVIS-003-UVSPEC15-SEED123 | BCE | 123 | `runs/EXP-UVVIS-003-UVSPEC15-SEED123/` |
| EXP-UVVIS-003-UVSPEC15-SEED2026 | BCE | 2026 | `runs/EXP-UVVIS-003-UVSPEC15-SEED2026/` |
| EXP-UVVIS-004-UVSPEC15-WBCE | WBCE | 42 | `runs/EXP-UVVIS-004-UVSPEC15-WBCE/` |
| EXP-UVVIS-004-UVSPEC15-WBCE-SEED123 | WBCE | 123 | `runs/EXP-UVVIS-004-UVSPEC15-WBCE-SEED123/` |
| EXP-UVVIS-004-UVSPEC15-WBCE-SEED2026 | WBCE | 2026 | `runs/EXP-UVVIS-004-UVSPEC15-WBCE-SEED2026/` |

## 3. 24 条件的数值来源

| 条件 | 数值来源（Test；Validation 同构） |
|---|---|
| Original BN + 阈值 0.5 | `<run>/test_metrics.json`（test_micro_f1、macro_f1、mAP、macro_AUROC）；逐标签 `<run>/test_per_label_metrics.csv` |
| Original BN + Validation-F1 | `<run>/threshold_calibration_summary.json` 的 `test_calibrated`；逐标签 `<run>/test_per_label_metrics_calibrated.csv`；阈值 `<run>/optimal_thresholds.json` |
| Recalibrated BN + 阈值 0.5 | `runs/UVSPEC15-BN-RECALIBRATED/<run>/threshold_calibration/threshold_calibration_summary.json` 的 `test_default`（与 `runs/UVSPEC15-BN-DIAGNOSTICS/<run>/batchnorm_full_diagnostic.json` 的 `after` 一致）；逐标签 P/R/F1（默认阈值）由汇总脚本从 `<run>/threshold_calibration/threshold_comparison.csv` 读取并写入 `per_label_720_rows.csv`；mAP/AUROC 取该 summary 的 `test_threshold_free` |
| Recalibrated BN + Validation-F1 | 同上 summary 的 `test_calibrated`；逐标签 `<run>/threshold_calibration/test_per_label_metrics_calibrated.csv`；阈值 `optimal_thresholds.json` |

## 4. 辅助与核验文件（`runs/UVSPEC15-ABLATION-SUMMARY/`）

| 文件 | 内容 |
|---|---|
| `README_summary.txt` | 汇总表与决策要点（Selection policy 等） |
| `mean_sd_8_conditions_each_split.csv` | 8 组（损失 × BN × 阈值）× 3 seeds 的 mean ± 样本 SD（验证 SD 为 `statistics.stdev`，ddof=1） |
| `test_24_conditions.csv` / `validation_24_conditions.csv` | 24 条件逐条记录（run_name, loss, seed, split, bn, threshold, 指标） |
| `per_label_720_rows.csv` | 15 标签 × 24 条件 × 2 splits 的逐标签指标 |
| `paired_effects.csv` | 配对效应（如 BCE vs WBCE 同条件对比） |
| `audit_checks.csv` | 三项审计：original_baseline_reproduced / bn_saved_diagnostic_reproduced / label_order_verified（全 True） |

## 5. 说明与局限

- **mAP/AUROC 为阈值无关指标**，每个模型只报告一次（`test_threshold_free`），不随阈值变化。
- **逐标签 F1 类指标**（Precision/Recall/F1）：默认阈值 0.5 与校准阈值下的逐标签值均已落盘——原始 checkpoint 在 `<run>/test_per_label_metrics.csv`（默认）与 `<run>/test_per_label_metrics_calibrated.csv`（校准）；BN 重估版由汇总脚本从 `<run>/threshold_calibration/threshold_comparison.csv`（及 Validation 版 `validation_threshold_comparison.csv`）读取默认阈值与校准阈值下的 P/R/F1，写入 `per_label_720_rows.csv`（Recalibrated + 0.5 条件对应 180 行 = 15 标签 × 3 seeds × 2 splits × 2 损失组合的相应子集）。
- **逐标签 AP/AUROC**：仅原始 checkpoint 的 `test_per_label_metrics.csv` 落盘（含 average_precision、auroc 列）；BN 重估版的逐标签 AP/AUROC 未单独落盘，整体 mAP/AUROC 见 summary 的 `test_threshold_free`。mAP/AUROC 为阈值无关指标，无需按阈值重复报告。
- 阈值列与 checkpoint 的配套关系见 `docs/experiments/uvspec15_repro_protocol.md` 检查清单。
- 本文件仅为映射索引；若与原始 JSON/CSV 冲突，以原始文件为准。
