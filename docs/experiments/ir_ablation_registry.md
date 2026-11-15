# IR 消融实验与原始文件对应关系（24 条件）

> **用途**：将 `runs/IR-ABLATION-SUMMARY/` 中的 24 个评估条件逐一映射回原始实验文件、checkpoint 与阈值，供审计与论文引用。
> **权威来源**：**所有数值以原始 JSON/CSV 为唯一来源**；本文件不重算、不修改任何实验产物。
> **对应模态**：QM9S 计算域 IR，14 类官能团多标签。
> **归档状态**：IR v1 已冻结；最终配置见 `IR_v1_final_decision.md`。
> **对照文档**：`uvspec15_ablation_registry.md`（UVSPEC15 同构登记表）。**IR 的目录布局与其不同，见 §9.3。**

---

## 1. 实验矩阵

- **6 次独立训练** = 损失（普通 BCE / Weighted BCE）× seed（42 / 123 / 2026）；
- 每模型 **2 种 BN 推理统计量**：Original / Recalibrated（train-only 重估 running stats）；
- 每模型 **2 种阈值**：0.5（默认）/ Validation-F1（Validation 上逐标签 F1 最优）；
- 合计每个 split **24 个评估条件**（Validation 24、Test 24）；
- 逐标签结果 **672 行**（14 标签 × 24 条件 × 2 splits）。

**计数纪律**：12 次 Threshold Calibration 是**执行任务**，其输出构成 24 条件矩阵中的 Validation-F1 分支，**不得**与 24 个条件相加表述为"36 个互不重复的消融条件"，也不得把训练次数与评估条件数混谈。

---

## 2. 六个训练 run

| run_name | 损失 | Seed | 训练产物目录 | 训练脚本 |
|---|---|---|---|---|
| `EXP-IR-001-FULL` | BCE | 42 | `runs/EXP-IR-001-FULL/` | `training/train_ir_full.py` |
| `EXP-IR-001-SEED123` | BCE | 123 | `runs/EXP-IR-001-SEED123/` | `training/train_ir_seed123.py` |
| `EXP-IR-001-SEED2026` | BCE | 2026 | `runs/EXP-IR-001-SEED2026/` | `training/train_ir_seed2026.py` |
| `EXP-IR-002-FULL` | WBCE | 42 | `runs/EXP-IR-002-FULL/` | `training/train_ir_weighted.py` |
| `EXP-IR-002-SEED123` | WBCE | 123 | `runs/EXP-IR-002-SEED123/` | `training/train_ir_weighted_seed123.py` |
| `EXP-IR-002-SEED2026` | WBCE | 2026 | `runs/EXP-IR-002-SEED2026/` | `training/train_ir_weighted_seed2026.py` |

每个 run 目录包含：`config.json`、`training_log.csv`、`best_model.pt`、`test_metrics.json`、`test_per_label_metrics.csv`。
`EXP-IR-001-FULL` 另含 2026-08-14 的 4 个历史校准文件（`optimal_thresholds.json`、`threshold_calibration_summary.json`、`threshold_comparison.csv`、`test_per_label_metrics_calibrated.csv`，1024/0 推理口径）。**这 4 个文件不参与 24 条件矩阵**，仅为历史记录保留，见 §9.2。

---

## 3. 24 条件的数值来源

**四类条件，各自的取值路径不同。** 记：

- `OTH = runs/IR-THRESHOLD-CALIBRATION/<run>/`（Original BN 校准目录）
- `RTH = runs/IR-BN-THRESHOLD-CALIBRATION/<run>/`（Recalibrated BN 校准目录）
- `DIAG = runs/IR-BN-DIAGNOSTICS/<run>/batchnorm_full_diagnostic.json`
- `S = <校准目录>/threshold_calibration_summary.json`

| 条件类型 | F1 相关指标（Micro/Macro-F1、Micro-Precision、Micro-Recall） | mAP / Macro-AUROC（Validation） | mAP / Macro-AUROC（Test） |
|---|---|---|---|
| **Original + 0.5** | `OTH/S` → `validation_default` / `test_default` | `DIAG` → `before.validation` | `OTH/S` → `test_threshold_free` |
| **Original + Val-F1** | `OTH/S` → `validation_calibrated` / `test_calibrated` | `DIAG` → `before.validation` | `OTH/S` → `test_threshold_free` |
| **Recalibrated + 0.5** | `RTH/S` → `validation_default` / `test_default` | `DIAG` → `after.validation` | `RTH/S` → `test_threshold_free` |
| **Recalibrated + Val-F1** | `RTH/S` → `validation_calibrated` / `test_calibrated` | `DIAG` → `after.validation` | `RTH/S` → `test_threshold_free` |

**为什么 Validation 的 mAP/AUROC 来自诊断 JSON**：两份 `threshold_calibration_summary.json` 只记录 `test_threshold_free`，**没有** `validation_threshold_free`；Validation 的阈值无关指标只能取自诊断记录的 `before`/`after` 块。

**交叉核验**：`summarize_ir_ablation.py` 在写任何一行之前，会校验
（a）`test_metrics.json` ↔ `OTH/S.test_default` + `test_threshold_free`；
（b）`DIAG.before.test` ↔ `test_metrics.json`；
（c）`DIAG.before.validation` ↔ `OTH/S.validation_default`；
（d）`DIAG.after.test` ↔ `RTH/S.test_default` + `test_threshold_free`；
（e）`DIAG.after.validation` ↔ `RTH/S.validation_default`；
全部需在 **0.0025** 容差内，否则脚本拒绝创建输出目录。

---

## 4. 每类条件对应的 checkpoint / 诊断 / 阈值 / 逐标签文件

| 条件类型 | checkpoint | 诊断 JSON | threshold summary | 阈值文件 | 逐标签（Validation） | 逐标签（Test） |
|---|---|---|---|---|---|---|
| Original + 0.5 | `runs/<run>/best_model.pt` | `DIAG`（`before` 段） | `OTH/S`（`*_default`） | 固定 0.5 | `OTH/validation_threshold_comparison.csv` 的 `default_*` 列 | `OTH/threshold_comparison.csv` 的 `default_*` 列 |
| Original + Val-F1 | 同上 | `DIAG`（`before` 段） | `OTH/S`（`*_calibrated`） | `OTH/optimal_thresholds.json` | 同 CSV 的 `calibrated_*` 列 | 同 CSV 的 `calibrated_*` 列 |
| Recalibrated + 0.5 | `runs/IR-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt` | `DIAG`（`after` 段） | `RTH/S`（`*_default`） | 固定 0.5 | `RTH/validation_threshold_comparison.csv` 的 `default_*` 列 | `RTH/threshold_comparison.csv` 的 `default_*` 列 |
| Recalibrated + Val-F1 | 同上 | `DIAG`（`after` 段） | `RTH/S`（`*_calibrated`） | `RTH/optimal_thresholds.json` | 同 CSV 的 `calibrated_*` 列 | 同 CSV 的 `calibrated_*` 列 |

**补充说明**

- 每个校准目录另有 `test_per_label_metrics_default.csv` 与 `test_per_label_metrics_calibrated.csv`，内容与 `threshold_comparison.csv` 的对应列等价（前者是后者的子集投影）。汇总脚本统一从 `threshold_comparison.csv` / `validation_threshold_comparison.csv` 读取逐标签值。
- Original checkpoint 的零后处理评估（阈值 0.5、无 Validation 拟合成分）以 `runs/<run>/test_metrics.json` 与 `test_per_label_metrics.csv` 为准，与 `OTH/S.test_default` 在 1e-3 内一致（实测最大差 8.75e-07 ~ 6.98e-05）。

---

## 5. BN manifest 与 checkpoint SHA256 对应关系

**来源链**：`runs/<run>/best_model.pt` --(train-only BN 重估)--> `runs/IR-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt`，由 `bn_recalibration_manifest.json` 记录两端 SHA256。

| run | Original checkpoint SHA256 | Recalibrated checkpoint SHA256 | BN 层数 | 重估样本 | 变化 buffer |
|---|---|---|---|---:|---:|
| `EXP-IR-001-FULL` | `d60cc7b159891728d46597d15826f29028d00a84801475ab07120f3255613239` | `22db5c67ec82b8965ee06de94bd2aad2e4dd9b0bfb69dcdfa81c64838578bb4f` | 12 | 103855 | 36 |
| `EXP-IR-001-SEED123` | `d3856f25c0d1cbb9ddbd3e2b031e5bcdc1c43825b1e60fa26ae9bb4e048e82a2` | `666e9eed5469ed5e3d611bc34e107c3de2090696a139d7f4e4720c80cce6c8ce` | 12 | 103855 | 36 |
| `EXP-IR-001-SEED2026` | `d7945c78d432fa907bbc0c8b39e3f4e7ae78088a1185931e625a8eb06cc4a274` | `b326e234f6464c55a818e5036f7278c05bceafd3b7c5cd7c29589328d2ef00c3` | 12 | 103855 | 36 |
| `EXP-IR-002-FULL` | `cda0b8019588874c33769f77741cb4cbe7804bb3cbac36b59458bfd4e0f4218c` | `9433c97510f698e13ec463f33f2a68faa201b438929e46fd8c235154bcc17b3b` | 12 | 103855 | 36 |
| `EXP-IR-002-SEED123` | `6c1f0ae3ad01a03a1dd1d094ae614460af56d9f1973016de5fe6709cd8ea94d8` | `f2063337ece8d13f20f676cb5666265fce9ac3e282bf485a0c4a97a3e0314dac` | 12 | 103855 | 36 |
| `EXP-IR-002-SEED2026` | `9cd4be28f820c4295afbc60eb53af6da497b3fb36835883f22c31c23b2455fe4` | `ef72872a1fca4d57b4073fd542697bdb6056fe84f0a075cced7e01ffa692006b` | 12 | 103855 | 36 |

**四向一致性（`summarize_ir_ablation.py` 逐 run 断言）**

1. `manifest.source_sha256` == 当前 `runs/<run>/best_model.pt` 的实测 SHA256；
2. `manifest.saved_sha256` == 当前 `best_model_bn_recalibrated.pt` 的实测 SHA256；
3. `OTH/threshold_calibration_summary.json.checkpoint_sha256` == 原 checkpoint 哈希；
4. `RTH/threshold_calibration_summary.json.checkpoint_sha256` == 重估 checkpoint 哈希。

审计结果写入 `runs/IR-ABLATION-SUMMARY/audit_checks.csv`（6 行，`manifest_hash_chain_verified` 全为 `True`）。

---

## 6. `runs/IR-ABLATION-SUMMARY/` 的 11 份文件用途

| 文件 | 行数 | 用途 |
|---|---:|---|
| `test_24_conditions.csv` | 24 | 24 个条件的 Test 六指标 + 条件标识 |
| `validation_24_conditions.csv` | 24 | 24 个条件的 Validation 六指标 |
| `mean_sd_8_conditions_each_split.csv` | 16 | 8 配置 × 2 split 的三 Seed Mean ± 样本 SD（ddof=1） |
| `paired_effects.csv` | 84 | 逐 Seed 配对差（4 个比较族 × split × seed） |
| `paired_effects_summary.csv` | 28 | 上述配对差的 Mean ± SD |
| `per_label_672_rows.csv` | 672 | 逐标签 P/R/F1（含阈值取值与正样本数） |
| `per_label_f1_levels.csv` | 14 | 每标签 × 8 配置 × 2 split 的 F1 均值 + 支持度分层 |
| `per_label_deltas.csv` | 224 | 每标签的 F1 变化（阈值校准 / BN 重估分族） |
| `condition_registry.csv` | 48 | 每个 split-level 条件的指标来源、阈值无关来源、逐标签来源文件路径 |
| `audit_checks.csv` | 6 | 每 run 的四项审计（基线复现 / BN 保存一致性 / 标签顺序 / manifest 哈希链） |
| `README_summary.txt` | — | 可读总结：Test/Validation Mean±SD 表、Validation 配对效应、选择政策、指标来源归档说明、支持度分层 |

---

## 7. 48 条 split-level 与 672 条逐标签记录的来源与数量核验

| 记录类型 | 定义 | 数量 | 来源 |
|---|---|---:|---|
| split-level | 6 run × 2 BN × 2 阈值 × 2 split | **48** | §3 的 summary / 诊断组合 |
| 逐标签 | 14 标签 × 24 条件 × 2 split | **672** | §4 的两份 comparison CSV |

脚本内断言（任一不满足即中止，不创建输出目录）：

- `len(records) == 48`；
- `len(per_label) == 672`；
- 唯一条件键 `(loss, seed, bn, threshold, split)` 数量 == 48（防重复计数）；
- 聚合组 `(split, loss, bn, threshold)` 数量 == 16（8 配置 × 2 split）；
- 每个聚合组恰好包含 3 个 Seed。

---

## 8. 阈值无关指标

`mAP` 与 `Macro-AUROC` 只依赖模型输出的排序，**与分类阈值无关**。因此在 24 条件矩阵中：

- 同一 (模型, BN 状态) 的两条条件（0.5 与 Validation-F1）具有**完全相同**的 mAP / Macro-AUROC；
- 汇总脚本对此做了显式断言（差 < 1e-9），并写入 `README_summary.txt` 的 Metric sources 段；
- 论文引用时**不应**把 mAP/AUROC 按阈值重复计数为 48 个独立数值，实际只有 **8 个**（4 配置 × 2 split，因 BN 状态改变会改变排序，所以 Original 与 Recalibrated 数值可以不同）。

---

## 9. 说明与局限

### 9.1 原始 JSON/CSV 的优先级

1. **最高**：`runs/<run>/`、`runs/IR-BN-DIAGNOSTICS/`、`runs/IR-THRESHOLD-CALIBRATION/`、`runs/IR-BN-THRESHOLD-CALIBRATION/`、`runs/IR-BN-RECALIBRATED/` 下的原始 JSON/CSV 与 manifest；
2. **次之**：`runs/IR-ABLATION-SUMMARY/` 的汇总 CSV（由脚本从 1 生成，可完全重放）；
3. **最低**：本文档与 `README_summary.txt` 的叙述性文字。

**若本文件与原始文件冲突，以原始文件为准。**

### 9.2 不参与矩阵的历史产物

`runs/EXP-IR-001-FULL/` 下的 4 个 2026-08-14 校准文件使用 `batch_size=1024 / workers=0` 推理口径，与统一口径（512/4）存在 2.1e-5 级数值差异。它们是 IR-v1 第一阶段的历史记录，**不计入 24 条件矩阵**，也不得与新校准结果混用。`runs/IR-v1-FINAL/` 同样作为历史版本保留，未改写。

### 9.3 与 UVSPEC15 目录布局的差异（重要）

| 项 | UVSPEC15 | IR |
|---|---|---|
| Original BN 校准结果位置 | 训练 run 目录内（`runs/EXP-UVVIS-003-*/threshold_calibration_summary.json`） | **独立根目录** `runs/IR-THRESHOLD-CALIBRATION/<run>/` |
| Recalibrated BN 校准结果位置 | `runs/UVSPEC15-BN-RECALIBRATED/<run>/threshold_calibration/` | **独立根目录** `runs/IR-BN-THRESHOLD-CALIBRATION/<run>/` |
| 历史校准产物 | 训练目录内，作为当次正式结果 | `runs/EXP-IR-001-FULL/` 内 4 个文件，**仅历史**（1024/0 口径） |
| 决策文档 | 与策略文档合并（`UVSPEC15_v1_experimental_strategy.md` 兼作决策记录） | 策略与决策**分立**（`IR_v1_experimental_strategy.md` + `IR_v1_final_decision.md`） |

**原因**：IR 的第一阶段已有历史校准产物，为避免覆盖（旧脚本以 `"w"` 模式写盘），统一口径的新校准结果必须另存独立目录；该设计同时使"防覆盖"成为结构约束而非流程约束。**IR 未沿用 UVSPEC15 的 in-run 布局，任何跨模态脚本复用都必须改路径。**

### 9.4 实验局限

- 结论限于 **QM9S 计算光谱域 + scaffold split**，未做实验域（NIST/API-Raman）验证。
- 4 个独立训练模型之外无额外重复；**n = 3 个 Seed**，SD 为描述性统计，**不构成显著性结论**。
- 部分标签样本量极小（nitro：Validation 11 / Test 15；c_f_bond：230 / 144），其逐标签阈值与 F1 的不确定度较大。
- 24 条件矩阵只覆盖「损失 × Seed × BN 统计量 × 阈值」四个因素，未覆盖架构、embedding 维度、数据增强、校准方法等其他维度。
- 本登记表不评价方法优劣；配置选择见 `IR_v1_final_decision.md`。

---

## 10. 复现入口

```powershell
# 只读重放汇总（会重新校验全部来源链；不训练、不前向、不拟合阈值）
python training\summarize_ir_ablation.py
```

预期末行：`PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows`
若输出目录已存在，脚本拒绝覆盖并退出。完整步骤见 `ir_repro_protocol.md`。
