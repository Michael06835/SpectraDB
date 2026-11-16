# Raman 消融实验与原始文件对应关系（24 条件）

> **用途**：将 `runs/RAMAN-ABLATION-SUMMARY/` 中的 24 个评估条件逐一映射回原始实验文件、checkpoint 与阈值，供审计与论文引用。
> **权威来源**：**所有数值以原始 JSON/CSV 为唯一来源**；本文件不重算、不修改任何实验产物。
> **对应模态**：QM9S 计算域 Raman，14 类官能团多标签，3501 维（500–4000 cm⁻¹，步长 1.0）。
> **归档状态**：Raman v1 已冻结；最终配置见 `Raman_v1_final_decision.md`。
> **对照文档**：`ir_ablation_registry.md`、`uvspec15_ablation_registry.md`（同构登记表）。目录布局差异见 §9.3。

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
| `EXP-RAMAN-001-FULL` | BCE | 42 | `runs/EXP-RAMAN-001-FULL/` | `training/train_raman_full.py` |
| `EXP-RAMAN-001-SEED123` | BCE | 123 | `runs/EXP-RAMAN-001-SEED123/` | `training/train_raman_seed123.py` |
| `EXP-RAMAN-001-SEED2026` | BCE | 2026 | `runs/EXP-RAMAN-001-SEED2026/` | `training/train_raman_seed2026.py` |
| `EXP-RAMAN-002-FULL` | WBCE | 42 | `runs/EXP-RAMAN-002-FULL/` | `training/train_raman_weighted.py` |
| `EXP-RAMAN-002-SEED123` | WBCE | 123 | `runs/EXP-RAMAN-002-SEED123/` | `training/train_raman_weighted_seed123.py` |
| `EXP-RAMAN-002-SEED2026` | WBCE | 2026 | `runs/EXP-RAMAN-002-SEED2026/` | `training/train_raman_weighted_seed2026.py` |

每个 run 目录包含：`config.json`、`training_log.csv`、`best_model.pt`、`test_metrics.json`、`test_per_label_metrics.csv`。
`EXP-RAMAN-001-FULL` 另含 2026-08-14 的 4 个历史校准文件（`optimal_thresholds.json`、`threshold_calibration_summary.json`、`threshold_comparison.csv`、`test_per_label_metrics_calibrated.csv`，1024/0 推理口径）。**这 4 个文件不参与 24 条件矩阵**，仅为历史记录保留（与 IR v1 同类情况，见 §9.2）。

---

## 3. 24 条件的数值来源

记：

- `OTH = runs/RAMAN-THRESHOLD-CALIBRATION/<run>/`（Original BN 校准目录）
- `RTH = runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run>/`（Recalibrated BN 校准目录）
- `DIAG = runs/RAMAN-BN-DIAGNOSTICS/<run>/batchnorm_full_diagnostic.json`
- `S = <校准目录>/threshold_calibration_summary.json`

| 条件类型 | F1 相关指标（Micro/Macro-F1、Micro-Precision、Micro-Recall） | mAP / Macro-AUROC（Validation） | mAP / Macro-AUROC（Test） |
|---|---|---|---|
| **Original + 0.5** | `OTH/S` → `validation_default` / `test_default` | `DIAG` → `before.validation` | `OTH/S` → `test_threshold_free` |
| **Original + Val-F1** | `OTH/S` → `validation_calibrated` / `test_calibrated` | `DIAG` → `before.validation` | `OTH/S` → `test_threshold_free` |
| **Recalibrated + 0.5** | `RTH/S` → `validation_default` / `test_default` | `DIAG` → `after.validation` | `RTH/S` → `test_threshold_free` |
| **Recalibrated + Val-F1** | `RTH/S` → `validation_calibrated` / `test_calibrated` | `DIAG` → `after.validation` | `RTH/S` → `test_threshold_free` |

**为什么 Validation 的 mAP/AUROC 来自诊断 JSON**：两份 `threshold_calibration_summary.json` 只记录 `test_threshold_free`，**没有** `validation_threshold_free`；Validation 的阈值无关指标只能取自诊断记录的 `before`/`after` 块。这与 IR v1 完全一致。

**交叉核验**：`summarize_raman_ablation.py` 在写任何一行之前会校验
（a）`test_metrics.json` ↔ `OTH/S.test_default` + `test_threshold_free`；
（b）`DIAG.before.test` ↔ `test_metrics.json`；
（c）`DIAG.before.validation` ↔ `OTH/S.validation_default`；
（d）`DIAG.after.test` ↔ `RTH/S.test_default` + `test_threshold_free`；
（e）`DIAG.after.validation` ↔ `RTH/S.validation_default`；
全部需在 **0.0025** 容差内，否则脚本拒绝创建输出目录。该容差与 IR 汇总器逐字一致，**未对 Raman 做任何放宽**。

---

## 4. 每类条件对应的 checkpoint / 诊断 / 阈值 / 逐标签文件

| 条件类型 | checkpoint | 诊断 JSON | threshold summary | 阈值文件 | 逐标签（Validation） | 逐标签（Test） |
|---|---|---|---|---|---|---|
| Original + 0.5 | `runs/<run>/best_model.pt` | `DIAG`（`before` 段） | `OTH/S`（`*_default`） | 固定 0.5 | `OTH/validation_threshold_comparison.csv` 的 `default_*` 列 | `OTH/threshold_comparison.csv` 的 `default_*` 列 |
| Original + Val-F1 | 同上 | `DIAG`（`before` 段） | `OTH/S`（`*_calibrated`） | `OTH/optimal_thresholds.json` | 同 CSV 的 `calibrated_*` 列 | 同 CSV 的 `calibrated_*` 列 |
| Recalibrated + 0.5 | `runs/RAMAN-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt` | `DIAG`（`after` 段） | `RTH/S`（`*_default`） | 固定 0.5 | `RTH/validation_threshold_comparison.csv` 的 `default_*` 列 | `RTH/threshold_comparison.csv` 的 `default_*` 列 |
| Recalibrated + Val-F1 | 同上 | `DIAG`（`after` 段） | `RTH/S`（`*_calibrated`） | `RTH/optimal_thresholds.json` | 同 CSV 的 `calibrated_*` 列 | 同 CSV 的 `calibrated_*` 列 |

**每个校准目录的 9 个文件**

| 文件 | 说明 |
|---|---|
| `optimal_thresholds.json` | 14 个逐标签阈值（`source` 声明仅用 Validation；`frozen_before_test_metrics: true`） |
| `validation_threshold_comparison.csv` | 12 列（IR 同构） |
| `threshold_comparison.csv` | 13 列（IR 同构） |
| `test_per_label_metrics_default.csv` | 0.5 条件逐标签 |
| `test_per_label_metrics_calibrated.csv` | Val-F1 条件逐标签 |
| `threshold_calibration_summary.json` | 六指标 + 来源 + `numeric_audit` |
| `probability_artifacts.json` | **Raman 新增**：数组 SHA256、checkpoint SHA256、split 规模、标签顺序、评估口径 |
| `<run>__validation__probabilities.npy` | **Raman 新增**：Validation 概率（float32） |
| `<run>__test__probabilities.npy` | **Raman 新增**：Test 概率（float32） |

前 6 个与 `runs/IR-THRESHOLD-CALIBRATION/EXP-IR-001-FULL/` 的同名文件**列名逐字相同**；后 3 个为 Raman 追加的溯源产物，不改变任何数值口径，也不参与 24 条件矩阵的取值。

---

## 5. checkpoint SHA256 对应关系

**来源链**：`runs/<run>/best_model.pt` --(train-only BN 重估，见 `save_raman_bn_checkpoint.py`, gate policy v5)--> `runs/RAMAN-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt`，由 `bn_recalibration_manifest.json` 记录两端 SHA256。

| run | Original checkpoint SHA256 | Recalibrated checkpoint SHA256 | BN 层 | 重估样本 | 变化 buffer | Gate B 最大偏差 |
|---|---|---|---:|---:|---:|---|
| `EXP-RAMAN-001-FULL` | `573ae7e2a3351252c5f1ad7e0ec060c77435468d3c7bddb37a641e1206cdfcd3` | `a9f9d55f1239abd7c91f52a4900f3ba5f23a9f4ced799b626802d956197022f5` | 12 | 103855 | 36 | 3.769e-05 |
| `EXP-RAMAN-001-SEED123` | `84334c5e013490de75bc23f0f844880062150759947835cb29e18e80c9158298` | `f84f0d32e2f24b411741f7006578daa8cd0621e0f0a5eba7a124c989abe65ec1` | 12 | 103855 | 36 | 6.653e-05 |
| `EXP-RAMAN-001-SEED2026` | `eac01833a7f618e18021856e94eeec21ef09802af428deae5a6c33e8f9e590ef` | `f86a1ac09bde8e8a62c1fceafff92442848a572ab8da8faa609aa78eda4c5693` | 12 | 103855 | 36 | 5.862e-07 |
| `EXP-RAMAN-002-FULL` | `ece69bbc763f9c3336d8c3b68e8f137f9d135a938796fbc4ba6ecdac3d14d9aa` | `9b100596a0669e3d491b7479e7a7d4f90b4d5a813c7612b596b5e8956ab0ed74` | 12 | 103855 | 36 | 6.653e-05 |
| `EXP-RAMAN-002-SEED123` | `6f1de2eec40dc3331d87078744d5a08f0cfcfc4b55a96a629707094a82625fae` | `6ece02b72c59f408b8d1f06a4580d68cde03a6177ea1edeb745d7aa2fc918a31` | 12 | 103855 | 36 | 3.459e-06 |
| `EXP-RAMAN-002-SEED2026` | `a1688d5b545156987266295d495550f4d0fa892344b3326f4226c12ae1a6b9a3` | `7c34c5d8a790d7b150e2bb1fb63b8c5bce4261510a8c04656732c4d485dabea1` | 12 | 103855 | 36 | 3.142e-05 |

（"Gate B 最大偏差"= 官方 FP16 六项 Validation 指标与 `DIAG.after.validation` 的**最大绝对偏差**，容差 1e-3。）

**四向一致性（`summarize_raman_ablation.py` 逐 run 断言）**

1. `manifest.source_sha256` == 当前 `runs/<run>/best_model.pt` 的实测 SHA256；
2. `manifest.saved_sha256` == 当前 `best_model_bn_recalibrated.pt` 的实测 SHA256；
3. `OTH/threshold_calibration_summary.json.checkpoint_sha256` == 原 checkpoint 哈希；
4. `RTH/threshold_calibration_summary.json.checkpoint_sha256` == 重估 checkpoint 哈希。

审计结果写入 `runs/RAMAN-ABLATION-SUMMARY/audit_checks.csv`（6 行 × 4 项 = **24 项全为 True**）。

---

## 6. `runs/RAMAN-ABLATION-SUMMARY/` 的 11 份文件用途

| 文件 | 行数 | 用途 |
|---|---:|---|
| `test_24_conditions.csv` | 24 | 24 个条件的 Test 六指标 + 条件标识 |
| `validation_24_conditions.csv` | 24 | 24 个条件的 Validation 六指标 |
| `mean_sd_8_conditions_each_split.csv` | 16 | 8 配置 × 2 split 的三 Seed Mean ± 样本 SD（ddof=1） |
| `paired_effects.csv` | 84 | 逐 Seed 配对差（4 个比较族 × 2 split × …，含六指标） |
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
- 汇总脚本对此做了显式断言（差 < 1e-9），并在 `README_summary.txt` 的 Metric sources 段与 `paired_effects.csv` 中体现（相关 Δ 恒为 0.000000 ± 0.000000）；
- 论文引用时**不应**把 mAP/AUROC 按阈值重复计数为 48 个独立数值，实际只有 **8 个**（4 配置 × 2 split，因 BN 状态改变会改变排序，所以 Original 与 Recalibrated 数值可以不同）。

---

## 9. 说明与局限

### 9.1 原始 JSON/CSV 的优先级

1. **最高**：`runs/<run>/`、`runs/RAMAN-BN-DIAGNOSTICS/`、`runs/RAMAN-THRESHOLD-CALIBRATION/`、`runs/RAMAN-BN-THRESHOLD-CALIBRATION/`、`runs/RAMAN-BN-RECALIBRATED/`、`runs/RAMAN-NUMERIC-AUDIT/` 下的原始 JSON/CSV、manifest 与概率数组；
2. **次之**：`runs/RAMAN-ABLATION-SUMMARY/` 的汇总 CSV（由脚本从 1 生成，可完全重放）；
3. **最低**：本文档与 `README_summary.txt` 的叙述性文字。

**若本文件与原始文件冲突，以原始文件为准。**

### 9.2 不参与矩阵的历史产物

`runs/EXP-RAMAN-001-FULL/` 下的 4 个 2026-08-14 校准文件使用 `batch_size=1024 / workers=0` 推理口径，与统一口径（512/4）存在 1e-5 级数值差异。它们是 Raman-v1 第一阶段的历史记录，**不计入 24 条件矩阵**，也不得与新校准结果混用。

### 9.3 与 IR / UVSPEC15 目录布局的差异

| 项 | IR v1 | Raman v1 |
|---|---|---|
| Original BN 校准结果位置 | 独立根目录 `runs/IR-THRESHOLD-CALIBRATION/<run>/` | 同构：`runs/RAMAN-THRESHOLD-CALIBRATION/<run>/` |
| Recalibrated BN 校准结果位置 | 独立根目录 `runs/IR-BN-THRESHOLD-CALIBRATION/<run>/` | 同构：`runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run>/` |
| 每个校准目录的文件数 | 6 | **9**（+ `probability_artifacts.json` + 2 个 `.npy`） |
| 数值审计记录 | 无独立目录（结论写在 manifest 的 `diagnostic_reference`） | **新增** `runs/RAMAN-NUMERIC-AUDIT/<run>/{original,recalibrated}_thresholds.json` |
| 决策文档 | 与策略文档分立 | 同构：`Raman_v1_final_decision.md` + `raman_repro_protocol.md` |

**新增 3 个文件与审计目录的理由**：Raman 在 Phase 3 曾出现一次 **Gate B 数值复现失败**——BCE seed 42 的保存后读数为 0.946294，与诊断记录的 0.945208 相差 **1.086e-3**，超出 1e-3 判据；随后的 T1/T2 跨进程对照显示，同一文件在独立进程中可读到 0.946288（与 0.945208 相差 **1.080e-3**）。两个差值对应不同的读数对，引用时须分别标注。为使"哪一组概率产生了哪一个指标"可被独立复核，Raman 额外落盘概率数组与首次观测即定论的审计记录。**这些新增产物不改变任何容差、指标定义或条件数量**，24 条件矩阵与 IR 完全同构。

### 9.4 实验局限

- 结论限于 **QM9S 计算光谱域 + scaffold split**，未做实验域（API-Raman、NIST）验证。
- 六个独立训练模型之外无额外重复；**n = 3 个 Seed**，SD 为描述性统计，**不构成显著性结论**。
- 部分标签样本量极小（nitro：Validation 11 / Test 15；c_f_bond：Test 144），其逐标签阈值与 F1 的不确定度较大；nitro 的 AP 还是 mAP 跨进程波动的主要载体。
- 24 条件矩阵只覆盖「损失 × Seed × BN 统计量 × 阈值」四个因素，未覆盖架构、embedding 维度、数据增强、校准方法等其他维度。
- 本登记表不评价方法优劣；配置选择见 `Raman_v1_final_decision.md`。

### 9.5 历史描述性元数据勘误（不改写 manifest）

**对象**：六份 `runs/RAMAN-BN-RECALIBRATED/<run_name>/bn_recalibration_manifest.json` 的 `gate_policy.measured` 块。

1. **存在一处历史算术归属错误**：`measured.official_fp16_map_same_file_two_processes`（T1/T2 那对读数）与 `measured.official_fp16_map_cross_process_spread` 被配成了一组，但该 spread 的数值写成了另一对读数之差。
2. **正确算术**：`[0.945208, 0.946288]` 的实际差值为 **0.001080**（1.080e-3），不是 0.001086。
3. **`0.001086`（1.086e-3）的真实归属**：最初 Gate B 失败时的那一对读数 **`[0.945208, 0.946294]`**（Phase 3 保存后读数 vs Phase 2 诊断记录）。
4. **影响范围**：该字段**仅用于历史描述**。经静态引用审计确认，`gate_policy.measured` 与其中任一字段都不参与 Gate A（10 项确定性检查）、Gate B（六指标 1e-3 判定）、数值容差、异常分类（`PASS` / `MAP_ONLY_NUMERIC_MISMATCH` / `OTHER_ANOMALY`）或放行判断（`ablation_release`）；它只被 `official_metrics_report()` 复制进 manifest 的 `official_fp16_metrics` 描述块。因此：**不影响模型权重、指标计算、Gate A/B 判定结果或放行状态**。
5. **处置**：六份历史 manifest **原样保留，不进行事后覆盖**。勘误以本文档为解释依据；代码常量已在 `training/save_raman_bn_checkpoint.py` 中更正为 `1.080e-3` 并附算术归属注释，仅影响此后新生成的 manifest。

> 引用规则：凡引用 1.086e-3，须注明是 Gate B 失败对 `[0.945208, 0.946294]`；凡引用 1.080e-3，须注明是 T1/T2 跨进程对 `[0.945208, 0.946288]`。详见 `Raman_v1_final_decision.md` §8.6「两个必须区分的数值」。

---

## 10. 复现入口

```powershell
# 重放汇总（对来源只读：不训练、不前向、不拟合阈值；但会新建汇总目录并写入 11 个文件）
python training\summarize_raman_ablation.py
```

- **对来源只读**：脚本只读取既有 JSON/CSV 并校验 12 个 checkpoint 的 SHA256，不修改 checkpoint、校准摘要或诊断记录。
- **但会写盘**：它会在 `runs/RAMAN-ABLATION-SUMMARY/` **新建**汇总目录并写入 11 个文件，因此不属于"全程零写盘"。
- **不允许覆盖**：若该目录已存在，脚本拒绝执行并退出；既有汇总目录**不得删除或替换**。若需验证性重放，请用 `--out-dir <尚不存在的新目录>`。

预期末行：`PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows`。
完整步骤见 `raman_repro_protocol.md`。
