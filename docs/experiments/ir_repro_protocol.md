# IR v1 实验复现协议

> **状态**：与 `IR_v1_experimental_strategy.md`（流程）、`ir_ablation_registry.md`（溯源）、`IR_v1_final_decision.md`（决策）配套的**可执行复现指南**。
> **范围**：QM9S 计算域 IR → 14 类官能团多标签的单模态 1D-CNN 基线；不包含多模态融合。
> **性质**：IR v1 已冻结；本协议描述**如何从零重放**，不改变任何既有结果。

---

## 1. 环境

| 项 | 版本 / 值 |
|---|---|
| Python | **3.12.13** |
| PyTorch | **2.11.0+cu128** |
| CUDA | 12.8（cuDNN 91900） |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.0（指标与 `precision_recall_curve`） |
| RDKit | 2026.03.3（仅数据准备阶段使用；训练/后处理/汇总不需要） |
| 环境名 | `spectra-db`（conda，路径示例 `D:\Miniconda3\envs\spectra-db`） |
| 硬件 | 单卡 **NVIDIA RTX 5070 Laptop 8 GB**（训练 batch 512 + AMP；后处理峰值显存约 1.8–2.0 GB） |
| 锁定文件 | `environment.yml`、`explicit-win-64.txt`、`requirements-pip-exact.txt`（其中 `torch==2.11.0+cu128`） |

**平台注意事项**：本项目在 Windows 上运行。脚本均带 `if __name__ == "__main__":` 守卫；`DataLoader(num_workers=4)` 依赖该守卫，**不要在交互式解释器中 import 后直接调用 `main()`**，否则会触发 `spawn` 引导错误。请始终以命令行方式执行。

---

## 2. 数据与标签

| 文件 | 内容 |
|---|---|
| `processed/qm9s/prepared/ir_float32.npy` | 光谱矩阵，`(129817, 3501)`，float32，**未归一化** |
| `processed/qm9s/prepared/ir_axis.npy` | 波数轴，500.0–4000.0 cm⁻¹，步长 1.0，单调递增 |
| `processed/qm9s/prepared/functional_group_labels.npy` | 标签矩阵，`(129817, 14)`，uint8 |
| `processed/qm9s/prepared/functional_group_label_names.json` | 14 类标签顺序（唯一权威） |
| `processed/qm9s/prepared/ir_scaffold_split_valid.npz` | scaffold split：`train/val/test` 索引数组 |

- **划分规模**：Train **103855** / Validation **12981** / Test **12981**，合计 129817；三集互斥、scaffold 零重叠。**不得重新划分数据**，也不得替换为 UV 的 103853 版本（后者排除了 `row_index 13998/14000` 两条 UV 全零谱，而这两条 IR 谱有效）。
- **标签顺序**（14 类）：aldehyde、alkene、alkyne、amide、amine、c_f_bond、carbonyl、ester、ether、hydroxyl、imine、ketone、nitrile、nitro。标签**非互斥**。
- **归一化**：存储的 NPY 未归一化；`QM9SSpectrumDataset(normalization="max")` 在 `__getitem__` 内按**每条谱自身的 max(|x|)** 归一。后处理与汇总必须沿用同一规则。
- **索引语义**：split 中的整数是 NPY 的行索引；Dataset 逐样本返回 `row_index` 用于与结构库对齐。

**质量事实**：IR 全库 0 条全零谱、0 个非有限值、0 个负值；`valid_ir` 掩码 129817/129817 全部有效。

---

## 3. 模型训练（每个 Seed 独立一次，共 6 次）

| 损失 | Seed | 脚本 | 输出目录 |
|---|---|---|---|
| BCE | 42 | `training/train_ir_full.py` | `runs/EXP-IR-001-FULL/` |
| BCE | 123 | `training/train_ir_seed123.py` | `runs/EXP-IR-001-SEED123/` |
| BCE | 2026 | `training/train_ir_seed2026.py` | `runs/EXP-IR-001-SEED2026/` |
| WBCE | 42 | `training/train_ir_weighted.py` | `runs/EXP-IR-002-FULL/` |
| WBCE | 123 | `training/train_ir_weighted_seed123.py` | `runs/EXP-IR-002-SEED123/` |
| WBCE | 2026 | `training/train_ir_weighted_seed2026.py` | `runs/EXP-IR-002-SEED2026/` |

```powershell
python training\train_ir_full.py
python training\train_ir_seed123.py
python training\train_ir_seed2026.py
python training\train_ir_weighted.py
python training\train_ir_weighted_seed123.py
python training\train_ir_weighted_seed2026.py
```

**固定超参数**（六模型一致，不得逐模型调整）：`SpectrumStructureModel`（1D-CNN，embedding 256，12 层 BatchNorm1d，14 输出）；AdamW（lr 1e-3、wd 1e-4）；batch 512；num_workers 4；max 50 epochs；Early Stopping patience 8；AMP fp16；`normalization="max"`；最优轮按 **Validation mAP**。

**WBCE 专有**：`pos_weight` 由 **Training split** 统计量算 `neg/pos`，再取 `sqrt` 并上限截断 20；`config.json` 记录 `final_pos_weights`。

**产物**：`config.json`、`training_log.csv`、`best_model.pt`、`test_metrics.json`、`test_per_label_metrics.csv`。
**实测耗时**：epoch 均值 15.8–16.7 s；实际 epoch 数 18–38，单模型约 **5–10 分钟**。

> 六份已有 checkpoint 的 SHA256 登记在 `ir_ablation_registry.md` §5。重放时若哈希不同，说明环境或数据已变，**不得**用新结果替换归档结果。

---

## 4. 后处理（每个模型独立执行；严格串行）

> **通用纪律**：四个阶段的脚本都**只写新目录**，目录已存在即报错退出；原始 checkpoint 全程只读；任一验收断言失败即中止且不落盘。

### 4.1 步骤 A：Full BN Diagnostic（6 次）

```powershell
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 42
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 123
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 2026
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 42
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 123
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 2026
```

- 输入：`runs/<run>/best_model.pt`、`test_metrics.json`、IR 数据与 split。
- 算法：`model.eval()` → 全部 BatchNorm `reset_running_stats()` + `momentum=None` + train 模式 → `no_grad`+autocast 下用 **Training split 103855 条**前向 203 批 → 恢复 momentum → eval。**只重估 BN buffer，不更新任何参数。**
- 输出：`runs/IR-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json`（含 `before`/`after` 的 Validation 与 Test 六指标、12 层 BN 摘要、`recalibration` 元数据）。
- 内建断言：基线复现（Test 指标 vs `test_metrics.json`，容差 0.002）、参数逐位不变、`train_samples_used == 103855`。
- **不做 Quick Diagnostic。**

### 4.2 步骤 B：保存 Training-only BN Recalibrated checkpoint（6 份）

```powershell
python training\save_ir_bn_checkpoint.py --loss bce  --seed 42
python training\save_ir_bn_checkpoint.py --loss bce  --seed 123
python training\save_ir_bn_checkpoint.py --loss bce  --seed 2026
python training\save_ir_bn_checkpoint.py --loss wbce --seed 42
python training\save_ir_bn_checkpoint.py --loss wbce --seed 123
python training\save_ir_bn_checkpoint.py --loss wbce --seed 2026
```

- 依赖：步骤 A 的诊断 JSON（用于保存后的一致性对照）。
- 断言：仅 36 个 BN buffer 变化、非 BN 张量逐位不变、12 层 `num_batches_tracked == 203`、全部 buffer 有限、`open("xb")` 防覆盖、保存后回读逐张量一致、源 checkpoint SHA256 前后一致、**重估模型的 Validation 指标与诊断记录 `after.validation` 差 ≤ 1e-3**。
- 输出：`runs/IR-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` 与 `bn_recalibration_manifest.json`。

### 4.3 步骤 C：Original BN 阈值校准（6 次）

```powershell
python training\calibrate_ir_original_thresholds.py --loss bce  --seed 42
python training\calibrate_ir_original_thresholds.py --loss bce  --seed 123
python training\calibrate_ir_original_thresholds.py --loss bce  --seed 2026
python training\calibrate_ir_original_thresholds.py --loss wbce --seed 42
python training\calibrate_ir_original_thresholds.py --loss wbce --seed 123
python training\calibrate_ir_original_thresholds.py --loss wbce --seed 2026
```

- 输入：`runs/<run>/best_model.pt`、`test_metrics.json`。
- 流程：在 Validation 上逐标签最大化 F1（`precision_recall_curve` 候选阈值）→ **在计算任何 Test 指标之前**写出 `optimal_thresholds.json` → 再计算 Test 指标。
- 输出目录：`runs/IR-THRESHOLD-CALIBRATION/<run_name>/`（6 文件）。

### 4.4 步骤 D：Recalibrated BN 阈值校准（6 次）

```powershell
python training\calibrate_ir_recalibrated_thresholds.py --loss bce  --seed 42
python training\calibrate_ir_recalibrated_thresholds.py --loss bce  --seed 123
python training\calibrate_ir_recalibrated_thresholds.py --loss bce  --seed 2026
python training\calibrate_ir_recalibrated_thresholds.py --loss wbce --seed 42
python training\calibrate_ir_recalibrated_thresholds.py --loss wbce --seed 123
python training\calibrate_ir_recalibrated_thresholds.py --loss wbce --seed 2026
```

- 输入：`runs/IR-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt` + manifest、步骤 A 的诊断 JSON。
- 额外断言：manifest 的 `source_sha256` / `saved_sha256` 与实际文件一致、`loss` / `seed` / `run_directory` 一致、`bn_state == "recalibrated"`、checkpoint 内 `bn_recalibration.source_sha256` 等于当前原 checkpoint 哈希、**0.5 基线指标与诊断 `after` 差 ≤ 1e-3**。
- **阈值由重估后的概率重新拟合，不复用 Original 的阈值文件。**
- 输出目录：`runs/IR-BN-THRESHOLD-CALIBRATION/<run_name>/`（6 文件）。

**四个阶段的输出文件（每个校准目录 6 个）**：`optimal_thresholds.json`、`validation_threshold_comparison.csv`、`threshold_comparison.csv`、`test_per_label_metrics_default.csv`、`test_per_label_metrics_calibrated.csv`、`threshold_calibration_summary.json`。

**实测耗时**：诊断 78.7–88.8 s/模型；BN 保存约 59 s/模型；校准 38.9–44.2 s/模型。

---

## 5. 消融汇总

```powershell
python training\summarize_ir_ablation.py
```

- 输出目录：`runs/IR-ABLATION-SUMMARY/`（**11 个文件**）；目录已存在则拒绝覆盖。
- **只读**：不加载 PyTorch、不执行前向推理、不重新拟合阈值、不修改 checkpoint；仅读取 JSON/CSV 并校验 12 个 checkpoint 的 SHA256。
- 写出前必须通过全部交叉核验（见 `ir_ablation_registry.md` §3），否则不创建输出目录。

**预期结果（末行）**：

```
PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows
```

**各文件行数**：`test_24_conditions` 24、`validation_24_conditions` 24、`mean_sd_8_conditions_each_split` 16、`per_label_672_rows` 672、`per_label_f1_levels` 14、`per_label_deltas` 224、`paired_effects` 84、`paired_effects_summary` 28、`condition_registry` 48、`audit_checks` 6、`README_summary.txt` 为叙述文件。

如需试跑到其它位置：`--out-dir <新目录>`（不得已存在）。

---

## 6. 复现验收检查清单

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | **Seed 与目录一致** | 每个 run 的 `config.seed` 与目录名（42/123/2026）一致，且与 checkpoint 内 `config.seed` 一致 |
| 2 | **Loss 与目录一致** | `EXP-IR-001-*` 的 `use_pos_weight == False`；`EXP-IR-002-*` 为 `True` 且 `pos_weight_transform=="sqrt"`、`pos_weight_cap==20.0` |
| 3 | **标签顺序** | `config.label_names` 与 checkpoint `label_names` 均等于 `functional_group_label_names.json`（14 类，顺序一致）；12 份逐标签 CSV 的行序与之一致 |
| 4 | **checkpoint 来源** | 诊断 JSON 的 `source_checkpoint` 指向 `runs/<run_name>/best_model.pt`；`experiment` 与 `requested_seed` 正确 |
| 5 | **manifest SHA256** | `source_sha256` == 原 checkpoint 实测哈希；`saved_sha256` == 重估 checkpoint 实测哈希（`audit_checks.csv` 的 `manifest_hash_chain_verified` 全 True） |
| 6 | **BN 参数不变性** | 诊断 `model_weights_unchanged == True`；保存日志含 `PASS: trainable parameters unchanged`；仅 36 个 BN buffer 变化 |
| 7 | **阈值来源** | Original 与 Recalibrated 各用各自目录的 `optimal_thresholds.json`；`checkpoint_sha256` 字段与对应 checkpoint 一致；`frozen_before_test_metrics == True` |
| 8 | **条件数量** | Validation 24 + Test 24 = 48 条；唯一条件键 48 个；聚合组 16 个 |
| 9 | **逐标签行数** | 672 行；每条件每 split 恰好 14 行且标签顺序正确 |
| 10 | **Test 隔离** | 阈值 JSON 在 Test 指标计算前写出；诊断与校准脚本中不存在"依据 Test 改变流程/参数"的分支；本协议的验收不依赖 Test 数值是否理想 |
| 11 | **阈值无关指标一致** | 同一 (模型, BN) 的两条条件 mAP/AUROC 完全相同（差 < 1e-9） |

---

## 7. 注意事项

1. **Windows 多进程内存风险**：`num_workers=4` + 1.8 GB 光谱 mmap 在运行期会把物理内存推向高位（实测运行期"可用物理内存"采样可低至数 MB，属文件缓存占用）。**所有阶段必须严格串行**，一次只跑一个模型；不要在跑诊断/校准的同时开其它重内存任务。若系统"已提交内存"持续逼近上限，应停止后续任务而非降低 `num_workers`/`batch_size`——后者会改变已验证的推理配置，破坏与归档结果的可比性。
2. **严格串行**：前一模型必须正常结束、产物校验通过、Python 子进程退出后，才能启动下一个。
3. **防覆盖**：四个后处理脚本均以"目标目录不存在"为前提。旧的单 Seed 校准脚本已**废弃归档**：实现迁至 `archive/2026-11-15/calibrate_ir_thresholds.py`（该目录不入 Git，可经 `git show 9e4b743:training/calibrate_ir_thresholds.py` 从历史恢复），原路径仅保留弃用入口 `training/calibrate_ir_thresholds.py`，直接运行会立即以非零状态退出且不做任何事。**不要重跑归档版本**：它以 `"w"` 模式写盘、硬编码 Seed 42、推理用 1024/0，会覆盖 `runs/EXP-IR-001-FULL/` 的 4 个历史校准文件。
4. **大文件不进 Git**：`.pt` 被 `.gitignore` 排除。原始与重估共 12 份 checkpoint（约 140 MB）必须**在仓库外备份**，并以 manifest 的 `saved_sha256` 作为备份校验依据。
5. **历史实验保护**：`runs/IR-v1-FINAL/`（2026-08-14）与 `runs/EXP-IR-001-FULL/` 的 4 个历史校准文件均**不得改写**；它们不参与 24 条件矩阵。
6. **Test 不参与方法选择**：阈值只在 Validation 拟合；配置选择只依据 Validation；Test 仅用于评价、记录与一致性核验。**不得**根据 Test 结果回头看改阈值、BN 处理或 Seed 选择。
7. **n = 3 的统计边界**：三 Seed 的 Mean ± 样本 SD 只描述训练随机性，不是置信区间，也不支持显著性断言。
8. **WBCE 非默认**：Weighted BCE 保留为消融对照；其 Validation/Test Micro-F1 在八个配置中一致低于对应 BCE 配置，因此**不作为**默认损失。
9. **BN 重估非默认**：主推理方案使用 Original BN（见 `IR_v1_final_decision.md`），不预设 BN 重估更优；重估 checkpoint 及其阈值必须成对使用。
10. **数据不得重划分**：`ir_scaffold_split_valid.npz` 是唯一权威划分；任何"重新 split"都会使归档结果不可比。
