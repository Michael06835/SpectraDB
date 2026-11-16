# Raman v1 实验复现协议

> **状态**：与 `Raman_v1_final_decision.md`（决策）、`raman_ablation_registry.md`（溯源）配套的**可执行复现指南**。
> **范围**：QM9S 计算域 Raman → 14 类官能团多标签的单模态 1D-CNN 基线；不包含多模态融合。
> **性质**：Raman v1 已冻结；本协议描述**如何从零重放**，不改变任何既有结果。
> **对照协议**：`ir_repro_protocol.md`、`uvspec15_repro_protocol.md`（步骤编号与验收结构同构）。

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
| 硬件 | 单卡 **NVIDIA RTX 5070 Laptop 8 GB**（训练 batch 512 + AMP；后处理峰值显存约 2.0 GB） |
| 锁定文件 | `environment.yml`、`explicit-win-64.txt`、`requirements-pip-exact.txt` |

**平台注意事项**：本项目在 Windows 上运行。所有脚本均带 `if __name__ == "__main__":` 守卫；`DataLoader(num_workers=4)` 依赖该守卫，**不要在交互式解释器中 import 后直接调用 `main()`**，否则会触发 `spawn` 引导错误。请始终以命令行方式执行。

---

## 2. 数据与标签

| 文件 | 内容 |
|---|---|
| `processed/qm9s/prepared/raman_float32.npy` | 光谱矩阵，`(129817, 3501)`，float32，**未归一化** |
| `processed/qm9s/prepared/raman_axis.npy` | 波数轴，500.0–4000.0 cm⁻¹，步长 1.0，单调递增 |
| `processed/qm9s/prepared/functional_group_labels.npy` | 标签矩阵，`(129817, 14)`，uint8 |
| `processed/qm9s/prepared/functional_group_label_names.json` | 14 类标签顺序（唯一权威） |
| `processed/qm9s/prepared/raman_scaffold_split_valid.npz` | scaffold split：`train/val/test` 索引数组 |

- **划分规模**：Train **103855** / Validation **12981** / Test **12981**，合计 129817；三集互斥、scaffold 零重叠。**不得重新划分数据**。该索引与 `ir_scaffold_split_valid.npz` **逐元素相同**（已用 `array_equal` 独立验证），不得替换为 UV 的 103853 版本。
- **标签顺序**（14 类）：aldehyde、alkene、alkyne、amide、amine、c_f_bond、carbonyl、ester、ether、hydroxyl、imine、ketone、nitrile、nitro。标签**非互斥**。
- **归一化**：存储的 NPY 未归一化；`QM9SSpectrumDataset(normalization="max")` 在 `__getitem__` 内按**每条谱自身的 max(|x|)** 归一。后处理与汇总必须沿用同一规则。
- **索引语义**：split 中的整数是 NPY 的行索引；Dataset 逐样本返回 `row_index` 用于与结构库对齐。

---

## 3. 模型训练（已冻结；重放仅供核对）

| 损失 | Seed | 脚本 | 输出目录 |
|---|---|---|---|
| BCE | 42 | `training/train_raman_full.py` | `runs/EXP-RAMAN-001-FULL/` |
| BCE | 123 | `training/train_raman_seed123.py` | `runs/EXP-RAMAN-001-SEED123/` |
| BCE | 2026 | `training/train_raman_seed2026.py` | `runs/EXP-RAMAN-001-SEED2026/` |
| WBCE | 42 | `training/train_raman_weighted.py` | `runs/EXP-RAMAN-002-FULL/` |
| WBCE | 123 | `training/train_raman_weighted_seed123.py` | `runs/EXP-RAMAN-002-SEED123/` |
| WBCE | 2026 | `training/train_raman_weighted_seed2026.py` | `runs/EXP-RAMAN-002-SEED2026/` |

**固定超参数**（六模型一致，不得逐模型调整）：`SpectrumStructureModel`（1D-CNN，embedding 256，12 层 BatchNorm1d，14 输出）；AdamW（lr 1e-3、wd 1e-4）；batch 512；num_workers 4；max 50 epochs；Early Stopping patience 8；AMP fp16；`normalization="max"`；最优轮按 **Validation mAP**。

**WBCE 专有**：`pos_weight` 由 **Training split** 统计量算 `neg/pos`，再取 `sqrt` 并上限截断 20；`config.json` 记录 `final_pos_weights`（三份 WBCE checkpoint 完全一致，nitro 取到上限 20.0）。

**产物**：`config.json`、`training_log.csv`、`best_model.pt`、`test_metrics.json`、`test_per_label_metrics.csv`。
**实测 epoch 数**：23–36；单模型约 **5–10 分钟**。

> **警告**：六份 Original checkpoint 的 SHA256 登记在 `raman_ablation_registry.md` §5。重放时若哈希不同，说明环境或数据已变，**不得**用新结果替换归档结果。**Raman v1 已冻结，禁止为本轮任何目的重新训练。**

---

## 4. 后处理（每个模型独立执行；严格串行）

> **通用纪律**：四个后处理阶段的脚本都**只写新目录**，目录已存在即报错退出；来源 checkpoint 与诊断/校准摘要全程只读。写盘前的验收断言失败即中止且不落盘；写盘后发现第 09 项失败时的文件状态另有规定，见 §4.2。

### 4.1 步骤 A：Full BN Diagnostic（6 次）

```powershell
python training\diagnose_raman_batchnorm_full.py --loss bce  --seed 42
python training\diagnose_raman_batchnorm_full.py --loss bce  --seed 123
python training\diagnose_raman_batchnorm_full.py --loss bce  --seed 2026
python training\diagnose_raman_batchnorm_full.py --loss wbce --seed 42
python training\diagnose_raman_batchnorm_full.py --loss wbce --seed 123
python training\diagnose_raman_batchnorm_full.py --loss wbce --seed 2026
```

- 输入：`runs/<run>/best_model.pt`、`test_metrics.json`、Raman 数据与 split。
- 算法：`model.eval()` → 全部 BatchNorm `reset_running_stats()` + `momentum=None` + train 模式 → `no_grad`+autocast 下用 **Training split 103855 条**前向 203 批 → 恢复 momentum → eval。**只重估 BN buffer，不更新任何参数。**
- 输出：`runs/RAMAN-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json`（含 `before`/`after` 的 Validation 与 Test 六指标、12 层 BN 摘要、`recalibration` 元数据）。
- 内建断言：基线复现（Test 指标 vs `test_metrics.json`，容差 0.002）、参数逐位不变、`train_samples_used == 103855`。
- **不做 Quick Diagnostic。**

### 4.2 步骤 B：保存 Training-only BN Recalibrated checkpoint（6 份，gate policy v5）

```powershell
python training\save_raman_bn_checkpoint.py --loss bce  --seed 42
python training\save_raman_bn_checkpoint.py --loss bce  --seed 123
python training\save_raman_bn_checkpoint.py --loss bce  --seed 2026
python training\save_raman_bn_checkpoint.py --loss wbce --seed 42
python training\save_raman_bn_checkpoint.py --loss wbce --seed 123
python training\save_raman_bn_checkpoint.py --loss wbce --seed 2026
```

**两道独立门，不再用单一布尔掩盖差异：**

- **Gate A（制品完整性，10 项确定性检查）**：来源 SHA256；40 个非 BN 状态张量逐位不变；可训练参数逐位不变；36/36 BN buffer 合法变化；12 层名称与 `num_batches_tracked=203` 对齐诊断；Training-only 范围（103855/203）与诊断互证；buffer 有限且 `running_var > 0`；BN 摘要 vs 诊断（rtol **1e-3**，IR 同标准）；写盘后回读逐张量一致；Original checkpoint 前后 SHA256 不变。

  **失败时的文件状态必须按两个阶段区分**（第 09 项依赖写盘后的文件，因此两阶段的后果不同）：

  | 阶段 | 失败项 | 磁盘状态 |
  |---|---|---|
  | **写盘前**（第 01–08、10 项） | 任一失败 | **不生成正式 checkpoint，不写 manifest**；目录通常尚未创建 |
  | **写盘后**（第 09 项 `09_checkpoint_reload_integrity`） | 回读比对不一致 | **可能留下已写盘但未获完整验收的孤儿 checkpoint**（无 manifest） |

  写盘后发现第 09 项失败时：**禁止直接覆盖、禁止重复执行同一命令**；应先检查该文件是否存在、其 SHA256、mtime 与运行日志，确认是"写盘损坏"还是"比较逻辑问题"后再决定后续动作（移入 `archive/` 或另行审计）。manifest 只有在第 09 项完成之后，才按既有状态机（`ELIGIBLE` / `NOT_ELIGIBLE`）写入；**不允许**在 09 未完成时补写 manifest。
- **Gate B（数值复现）**：官方 FP16 六项 Validation 指标 vs `after.validation`，**全部 1e-3**。PASS/FAIL 二态；FAIL 时 manifest 记为 `ARTIFACT_PASS__NUMERIC_FAIL`、`ablation_release=NOT_ELIGIBLE`，退出码 3（**不是崩溃**）。

**制品与放行的分离**：manifest 存在 ≠ 获准。数值未通过时，human release 需单独落 `manual_approval.json`（`--approve-release --manual-approval-note "..." --approve-checkpoint-sha256 <sha>`），绑定单一 checkpoint SHA256；`OTHER_ANOMALY` 类**不可放行**。

**孤儿 checkpoint 的 finalize（仅用于解释历史处理方式；不重算 BN、不改写文件）**

`^` 是 cmd.exe 的续行符，在 PowerShell 中不可用。以下是**一行完整命令**：

```powershell
python training\save_raman_bn_checkpoint.py --loss bce --seed 42 --finalize-existing --expect-checkpoint-sha256 a9f9d55f1239abd7c91f52a4900f3ba5f23a9f4ced799b626802d956197022f5
```

（若确实需要换行，PowerShell 使用反引号 `` ` `` 结尾续行，或使用 splatting；不要在 PowerShell 里用 `^`。）

> **重要**：该命令**仅用于解释历史孤儿 checkpoint 的处理方法**。当前 Seed 42 已完成验收（`bn_recalibration_manifest.json` 存在、`ARTIFACT_PASS__NUMERIC_PASS`、`ELIGIBLE`），**禁止对已冻结目录重新执行 finalize**——脚本会以"manifest 已存在"拒绝，且任何绕过该拒绝的操作都会破坏冻结状态。

加 `--dry-run` 可在**纯 CPU** 下只跑 Gate A（不触发 GPU、不写文件）；该开关同样适用于上面的 finalize 命令形态。

**实测耗时**：约 90–150 s/模型（BN 203 批 + Gate B 一次 Validation 前向）。

> **历史事故（已修复，勿重演）**：门禁曾在写盘前评估尚处于 `pending` 的第 09 项，导致 `cmd_save` 必然以 `09_checkpoint_reload_integrity` 失败（BCE seed 123 首次暴露）。修复为写盘前只排除 09、写盘后以完整清单判定；09 占位仍为 FAIL，遗漏完成调用依旧 fail-closed。回归测试见 `training/test_raman_bn_reload_integrity.py`。

### 4.3 步骤 C：Original BN 阈值校准（6 次）

```powershell
python training\calibrate_raman_original_thresholds.py --loss bce  --seed 42
python training\calibrate_raman_original_thresholds.py --loss bce  --seed 123
python training\calibrate_raman_original_thresholds.py --loss bce  --seed 2026
python training\calibrate_raman_original_thresholds.py --loss wbce --seed 42
python training\calibrate_raman_original_thresholds.py --loss wbce --seed 123
python training\calibrate_raman_original_thresholds.py --loss wbce --seed 2026
```

- 输入：`runs/<run>/best_model.pt`（**Original**）、`test_metrics.json`。
- 流程：Validation 上逐标签最大化 F1 → **在计算任何 Test 指标之前**写出 `optimal_thresholds.json` → 再计算 Test 指标。
- 每个 split **只推理一次**；0.5 与 Val-F1 两个条件共用同一组概率，概率数组连同 SHA256/checkpoint SHA256/split/label 顺序/口径落盘。
- Test 侧复现检查按 v5 设计**只记录**（`test_anomaly`），不阻断；一致性由步骤 E 的 0.0025 统一核验兜底。
- 输出目录：`runs/RAMAN-THRESHOLD-CALIBRATION/<run_name>/`（**9 文件**）。

### 4.4 步骤 D：Recalibrated BN 阈值校准（6 次）

```powershell
python training\calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 42
python training\calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 123
python training\calibrate_raman_recalibrated_thresholds.py --loss bce  --seed 2026
python training\calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 42
python training\calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 123
python training\calibrate_raman_recalibrated_thresholds.py --loss wbce --seed 2026
```

- 输入：`runs/RAMAN-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt` + manifest、步骤 A 的诊断 JSON。
- 额外断言：manifest 的 `source_sha256` / `saved_sha256` 与实际一致、`loss`/`seed`/`run_directory` 一致、`bn_state == "recalibrated"`、`artifact_integrity == PASS`、**`ablation_release` 允许继续**（`ELIGIBLE`，或 `NOT_ELIGIBLE` 但有 sha 绑定的 `manual_approval.json`）；checkpoint 内 `bn_recalibration.source_sha256` 等于当前原 checkpoint 哈希。
- 审计分流：`validation_release` 必须 PASS（Validation 侧无放行入口）；`test_anomaly` 仅记录。
- **阈值由重估后的概率重新拟合，不复用 Original 的阈值文件。**
- 输出目录：`runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/`（**9 文件**）。

**两个校准阶段合计 12 次**；每次都会在 `runs/RAMAN-NUMERIC-AUDIT/<run_name>/` 写一份**首次观测即定论**的审计记录（`original_thresholds.json` / `recalibrated_thresholds.json`），后续运行只读取、不覆盖。

**实测耗时**：校准约 60–90 s/模型。

---

## 5. 消融汇总

```powershell
python training\summarize_raman_ablation.py
```

- 输出目录：`runs/RAMAN-ABLATION-SUMMARY/`（**11 个文件**）；目录已存在则拒绝覆盖。
- **对来源数据只读**：不加载 PyTorch、不执行前向推理、不重新拟合阈值、不修改任何 checkpoint / 校准摘要 / 诊断 JSON；仅读取 JSON/CSV 并校验 12 个 checkpoint 的 SHA256。
- **但脚本本身会写盘**：它会**新建**上述汇总目录并写入 11 个文件，因此"整个过程零写盘"的说法不成立。既有 `runs/RAMAN-ABLATION-SUMMARY/` 不允许覆盖，**也不允许删除或替换**；如需验证性重放，请用 `--out-dir` 指向一个**尚不存在**的新目录。
- 写出前必须通过全部交叉核验（见 `raman_ablation_registry.md` §3），否则不创建输出目录。

**预期结果（末行）**：

```
PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows
```

**各文件行数**：`test_24_conditions` 24、`validation_24_conditions` 24、`mean_sd_8_conditions_each_split` 16、`per_label_672_rows` 672、`per_label_f1_levels` 14、`per_label_deltas` 224、`paired_effects` 84、`paired_effects_summary` 28、`condition_registry` 48、`audit_checks` 6、`README_summary.txt` 为叙述文件。

如需验证性重放：`--out-dir <新目录>`（必须尚不存在，且不要指向现有汇总目录）。

---

## 6. 复现验收检查清单

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | **Seed 与目录一致** | 每个 run 的 `config.seed` 与目录名（42/123/2026）一致，且与 checkpoint 内 `config.seed` 一致 |
| 2 | **Loss 与目录一致** | `EXP-RAMAN-001-*` 的 `use_pos_weight == False`；`EXP-RAMAN-002-*` 为 `True` 且 `pos_weight_transform=="sqrt"`、`pos_weight_cap==20.0` |
| 3 | **标签顺序** | `config.label_names` 与 checkpoint `label_names` 均等于 `functional_group_label_names.json`（14 类）；12 份逐标签 CSV 的行序与之一致 |
| 4 | **checkpoint 来源** | 诊断 JSON 的 `source_checkpoint` 指向 `runs/<run_name>/best_model.pt`；`experiment` 与 `requested_seed` 正确 |
| 5 | **manifest SHA256** | `source_sha256` == 原 checkpoint 实测哈希；`saved_sha256` == 重估 checkpoint 实测哈希；`audit_checks.csv` 的 `manifest_hash_chain_verified` 全 True |
| 6 | **Gate A** | 六份 manifest 的 `artifact_integrity.status == "PASS"`，且 10 项检查均为 `true` |
| 7 | **Gate B** | 六份 `numeric_reproduction.status == "PASS"`、`acceptance.status == "ARTIFACT_PASS__NUMERIC_PASS"`、`ablation_release.status == "ELIGIBLE"` |
| 8 | **BN 参数不变性** | 诊断 `model_weights_unchanged == True`；仅 36 个 BN buffer 变化；`num_batches_tracked == 203` |
| 9 | **阈值来源** | Original 与 Recalibrated 各用各自目录的 `optimal_thresholds.json`；`checkpoint_sha256` 字段与对应 checkpoint 一致；`frozen_before_test_metrics == True` |
| 10 | **概率溯源** | 每个校准目录的 `probability_artifacts.json` 的两个数组 SHA256 与 `.npy` 实测一致；`meta.checkpoint_sha256` 与对应 checkpoint 一致 |
| 11 | **数值审计** | `runs/RAMAN-NUMERIC-AUDIT/` 下 12 份记录 `status == "PASS"`、`exceeding_metrics == []` |
| 12 | **条件数量** | Validation 24 + Test 24 = 48 条；唯一条件键 48 个；聚合组 16 个；逐标签 672 行 |
| 13 | **Test 隔离** | 阈值 JSON 在 Test 指标计算前写出；诊断与校准脚本中不存在"依据 Test 改变流程/参数"的分支；本协议的验收不依赖 Test 数值是否理想 |
| 14 | **阈值无关指标一致** | 同一 (模型, BN) 的两条条件 mAP/AUROC 完全相同（差 < 1e-9） |

---

## 7. 注意事项

1. **Windows 多进程内存风险**：`num_workers=4` + 1.8 GB 光谱 mmap 在运行期会把物理内存推向高位。**所有阶段必须严格串行**，一次只跑一个模型；不要并行、不要在跑的时候开其它重内存任务。若"已提交内存"持续逼近上限，应停止后续任务而非降低 `num_workers`/`batch_size`——后者会改变已验证的推理配置，破坏与归档结果的可比性。
2. **严格串行**：前一模型必须正常结束、产物校验通过、Python 子进程退出后，才能启动下一个。
3. **防覆盖**：所有后处理脚本均以"目标目录不存在"为前提；已完成目录一律拒绝重跑。中断目录（只剩 `optimal_thresholds.json`）续跑必须显式 `--manual-approval-note "..."`，且**复用已冻结阈值、不重拟合**；若冻结概率数组存在，续跑**完全不做推理**。
4. **大文件不进 Git**：`.pt` 被 `.gitignore` 排除。原始与重估共 12 份 checkpoint（约 **147 MB**）必须**在仓库外备份**，并以 manifest 的 `saved_sha256` 作为备份校验依据（见 `Raman_v1_final_decision.md` 与备份清单）。
5. **历史实验保护**：`runs/EXP-RAMAN-001-FULL/` 的 4 个历史校准文件（2026-08-14，1024/0 口径）**不得改写**；它们不参与 24 条件矩阵。
6. **禁止直接执行（会覆盖冻结结果）**：`training/evaluate_raman_*.py`（6 个）以 `"w"` 模式**原地重写**对应 run 目录的 `test_metrics.json` 与 `test_per_label_metrics.csv`。如需复核，请先把 checkpoint 复制到临时目录。
7. **Test 不参与方法选择**：阈值只在 Validation 拟合；配置选择只依据 Validation；Test 仅用于评价、记录与一致性核验。**不得**根据 Test 结果回头看改阈值、BN 处理或 Seed 选择。
8. **FP16 数值复现的已知表现**：已实测到**同一 checkpoint 在不同进程间的官方 FP16 mAP 存在差异**（BCE seed 42 的 T1/T2 对照：0.945208 vs 0.946288，差 **1.080e-3**），而同一两次运行的 FP32 sigmoid 结果稳定在 2e-6 量级、进程内重复推理概率逐位一致；逐标签比对显示该差异主要由 `nitro` 的 AP 承载。另需注意，Phase 3 那次 **Gate B 失败**用的是另一对读数（0.946294 vs 0.945208，差 **1.086e-3**），不可与上述跨进程测量混用（详见 `Raman_v1_final_decision.md` §8.6 的"两个必须区分的数值"）。**一种合理解释**是：FP16 sigmoid 的输出落在量化格点上，近邻预测值的微小变化可能改变并列样本的相对次序，并在 AP 这类秩统计量上被放大；但具体算子级根因与误差分布**均未证明**（详见 `Raman_v1_final_decision.md` §8.6 的"尚未证明"）。这是**参考记录方式的已知性质**，不是制品损坏。**不得**以任何形式放宽 1e-3 判据，也不得"重跑到通过"。若需归因，使用非门控的 `training/verify_raman_bn_reproducibility.py`（固定算法 + FP32 sigmoid + 概率逐位比对），其结果**不得**替代官方指标。
9. **n = 3 的统计边界**：三 Seed 的 Mean ± 样本 SD 只描述训练随机性，不是置信区间，也不支持显著性断言。
10. **数据不得重划分**：`raman_scaffold_split_valid.npz` 是唯一权威划分；任何"重新 split"都会使归档结果不可比。
