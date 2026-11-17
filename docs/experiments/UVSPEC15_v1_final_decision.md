# UVSPEC15 v1 最终科研决策记录（冻结稿）

> **文档性质**：实验完成后形成的配置决策记录（exploratory, post-hoc decision record），**不是事先注册的实验方案**。
> **冻结日期**：2026-11-15（决策内容沿用 2026-11-14 归档的 `UVSPEC15_v1_experimental_strategy.md`，本文件为其正式决策记录形式）
> **适用范围**：QM9S 计算域 UV-Vis 单模态 1D-CNN → UVSPEC15 的 15 类结构语义标签识别。
> **不适用范围**：实验域（NIST 等）泛化、多模态融合、完整分子结构 Top-k 推断、开放集拒识与可解释性。
> **上游文档**：`UVSPEC15_v1_experimental_strategy.md`（实验策略与决策依据）、`uvspec15_ablation_registry.md`（溯源）、`uvspec15_repro_protocol.md`（复现）、`runs/UVSPEC15-ABLATION-SUMMARY/`（汇总）。

---

## 1. 研究任务、数据与评价协议

**任务**：由 QM9S 计算紫外-可见吸收光谱预测 **UVSPEC15 的 15 类二值结构语义标签**（共轭体系、芳香体系及与电子结构相关的特征）。该任务是后续多模态融合的 UV 专属表征基础，**不等同于**完整分子结构识别。

**数据**

| 项 | 值 |
|---|---|
| 光谱矩阵 | `processed/qm9s/prepared/uvvis_float32.npy`，`(129817, 701)`，float32，1–15 eV |
| 标签矩阵 | `processed/qm9s/prepared/uvspec15_labels.npy`，`(129817, 15)`，float32 |
| 标签顺序 | `processed/qm9s/prepared/uvspec15_label_names.json`（唯一权威，15 类） |
| 归一化 | 逐谱最大绝对值归一化（`normalization="max"`），在 Dataset 内施加 |
| 划分 | `processed/qm9s/prepared/uvvis_scaffold_split_valid.npz`，**Train 103853 / Validation 12981 / Test 12981**；三集互斥、scaffold 零重叠 |

> **划分口径说明**：UV 的 Train 为 **103853**，比全库 129817 少 2 条（`row_index 13998`、`14000` 为全零 UV 谱，已由 `valid_uvvis` 掩码排除）。这是 UV 的正式划分，**不得**替换为 IR 的 103855 版本。

**15 类标签**：acyclic_conjugated_system、alpha_beta_unsaturated_carbonyl、aryl_carbonyl、cyclic_conjugated_system、donor_acceptor_same_conjugated_system、has_acceptor_substituted_pi_system、has_aromatic_system、has_conjugation、has_donor_substituted_pi_system、has_nonaromatic_conjugation、heteroaromatic、mixed_aromatic_nonaromatic_conjugation、pi_conjugated_carbonyl、pi_conjugated_imine、pi_conjugated_nitrile。标签**非互斥**（同一分子可同时属于多个共轭/芳香子体系类别）。

> **列顺序说明（2026-11-17 勘误）**：本段按**字母序**列举标签，仅为叙述顺序，**不是**模型输出列顺序。真实列顺序/输出头顺序以 `processed/qm9s/prepared/uvspec15_label_names.json` 为唯一权威；不得根据本段顺序或任何文档中的标签列举顺序推断张量列索引。

**评价指标**：Micro-F1、Macro-F1、Micro-Precision、Micro-Recall、mAP、Macro-AUROC；另记录逐标签 Precision/Recall/F1。mAP 与 Macro-AUROC 为阈值无关指标。

**数据使用纪律**：Training 用于参数训练与 BN 统计量重估；Validation 用于 Early Stopping、方法比较与逐标签阈值拟合；Test 仅用于最终评价、记录与一致性核验。本文档第 6 节的配置选择**全部基于 Validation**；Test 数值在第 7 节独立报告，**未用于**改变任何配置、阈值或 Seed 选择。

---

## 2. 六个独立训练模型

| 模型目录 | 损失 | Seed | 最优 epoch | Validation mAP（选轮指标） |
|---|---|---:|---:|---:|
| `runs/EXP-UVVIS-003-UVSPEC15-FULL` | 普通 BCE | 42 | 6 | 0.517480 |
| `runs/EXP-UVVIS-003-UVSPEC15-SEED123` | 普通 BCE | 123 | 4 | 0.450397 |
| `runs/EXP-UVVIS-003-UVSPEC15-SEED2026` | 普通 BCE | 2026 | 10 | 0.537150 |
| `runs/EXP-UVVIS-004-UVSPEC15-WBCE` | Weighted BCE | 42 | 3 | 0.462269 |
| `runs/EXP-UVVIS-004-UVSPEC15-WBCE-SEED123` | Weighted BCE | 123 | 5 | 0.497287 |
| `runs/EXP-UVVIS-004-UVSPEC15-WBCE-SEED2026` | Weighted BCE | 2026 | 8 | 0.502330 |

**共同训练配置**（六模型一致，未做任何逐模型调整）：

- 模型：`SpectrumStructureModel`，1D-CNN，embedding 256，12 层 BatchNorm1d，15 维输出头。
- 优化：AdamW（lr 1e-3，weight decay 1e-4），batch 512，num_workers 4，最多 50 epochs，Early Stopping patience 8，AMP（fp16）。
- 最优轮选择：Validation mAP（阈值无关）。
- Weighted BCE：`sqrt(pos_weight)` 变换、权重上限 20；`pos_weight` 仅由 Training split 统计量计算。
- 实际训练轮数明显少于 IR（最优 epoch 3–10），反映 UV 任务在 **UV 专属语义标签**上的收敛更快，但也意味着早停窗口较短，Seed 间差异需谨慎解读。

**Seed 政策**：Seed 是重复实验因素，用于估计训练稳定性，不是需要优化的超参数。论文统计必须覆盖三个 Seed；**不依据 Test 分数选择 Seed**。Seed 42 为预先固定的 canonical 演示模型。

---

## 3. BN 诊断、BN 重估与阈值校准的完成状态

| 环节 | 数量 | 产物位置 |
|---|---:|---|
| Full BN Diagnostic | **6/6** | `runs/UVSPEC15-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| Training-only BN 重估 checkpoint + manifest | **6/6** | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| BN 后阈值校准 | **6/6** | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/threshold_calibration/` |
| Original BN 阈值校准 | **6/6** | 训练 run 目录内（见下） |
| 阈值校准合计 | **12/12** | — |

**目录布局（与 IR 不同，须注意）**：UVSPEC15 的 Original 阈值校准结果**存放在训练 run 目录内**（`optimal_thresholds.json`、`validation_threshold_comparison.csv`、`threshold_comparison.csv`、`test_per_label_metrics_calibrated.csv`、`threshold_calibration_summary.json`，共 5 个文件）；Recalibrated 的对应结果存放在 `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/threshold_calibration/`。IR 则使用两个独立根目录。**跨模态脚本复用必须先改路径。**

**BN 重估协议**：在原始 checkpoint 的独立深拷贝上进行；`reset_running_stats()` 后将 `momentum` 置为 `None`（按 batch 等权累计平均），仅将 BatchNorm 层置于 train 模式，用**完整 Training split 的 103853 条样本**前向一遍（203 批），恢复 momentum 后回到 eval。可训练参数与优化器状态不变，原始 checkpoint 全程只读。未执行 Quick Diagnostic。

**安全检查（六份全部通过）**：仅 BN buffer 允许变化；12 层 `num_batches_tracked` 由 `最优 epoch × 203` 变为 203（实测 1218 / 812 / 2030 / 609 / 1015 / 1624 → 203）；所有 running statistics 有限；回读一致；manifest 的 `source_sha256` 与 `saved_sha256` 分别与原始/重估 checkpoint 一致。

**阈值校准协议**：对每一种 BN 状态分别、独立地在 Validation 上逐标签最大化 F1；阈值在计算任何 Test 指标之前落盘；Test 不参与拟合。Recalibrated 阈值由重估后的概率重新拟合，未复用 Original 阈值。

**审计结果**：`runs/UVSPEC15-ABLATION-SUMMARY/audit_checks.csv` 的 6 行 `original_baseline_reproduced`、`bn_saved_diagnostic_reproduced`、`label_order_verified` **全部为 True**。

---

## 4. 24 条件消融矩阵与三 Seed 统计

矩阵维度：2 损失 × 3 Seed × 2 BN 状态 × 2 阈值策略 = **24 个评估条件**；Validation 24 条 + Test 24 条 = **48 条 split-level 记录**；逐标签 15 × 24 × 2 = **720 行**。

**Validation，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | Micro-P | Micro-R | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | 0.5743 ± 0.0588 | 0.4126 ± 0.0463 | 0.5681 ± 0.1371 | 0.6023 ± 0.0494 | 0.5017 ± 0.0455 | 0.9119 ± 0.0168 |
| BCE | Original | Val-F1 | 0.6005 ± 0.0259 | 0.4908 ± 0.0393 | 0.5646 ± 0.0286 | 0.6416 ± 0.0288 | 0.5017 ± 0.0455 | 0.9119 ± 0.0168 |
| BCE | Recalibrated | 0.5 | 0.6167 ± 0.0048 | 0.4355 ± 0.0243 | 0.7003 ± 0.0202 | 0.5516 ± 0.0192 | 0.5321 ± 0.0163 | 0.9270 ± 0.0014 |
| **BCE** | **Recalibrated** | **Val-F1** | **0.6283 ± 0.0059** | **0.5229 ± 0.0087** | 0.5890 ± 0.0222 | 0.6741 ± 0.0148 | 0.5321 ± 0.0163 | 0.9270 ± 0.0014 |
| WBCE | Original | 0.5 | 0.5357 ± 0.0870 | 0.3867 ± 0.0666 | 0.4911 ± 0.1653 | 0.6616 ± 0.1433 | 0.4873 ± 0.0218 | 0.9154 ± 0.0033 |
| WBCE | Original | Val-F1 | 0.5968 ± 0.0169 | 0.4889 ± 0.0139 | 0.5486 ± 0.0243 | 0.6548 ± 0.0139 | 0.4873 ± 0.0218 | 0.9154 ± 0.0033 |
| WBCE | Recalibrated | 0.5 | 0.6106 ± 0.0095 | 0.5042 ± 0.0134 | 0.5364 ± 0.0218 | 0.7102 ± 0.0290 | 0.5285 ± 0.0214 | 0.9282 ± 0.0040 |
| WBCE | Recalibrated | Val-F1 | 0.6241 ± 0.0136 | 0.5214 ± 0.0110 | 0.5840 ± 0.0258 | 0.6708 ± 0.0031 | 0.5285 ± 0.0214 | 0.9282 ± 0.0040 |

**Test，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | Micro-P | Micro-R | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | 0.6136 ± 0.0845 | 0.4397 ± 0.0744 | 0.6136 ± 0.1548 | 0.6302 ± 0.0305 | 0.5389 ± 0.0599 | 0.9259 ± 0.0204 |
| BCE | Original | Val-F1 | 0.6325 ± 0.0418 | 0.5104 ± 0.0545 | 0.5959 ± 0.0474 | 0.6746 ± 0.0424 | 0.5389 ± 0.0599 | 0.9259 ± 0.0204 |
| BCE | Recalibrated | 0.5 | 0.6580 ± 0.0154 | 0.4721 ± 0.0271 | 0.7400 ± 0.0026 | 0.5926 ± 0.0245 | 0.5728 ± 0.0203 | 0.9384 ± 0.0040 |
| **BCE** | **Recalibrated** | **Val-F1** | **0.6568 ± 0.0138** | **0.5395 ± 0.0143** | 0.6179 ± 0.0336 | 0.7020 ± 0.0120 | 0.5728 ± 0.0203 | 0.9384 ± 0.0040 |
| WBCE | Original | 0.5 | 0.5682 ± 0.1017 | 0.4009 ± 0.0736 | 0.5295 ± 0.1891 | 0.6858 ± 0.1268 | 0.5095 ± 0.0204 | 0.9262 ± 0.0035 |
| WBCE | Original | Val-F1 | 0.6210 ± 0.0097 | 0.4933 ± 0.0126 | 0.5742 ± 0.0152 | 0.6765 ± 0.0157 | 0.5095 ± 0.0204 | 0.9262 ± 0.0035 |
| WBCE | Recalibrated | 0.5 | 0.6465 ± 0.0089 | 0.5264 ± 0.0166 | 0.5760 ± 0.0224 | 0.7379 ± 0.0252 | 0.5596 ± 0.0226 | 0.9382 ± 0.0039 |
| WBCE | Recalibrated | Val-F1 | 0.6520 ± 0.0116 | 0.5330 ± 0.0154 | 0.6144 ± 0.0239 | 0.6949 ± 0.0049 | 0.5596 ± 0.0226 | 0.9382 ± 0.0039 |

---

## 5. 主要配对实验发现（同 Seed 配对，Mean ± 样本 SD）

> 配对统计由 `runs/UVSPEC15-ABLATION-SUMMARY/paired_effects.csv` 的逐 Seed 差值计算（该目录未提供预聚合的配对汇总文件）。

**5.1 BN 重估（Recalibrated − Original，同阈值）**

| 损失 / 阈值 | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 | Test ΔMacro-F1 |
|---|---|---|---|---|
| BCE / 0.5 | +0.0424 ± 0.0544 | +0.0229 ± 0.0234 | +0.0444 ± 0.0733 | +0.0323 ± 0.0499 |
| BCE / Val-F1 | +0.0279 ± 0.0235 | +0.0321 ± 0.0327 | +0.0243 ± 0.0315 | +0.0291 ± 0.0420 |
| WBCE / 0.5 | +0.0749 ± 0.0806 | +0.1175 ± 0.0532 | +0.0783 ± 0.0949 | +0.1254 ± 0.0571 |
| WBCE / Val-F1 | +0.0273 ± 0.0033 | +0.0325 ± 0.0030 | +0.0310 ± 0.0037 | +0.0398 ± 0.0028 |

BN 重估在本模态上**四个组合、两个 split、两个指标方向全部为正**，且幅度（+0.024 至 +0.078 Micro-F1）远大于 IR 上观察到的量级（≤0.003）。同时它显著压缩了跨 Seed 方差：BCE 的 Validation Micro-F1 SD 从 0.0588（Original/0.5）降至 0.0048（Recalibrated/0.5），Test 从 0.0845 降至 0.0154。

**5.2 阈值校准（Validation-F1 − 默认 0.5）**

| 损失 / BN | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 | Test ΔMacro-F1 |
|---|---|---|---|---|
| BCE / Original | +0.0262 ± 0.0337 | +0.0782 ± 0.0081 | +0.0189 ± 0.0429 | +0.0707 ± 0.0200 |
| BCE / Recalibrated | +0.0116 ± 0.0046 | +0.0874 ± 0.0172 | −0.0013 ± 0.0015 | +0.0674 ± 0.0129 |
| WBCE / Original | +0.0611 ± 0.0886 | +0.1022 ± 0.0531 | +0.0528 ± 0.1067 | +0.0924 ± 0.0613 |
| WBCE / Recalibrated | +0.0135 ± 0.0098 | +0.0172 ± 0.0051 | +0.0055 ± 0.0110 | +0.0067 ± 0.0056 |

与 IR 不同，UV 的阈值校准收益**大部分迁移到了 Test**：四个组合中 3 个 Test ΔMicro-F1 为正，4 个 Test ΔMacro-F1 全为正；唯一为负的是 BCE/Recalibrated（−0.0013 ± 0.0015，量级接近零）。该差异将在第 8 节讨论。

**5.3 交互效应（BN × 阈值）**

阈值校准在 Original BN 上的 Validation 增益（+0.0262）大于在 Recalibrated BN 上的增益（+0.0116）：即 **BN 重估已吸收了相当一部分操作点修正**，两者的作用并非简单可加。WBCE 下同一模式更明显（+0.0611 → +0.0135）。

**5.4 Weighted BCE 与普通 BCE**

在八个配置中，WBCE 的 Validation Micro-F1 全部低于对应 BCE 配置（配对差 −0.0036 至 −0.0386），Macro-F1 亦全部偏低。因此**不采用**类别加权作为默认损失，保留为消融对照。仅有三 Seed，该比较不作显著性断言。

---

## 6. 决策：主方案、备选方案与固定基线

**决策依据的唯一来源是 Validation**。

| 配置 | Validation Micro-F1 | Validation Macro-F1 | Validation mAP | Validation AUROC |
|---|---|---|---|---|
| **BCE + Recalibrated + Val-F1（主方案）** | **0.6283 ± 0.0059** | 0.5229 ± 0.0087 | 0.5321 ± 0.0163 | 0.9270 ± 0.0014 |
| BCE + Recalibrated + 0.5（备选：更简链路） | 0.6167 ± 0.0048 | 0.4355 ± 0.0243 | 0.5321 ± 0.0163 | 0.9270 ± 0.0014 |
| WBCE + Recalibrated + Val-F1（备选：加权对照） | 0.6241 ± 0.0136 | 0.5214 ± 0.0110 | 0.5285 ± 0.0214 | **0.9282 ± 0.0040** |
| BCE + Original + 0.5（固定基线） | 0.5743 ± 0.0588 | 0.4126 ± 0.0463 | 0.5017 ± 0.0455 | 0.9119 ± 0.0168 |

### 6.1 主推理方案：**BCE + Training-only BN Recalibrated + Validation-F1 Threshold**

理由（按权重排序）：

1. 三 Seed Validation Micro-F1 = **0.6283 ± 0.0059**，为八组配置中的最高均值。
2. 三 Seed Validation Macro-F1 = 0.5229 ± 0.0087，相对固定基线提升 +0.1103。
3. 相比 Original BN，BN 重估在 Validation 与 Test 上、两种阈值下、两种损失下的配对方向**全部为正**，且同时把跨 Seed 方差降低约一个数量级——这是本模态最稳健的单项证据。
4. 阈值校准在此基础上仍带来 Validation Macro-F1 的额外增益（+0.0874），并基本保留到 Test（+0.0674）。
5. 与备选配置的 Validation Micro-F1 差值仅 0.0042（WBCE）与 0.0116（BCE/0.5），但方向与多指标一致，故选主方案。

### 6.2 备选方案 A：**BCE + Recalibrated BN + Threshold 0.5**

- Validation Micro-F1 = 0.6167 ± 0.0048；Macro-F1 = 0.4355 ± 0.0243。
- **定位**：需要"零 Validation 拟合成分"的推理链路时使用（阈值固定 0.5），代价是 Macro-F1 低约 0.087。
- 其 Test Micro-F1（0.6580 ± 0.0154）是八组中最高——**但该事实不构成选择依据**（见第 7、8 节）。

### 6.3 备选方案 B：**WBCE + Recalibrated BN + Validation-F1**

- Validation Micro-F1 = 0.6241 ± 0.0136；Macro-F1 = 0.5214 ± 0.0110；**Validation AUROC = 0.9282 ± 0.0040（八组最高）**。
- **定位**：当关注 Macro-AUROC 或需要类别加权视角时的对照方案。默认不采用加权损失（见 5.4）。

### 6.4 固定参考基线：**BCE + Original BN + Threshold 0.5**

- 作为后续研究、消融与融合实验的**标准比较对象**保留。
- 其价值：不含任何 Validation 拟合成分、不含 BN 后处理，是最"朴素"的参照点；也是方差最大的配置（Validation Micro-F1 SD 0.0588，Test 0.0845）。
- **不因任何单次 Test 数值调整。**

### 6.5 其余条件全部保留

Original BN、Recalibrated BN、0.5 与 Validation-F1 阈值、BCE 与 WBCE 的 24 个条件全部保留在归档中，任何后续研究均可复现。

---

## 7. 主方案的独立 Test 评价

**声明**：本节数值为**固定的**主方案在 Test 上的独立评价，用于报告与一致性核验。Test 数值**未参与**第 6 节的任何选择。

| 指标 | Test（Mean ± 样本 SD，n=3） |
|---|---|
| Micro-F1 | **0.6568 ± 0.0138** |
| Macro-F1 | **0.5395 ± 0.0143** |
| Micro-Precision | 0.6179 ± 0.0336 |
| Micro-Recall | 0.7020 ± 0.0120 |
| mAP | **0.5728 ± 0.0203** |
| Macro-AUROC | **0.9384 ± 0.0040** |

**逐标签 Test F1（主方案，三 Seed 均值，按 F1 降序）**

| 标签 | Test 正样本 | Test F1 |
|---|---:|---:|
| has_conjugation | 3000 | 0.8043 |
| cyclic_conjugated_system | 2693 | 0.7644 |
| heteroaromatic | 2198 | 0.7304 |
| has_aromatic_system | 2208 | 0.7249 |
| pi_conjugated_carbonyl | 1036 | 0.6338 |
| has_acceptor_substituted_pi_system | 1215 | 0.6227 |
| alpha_beta_unsaturated_carbonyl | 308 | 0.5612 |
| has_donor_substituted_pi_system | 2138 | 0.5542 |
| has_nonaromatic_conjugation | 809 | 0.5354 |
| mixed_aromatic_nonaromatic_conjugation | 695 | 0.5266 |
| acyclic_conjugated_system | 307 | 0.4086 |
| aryl_carbonyl | 178 | 0.3997 |
| donor_acceptor_same_conjugated_system | 370 | 0.3882 |
| **pi_conjugated_nitrile** | 222 | 0.2292 |
| **pi_conjugated_imine** | **148** | 0.2089 |

**Validation 决策 vs Test 报告的差异（如实记录）**：主方案在 Validation（0.6283）与 Test（0.6568）上的 Micro-F1 均**高于**固定基线（Validation 0.5743、Test 0.6136），方向一致。**但需强调**：这一致的转移方向是观测结果，不是选择依据；选择依据只有 Validation（第 6 节）。相反，备选方案 A 在 Test 上的 Micro-F1（0.6580）略高于主方案（0.6568），而其在 Validation 上更低（0.6167 vs 0.6283）——**该反例说明 Test 排序与 Validation 排序并不一致，因此不能用 Test 反选配置**。

---

## 8. 阈值校准与 BN 重估的收益迁移讨论

**观察一：BN 重估的收益在两个 split 上一致。** 四个组合的 Validation 与 Test 配对差同号且量级相近（见 5.1），并伴随跨 Seed 方差的大幅下降。这使 BN 重估成为本模态中最稳健的后处理。

**观察二：阈值校准的 Test Micro-F1 增益弱于 Validation，但方向多为正。** 四个组合中 3 个 Test ΔMicro-F1 为正（+0.0055 至 +0.0528），仅 BCE/Recalibrated 为 −0.0013（量级接近零）；Test ΔMacro-F1 四个组合全部为正。与 IR（四个组合 Test ΔMicro-F1 全为负）形成对比。

**解释（按可支持程度排序）**

1. **样本内优化偏差（可直接支持）**：阈值直接最大化 Validation 的逐标签 F1，其 Validation 增益在定义上包含乐观偏差。Test 才是独立评价，迁移程度取决于 Validation 与 Test 的分布一致性。
2. **类别先验差异（部分支持）**：部分标签的正样本比例在两集之间不同（例如 alpha_beta_unsaturated_carbonyl：Validation 392 / Test 308；aryl_carbonyl：155 / 178），单一全局操作点在两集上的最优性不同。
3. **与 BN 重估的交互（可直接支持）**：阈值校准在 Original BN 上的增益大于在 Recalibrated BN 上（5.3），说明 BN 重估已经修正了部分操作点；因此在主方案（Recalibrated）下，阈值校准的边际贡献小于在 Original 下。
4. **不可下结论的部分**：本文档**不主张**阈值校准是可泛化的方法学改进。其收益在 UV 上虽多数为正，但仍以 Validation 拟合为前提，且幅度与跨 Seed 波动同阶；要作为方法学主张，必须另行设计嵌套/交叉拟合评估。

**对后续研究的要求**：任何"阈值校准提升泛化"的主张都需要独立校准集或嵌套验证；在获得该证据前，主方案的定位是"在 Validation 上表现最佳、且 BN 重估带来稳定方差下降"的配置。

---

## 9. 低样本量标签与数据局限

**支持度分层**（与 IR Final Decision 使用**完全相同**的分层定义：min(Validation, Test) 正样本 < 30 记为 very_low，< 150 记为 low，其余为 normal）：

| 标签 | Validation 正样本 | Test 正样本 | 分层 |
|---|---:|---:|---|
| pi_conjugated_imine | 214 | **148** | **low** |
| aryl_carbonyl | 155 | 178 | normal（接近 low 边界） |
| pi_conjugated_nitrile | 281 | 222 | normal |
| acyclic_conjugated_system | 382 | 307 | normal |
| alpha_beta_unsaturated_carbonyl | 392 | 308 | normal |
| donor_acceptor_same_conjugated_system | 318 | 370 | normal |
| mixed_aromatic_nonaromatic_conjugation | 636 | 695 | normal |
| 其余 8 类 | 809–3147 | 809–3000 | normal |

**必要说明**：

- 按与 IR 相同的分层定义，UVSPEC15 只有 1 个标签落入 `low`，**没有** `very_low`。但**绝对样本量明显小于 IR**：UVSPEC15 的 Validation 正样本中位数约 **1137**（IR 约 **1703**），最低为 **155**（aryl_carbonyl；IR 最低为 nitro 的 11）。因此同一分层定义在 UV 上**可能低估**其稀疏性——15 个标签中有 **6 个**的 Validation 正样本少于 400。
- 已有记录显示 `pi_conjugated_nitrile`（Test F1 0.2292）与 `pi_conjugated_imine`（0.2089）明显低于其余 13 类（0.39–0.80），其逐标签阈值与 F1 的不确定度较大。
- 由于采用宏平均（Macro-F1、mAP、Macro-AUROC），稀疏标签的波动会以不可忽略的权重进入汇总；本文件的宏平均结论应结合逐标签表（`per_label_720_rows.csv`）阅读。
- 本文档**不主张**移除稀疏标签后性能如何变化；该敏感性分析未在本阶段执行。
- **数据层局限**：2 条全零 UV 谱已在划分中排除（Train 因此为 103853）；另有 129 条强度离群谱被标记但保留（见数据准备阶段的 `qc_flagged_samples.csv`）。这些样本未做剔除，其影响未单独量化。

---

## 10. 研究边界

- **域边界**：全部结论限于 **QM9S 计算光谱**与 **scaffold split**；计算谱与实验谱（NIST 等）在峰位、峰宽、基线与噪声上存在物理差异，本阶段**未做**实验域验证或迁移。
- **任务边界**：输出为 15 类 UV 专属结构语义标签，属**结构特征识别**，不是完整分子结构推断；不涉及 Top-k 候选生成、结构渲染或化学规则重排。
- **系统边界**：当前为 UV-Vis **单模态**模型；多模态证据融合、开放集拒识、XAI 与自然语言解释均未完成。
- 因此，"UVSPEC15 v1 冻结"指**该单模态基线的流程、权重与配置被冻结并可复现**，不表示项目整体目标达成。

### 10.1 面向多模态融合的接口与产物

**可复用的部分**

- **编码器表征**：`SpectrumStructureModel.encoder` 输出的 **256 维 embedding**。
- **样本对齐标识**：`row_index`（0–129816），与光谱矩阵、标签矩阵、结构库一致。

**不可误用的部分**

- **15 个分类阈值属于 UVSPEC15 的多标签分类输出头**，**不是 embedding 的组成部分**，不得作为融合特征的阈值或归一化参数。
- **BN running statistics 不得跨模态迁移**：UV 的 12 层 BN 统计量是在 UV 光谱分布上估计的；融合训练必须重新估计。
- **两套 BN 状态（Original / Recalibrated）的 embedding 不同**；融合实验必须显式声明使用哪一套，并保证训练与推理一致。

**样本对齐的已知差异**

- UV 的 split 覆盖 **129815** 行（排除 2 条全零谱）；IR 覆盖 129817 行。
- 多模态融合必须使用 `processed/qm9s/prepared/multimodal_complete_scaffold_split.npz`（129815：Train 103853 / Val 12981 / Test 12981，Val/Test 与 IR 同集），**不得**把两份不同规模的 train 直接混用。
- 若只做 UV-only 实验，应继续使用 `uvvis_scaffold_split_valid.npz`（103853）。

**融合实验应引用的产物**

| 用途 | 路径 |
|---|---|
| 主方案权重 | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt`（+ manifest） |
| 主方案阈值 | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/threshold_calibration/optimal_thresholds.json` |
| 原始权重（对照） | `runs/EXP-UVVIS-003-UVSPEC15-*/best_model.pt`、`runs/EXP-UVVIS-004-UVSPEC15-WBCE*/best_model.pt` |
| 原始阈值（对照） | 对应 run 目录内的 `optimal_thresholds.json` |
| 单模态性能基线 | `runs/UVSPEC15-ABLATION-SUMMARY/test_24_conditions.csv`、`mean_sd_8_conditions_each_split.csv` |

**对齐要求**：融合实验须以"固定基线（BCE + Original BN + 0.5）"作为单模态下限对照，并记录其与融合模型的差异；不得用 Test 数值挑选参与融合的单模态 Seed。

---

## 11. 决策性质：事后形成，非预注册

本文档的配置决策是在 24 条件消融**全部完成并已查看 Validation 与 Test 结果之后**形成的（上游策略文档自述为"在完成消融实验后形成的研究决策记录，而非事先注册的实验方案"）。因此：

- 它是**探索性决策记录**，不得表述为事先注册的分析方案或预注册假设检验。
- 取舍理由被限定为可核查的 Validation 数值 + 稳定性观察 + 链路性质，并明确承诺**不因 Test 数值改选**。
- 未来若以 UVSPEC15 v1 为基线开展新实验，该配置应作为**已知固定基线**引用，而不是重新作为待优化对象；如需改变，应以新实验编号登记并说明与 v1 的可比性。

---

## 12. 统计解释边界

- 全部区间为 **n = 3 个随机 Seed 的均值 ± 样本标准差**（ddof = 1），用于描述训练随机性，**不是**置信区间。
- **本文档不作任何显著性断言**：不声称 BN 重估在统计上显著优于 Original（尽管四个组合方向一致、幅度较大），不声称阈值校准显著优于 0.5，不声称 BCE 显著优于 WBCE。上述差值均只有三个配对观测。
- n = 3 时样本标准差自身的估计误差很大（相对量级约 50%），因此"某个配置 SD 更小"同样不能作为稳健性结论。BN 重估使 SD 从 0.0588 降至 0.0048 的幅度远超该估计误差，是本文件中少数可以明确描述的稳定性现象。
- 若需要方法学层面的优劣判断，必须增加独立重复次数并预先规定推断程序。

---

## 附录 A：冻结清单（引用路径）

| 类别 | 路径 |
|---|---|
| 训练脚本 | `training/train_uvvis_uvspec15.py`、`_seed123`、`_seed2026`、`_weighted.py`、`_weighted_seed123.py`、`_weighted_seed2026.py` |
| BN 诊断脚本 | `training/diagnose_uvspec15_batchnorm_multiseed.py` |
| BN checkpoint 保存 | `training/save_uvspec15_bn_checkpoint.py` |
| 阈值校准脚本 | `training/calibrate_uvvis_uvspec15_thresholds.py`（及 `_seed123`、`_seed2026`、`_weighted*`）、`training/calibrate_uvspec15_bn_thresholds.py` |
| 汇总脚本 | `training/summarize_uvspec15_ablation.py` |
| 原始 checkpoint | `runs/EXP-UVVIS-003-UVSPEC15-{FULL,SEED123,SEED2026}/best_model.pt`、`runs/EXP-UVVIS-004-UVSPEC15-WBCE{,-SEED123,-SEED2026}/best_model.pt` |
| BN 重估 checkpoint | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| BN 诊断记录 | `runs/UVSPEC15-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| 阈值（Original） | `runs/<run_name>/optimal_thresholds.json` |
| 阈值（Recalibrated） | `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/threshold_calibration/optimal_thresholds.json` |
| 消融汇总 | `runs/UVSPEC15-ABLATION-SUMMARY/`（7 个文件） |

## 附录 B：主方案的推理调用链

1. 加载 `runs/UVSPEC15-BN-RECALIBRATED/EXP-UVVIS-003-UVSPEC15-FULL/best_model_bn_recalibrated.pt`（Seed 42 canonical）或对应 Seed 的重估 checkpoint。
2. 输入 701 点 UV-Vis 光谱（1–15 eV），按逐谱最大绝对值归一化，送入 `SpectrumStructureModel`（15 输出）。
3. 取 `runs/UVSPEC15-BN-RECALIBRATED/<run_name>/threshold_calibration/optimal_thresholds.json` 中的 15 个阈值（**不得**与 Original BN 的阈值混用）。
4. 概率 ≥ 对应标签阈值即判为正。

## 附录 C：复现入口

```powershell
# 只读汇总（不加载 PyTorch、不重算预测）
python training\summarize_uvspec15_ablation.py
```

预期末行：`PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 720 per-label rows`
完整步骤见 `uvspec15_repro_protocol.md`。

## 附录 D：版本记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-11-15 | 首版正式决策记录：确认主方案（BCE + Recalibrated BN + Val-F1）、备选方案 A/B 与固定基线；整理 24 条件矩阵、配对效应、Test 独立评价与迁移讨论；补充分层与融合接口 |

---

**决策边界**：本记录冻结的是 UVSPEC15 v1 **单模态基线的配置与归档状态**。任何改变损失、数据、标签、BN 处理或阈值策略的后续实验，必须以新实验编号登记，并明确与本文档的可比性。
