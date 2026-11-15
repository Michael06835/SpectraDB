# IR v1 最终科研决策记录（冻结稿）

> **文档性质**：实验完成后形成的配置决策记录（exploratory, post-hoc decision record），**不是事先注册的实验方案**。
> **冻结日期**：2026-11-15
> **适用范围**：QM9S 计算域 IR 单模态 1D-CNN → 14 类官能团多标签识别。
> **不适用范围**：实验域（NIST/API-Raman）泛化、多模态融合、完整分子结构 Top-k 推断、开放集拒识与可解释性。
> **上游文档**：`docs/experiments/IR_v1_experimental_strategy.md`（实验流程）、`runs/IR-ABLATION-SUMMARY/README_summary.txt`（汇总）。

---

## 1. 研究任务、数据与评价协议

**任务**：由 QM9S 计算红外光谱预测 14 类官能团多标签二值标签。该任务是后续多模态结构推断的中间监督信号，本身不等于完整分子结构识别。

**数据**

| 项 | 值 |
|---|---|
| 光谱矩阵 | `processed/qm9s/prepared/ir_float32.npy`，`(129817, 3501)`，float32，500–4000 cm⁻¹，步长 1.0 cm⁻¹ |
| 标签矩阵 | `processed/qm9s/prepared/functional_group_labels.npy`，`(129817, 14)`，uint8 |
| 标签顺序 | `functional_group_label_names.json`（字母序，14 类） |
| 归一化 | 逐谱最大绝对值归一化（`normalization="max"`），在 Dataset 内施加 |
| 划分 | `ir_scaffold_split_valid.npz`，Bemis–Murcko scaffold split，Train 103855 / Validation 12981 / Test 12981，三集互斥、无 scaffold 重叠 |

**14 类标签**：aldehyde、alkene、alkyne、amide、amine、c_f_bond、carbonyl、ester、ether、hydroxyl、imine、ketone、nitrile、nitro。标签之间**非互斥**（例如 carbonyl 与 ketone/ester/amide/aldehyde 天然共现）。

**评价指标**：Micro-F1、Macro-F1、Micro-Precision、Micro-Recall、mAP、Macro-AUROC；另记录逐标签 Precision/Recall/F1/AP/AUROC。mAP 与 Macro-AUROC 为阈值无关指标。

**数据使用纪律**：Training 用于参数训练与 BN 统计量重估；Validation 用于 Early Stopping、方法比较与逐标签阈值拟合；Test 仅用于最终评价、记录与一致性核验。本文档第 6 节的配置选择**全部基于 Validation**；Test 数值在第 7 节独立报告，**未用于**改变任何配置、阈值或 Seed 选择。

---

## 2. 六个独立训练模型

| 模型目录 | 损失 | Seed | 最优 epoch | 训练 epoch 数 | Validation mAP（选轮指标） |
|---|---|---:|---:|---:|---:|
| `runs/EXP-IR-001-FULL` | 普通 BCE | 42 | 21 | 29 | 0.967973 |
| `runs/EXP-IR-001-SEED123` | 普通 BCE | 123 | 30 | 38 | 0.956755 |
| `runs/EXP-IR-001-SEED2026` | 普通 BCE | 2026 | 15 | 23 | 0.956584 |
| `runs/EXP-IR-002-FULL` | Weighted BCE | 42 | 10 | 18 | 0.958067 |
| `runs/EXP-IR-002-SEED123` | Weighted BCE | 123 | 23 | 31 | 0.961716 |
| `runs/EXP-IR-002-SEED2026` | Weighted BCE | 2026 | 11 | 19 | 0.956467 |

**共同训练配置**（六模型一致，未做任何逐模型调整）：

- 模型：`SpectrumStructureModel`，1D-CNN，embedding 256，12 层 BatchNorm1d，14 维输出头。
- 优化：AdamW（lr 1e-3，weight decay 1e-4），batch 512，num_workers 4，最多 50 epochs，Early Stopping patience 8，AMP（fp16）。
- 最优轮选择：Validation mAP（阈值无关）。
- Weighted BCE：`sqrt(pos_weight)` 变换、权重上限 20；`pos_weight` **仅由 Training split 统计量计算**。
- **未执行任何一次重训练**；六份 checkpoint 自训练完成后保持一致（SHA256 已登记）。

**Seed 政策**：Seed 是重复实验因素，用于估计训练稳定性，不是需要优化的超参数。论文统计必须覆盖全部三个 Seed；**不依据 Test 分数选择 Seed**。Seed 42 为项目既定的 canonical 实例，仅用于工程演示与样例输出。

---

## 3. BN 诊断、BN 重估与阈值校准的完成状态

| 环节 | 数量 | 产物位置 |
|---|---:|---|
| Full BN Diagnostic | **6/6** | `runs/IR-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| Training-only BN 重估 checkpoint | **6/6** | `runs/IR-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| Original BN 阈值校准 | **6/6** | `runs/IR-THRESHOLD-CALIBRATION/<run_name>/`（各 6 文件） |
| Recalibrated BN 阈值校准 | **6/6** | `runs/IR-BN-THRESHOLD-CALIBRATION/<run_name>/`（各 6 文件） |
| 阈值校准合计 | **12/12** | 上述两个目录 |

**BN 重估协议**：在原始 checkpoint 的独立深拷贝上进行；`reset_running_stats()` 后将 `momentum` 置为 `None`（按 batch 等权累计平均），仅将 BatchNorm 层置于 train 模式，用**完整 Training split 的 103855 条样本**前向一遍，恢复 momentum 后回到 eval。可训练参数与优化器状态不变，原始 checkpoint 全程只读。该算法与 UVSPEC15 v1 已归档实现完全一致，本阶段未做任何算法改动。未执行 Quick Diagnostic。

**安全核验**（六份全部通过）：36 个 BN buffer 是唯一允许变化的张量；非 BN 张量 0 变化；12 层 `num_batches_tracked` 由 `best_epoch × 203` 变为 203；所有 running statistics 有限且方差非负；保存后可回读；manifest 的 `source_sha256` 与 `saved_sha256` 分别与原始/重估 checkpoint 实测哈希一致；原始 checkpoint 的 SHA256 与 mtime 未变。

**阈值校准协议**：对每一种 BN 状态分别、独立地在 Validation 上逐标签最大化 F1（`precision_recall_curve` 的阈值候选），阈值在**计算任何 Test 指标之前**落盘；Test 不参与拟合。Recalibrated BN 的阈值由重估后的概率重新拟合，**未复用** Original BN 的阈值文件。

---

## 4. 24 条件消融矩阵与三 Seed 统计

矩阵维度：2 损失 × 3 Seed × 2 BN 状态 × 2 阈值策略 = **24 个评估条件**；Validation 24 条 + Test 24 条 = **48 条 split-level 记录**；逐标签 14 × 24 × 2 = **672 行**。12 次阈值校准是**执行任务**，其结果构成 24 条件中的 Validation-F1 分支，**不额外计为独立条件**。

**Validation，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | Micro-P | Micro-R | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | 0.9470 ± 0.0018 | 0.9165 ± 0.0066 | 0.9526 | 0.9416 | 0.9604 ± 0.0065 | 0.9946 ± 0.0020 |
| **BCE** | **Original** | **Val-F1** | **0.9505 ± 0.0025** | **0.9268 ± 0.0045** | 0.9545 | 0.9465 | 0.9604 ± 0.0065 | 0.9946 ± 0.0020 |
| BCE | Recalibrated | 0.5 | 0.9438 ± 0.0037 | 0.9179 ± 0.0087 | — | — | 0.9607 ± 0.0067 | 0.9957 ± 0.0001 |
| **BCE** | **Recalibrated** | **Val-F1** | **0.9503 ± 0.0023** | **0.9299 ± 0.0035** | 0.9516 | 0.9491 | 0.9607 ± 0.0067 | 0.9957 ± 0.0001 |
| WBCE | Original | 0.5 | 0.9379 ± 0.0067 | 0.9070 ± 0.0102 | — | — | 0.9588 ± 0.0027 | 0.9940 ± 0.0014 |
| WBCE | Original | Val-F1 | 0.9450 ± 0.0064 | 0.9221 ± 0.0085 | — | — | 0.9588 ± 0.0027 | 0.9940 ± 0.0014 |
| WBCE | Recalibrated | 0.5 | 0.9398 ± 0.0068 | 0.9068 ± 0.0049 | — | — | 0.9595 ± 0.0022 | 0.9952 ± 0.0008 |
| WBCE | Recalibrated | Val-F1 | 0.9458 ± 0.0060 | 0.9223 ± 0.0072 | — | — | 0.9595 ± 0.0022 | 0.9952 ± 0.0008 |

（"WBCE" 即 Weighted BCE；"—" 表示该四项 Precision/Recall 见 `mean_sd_8_conditions_each_split.csv`，本文档不重复列出。）

**Test，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | **0.9405 ± 0.0048** | 0.9193 ± 0.0050 | 0.9688 ± 0.0050 | 0.9954 ± 0.0007 |
| **BCE** | **Original** | **Val-F1** | **0.9364 ± 0.0054** | **0.9199 ± 0.0076** | **0.9688 ± 0.0050** | **0.9954 ± 0.0007** |
| BCE | Recalibrated | 0.5 | 0.9376 ± 0.0071 | 0.9179 ± 0.0118 | 0.9685 ± 0.0067 | 0.9953 ± 0.0006 |
| **BCE** | **Recalibrated** | **Val-F1** | **0.9350 ± 0.0068** | **0.9198 ± 0.0076** | **0.9685 ± 0.0067** | **0.9953 ± 0.0006** |
| WBCE | Original | 0.5 | 0.9328 ± 0.0085 | 0.9091 ± 0.0169 | 0.9670 ± 0.0069 | 0.9949 ± 0.0009 |
| WBCE | Original | Val-F1 | 0.9317 ± 0.0076 | 0.9083 ± 0.0227 | 0.9670 ± 0.0069 | 0.9949 ± 0.0009 |
| WBCE | Recalibrated | 0.5 | 0.9347 ± 0.0081 | 0.9097 ± 0.0120 | 0.9676 ± 0.0058 | 0.9951 ± 0.0007 |
| WBCE | Recalibrated | Val-F1 | 0.9324 ± 0.0079 | 0.9091 ± 0.0216 | 0.9676 ± 0.0058 | 0.9951 ± 0.0007 |

---

## 5. 主要配对实验发现（同 Seed 配对，Mean ± 样本 SD）

**5.1 阈值校准（Validation-F1 阈值 − 默认 0.5）**

| 损失 / BN | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 |
|---|---|---|---|
| BCE / Original | +0.0035 ± 0.0012 | +0.0103 ± 0.0043 | **−0.0042 ± 0.0010** |
| WBCE / Original | +0.0071 ± 0.0011 | +0.0151 ± 0.0056 | −0.0011 ± 0.0025 |
| BCE / Recalibrated | +0.0066 ± 0.0016 | +0.0119 ± 0.0059 | −0.0026 ± 0.0004 |
| WBCE / Recalibrated | +0.0060 ± 0.0016 | +0.0154 ± 0.0023 | −0.0022 ± 0.0005 |

mAP 与 Macro-AUROC 在两种阈值下**严格相等**（阈值无关指标的恒等性在汇总阶段被断言检查）。

**5.2 BN 重估（Recalibrated − Original，同阈值）**

| 损失 / 阈值 | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 |
|---|---|---|---|
| BCE / 0.5 | −0.0032 ± 0.0032 | +0.0014 ± 0.0023 | −0.0030 ± 0.0032 |
| WBCE / 0.5 | +0.0020 ± 0.0003 | −0.0002 ± 0.0066 | +0.0019 ± 0.0006 |
| BCE / Val-F1 | −0.0001 ± 0.0007 | +0.0030 ± 0.0016 | −0.0014 ± 0.0019 |
| WBCE / Val-F1 | +0.0008 ± 0.0003 | +0.0001 ± 0.0013 | +0.0007 ± 0.0015 |

BN 重估对**排序能力**（mAP / Macro-AUROC）的影响极小：六个模型全部 |Δ| ≤ 0.0033，方向在 BCE 与 WBCE 之间不一致。阈值无关指标几乎不动、而固定 0.5 阈值下的 Micro-F1 明显变化，说明 BN 重估主要改变**分数分布（操作点）**，而非排序本身。

**5.3 BN 与阈值校准的交互**

| 损失 | 交互项（(Recal−Orig|ValF1) − (Recal−Orig|0.5)）ΔMicro-F1 |
|---|---|
| BCE | +0.0031 ± 0.0028 |
| WBCE | −0.0012 ± 0.0005 |

即：在 BCE 下，BN 重估在 0.5 阈值下的小幅退化会在阈值校准后被部分抵消；WBCE 下方向相反。该交互量级与主效应同阶，说明**BN 处理与阈值策略不宜分开单独评价**。

**5.4 Weighted BCE 与普通 BCE**

在八个配置中的每一个上，WBCE 的 Validation Micro-F1 均低于对应 BCE（配对差 −0.0040 至 −0.0092），Macro-F1 亦一致偏低。因此**不采用**类别加权作为默认损失。仅有三 Seed，该比较不作显著性断言。

---

## 6. 决策：主方案、备选方案与固定基线

**决策依据的唯一来源是 Validation**。下表列出八组配置在 Validation 上的表现（与第 4 节一致）：

| 配置 | Validation Micro-F1 | Validation Macro-F1 | Validation mAP | Validation Macro-AUROC |
|---|---|---|---|---|
| **BCE + Original + Val-F1（主方案）** | **0.9505 ± 0.0025** | 0.9268 ± 0.0045 | 0.9604 ± 0.0065 | 0.9946 ± 0.0020 |
| **BCE + Recalibrated + Val-F1（备选）** | 0.9503 ± 0.0023 | **0.9299 ± 0.0035** | **0.9607 ± 0.0067** | **0.9957 ± 0.0001** |
| BCE + Original + 0.5（固定基线） | 0.9470 ± 0.0018 | 0.9165 ± 0.0066 | 0.9604 ± 0.0065 | 0.9946 ± 0.0020 |
| BCE + Recalibrated + 0.5 | 0.9438 ± 0.0037 | 0.9179 ± 0.0087 | 0.9607 ± 0.0067 | 0.9957 ± 0.0001 |

### 6.1 主推理方案：**BCE + Original BN + Validation-F1 Threshold**

理由（按权重排序）：

1. 三 Seed Validation Micro-F1 = **0.9505 ± 0.0025**，为八组配置中的最高均值。
2. 三 Seed Validation Macro-F1 = 0.9268 ± 0.0045，相对固定基线提升 +0.0103。
3. 不需要额外保存与加载 BN 重估 checkpoint，推理链路最短，配置与复现成本最低；重估 checkpoint 的加载与其阈值文件必须成对使用，减少一个失效点。
4. 与备选方案的 Validation Micro-F1 差值仅 0.0002（远小于 0.0025 的跨 Seed 波动量级），不构成可区分的优势；在此近似并列的情况下，优先选择更简单的链路。

### 6.2 备选方案：**BCE + Recalibrated BN + Validation-F1 Threshold**

- Validation Micro-F1 = 0.9503 ± 0.0023（与主方案并列）。
- Validation Macro-F1 = **0.9299 ± 0.0035**（八组最高）。
- Validation mAP = **0.9607 ± 0.0067**（八组最高）。
- **定位**：当研究关注 Macro-F1、稀有类均衡性或排序质量而非 Micro-F1 时采用。其 Macro-F1 优势为 +0.0031 且跨 Seed 方差更小（0.0035 vs 0.0045），但同样不宣称具有统计显著性。
- 使用时必须与 `runs/IR-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` 成对加载，不得与 Original BN 的阈值混用。

### 6.3 固定参考基线：**BCE + Original BN + Threshold 0.5**

- 作为后续研究、消融与融合实验的**标准比较对象**保留，不因任何单次 Test 数值而调整。
- 其价值在于：不含任何 Validation 拟合成分（阈值固定为 0.5），是唯一"零后处理"的参照点。
- **不根据 Test 上最高 Micro-F1（该配置 Test Micro-F1 = 0.9405，确为八组最高）事后改变研究决策。** 该现象在第 7、8 节如实记录，而不是作为改选主方案的理由。

### 6.4 三个配置全部保留

三种配置的 checkpoint、阈值文件与评估记录全部保留在归档中，任何后续研究均可复现。

---

## 7. 主方案的独立 Test 评价

**声明**：本节数值为**固定的**主方案在 Test 上的独立评价，用于报告与一致性核验。Test 数值**未参与**第 6 节的任何选择。

| 指标 | Test（Mean ± 样本 SD，n=3） |
|---|---|
| Micro-F1 | **0.9364 ± 0.0054** |
| Macro-F1 | **0.9199 ± 0.0076** |
| Micro-Precision | 0.9302 |
| Micro-Recall | 0.9427 |
| mAP | **0.9688 ± 0.0050** |
| Macro-AUROC | **0.9954 ± 0.0007** |

**逐标签 Test F1（主方案，三 Seed 均值）**

| 标签 | Test 正样本 | Test F1 |
|---|---:|---:|
| hydroxyl | 4334 | 0.9963 |
| alkyne | 1514 | 0.9867 |
| nitrile | 1517 | 0.9857 |
| carbonyl | 3898 | 0.9819 |
| aldehyde | 1284 | 0.9646 |
| alkene | 1745 | 0.9406 |
| ether | 5260 | 0.9331 |
| imine | 1295 | 0.9068 |
| amide | 1053 | 0.9014 |
| ester | 622 | 0.8907 |
| **nitro** | **15** | **0.8834** |
| ketone | 1272 | 0.8746 |
| amine | 3491 | 0.8287 |
| **c_f_bond** | **144** | **0.8044** |

**Validation 决策 vs Test 报告的差异（如实记录）**：主方案的 Test Micro-F1（0.9364）**低于**固定基线（0.9405），差 −0.0042 ± 0.0010；而其 Validation Micro-F1（0.9505）高于基线（0.9470）。Test Macro-F1（0.9199）与基线（0.9193）基本持平。阈值无关指标完全相同（同一 checkpoint）。这一现象在第 8 节讨论。

---

## 8. 阈值校准收益未能稳定迁移至 Test 的讨论

**观察**：在 Validation 上，逐标签阈值校准一致地提升 Micro-F1（+0.0035 至 +0.0071）与 Macro-F1（+0.0103 至 +0.0154）；但在 Test 上，同一校准使四个配置的 Micro-F1 一致下降（−0.0011 至 −0.0042），Macro-F1 则一升一降、总体接近持平。

**解释（按可支持程度排序）**

1. **样本内优化偏差（可直接支持）**：阈值直接最大化 Validation 上的逐标签 F1，其 Validation 增益在定义上包含乐观偏差，不能作为泛化证据。Test 才是独立评价。
2. **操作点位移（可直接支持）**：校准阈值整体低于 0.5（例如主方案下多数标签阈值在 0.2–0.9 之间，且部分显著低于 0.5），在 Test 上表现为 Precision 下降、Recall 基本不变（主方案 Test P 0.9302 vs 基线 0.9384，R 均为 0.9427）。因此 Micro-F1 的下降可由精确率损失解释，而非排序能力退化——两者的 mAP/AUROC 完全相同。
3. **Validation 与 Test 的类别先验差异（部分支持）**：部分标签的正样本比例在两集之间不同（例如 alkene：Validation 2914、Test 1745），单一全局操作点难以同时最优。
4. **不可下结论的部分**：本文档**不主张**"阈值校准无效"。校准提升 Macro-F1 的方向在 Validation 与 Test 上一致；且 Validation-F1 阈值本身是方法选择的一部分，其合理评价需要嵌套验证或独立校准集，本阶段未设计该实验。

**对后续研究的要求**：若要把阈值策略作为可泛化的方法学主张，必须另行设计嵌套/交叉拟合式的校准评估；在获得该证据之前，主方案定位为"在 Validation 上表现最佳、链路最简单"的配置，而非"经证明优于 0.5 基线"的方法。

---

## 9. 低样本量标签的局限

| 标签 | Validation 正样本 | Test 正样本 | 支持度分层 |
|---|---:|---:|---|
| **nitro** | **11** | **15** | very low |
| **c_f_bond** | **230** | **144** | low |
| ester | 633 | 622 | normal |
| 其余 11 类 | 1053–5794 | 1053–5260 | normal |

- `nitro` 的 Validation 正样本仅 11 个，其逐标签阈值在 11 个正样本上拟合，**阈值估计与 F1 估计的不确定度都很大**（主方案下 Validation F1 = 0.7980、Test F1 = 0.8834，两者差异本身即反映该不稳定）。任何以 `nitro` 为依据的结论都必须标注样本量。
- `c_f_bond` 的 Test 正样本 144 个，逐标签指标的抽样波动同样较大。
- 由于 14 类采用宏平均（Macro-F1、mAP、Macro-AUROC），少数稀疏标签的波动会以不可忽略的权重进入汇总；本文档的宏平均结论应结合逐标签表（`per_label_f1_levels.csv`）阅读。
- 本文档**不主张**移除稀疏标签后性能会如何变化；该敏感性分析未在本阶段执行。

---

## 10. 研究边界

- **域边界**：全部结论限于 **QM9S 计算光谱**与 **scaffold split**。计算谱与实验谱（NIST IR、API-Raman）在峰位、峰宽、基线与噪声特性上存在物理差异，本阶段**未做**实验域验证或迁移。
- **任务边界**：输出为 14 类官能团多标签，属**结构特征识别**，不是完整分子结构推断；不涉及 Top-k 候选生成、结构渲染或化学规则重排。
- **系统边界**：当前为 IR **单模态**模型；多模态证据融合、开放集拒识、XAI 与自然语言解释均未完成。
- 因此，"IR v1 冻结"指**单模态基线的流程、权重与配置被冻结并可复现**，不表示项目整体目标达成。

### 10.1 面向多模态融合的接口与产物

**可复用的部分**

- **编码器表征**：`SpectrumStructureModel.encoder` 输出的 **256 维 embedding**（`AdaptiveAvgPool1d(16)` + `Linear(2048→256)`）。
- **样本对齐标识**：`row_index`（0–129816），与光谱矩阵、标签矩阵、结构库一致。

**不可误用的部分**

- **14 个分类阈值属于 IR 的输出头**（`feature_head: Linear(256→14)`），**不是 embedding 的组成部分**，不得作为融合特征的阈值或归一化参数。
- **BN running statistics 不得跨模态迁移**：IR 的 12 层 BN 统计量是在 IR 光谱分布上估计的；多模态融合模型训练时必须重新估计（或在融合训练集上重新校准），不得直接加载 IR 的 BN buffer。
- **两套 BN 状态（Original / Recalibrated）的 embedding 不同**：Recalibrated 版本的 BN 统计量已改变，其 embedding 分布与 Original 不同。融合实验必须**显式声明**使用哪一套，并保证训练与推理一致。

**样本对齐的已知差异**

- IR 的 split 覆盖全部 **129817** 行；UV-Vis 因 2 条全零谱（`row_index 13998`、`14000`）只有 **129815** 行。
- 多模态融合必须使用 `processed/qm9s/prepared/multimodal_complete_scaffold_split.npz`（129815：Train 103853 / Val 12981 / Test 12981，Val/Test 与 IR 完全同集），**不得**直接把 IR 的 103855 条 train 与 UV 的 103853 条 train 混用。
- 若只做 IR-only 实验，应继续使用 `ir_scaffold_split_valid.npz`（103855）。

**融合实验应引用的产物**

| 用途 | 路径 |
|---|---|
| 主方案权重 | `runs/EXP-IR-001-FULL/best_model.pt`（及其余 Seed 的对应 checkpoint） |
| 主方案阈值（仅用于 IR 单模态报告） | `runs/IR-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 备选方案权重 | `runs/IR-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt`（+ manifest） |
| 备选方案阈值 | `runs/IR-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 单模态性能基线（用于融合增益对照） | `runs/IR-ABLATION-SUMMARY/test_24_conditions.csv`、`mean_sd_8_conditions_each_split.csv` |
| 推理配置元数据 | `runs/<run_name>/config.json` |

**对齐要求**：融合实验须以上表的"固定基线（BCE + Original BN + 0.5）"作为单模态下限对照，并记录其与融合模型的差异；不得用 Test 数值挑选参与融合的单模态 Seed。

---

## 11. 决策性质：事后形成，非预注册

本文档的配置决策是在 24 条件消融**全部完成、并已查看 Validation 与 Test 结果之后**形成的。因此：

- 它是**探索性决策记录**，不得表述为事先注册的分析方案或预注册假设检验。
- 六模型训练、BN 诊断、BN 重估与阈值校准的**流程**在多大程度上先于结果存在，已在 `docs/experiments/IR_v1_experimental_strategy.md` 的过程记录中留存；但"八组配置的最终取舍"这一**决策动作本身是事后的**。
- 为进一步降低解释弹性，本文档的取舍理由被限定为可核查的 Validation 数值 + 链路复杂度，并明确承诺**不因 Test 数值改选**。
- 未来若以 IR v1 为基线开展新实验，配置应作为**已知固定基线**引用，而不是重新作为待优化对象；如需改变，应以新实验编号登记并说明与 v1 的可比性。

---

## 12. 统计解释边界

- 全部区间为 **n = 3 个随机 Seed 的均值 ± 样本标准差**（ddof = 1），用于描述训练随机性，**不是**置信区间。
- **本文档不作任何显著性断言**：不声称主方案优于备选方案、不声称阈值校准优于 0.5、不声称 BCE 优于 WBCE、不声称 BN 重估有效或无效。上述比较的差值均与跨 Seed 波动同阶或更小。
- n = 3 时样本标准差自身的估计误差很大（相对量级约 50%），因此"哪个配置 SD 更小"同样不能作为稳健性结论。
- 若后续需要方法学层面的优劣判断，必须增加独立重复次数并预先规定推断程序。

---

## 附录 A：冻结清单（引用路径）

| 类别 | 路径 |
|---|---|
| 训练脚本 | `training/train_ir_full.py`、`train_ir_seed123.py`、`train_ir_seed2026.py`、`train_ir_weighted.py`、`train_ir_weighted_seed123.py`、`train_ir_weighted_seed2026.py` |
| BN 诊断脚本 | `training/diagnose_ir_batchnorm_full.py` |
| BN checkpoint 保存 | `training/save_ir_bn_checkpoint.py` |
| 阈值校准脚本 | `training/calibrate_ir_original_thresholds.py`、`calibrate_ir_recalibrated_thresholds.py` |
| 汇总脚本 | `training/summarize_ir_ablation.py` |
| 原始 checkpoint | `runs/EXP-IR-001-FULL/best_model.pt`、`runs/EXP-IR-001-SEED123/best_model.pt`、`runs/EXP-IR-001-SEED2026/best_model.pt`、`runs/EXP-IR-002-FULL/best_model.pt`、`runs/EXP-IR-002-SEED123/best_model.pt`、`runs/EXP-IR-002-SEED2026/best_model.pt` |
| BN 重估 checkpoint | `runs/IR-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| BN 诊断记录 | `runs/IR-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| 阈值（Original） | `runs/IR-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 阈值（Recalibrated） | `runs/IR-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 消融汇总 | `runs/IR-ABLATION-SUMMARY/`（11 个文件） |
| 历史汇总（保留） | `runs/IR-v1-FINAL/`（2026-08-14，未改写） |

## 附录 B：主方案的推理调用链

1. 加载 `runs/EXP-IR-001-FULL/best_model.pt`（Seed 42 canonical）或对应 Seed 的 checkpoint。
2. 输入按逐谱最大绝对值归一化，送入 `SpectrumStructureModel`（14 输出）。
3. 取 `runs/IR-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` 中的 14 个阈值（**不得**与 Recalibrated 或历史 1024/4 口径的阈值混用）。
4. 概率 ≥ 对应标签阈值即判为正。

## 附录 C：复现入口

```powershell
# 只读汇总（不加载 PyTorch、不重算预测）
python training\summarize_ir_ablation.py
```

## 附录 D：版本记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-11-15 | 首版冻结：确定主方案（BCE + Original BN + Val-F1）、备选方案（BCE + Recalibrated BN + Val-F1）与固定基线（BCE + Original BN + 0.5）；记录 Test 独立评价与阈值校准迁移讨论 |

---

**决策边界**：本记录冻结的是 IR v1 **单模态基线的配置与归档状态**。任何改变损失、数据、标签、BN 处理或阈值策略的后续实验，必须以新实验编号登记，并明确与本文档的可比性。
