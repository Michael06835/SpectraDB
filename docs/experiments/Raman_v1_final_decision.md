# Raman v1 最终科研决策记录（冻结稿）

> **文档性质**：实验完成后形成的配置决策记录（exploratory, post-hoc decision record），**不是事先注册的实验方案**。
> **冻结日期**：2026-11-16
> **适用范围**：QM9S 计算域 Raman 单模态 1D-CNN → 14 类官能团多标签识别。
> **不适用范围**：实验域（NIST/API-Raman）泛化、多模态融合、完整分子结构 Top-k 推断、开放集拒识与可解释性。
> **上游文档**：`docs/experiments/raman_repro_protocol.md`（流程与复现）、`runs/RAMAN-ABLATION-SUMMARY/README_summary.txt`（汇总）、`docs/experiments/raman_ablation_registry.md`（溯源）。
> **对照文档**：`docs/experiments/IR_v1_final_decision.md`、`docs/experiments/UVSPEC15_v1_final_decision.md`（同构方法学）。

---

## 1. 研究任务、数据与评价协议

**任务**：由 QM9S 计算拉曼光谱预测 14 类官能团多标签二值标签。该任务是后续多模态结构推断的中间监督信号，本身不等于完整分子结构识别。

**数据**

| 项 | 值 |
|---|---|
| 光谱矩阵 | `processed/qm9s/prepared/raman_float32.npy`，`(129817, 3501)`，float32，500–4000 cm⁻¹，步长 1.0 cm⁻¹（`raman_axis.npy`：n=3501，min=500.0，max=4000.0，step=1.000000） |
| 标签矩阵 | `processed/qm9s/prepared/functional_group_labels.npy`，`(129817, 14)`，uint8 |
| 标签顺序 | `functional_group_label_names.json`（字母序，14 类） |
| 归一化 | 逐谱最大绝对值归一化（`normalization="max"`），在 Dataset 内施加 |
| 划分 | `raman_scaffold_split_valid.npz`，Bemis–Murcko scaffold split，Train 103855 / Validation 12981 / Test 12981，三集互斥、无 scaffold 重叠；索引与 `ir_scaffold_split_valid.npz` 逐元素相同（已独立验证 `array_equal=True`，合计 129817 全唯一） |

**14 类标签**：aldehyde、alkene、alkyne、amide、amine、c_f_bond、carbonyl、ester、ether、hydroxyl、imine、ketone、nitrile、nitro。标签之间**非互斥**。

**评价指标**：Micro-F1、Macro-F1、Micro-Precision、Micro-Recall、mAP、Macro-AUROC；另记录逐标签 Precision/Recall/F1/AP/AUROC。mAP 与 Macro-AUROC 为阈值无关指标。

**数据使用纪律**：Training 用于参数训练与 BN 统计量重估；Validation 用于 Early Stopping、方法比较与逐标签阈值拟合；Test 仅用于最终评价、记录与一致性核验。本文档第 6 节的配置选择**全部基于 Validation**；Test 数值在第 7 节独立报告，**未用于**改变任何配置、阈值或 Seed 选择。

---

## 2. 六个独立训练模型

| 模型目录 | 损失 | Seed | 最优 epoch | 训练 epoch 数 | Validation mAP（选轮指标） |
|---|---|---:|---:|---:|---:|
| `runs/EXP-RAMAN-001-FULL` | 普通 BCE | 42 | 28 | 36 | 0.945821 |
| `runs/EXP-RAMAN-001-SEED123` | 普通 BCE | 123 | 21 | 29 | 0.950330 |
| `runs/EXP-RAMAN-001-SEED2026` | 普通 BCE | 2026 | 22 | 30 | 0.945433 |
| `runs/EXP-RAMAN-002-FULL` | Weighted BCE | 42 | 15 | 23 | 0.947035 |
| `runs/EXP-RAMAN-002-SEED123` | Weighted BCE | 123 | 24 | 32 | 0.957277 |
| `runs/EXP-RAMAN-002-SEED2026` | Weighted BCE | 2026 | 22 | 30 | 0.956228 |

**共同训练配置**（六模型一致，未做任何逐模型调整）：

- 模型：`SpectrumStructureModel`，1D-CNN，embedding 256，12 层 BatchNorm1d，14 维输出头。
- 优化：AdamW（lr 1e-3，weight decay 1e-4），batch 512，num_workers 4，最多 50 epochs，Early Stopping patience 8，AMP（fp16）。
- 最优轮选择：Validation mAP（阈值无关）。
- Weighted BCE：`sqrt(pos_weight)` 变换、权重上限 20；`pos_weight` **仅由 Training split 统计量计算**（三份 WBCE checkpoint 的 `final_pos_weights` 完全一致，nitro 取到上限 20.0）。
- **未执行任何一次重训练**；六份 Original checkpoint 自训练完成后保持一致（SHA256 见 `raman_ablation_registry.md` §5）。

**Seed 政策**：Seed 是重复实验因素，用于估计训练稳定性，不是需要优化的超参数。论文统计必须覆盖全部三个 Seed；**不依据 Test 分数选择 Seed**。Seed 42 为项目既定的 canonical 实例，仅用于工程演示与样例输出。

---

## 3. BN 诊断、BN 重估与阈值校准的完成状态

| 环节 | 数量 | 产物位置 |
|---|---:|---|
| Full BN Diagnostic | **6/6** | `runs/RAMAN-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| Training-only BN 重估 checkpoint | **6/6** | `runs/RAMAN-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| Original BN 阈值校准 | **6/6** | `runs/RAMAN-THRESHOLD-CALIBRATION/<run_name>/`（各 9 文件：6 个 IR 同构文件 + 3 个概率溯源文件） |
| Recalibrated BN 阈值校准 | **6/6** | `runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/`（各 9 文件） |
| 阈值校准合计 | **12/12** | 上述两个目录 |

**BN 重估协议**：在原始 checkpoint 的独立副本上进行；`reset_running_stats()` 后将 `momentum` 置为 `None`（按 batch 等权累计平均），仅将 BatchNorm 层置于 train 模式，用**完整 Training split 的 103855 条样本（203 批）**前向一遍，恢复 momentum 后回到 eval。可训练参数与优化器状态不变，原始 checkpoint 全程只读。算法与 Phase 2 诊断调用的是同一个函数 `diagnose_raman_batchnorm_full.reset_recompute_bn`，未做任何算法改动。未执行 Quick Diagnostic。

**安全核验（Gate A，10 项，六份全部 PASS）**：来源 SHA256 与 Original 一致；40 个非 BN 状态张量逐位不变；可训练参数逐位不变；36 个 BN buffer 是唯一允许变化的张量且全部发生变化；12 层名称与 `num_batches_tracked=203` 对齐诊断记录；Training-only 范围（103855 样本 / 203 批）与诊断互证；BN buffer 有限且 `running_var > 0`；BN 摘要与 Phase 2 记录一致（容差 1e-3，IR 同标准）；保存文件回读逐张量一致；Original checkpoint 的 SHA256 前后不变。

**数值复现（Gate B，六份全部 PASS）**：官方 FP16 协议、threshold 0.5、14 标签，六项 Validation 指标与 Phase 2 `after.validation` 的**最大绝对偏差仅 3.1e-5 ～ 6.7e-5**（容差 1e-3，用掉不到 7%）。详见 §9.5。

**阈值校准协议**：对每一种 BN 状态分别、独立地在 Validation 上逐标签最大化 F1（`precision_recall_curve` 的阈值候选），阈值在**计算任何 Test 指标之前**落盘；Test 不参与拟合。Recalibrated BN 的阈值由重估后的概率重新拟合，**未复用** Original BN 的阈值文件。每个 (模型, BN) 只做**一次**推理，两个阈值条件（0.5 与 Validation-F1）从同一组概率派生；该组概率连同 SHA256、checkpoint SHA256、split 规模、标签顺序、评估口径一并落盘（`probability_artifacts.json`）。

---

## 4. 24 条件消融矩阵与三 Seed 统计

矩阵维度：2 损失 × 3 Seed × 2 BN 状态 × 2 阈值策略 = **24 个评估条件**；Validation 24 条 + Test 24 条 = **48 条 split-level 记录**；逐标签 14 × 24 × 2 = **672 行**。12 次阈值校准是**执行任务**，其结果构成 24 条件中的 Validation-F1 分支，**不额外计为独立条件**。

**Validation，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | Micro-P | Micro-R | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | 0.910148 ± 0.002418 | 0.874560 ± 0.012002 | 0.936987 | 0.884808 | 0.947195 ± 0.002721 | 0.988253 ± 0.000093 |
| BCE | Original | Val-F1 | 0.917811 ± 0.001000 | 0.904803 ± 0.003325 | 0.921719 | 0.913961 | 0.947195 ± 0.002721 | 0.988253 ± 0.000093 |
| BCE | Recalibrated | 0.5 | 0.917119 ± 0.003632 | 0.882060 ± 0.016092 | 0.940413 | 0.894988 | 0.948835 ± 0.003363 | 0.989114 ± 0.000857 |
| BCE | Recalibrated | Val-F1 | 0.920907 ± 0.003517 | 0.907774 ± 0.005156 | 0.922488 | 0.919339 | 0.948835 ± 0.003363 | 0.989114 ± 0.000857 |
| WBCE | Original | 0.5 | 0.915862 ± 0.004744 | 0.903717 ± 0.005982 | 0.921584 | 0.910246 | 0.953513 ± 0.005634 | 0.989581 ± 0.000551 |
| WBCE | Original | Val-F1 | 0.921640 ± 0.002583 | 0.911725 ± 0.004536 | 0.917699 | 0.925627 | 0.953513 ± 0.005634 | 0.989581 ± 0.000551 |
| WBCE | Recalibrated | 0.5 | 0.914868 ± 0.006066 | 0.903369 ± 0.006663 | 0.927221 | 0.902872 | 0.954288 ± 0.004972 | 0.989701 ± 0.000428 |
| **WBCE** | **Recalibrated** | **Val-F1** | **0.922034 ± 0.002103** | **0.913559 ± 0.004126** | 0.913703 | 0.930572 | **0.954288 ± 0.004972** | **0.989701 ± 0.000428** |

Micro-P / Micro-R 的 SD 见 `mean_sd_8_conditions_each_split.csv`（本文档为排版省略）。

**Test，Mean ± 样本 SD（n = 3）**

| 损失 | BN | 阈值 | Micro-F1 | Macro-F1 | Micro-P | Micro-R | mAP | Macro-AUROC |
|---|---|---|---|---|---|---|---|---|
| BCE | Original | 0.5 | **0.915080 ± 0.001382** | 0.888480 ± 0.007879 | 0.925687 | 0.904715 | 0.954789 ± 0.004389 | 0.991103 ± 0.000418 |
| BCE | Original | Val-F1 | 0.914346 ± 0.001829 | 0.900003 ± 0.004514 | 0.900879 | 0.928242 | 0.954789 ± 0.004389 | 0.991103 ± 0.000418 |
| BCE | Recalibrated | 0.5 | **0.919004 ± 0.002212** | 0.898904 ± 0.009189 | 0.925910 | 0.912233 | 0.957060 ± 0.004043 | 0.991520 ± 0.000593 |
| BCE | Recalibrated | Val-F1 | 0.915002 ± 0.003411 | 0.903324 ± 0.003829 | 0.897922 | 0.932772 | 0.957060 ± 0.004043 | 0.991520 ± 0.000593 |
| WBCE | Original | 0.5 | 0.912447 ± 0.009032 | 0.906279 ± 0.009626 | 0.901302 | 0.923906 | 0.960983 ± 0.005204 | **0.991782 ± 0.000636** |
| WBCE | Original | Val-F1 | 0.915652 ± 0.005579 | 0.910872 ± 0.005765 | 0.895920 | 0.936282 | 0.960983 ± 0.005204 | **0.991782 ± 0.000636** |
| WBCE | Recalibrated | 0.5 | 0.912907 ± 0.011422 | 0.908201 ± 0.009865 | 0.907048 | 0.918865 | **0.962030 ± 0.005486** | 0.991753 ± 0.000922 |
| **WBCE** | **Recalibrated** | **Val-F1** | 0.914113 ± 0.006869 | **0.912085 ± 0.005732** | 0.889281 | 0.940436 | **0.962030 ± 0.005486** | 0.991753 ± 0.000922 |

---

## 5. 主要配对实验发现（同 Seed 配对，Mean ± 样本 SD，n = 3）

数据源：`runs/RAMAN-ABLATION-SUMMARY/paired_effects_summary.csv`（28 行 = 4 比较族 × 2 split × …）。

### 5.1 阈值校准（Validation-F1 阈值 − 默认 0.5）

| 损失 / BN | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 | Test ΔMacro-F1 |
|---|---|---|---|---|
| BCE / Original | +0.007663 ± 0.002870 | +0.030243 ± 0.009331 | −0.000734 ± 0.003167 | +0.011523 ± 0.004689 |
| WBCE / Original | +0.005779 ± 0.002262 | +0.008008 ± 0.001592 | +0.003205 ± 0.006499 | +0.004593 ± 0.005669 |
| BCE / Recalibrated | +0.003788 ± 0.000387 | +0.025714 ± 0.012915 | −0.004002 ± 0.002277 | +0.004420 ± 0.006749 |
| WBCE / Recalibrated | +0.007165 ± 0.004086 | +0.010190 ± 0.003915 | +0.001207 ± 0.005133 | +0.003884 ± 0.005438 |

mAP 与 Macro-AUROC 在两种阈值下**严格相等**（阈值无关指标的恒等性在汇总阶段被断言检查，Δ 恒为 0.000000 ± 0.000000）。

### 5.2 BN 重估（Recalibrated − Original，同阈值）

| 损失 / 阈值 | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Test ΔMicro-F1 | Test ΔMacro-F1 | Test ΔmAP |
|---|---|---|---|---|---|
| BCE / 0.5 | +0.006972 ± 0.004659 | +0.007500 ± 0.005029 | +0.003924 ± 0.002165 | +0.010424 ± 0.002843 | +0.002270 ± 0.002388 |
| WBCE / 0.5 | −0.000993 ± 0.002221 | −0.000348 ± 0.002343 | +0.000460 ± 0.002766 | +0.001922 ± 0.000978 | +0.001047 ± 0.000402 |
| BCE / Val-F1 | +0.003096 ± 0.002558 | +0.002971 ± 0.002556 | +0.000656 ± 0.002638 | +0.003322 ± 0.000916 | +0.002270 ± 0.002388 |
| WBCE / Val-F1 | +0.000393 ± 0.000575 | +0.001834 ± 0.001551 | −0.001538 ± 0.002729 | +0.001213 ± 0.001077 | +0.001047 ± 0.000402 |

BN 重估对**排序能力**的影响很小：Validation ΔmAP 为 BCE +0.001641 ± 0.002315、WBCE +0.000775 ± 0.001021；Test ΔmAP 为 +0.002270 / +0.001047。方向在损失之间不一致，量级与跨 Seed 波动同阶。

### 5.3 BN 与阈值校准的交互

| 损失 | 交互项（(Recal−Orig\|ValF1) − (Recal−Orig\|0.5)）ΔMicro-F1 | ΔMacro-F1 | Test ΔMicro-F1 | Test ΔMacro-F1 |
|---|---|---|---|---|
| BCE | −0.003876 ± 0.003254 | −0.004529 ± 0.003803 | −0.003268 ± 0.001234 | −0.007103 ± 0.002211 |
| WBCE | +0.001387 ± 0.002760 | +0.002182 ± 0.003876 | −0.001998 ± 0.002843 | −0.000709 ± 0.000268 |

即：BCE 下 BN 重估的收益在阈值校准后被部分抵消（方向为负），WBCE 下方向相反。交互量级与主效应同阶，说明 **BN 处理与阈值策略不宜分开单独评价**。

### 5.4 Weighted BCE 与普通 BCE

| 配置对比（WBCE − BCE） | Validation ΔMicro-F1 | Validation ΔMacro-F1 | Validation ΔmAP | Test ΔMicro-F1 | Test ΔMacro-F1 | Test ΔmAP |
|---|---|---|---|---|---|---|
| Original / 0.5 | +0.005714 ± 0.005892 | +0.029157 ± 0.010444 | +0.006318 ± 0.004819 | −0.002634 ± 0.009155 | +0.017800 ± 0.007050 | +0.006193 ± 0.003184 |
| Original / Val-F1 | +0.003829 ± 0.001610 | +0.006922 ± 0.002705 | +0.006318 ± 0.004819 | +0.001305 ± 0.003989 | +0.010869 ± 0.002025 | +0.006193 ± 0.003184 |
| Recalibrated / 0.5 | −0.002251 ± 0.003249 | +0.021309 ± 0.015032 | +0.005452 ± 0.001688 | −0.006097 ± 0.009544 | +0.009297 ± 0.010780 | +0.004970 ± 0.001452 |
| Recalibrated / Val-F1 | +0.001127 ± 0.001614 | +0.005785 ± 0.001236 | +0.005452 ± 0.001688 | −0.000889 ± 0.003467 | +0.008761 ± 0.001905 | +0.004970 ± 0.001452 |

**与 IR v1 的显著差异**：在 IR v1 中 WBCE 在八个配置上的 Validation Micro-F1 一致低于 BCE，因而被排除；在 Raman v1 中，WBCE 的 **Macro-F1 与 mAP 在四个配对比较上全部为正**（Macro-F1 +0.0058 ~ +0.0292，mAP +0.0055 ~ +0.0063），而 Micro-F1 方向不一致（三正一负）。结论必须按指标分开表述，不能笼统说"WBCE 更好"或"更差"。

---

## 6. 决策：主方案、备选方案与固定基线

**决策依据的唯一来源是 Validation。** 下表把八组配置按 Validation Micro-F1 降序排列（数值取自 `mean_sd_8_conditions_each_split.csv`）：

| 排名 | 配置 | Validation Micro-F1 | Macro-F1 | mAP | Macro-AUROC |
|---:|---|---|---|---|---|
| 1 | **WBCE + Recalibrated + Val-F1（主方案）** | **0.922034 ± 0.002103** | **0.913559 ± 0.004126** | **0.954288 ± 0.004972** | **0.989701 ± 0.000428** |
| 2 | WBCE + Original + Val-F1（备选） | 0.921640 ± 0.002583 | 0.911725 ± 0.004536 | 0.953513 ± 0.005634 | 0.989581 ± 0.000551 |
| 3 | BCE + Recalibrated + Val-F1 | 0.920907 ± 0.003517 | 0.907774 ± 0.005156 | 0.948835 ± 0.003363 | 0.989114 ± 0.000857 |
| 4 | BCE + Original + Val-F1 | 0.917811 ± 0.001000 | 0.904803 ± 0.003325 | 0.947195 ± 0.002721 | 0.988253 ± 0.000093 |
| 5 | BCE + Recalibrated + 0.5 | 0.917119 ± 0.003632 | 0.882060 ± 0.016092 | 0.948835 ± 0.003363 | 0.989114 ± 0.000857 |
| 6 | WBCE + Original + 0.5 | 0.915862 ± 0.004744 | 0.903717 ± 0.005982 | 0.953513 ± 0.005634 | 0.989581 ± 0.000551 |
| 7 | WBCE + Recalibrated + 0.5 | 0.914868 ± 0.006066 | 0.903369 ± 0.006663 | 0.954288 ± 0.004972 | 0.989701 ± 0.000428 |
| 8 | BCE + Original + 0.5（固定基线） | 0.910148 ± 0.002418 | 0.874560 ± 0.012002 | 0.947195 ± 0.002721 | 0.988253 ± 0.000093 |

### 6.1 主推理方案：**WBCE + Recalibrated BN + Validation-F1 Threshold**

理由（按可核查程度排序）：

1. 三 Seed Validation Micro-F1 = **0.922034 ± 0.002103**，为八组配置中的最高均值；Validation Macro-F1 = **0.913559 ± 0.004126**，亦为八组最高。Validation mAP = 0.954288 ± 0.004972 与 **同一个 Recalibrated checkpoint 的固定 0.5 阈值条件并列最高**，Macro-AUROC = 0.989701 ± 0.000428 同理并列最高。
2. **mAP 与 Macro-AUROC 不受分类阈值影响**：同一 (模型, BN 状态) 的 0.5 与 Validation-F1 两条条件共享同一组预测概率的排序，因而这两项数值完全相同。因此"主方案在 mAP/AUROC 上最高"与"该 checkpoint 的 0.5 条件在 mAP/AUROC 上最高"是同一件事，**不得**当作两条独立的增益证据；阈值策略的实际作用只体现在 F1 家族（Micro/Macro-F1、Precision、Recall）上。
3. 相对备选方案（第 2 名）的四项领先幅度分别为 **+0.000394 / +0.001834 / +0.000775 / +0.000120**，**全部远小于对应指标的跨 Seed SD**（0.0021 / 0.0041 / 0.0050 / 0.0004）。因此这不是"可区分的优势"，而是**在同一噪声水平内的微弱排序**；选择它是因为它在这四项上同时不落后，而不是因为它被证明更优，也**不宣称**具有统计显著性。
4. 它对"14 类宏平均 + 排序质量"这一类下游用途最匹配：在 Validation 上同时占据 Micro-F1 与 Macro-F1 的均值首位。
5. 其 Validation-F1 阈值文件与 Recalibrated checkpoint 必须**成对加载**（见附录 B），不得与 Original BN 的阈值混用。

### 6.2 备选方案：**WBCE + Original BN + Validation-F1 Threshold**

- Validation Micro-F1 = 0.921640 ± 0.002583（与主方案差 0.000394，见 §6.1 第 2 条）。
- **定位**：当部署链路需要最短、最少额外制品时采用。它只需 `runs/<run>/best_model.pt` + `runs/RAMAN-THRESHOLD-CALIBRATION/<run>/optimal_thresholds.json` 两个文件，不需要 BN 重估 checkpoint，少一个失效点。
- 损失与阈值策略与主方案相同，**唯独 BN 状态不同**，因此两者可以直接对照"是否值得为 +0.0004 的 Validation Micro-F1 引入 BN 重估链路"这一问题——本次不宣称值得，只记录数值。

### 6.3 BCE 家族最佳：**BCE + Recalibrated BN + Validation-F1 Threshold**

- Validation Micro-F1 = 0.920907 ± 0.003517（第 3 名，与主方案差 0.001127）。
- **定位**：保留为"不采用类别加权"的等价对照。若后续研究或部署场景不希望使用 `pos_weight`（例如标签分布改变、或需要与其他模态的损失保持一致），这是 Raman 上的首选 BCE 配置。
- 它与主方案的差距（0.0011）小于 Micro-F1 的跨 Seed SD（0.0021），同样**不构成可区分差异**。

### 6.4 固定参考基线：**BCE + Original BN + Threshold 0.5**

- 作为后续研究、消融与融合实验的**标准比较对象**保留，不因任何单次 Test 数值而调整。
- 其价值在于：不含任何 Validation 拟合成分（阈值固定 0.5）、不含 BN 后处理，是唯一"零后处理"的参照点，也是训练脚本的默认输出口径。
- **不根据 Test 上最高 Micro-F1 事后改变研究决策。** 事实上 Test Micro-F1 最高的配置是 BCE + Recalibrated + 0.5（0.919004），而它在 Validation 上仅列第 5；这一现象在 §7、§8 如实记录，**不作为**改选主方案的理由。

### 6.5 部署 checkpoint 的选择规则（与"方法组合选择"分开记录）

方法组合（损失 × BN × 阈值策略）由 §6.1–6.4 按 **Validation** 决定。**部署实例（用哪一个 Seed 的权重）遵循另一条规则，且不使用 Test 分数**：

| 用途 | 规则 | 理由 |
|---|---|---|
| 论文/报告的主数值 | 三 Seed 的 Mean ± 样本 SD | 反映训练随机性，不挑选"最好看的 Seed" |
| 工程演示与样例输出 | **Seed 42（项目 canonical）** | 与 IR v1、UVSPEC15 v1 的既有约定一致；与性能无关 |
| 若需单模型部署 | 预先指定 Seed 42；如需更强鲁棒性，改用三模型概率平均（本阶段未做该实验） | 避免"按 Test 挑 Seed" |
| 禁止 | 依据 Test Micro-F1/Macro-F1/mAP 挑选 Seed 或挑选用哪一次推理 | 见 §11、§12 |

### 6.6 全部配置保留

八种配置的 checkpoint、BN 重估文件、阈值文件、概率数组与评估记录全部保留在归档中，任何后续研究均可复现；**不删除任何表现较差的实验结果**。

---

## 7. 主方案的独立 Test 评价

**声明**：本节数值为**固定的**主方案在 Test 上的独立评价，用于报告与一致性核验。Test 数值**未参与**第 6 节的任何选择。

| 指标 | Test（Mean ± 样本 SD，n=3） |
|---|---|
| Micro-F1 | 0.914113 ± 0.006869 |
| Macro-F1 | **0.912085 ± 0.005732**（八组最高） |
| Micro-Precision | 0.889281 |
| Micro-Recall | 0.940436（八组最高） |
| mAP | **0.962030 ± 0.005486**（八组最高，与 WBCE/Recalibrated/0.5 并列，阈值无关） |
| Macro-AUROC | 0.991753 ± 0.000922 |

**逐标签 Test F1（主方案，三 Seed 均值；数据源 `per_label_f1_levels.csv`）**

| 标签 | Validation 正样本 | Test 正样本 | 支持度 | Validation F1 | Test F1 |
|---|---:|---:|---|---:|---:|
| alkyne | 1708 | 1514 | normal | 0.996688 | 0.997906 |
| nitrile | 1697 | 1517 | normal | 0.995290 | 0.994292 |
| hydroxyl | 3949 | 4334 | normal | 0.991461 | 0.991840 |
| carbonyl | 4238 | 3898 | normal | 0.984699 | 0.982791 |
| alkene | 2914 | 1745 | normal | 0.984019 | 0.959040 |
| aldehyde | 1447 | 1284 | normal | 0.979228 | 0.964911 |
| **nitro** | **11** | **15** | **very low** | 0.861284 | 0.964696 |
| imine | 1191 | 1295 | normal | 0.914351 | 0.907477 |
| amide | 1297 | 1053 | normal | 0.868521 | 0.860201 |
| ketone | 1223 | 1272 | normal | 0.858715 | 0.856771 |
| amine | 3728 | 3491 | normal | 0.844729 | 0.801984 |
| c_f_bond | 230 | 144 | low | 0.842798 | 0.836450 |
| ether | 5794 | 5260 | normal | 0.834197 | 0.854999 |
| ester | 633 | 622 | normal | 0.833844 | 0.795830 |

**Validation 决策 vs Test 报告的差异（如实记录）**

1. 主方案的 Test Micro-F1（0.914113）在**八组中排第 6**，**低于** Test Micro-F1 最高的 BCE + Recalibrated + 0.5（0.919004，差 −0.004891）。完整降序为：BCE/Recalibrated/0.5 > WBCE/Original/Val-F1 > BCE/Original/0.5 > BCE/Recalibrated/Val-F1 > BCE/Original/Val-F1 > **WBCE/Recalibrated/Val-F1** > WBCE/Recalibrated/0.5 > WBCE/Original/0.5。
2. 主方案的 Test Macro-F1（0.912085）为八组最高；Test mAP（0.962030）亦为最高，但与同一 Recalibrated checkpoint 的 0.5 条件并列（阈值无关指标，属同一 checkpoint 的重复读数，不是独立证据）。
3. 主方案的 Test Micro-Precision 仅 0.889281（八组最低），Micro-Recall 0.940436（八组最高）：校准后的整体操作点更偏向高召回、较低精确率，把更多样本判正。这与 Validation 上 Precision 同样偏低（0.913703）、Recall 偏高（0.930572）一致，属**同一操作点位移**在两组上的延续，而非两集评价方向相反。
4. **排名不得用于改选**：Test Micro-F1 的第 1 名（BCE + Recalibrated + 0.5）在 Validation 上仅列第 5，本记录**不因此**改变 §6 的决策（见 §11、§12）。

---

## 8. 讨论

### 8.1 阈值校准的收益在 Validation 上明显、在 Test 上按损失分叉

Validation 上，逐标签阈值校准一致提升四个配置的 Micro-F1（+0.0038 ~ +0.0077）与 Macro-F1（+0.0080 ~ +0.0302）；Test 上则出现**按损失分叉**：BCE 两组的 Test Micro-F1 略微下降（−0.0007、−0.0040），WBCE 两组略微上升（+0.0032、+0.0012），而四个配置的 Test Macro-F1 **全部为正**（+0.0039 ~ +0.0115）。

解释（按可支持程度排序）：
1. **样本内优化偏差（可直接支持）**：阈值直接在 Validation 上最大化逐标签 F1，其 Validation 增益在定义上含乐观偏差，不能作为泛化证据。
2. **决策边界改变导致的整体操作点位移（可直接支持）**：逐标签阈值校准改变了各标签的分类决策边界，在当前数据分布下使整体预测操作点向**更高召回率、较低精确率**的方向移动（如主方案 Test P 0.8893 / R 0.9404 vs 固定基线 P 0.9257 / R 0.9047）。**不能概括为"阈值整体低于 0.5"**：14 个标签的最优阈值方向并不一致，12 次 Raman 校准中"高于 0.5 的标签数"在 **3～11** 之间（例如主方案 Seed 42 为 9 个 >0.5 / 5 个 <0.5，而 BCE Seed 123 的 Recalibrated 校准为 3 个 >0.5 / 11 个 <0.5）。Micro-F1 的变化由精确率-召回率权衡决定，而排序能力（mAP/AUROC）完全相同。
3. **Micro-F1 与 Macro-F1 的汇总方式不同（可直接支持，但不指向具体标签）**：Micro-F1 汇总全部标签的 TP、FP、FN，通常更容易被高频标签的预测结果主导；Macro-F1 对 14 个标签的 F1 做等权平均，因此相对于 Micro-F1，低支持度标签的变化可能对均值产生更大影响。这可以解释"Test 端 Macro-F1 四项一致为正、而 Micro-F1 方向不定"的**方向性差异**；但**不得**把 Macro-F1 的提高直接等同于所有低频类别都得到改善——它只表明等权平均发生了移动，具体哪些标签改善必须查 `per_label_f1_levels.csv` 与 `per_label_deltas.csv`。
4. **不可下结论的部分**：本文档**不主张**"阈值校准无效"或"阈值校准有效"。若要把阈值策略作为可泛化的方法学主张，必须另设嵌套/交叉拟合式的校准评估；本阶段未设计该实验。

### 8.2 BCE 与 WBCE 的差异

- **Validation**：WBCE 在四个配对比较上的 Macro-F1 与 mAP 全部为正（Macro-F1 +0.0058 ~ +0.0292，mAP +0.0055 ~ +0.0063），Micro-F1 三正一负（−0.0023 ~ +0.0057）。
- **Test**：Macro-F1 四项全部为正（+0.0088 ~ +0.0178），Micro-F1 三项为负（−0.0009 ~ −0.0061）。
- **机制解释（弱支持）**：`pos_weight` 提高稀有类权重，直接改善宏平均（每类等权）与稀有类召回；而 Micro-F1 由样本量主导，加权后常见类的精确率可能下降。这与"WBCE 抬高 Recall、压低 Precision"的配对方向一致。
- **与 IR v1 的对照**：IR v1 中 WBCE 在八个配置的 Validation Micro-F1 上一致更低，故被排除；Raman v1 不能照搬该结论——Raman 上 WBCE 在 Macro-F1/mAP 上一致更好，这也是本次主方案与 IR 主方案不同的直接原因。
- **不宣称**：不声称 WBCE 在 Raman 上显著优于 BCE（n=3，差值 0.0004~0.0058 与跨 Seed SD 同阶）。

### 8.3 BN 重估的独立效应与交互

- 独立效应：Validation 上 BCE 组为 +0.0070（0.5）/ +0.0031（Val-F1）Micro-F1，WBCE 组为 −0.0010 / +0.0004；方向随损失变化，量级与 SD 同阶。
- 对排序能力几乎无影响：Validation mAP 变化 +0.0016（BCE）/ +0.0008（WBCE）。
- 与阈值的交互：BCE 为 −0.0039 ± 0.0033，WBCE 为 +0.0014 ± 0.0028，与主效应同阶 → **BN 与阈值策略必须联合评价**，单独看任一项都可能得到相反结论。
- 结论：BN 重估**不被假定为有益**；主方案采用它仅因为在 Validation 上四项同时最高（且幅度在噪声内）。

### 8.4 Validation 与 Test 的不一致

除 §7 已记录的 Micro-F1 排名差异外，还有一处内部排序差异值得记录。按 §4 的原始汇总：

- **Validation Micro-F1 前四名**：WBCE/Recalibrated/Val-F1 > WBCE/Original/Val-F1 > BCE/Recalibrated/Val-F1 > BCE/Original/Val-F1；
- **Validation Macro-F1 前四名**：**顺序完全相同**。

差异出现在第 5～7 名：Micro-F1 为 BCE/Recalibrated/0.5 > WBCE/Original/0.5 > WBCE/Recalibrated/0.5，而 Macro-F1 为 WBCE/Original/0.5 > WBCE/Recalibrated/0.5 > BCE/Recalibrated/0.5（第 8 名两种指标均为 BCE/Original/0.5）。

也就是说：**前四名的两种指标排序一致，中低名次出现互换**。这说明在 n=3、差值为 1e-3 量级时，单一指标对中低名次配置的排序并不稳定；本文档因此**同时报告四项 Validation 指标**，而不是只挑一个，也不从这种中低名次的互换中提炼任何新的结论。

### 8.5 稀疏标签 nitro

- Validation 正样本 **11**、Test 正样本 **15**，属 very low 支持度（`per_label_f1_levels.csv` 的 `support_tier`）。
- 主方案下 Validation F1 = 0.861284、Test F1 = 0.964696，两者差 0.103，是 14 类中最大的集间差异——**这本身就是小样本估计不稳定的直接证据**，不能解读为"Test 更容易"。
- 现有证据（`runs/RAMAN-BN-DIAGNOSTICS/` 的 FP16 诊断与 T1/T2 逐标签比对）显示：在已观测到的 mAP 跨进程波动中，**`nitro` 的 AP 是主要载体**（逐标签定位，不是机制断言）。因此凡是涉及 mAP 的微小差异，都要先检查该标签的杠杆效应；至于为何由它承载，见 §8.6 的"尚未证明"部分。
- 本文档**不主张**移除 nitro 或任何稀疏标签后的性能；该敏感性分析未在本阶段执行，且阈值拟合仍在 14 类上完成。

### 8.6 FP16 数值复现问题及其处理（重要方法学记录）

**现象**：Phase 3 保存第一份 BN checkpoint（BCE seed 42）时，官方 FP16 Validation mAP 读数为 0.946294，而 Phase 2 诊断为 0.945208，偏差 1.086e-3，超过当时的 1e-3 判据。

**直接观测（本阶段实测，未做任何推断）**（T1/T2 专项实验，`training/diagnose_raman_validation_repro.py`）：同一 checkpoint、不同 Python 进程，官方 FP16 Validation mAP 分别为 **0.945208** 与 **0.946288**（差 **1.080e-3**）；同一两次运行改用 FP32 sigmoid 则分别为 **0.946809** 与 **0.946807**（差 2e-6）；进程内重复推理的概率数组**逐位一致**；逐标签比对显示该差异几乎全部由 `nitro` 的 AP 承载（Validation 仅 11 个正样本）。

**两个必须区分的数值（不可互相替代）**：

| 事件 | 读数对 | 差值 |
|---|---|---|
| 最初 Gate B 失败（Phase 3 保存后读数 vs Phase 2 记录） | 0.946294 vs 0.945208 | **1.086e-3** |
| T1/T2 跨进程复现对照（同一文件的两次独立进程读数） | 0.946288 vs 0.945208 | **1.080e-3** |

两个读数本身相差 6e-6：0.946294 来自保存流程的那次读数，0.946288 来自 T1 独立进程。**引用时必须指明是哪一对读数**——不得把 1.086e-3 当作跨进程复现的测量值，也不得把 1.080e-3 当作 Gate B 的失败量。

**历史 manifest 的处理（与本节的对应关系）**：六份已冻结的 `bn_recalibration_manifest.json` 的 `gate_policy.measured` 块中仍保留旧值（把 `0.001086` 与 T1/T2 配对 `[0.945208, 0.946288]` 写在一起）。该字段仅用于历史描述，不参与任何门禁、容差或判定；**manifest 原样保留、不作事后覆盖**，勘误记录见 `raman_ablation_registry.md` §9.5。代码常量已在 `training/save_raman_bn_checkpoint.py` 的 `GATE_POLICY["measured"]` 中更正为 `1.080e-3` 并附算术归属注释（仅影响此后新生成的 manifest）。

**合理解释（有方向性证据，但未完成严格归因）**：进程内完全可复现、跨进程出现差异，说明差异来自**进程级数值路径**而非数据或权重；FP16 sigmoid 的输出落在约 4.9e-4 量级的量化格点上，近邻预测值的微小变化可能改变并列样本的相对次序，从而在 AP 这类秩统计量上被放大；`nitro` 仅 11 个正样本，单个样本的位次变动对其 AP 的杠杆最大。该解释与"FP32 sigmoid 稳定、FP16 波动"这一对照一致。

**尚未证明（不得写成结论）**：

- 差异具体由哪一个 CUDA/cuDNN 算子或算法选择造成，**未做算子级归因**；
- 该误差的统计分布形态（是否近似某种抽样分布、尾部多厚）**未测量**，因此**不能**把历史越界断言为"统计意义上的极端抽样"；
- 也不能排除其它进程级数值来源（归约顺序、算子实现选择等）与上述机制共同作用。

**处理**：
1. **拒绝**把 mAP 容差放宽到 3e-3（曾短暂实现为 gate v2，未投入使用即撤回）：这会把失败改判为通过，且破坏与 IR/UVSPEC15 的方法学一致性。
2. 保留 **1e-3 全指标容差**，把"制品完整性"与"数值复现"拆成两道独立门（Gate A / Gate B），并把 `MAP_ONLY_NUMERIC_MISMATCH` 作为**纯描述性**标签（只说明哪些指标越界，不宣称原因）。
3. 建立**非门控的辅助复现路径**（`training/verify_raman_bn_reproducibility.py`）：固定 CUDA 算法（`cudnn.benchmark=False`、`cudnn.deterministic=True`、`use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）+ FP32 sigmoid + 概率数组逐位比对。该路径**只用于归因**，从不替代官方 FP16 指标。

**结果**：六份**最终获准归档**的 checkpoint 均通过 v5 正式验收（Gate A 10/10、Gate B 六项均 ≤1e-3，最大偏差 6.7e-5，占容差不到 7%），归档记录中没有任何一份触发 `MAP_ONLY_NUMERIC_MISMATCH`。

**早期失败历史予以保留**（不因最终通过而删除）：seed 42 的首次保存曾以 mAP 偏差 1.086e-3 触犯 Gate B（本节上文"现象"段）；seed 123 的首次保存曾因 Gate A 第 09 项的执行顺序缺陷而中止（§8.7）。两者都不是制品损坏，且都经修复或重新验收后才进入最终归档。

上述事实说明：在本批制品与本判据下，数值复现具备充足余量。它**不构成**"历史越界属于极端抽样"的证明——那需要误差分布的测量，本阶段没有做——也**不改变**任何容差或判据。

### 8.7 Gate A 第 09 项的执行顺序缺陷（已修复）

**缺陷**：`cmd_save()` 在写盘**之前**评估包含第 09 项（重载完整性）的完整清单，而第 09 项在彼时按设计仍处于 `pending`（`pass=False`），导致正常保存路径必然以 `09_checkpoint_reload_integrity` 失败。BCE seed 123 首次走该路径时暴露；其时 BN 重估已完成但未落盘任何文件。

**修复**：新增 `Checklist.status_ignoring()`，写盘前的门禁**只排除** 09 这一项（其余 9 项照常判定），写盘后再以完整清单（含 09）决定是否写 manifest；09 的占位仍为 `False`，任何遗漏调用 `finish_reload_check()` 的路径依旧会 FAIL（fail-closed 性质保留）。

**验证**：`training/test_raman_bn_reload_integrity.py`（CPU）——门禁顺序 8 项断言；合成 checkpoint 回读 7/7 张量；**真实 Seed 123 checkpoint 76/76 张量回读一致**、36 个 BN buffer 保留；1e-6 单元素改动可被检出。修复后五份剩余模型与 seed 123 全部一次通过。

### 8.8 六份 BN checkpoint 与 Phase 4–6 的验收状态

- **Phase 3**：6/6 manifest 满足 `artifact_integrity=PASS`、`numeric_reproduction=PASS`、`acceptance=ARTIFACT_PASS__NUMERIC_PASS`、`ablation_release=ELIGIBLE`；`source_sha256` / `saved_sha256` 与磁盘实测一致；Original checkpoint 未被修改。
- **Phase 4–5**：12 次阈值校准全部完成；`runs/RAMAN-NUMERIC-AUDIT/` 下 **12 份数值审计记录全部 `PASS`**（`exceeding_metrics` 为空）；每份校准都落盘了 `probability_artifacts.json` 与两组概率数组，SHA256 与实际文件一致。
- **Phase 6**：24 Validation + 24 Test 条件、672 条逐标签记录、11 个汇总文件；`audit_checks.csv` 的 **6 模型 × 4 项 = 24 项全部 True**；`paired_effects.csv` 84 行、`per_label_deltas.csv` 224 行、`condition_registry.csv` 48 行。
- **阶段范围（避免与 Phase 2–3 的 BN 重估混淆）**：BN 的 Training-only 重估只发生在 **Phase 2**（诊断脚本在模型副本上重估一次）与 **Phase 3**（保存前对每个模型各重估一次）；**自 Phase 4 起至 Phase 7**，未再执行任何重训练、BN 重算或重新拟合。阈值仅在 Phase 4（Original BN）与 Phase 5（Recalibrated BN）各拟合一次，此后未再拟合；历史产物（checkpoint、诊断 JSON、校准摘要、汇总 CSV）在上述过程中均未被改写。

---

## 9. 低样本量标签与统计边界

### 9.1 支持度分层

| 标签 | Validation 正样本 | Test 正样本 | 分层 |
|---|---:|---:|---|
| **nitro** | **11** | **15** | very low |
| **c_f_bond** | **230** | **144** | low |
| ester | 633 | 622 | normal |
| 其余 11 类 | 1191–5794 | 1053–5260 | normal |

### 9.2 统计解释边界

- 全部区间为 **n = 3 个随机 Seed 的均值 ± 样本标准差**（ddof = 1），用于描述训练随机性，**不是**置信区间。
- **本文档不作任何显著性断言**：不声称主方案优于备选、不声称 WBCE 优于 BCE、不声称 BN 重估有效、不声称阈值校准优于 0.5。上述比较的差值均与跨 Seed 波动同阶或更小（§6.1 第 2 条给出四组具体对照）。
- n = 3 时样本标准差自身的估计误差很大（相对量级约 50%），因此"哪个配置 SD 更小"同样不能作为稳健性结论。
- `c_f_bond`（Test 144）与 `nitro`（Validation 11 / Test 15）的逐标签指标抽样波动显著，宏平均结论应结合 `per_label_f1_levels.csv` 阅读。

---

## 10. 研究边界

- **域边界**：全部结论限于 **QM9S 计算拉曼光谱 + scaffold split**。计算谱与实验谱（API-Raman、NIST）在峰位、峰宽、基线、荧光背景与噪声特性上存在物理差异，本阶段**未做**实验域验证或迁移。
- **任务边界**：输出为 14 类官能团多标签，属**结构特征识别**，不是完整分子结构推断；不涉及 Top-k 候选生成、结构渲染或化学规则重排。
- **系统边界**：当前为 Raman **单模态**模型；多模态证据融合、开放集拒识、XAI 与自然语言解释均未完成。
- 因此，"Raman v1 冻结"指**单模态基线的流程、权重与配置被冻结并可复现**，不表示项目整体目标达成。

### 10.1 面向多模态融合的接口与产物

**可复用的部分**

- **编码器表征**：`SpectrumStructureModel.encoder` 输出的 **256 维 embedding**（`AdaptiveAvgPool1d(16)` + `Linear(2048→256)`）。
- **样本对齐标识**：`row_index`（0–129816），与光谱矩阵、标签矩阵、结构库一致。Raman split 覆盖全部 **129817** 行，与 IR 逐元素相同。

**不可误用的部分**

- **14 个分类阈值属于 Raman 输出头**，不是 embedding 的组成部分，不得作为融合特征的阈值或归一化参数。
- **BN running statistics 不得跨模态迁移**：Raman 的 12 层 BN 统计量是在 Raman 光谱分布上估计的，与 IR 的分布不同；融合训练必须重新估计。
- **两套 BN 状态（Original / Recalibrated）的 embedding 不同**：融合实验必须显式声明使用哪一套，并保证训练与推理一致。
- **阈值与 BN 状态必须成对**：Recalibrated checkpoint 只能配 `runs/RAMAN-BN-THRESHOLD-CALIBRATION/` 的阈值，Original checkpoint 只能配 `runs/RAMAN-THRESHOLD-CALIBRATION/` 的阈值。

**融合实验应引用的产物**

| 用途 | 路径 |
|---|---|
| 主方案权重 | `runs/RAMAN-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt`（+ manifest） |
| 主方案阈值 | `runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 备选方案权重 | `runs/<run_name>/best_model.pt` |
| 备选方案阈值 | `runs/RAMAN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 固定基线（零后处理） | `runs/<run_name>/best_model.pt` + 阈值 0.5 |
| 单模态性能基线（融合增益对照） | `runs/RAMAN-ABLATION-SUMMARY/test_24_conditions.csv`、`mean_sd_8_conditions_each_split.csv` |
| 推理配置元数据 | `runs/<run_name>/config.json` |

**对齐要求**：融合实验须以固定基线（BCE + Original + 0.5）作为单模态下限对照，并记录其与融合模型的差异；不得用 Test 数值挑选参与融合的单模态 Seed。

---

## 11. 决策性质：事后形成，非预注册

本文档的配置决策是在 24 条件消融**全部完成、并已查看 Validation 与 Test 结果之后**形成的。因此：

- 它是**探索性决策记录**，不得表述为事先注册的分析方案或预注册假设检验。
- **历史策略文档没有预先定义唯一的主选择指标**（`docs/experiments/IR_v1_experimental_strategy.md` 明确写明这一空缺，Raman 沿用同一文本）。本文档采用"Validation 四项指标同时最高 + 链路复杂度"这一**事后明确写出**的取舍依据，而不是引用一个并不存在的预注册规则。
- 六模型训练、BN 诊断、BN 重估与阈值校准的**流程**先于结果存在（脚本与判据在跑之前已固定），但"八组配置的最终取舍"这一**决策动作本身是事后的**。
- 为进一步降低解释弹性，取舍理由被限定为可核查的 Validation 数值（§6）与链路复杂度（§6.2），并明确承诺**不因 Test 数值改选**（§6.4、§6.5）。
- 未来若以 Raman v1 为基线开展新实验，配置应作为**已知固定基线**引用；如需改变，应以新实验编号登记并说明与 v1 的可比性。

---

## 12. 附录 A：冻结清单（引用路径）

| 类别 | 路径 |
|---|---|
| 训练脚本 | `training/train_raman_full.py`、`train_raman_seed123.py`、`train_raman_seed2026.py`、`train_raman_weighted.py`、`train_raman_weighted_seed123.py`、`train_raman_weighted_seed2026.py` |
| BN 诊断脚本 | `training/diagnose_raman_batchnorm_full.py` |
| BN checkpoint 保存 | `training/save_raman_bn_checkpoint.py`（gate policy v5） |
| 数值审计模块 | `training/raman_numeric_audit.py` |
| 阈值校准脚本 | `training/calibrate_raman_original_thresholds.py`、`calibrate_raman_recalibrated_thresholds.py` |
| 汇总脚本 | `training/summarize_raman_ablation.py` |
| 辅助复现脚本（非门控） | `training/verify_raman_bn_reproducibility.py` |
| 回归测试（CPU） | `training/test_raman_bn_gate_policy.py`、`test_raman_bn_reload_integrity.py` |
| 原始 checkpoint | `runs/EXP-RAMAN-001-FULL/best_model.pt` 等六份（哈希见 registry §5） |
| BN 重估 checkpoint | `runs/RAMAN-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json` |
| BN 诊断记录 | `runs/RAMAN-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json` |
| 阈值（Original） | `runs/RAMAN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 阈值（Recalibrated） | `runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` |
| 概率溯源 | 上述两个校准目录下的 `probability_artifacts.json` + `.npy` |
| 数值审计记录 | `runs/RAMAN-NUMERIC-AUDIT/<run_name>/{original,recalibrated}_thresholds.json` |
| 消融汇总 | `runs/RAMAN-ABLATION-SUMMARY/`（11 个文件） |

## 附录 B：主方案的推理调用链

1. 加载 `runs/RAMAN-BN-RECALIBRATED/<run_name>/best_model_bn_recalibrated.pt`（Seed 42 canonical）；
2. 输入按逐谱最大绝对值归一化，送入 `SpectrumStructureModel`（14 输出）；
3. 取 `runs/RAMAN-BN-THRESHOLD-CALIBRATION/<run_name>/optimal_thresholds.json` 中的 14 个阈值（**不得**与 Original BN 的阈值混用）；
4. 概率 ≥ 对应标签阈值即判为正。

## 附录 C：复现入口

```powershell
# 重放汇总（对来源数据只读：不加载 PyTorch、不前向、不拟合阈值；但会新建汇总目录并写入 11 个文件）
python training\summarize_raman_ablation.py
```

**读写性质**：该脚本**对来源只读**（只读取既有 JSON/CSV 并校验 12 个 checkpoint 的 SHA256，不修改任何 checkpoint、诊断 JSON 或校准摘要），但**不是全程只读**——它会在 `runs/RAMAN-ABLATION-SUMMARY/` **新建**汇总目录并写入 11 个文件。既有汇总目录**不允许覆盖、删除或替换**（目录已存在时脚本拒绝执行并退出）。

**验证性重放**须用 `--out-dir` 指向一个**尚不存在**的新目录，避免触碰归档目录：

```powershell
python training\summarize_raman_ablation.py --out-dir runs\RAMAN-ABLATION-SUMMARY-REPLAY
```

（重放产物仅用于验证，不替代归档目录；如需再跑一次，请换用另一个新目录名。）

预期末行：`PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 672 per-label rows`。

## 附录 D：版本记录

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-11-16 | 首版冻结：主方案 WBCE + Recalibrated BN + Val-F1；备选 WBCE + Original BN + Val-F1；BCE 家族最佳 BCE + Recalibrated + Val-F1；固定基线 BCE + Original + 0.5；记录 Test 独立评价、四项指标领先幅度与噪声的关系、阈值校准的 Val/Test 分叉、FP16 复现史与 Gate 09 修复 |
| v1.1 | 2026-11-16 | 仅修订表述，**未改动任何指标、Mean±SD、checkpoint SHA256、配置决策与容差**：① §7 主方案 Test Micro-F1 排名由"第 4～5"更正为 **6/8** 并给出完整降序；② §8.4 更正为"前四名两种指标排序一致、第 5～7 名互换"；③ §8.1 改用多标签 Micro/Macro 汇总方式的准确解释，并删除"Macro-F1 对头部分布更敏感"；④ 删除"校准阈值整体低于 0.5"的未经支持表述，改为操作点位移，并给出 14 类阈值的实际分布（3～11 个 >0.5）；⑤ §6.1 明确 mAP/AUROC 与同 checkpoint 的 0.5 条件并列、属阈值无关指标，不构成独立增益证据；⑥ §8.6 把 FP16 差异拆为"直接观测／合理解释／尚未证明"，删除"极端抽样"等未完成归因的断言 |
| v1.2 | 2026-11-16 | 更正 §8.6 一处算术归属错误，**未改动任何观测数据与结论**：T1/T2 跨进程对照（0.946288 vs 0.945208）的差值由误写的 1.086e-3 更正为 **1.080e-3**；1.086e-3 仅对应最初 Gate B 失败那一对读数（0.946294 vs 0.945208）。同时新增"两个必须区分的数值"对照表，并在 `raman_ablation_registry.md` §9.3 与 `raman_repro_protocol.md` §7 第 8 条作同样的配对更正 |

---

**决策边界**：本记录冻结的是 Raman v1 **单模态基线的配置与归档状态**。任何改变损失、数据、标签、BN 处理或阈值策略的后续实验，必须以新实验编号登记，并明确与本文档的可比性。
