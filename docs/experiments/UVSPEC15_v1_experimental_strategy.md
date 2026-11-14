# UVSPEC15 v1：正式实验策略与决策记录

> **状态**：单模态 UV-Vis 基线策略 v1 已确定；三个随机种子的全部权重与评估结果继续保留。
> **记录日期**：2026-11-14
> **研究项目**：SpectraDB / 曦源项目
> **决策范围**：QM9S 计算域的 UV-Vis 结构语义多标签识别基线。**不代表**整个多模态分子结构推断系统已经完成。
> **性质**：在完成消融实验后形成的研究决策记录，而非事先注册的实验方案。

## 1. 任务、数据与评价协议

本阶段任务是根据 QM9S 计算 UV-Vis 光谱预测 **UVSPEC15 的 15 类二值结构语义标签**（共轭、芳香体系及与电子结构相关的特征），作为后续多模态融合的单模态表征基础。该任务不等同于完整分子结构识别。

- 光谱：`processed/qm9s/prepared/uvvis_float32.npy`；每条 701 点，输入按单谱最大绝对强度归一化（`normalization="max"`）。
- 标签：`processed/qm9s/prepared/uvspec15_labels.npy`；标签顺序由 `uvspec15_label_names.json` 固定；生成脚本为 `scripts/prepare_uvspec15_labels.py`。
- 划分：`uvvis_scaffold_split_valid.npz`，**Train 103,853 / Validation 12,981 / Test 12,981**；光谱有效样本合计 129,815。各模型使用同一划分，不允许重划分或跨集合拟合阈值。
- 模型：`training.models.spectrum_cnn1d.SpectrumStructureModel`，1D-CNN，embedding 维度 256、输出 15 个 logits。
- 固定训练条件（现有实验记录）：batch size 512，`num_workers=4`，AdamW（学习率 `1e-3`、weight decay `1e-4`），最多 50 epochs，early-stopping patience 8；使用各实验实际 checkpoint/config 作为最终权威来源。
- 评价：Micro-F1、Macro-F1、mAP、Macro-AUROC；同时保留 Precision、Recall 及逐标签结果。mAP 和 AUROC 不受推理阈值改变的影响。

## 2. 实验矩阵与结果来源

固定训练方法，分别运行随机种子 **42、123、2026**：

| 维度 | 条件 |
|---|---|
| 损失函数 | 普通 BCE（EXP-UVVIS-003）和 Weighted BCE（EXP-UVVIS-004；正类权重的平方根变换、上限 20） |
| 随机种子 | 42、123、2026，分别独立训练 |
| BN 推理统计量 | 原始 checkpoint；仅用 Training Set 重估后的 BN checkpoint |
| 分类阈值 | 统一 0.5；对应 checkpoint 在 Validation 上逐标签 F1 最优化 |

合计 **6 次独立训练、每个数据划分 24 个评估条件**（2 损失 × 3 Seed × 2 BN × 2 Threshold）。

原始训练目录：

- BCE：`runs/EXP-UVVIS-003-UVSPEC15-FULL/`、`...-SEED123/`、`...-SEED2026/`。
- WBCE：`runs/EXP-UVVIS-004-UVSPEC15-WBCE/`、`...-SEED123/`、`...-SEED2026/`。
- BN 完整诊断：`runs/UVSPEC15-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json`。
- BN 重估 checkpoint 与 manifest：`runs/UVSPEC15-BN-RECALIBRATED/<run_name>/`。
- BN 后阈值：上述目录下的 `threshold_calibration/optimal_thresholds.json`。
- 整体汇总：`runs/UVSPEC15-ABLATION-SUMMARY/`，由 `training/summarize_uvspec15_ablation.py` 生成。

> 模型 `.pt` 文件和原始大规模光谱数据不必进入普通 Git 仓库，但必须另行安全备份，并保留与代码、manifest、配置和阈值文件的对应关系。

## 3. 正式采用的 v1 方法策略

**当前默认实验策略：普通 BCE + Train-only BN 重估 + Validation 逐标签 F1 阈值校准。**

具体流程：

1. 按固定架构、数据划分、预处理和训练配置训练普通 BCE 模型；分别保留 Seed 42、123、2026 的原始 checkpoint。
2. 在原始 checkpoint 的**独立副本**中，仅使用完整 Training Set 前向传播重估 BatchNorm 运行统计量；保持可训练参数不变，不使用 Validation/Test 更新 BN，不覆盖原始权重。BN 重估使用项目已实现的累计 batch 统计量方案，需保留对应实现和诊断 manifest。
3. 重新加载该 Seed 的 BN 重估 checkpoint，仅在 Validation 上为 15 个标签分别寻找 F1 最优分类阈值；冻结后用于相应 Test 评估。**原始 checkpoint 的阈值不能混用于 BN 重估 checkpoint**。
4. 对三 Seed **逐一独立**执行上述后处理，报告三次结果的 Mean ± 样本 SD，而非挑选分数最高的 Seed 作为论文主结果。
5. WBCE 及其他三种 BN/Threshold 处理组合保留为消融对照，不删除、不覆盖。

> **特别说明**：随机种子仅控制训练随机过程，作为重复实验与稳定性检查的因素，不是需要优化的模型超参数。此处“冻结”指确定 v1 **方法与评价协议**，不是宣称某一随机种子具有科学上的最优性。工程演示若必须加载一套具体权重，可以预先固定 Seed 42 作为代表实例；应标注其代表性用途，不把该次结果当成三 Seed 的整体性能。融合模型的稳健性仍应通过重复实验检验。

> **接口说明**：对后续多模态融合而言，主要复用 BN 重估后的 **编码器表征**；15 个分类阈值属于 UVSPEC15 **多标签分类输出头**，不是 embedding 本身的组成部分，不应误用为融合特征的阈值。

## 4. 决策依据（来自已完成的汇总文件）

### Validation，三 Seed Mean ± 样本 SD

| 策略（均为 BN 重估 + Validation-F1） | Micro-F1 | Macro-F1 | mAP | Macro-AUROC |
|---|---:|---:|---:|---:|
| **BCE** | **0.6283 ± 0.0060** | **0.5229 ± 0.0087** | **0.5321 ± 0.0163** | 0.9270 ± 0.0014 |
| WBCE | 0.6241 ± 0.0136 | 0.5214 ± 0.0110 | 0.5285 ± 0.0214 | **0.9282 ± 0.0040** |

### Test，仅用于评价所形成的方案，不反向选择配置

| 策略（均为 BN 重估 + Validation-F1） | Micro-F1 | Macro-F1 | mAP | Macro-AUROC |
|---|---:|---:|---:|---:|
| **BCE** | **0.6568 ± 0.0138** | **0.5395 ± 0.0143** | **0.5728 ± 0.0203** | **0.9384 ± 0.0040** |
| WBCE | 0.6520 ± 0.0116 | 0.5330 ± 0.0154 | 0.5596 ± 0.0226 | 0.9382 ± 0.0039 |

据现有 Validation 对比，BCE 的主要指标略高且三 Seed 波动较小，因此选择普通 BCE 作为 **v1 的简洁默认基线**；WBCE 作为完整对照保留。差异较小且只有三个 Seed，**不能声称 BCE 在统计意义上显著优于 WBCE**。阈值校准在 Validation 上直接优化 F1，该集合上的 F1 改善有内生的乐观偏差；Test 用来评价迁移后的效果。

## 5. 结果有效性与未解决问题

- 所有 24 条条件均已完成汇总和交叉核验；统计脚本输出 `PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 720 per-label rows`。
- BN 重估改善冻结 checkpoint 的表现与跨 Seed 一致性；**不代表**已经证明训练中逐 Epoch Validation 指标震荡被消除。
- 现有结论限于当前 QM9S **计算光谱域与 Scaffold Split**，不能直接外推至 NIST/SDBS 等实验域。
- **数据划分使用纪律**：Training 用于参数学习与 BN 统计量重估；Validation 用于 Early Stopping、模型与后处理策略（损失函数、BN 处理方式、阈值）选择及逐标签阈值校准；Test 作为独立分割，仅用于性能评价、实验记录与一致性核验。本研究中 Test 结果未被用于调整模型架构、训练策略、BN 处理方法、阈值或随机种子选择，所有配置选择均依据 Validation（见第 3、4 节）。
- **跨域验证的必要性**：QM9S 为计算光谱，与真实实验光谱（NIST、API-Raman 等）在谱形、峰位、强度与噪声特性上存在域差异，因此后续仍需在独立数据及实验域上验证泛化能力。该验证需求源于计算域与实验域的物理差异，与 Test 的使用方式无关。
- 三 Seed 的 Mean ± SD 不等于对模型性能统计显著性的证明；如需显著性结论，须另行设计适当的重复和推断程序。
- 当前 UVSPEC15 仅输出结构特征标签；尚未完成多模态融合、Top-k 完整结构候选和开放集拒识。

## 6. 复现入口与归档要求

根目录运行（只读汇总，已验证本地能够成功执行）：

```powershell
python training\summarize_uvspec15_ablation.py
```

需要保留并互相匹配的材料：

- 代码：`scripts/prepare_uvspec15_labels.py`、六种子训练脚本、原始阈值校准脚本、`training/diagnose_uvspec15_batchnorm_multiseed.py`、`training/save_uvspec15_bn_checkpoint.py`、`training/calibrate_uvspec15_bn_thresholds.py`、`training/summarize_uvspec15_ablation.py`。
- 轻量级结果：六实验的 `config.json`、`test_metrics.json`、阈值 JSON/CSV、训练日志、Full BN diagnostic、BN manifest、消融汇总 CSV。
- 大文件（仓库外安全备份）：六组原始及六组 BN 重估 `.pt`、QM9S 的光谱/标签矩阵以及划分文件；确保能够通过标签版本、来源目录和文件哈希追溯。
- Git：本记录、汇总脚本和轻量级统计文件应纳入版本控制，提交前执行 `git diff --check` 并复核新增文件清单；不使用无审查的全量 `git add .`。

## 7. 下一研究阶段

1. 审查并归档 UVSPEC15 v1 的汇总结果及决策记录。
2. 对已有 IR checkpoint 开展 **Full BN Diagnostic**；不再安排 Quick Diagnostic；根据 Validation 结果决定是否保存 BN 重估版并重新校准阈值。
3. 对 Raman 做相同的完整审计。
4. 建立 IR/Raman/UV-Vis 编码器统一推理与样本对齐接口；先实现 Concat 融合对照，再开展 Chemistry-aware Fusion、结构候选 Top-k、拒识和可解释性模块。

---

**决策边界**：上述“正式策略”是 UVSPEC15 **v1 单模态基线**，不是对后续架构改进的禁止；任何改变损失、数据、标签、BN 处理或阈值选择的后续实验应以新的实验编号记录，并明确与 v1 的可比性。
