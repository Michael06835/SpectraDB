# Line B 实验协议草案（Phase B0）

> **状态**：DRAFT v0.2 — 待人工审核（B0 最终方法学修订稿）。**本轮不执行新的训练/推理、不重估 BN、不校准阈值、不修改历史数据/checkpoint/指标/实验结论；仅按本轮要求做文档勘误。**
>
> **日期**：2026-11-17（Asia/Shanghai）
>
> **版本**：v0.2（2026-11-17；修订 Oracle 定义、Bootstrap 解释、Test 接触声明、UV 优势/互补拆分、B1–B5 阶段编号）
>
> **上游审计**：`docs/experiments/line_b_b0_asset_audit.md`
>
> **机器可读矩阵**：`metadata/line_b_b0_experiment_matrix.csv`
>
> **适用范围**：QM9S 计算域、IR/Raman/UV-Vis 三模态、Shared14 与 UVSPEC15 两个独立任务。
>
> **不适用范围**：实验域（NIST/API-Raman）泛化、多模态融合训练、完整结构 Top-k、开放集拒识、可解释性。以上分别属于 Line C/D/F/G/H/I。

---

## 0. 文档性质与结论边界

### 0.1 这是草案，不是已执行的实验

本文件只定义：

- 研究问题与假设；
- 数据/标签/划分/模型/超参数；
- 实验矩阵；
- 评价指标与统计口径；
- Test 使用纪律；
- 输出目录与脚本任务；
- 阶段验收标准。

本文件**不**包含任何新训练结果、新推理结果或新阈值。

### 0.2 不是盲法预注册确认性实验

三套 v1 的 Test 指标已经在本项目历史报告中被查看，且 v1 Final Decision 是在 24 条件消融完成后写成的。因此：

- Line B 的 Test 分析只能定位为 **exploratory / post-hoc**；
- 本协议在 B1 启动前冻结，可以**减少**分析自由度，但不能改变“Test 已开启”的事实；
- 不得使用“盲法”“预注册确认性”“独立确认”等表述；
- 3 Seed 的 Mean±SD 是训练稳定性描述，不是显著性检验；
- scaffold Bootstrap 是探索性不确定度分析，不构成确认性推断；
- 若需要确认性结论，必须使用外部实验域数据（NIST/API-Raman）或新的锁定评估集，另行立项。

### 0.3 本轮禁止事项

- 禁止任何训练、GPU 计算、BN 重估、阈值校准；
- 禁止修改/覆盖 IR v1、Raman v1、UVSPEC15 v1 的 checkpoint、manifest、阈值、审计结果、冻结报告；**唯一例外**是本轮任务 6 要求的两份纯文档勘误（`uvspec15_label_guide.md` 计数、`UVSPEC15_v1_final_decision.md` 列顺序说明），不改数据/指标/结论；
- 禁止修改任何 split 或标签；
- 禁止改变历史实验的配置选择；
- 禁止 Git commit / push；
- 禁止直接进入 B1 的 GPU 推理或 B2 的训练；B1–B5 均需本修订稿审核通过后另行启动。

---

## 1. 研究问题与假设

### 1.1 总问题

在**完全相同**的标签、样本划分、模型骨架、损失家族、BN 状态与阈值政策下，IR、Raman、UV-Vis 三种光谱分别能提供多少结构信息？它们的错误是否互补？这种互补性是否随结构子群变化？

### 1.2 两个独立任务

| 任务 | 标签集 | 标签数 | 目标 |
|---|---|---:|---|
| **A. Shared14** | 14 类官能团/局部结构特征 | 14 | 比较三模态对“通用官能团/局部结构”的表征能力 |
| **B. UVSPEC15** | 15 类共轭/芳香/供受体结构语义 | 15 | 比较三模态对“UV 专属电子结构语义”的表征能力 |

两个任务**独立**：

- 不训练多任务头；
- 不把 14 类与 15 类的指标直接横向比较（标签集不同，Micro/Macro 不可比）；
- 只在每个任务内部比较三模态；
- 两个任务的结论可以相互参照，但不能合并成一个“总排名”。

### 1.3 假设

**H1（振动模态的通用结构优势）**

在 Shared14 上，IR 与 Raman 的 Macro AP / Macro AUROC 高于 UV-Vis；差距在羰基、腈、炔、胺、羟基等振动敏感标签上更明显。

**H2（UV 的电子结构优势）**

在 UVSPEC15 上，UV-Vis 的 Macro AP / Macro AUROC 高于 IR 与 Raman；优势集中在 `has_conjugation`、`has_aromatic_system`、`heteroaromatic`、`cyclic_conjugated_system`、供/受体取代等粗粒度共轭/芳香语义。

**H3（错误互补）**

同一任务、同一 Test 样本上，IR/Raman 与 UV 的错误集合不完全重叠；三模态的 `positive_or_fusion`（任一模型判正即为正）可能高于任一单模态，但这只是融合规则，不是 Oracle；真正的互补性由使用 `y_true` 的 Oracle 正确性分析（至少一个模态正确、仅一个模态正确、全部模态错误）评估。UV 的独家正确样本应更多出现在共轭/芳香结构子群，IR/Raman 的独家正确样本应更多出现在羰基/腈/炔等局部官能团子群。

**H4（结构子群依赖）**

模态差距随结构子群变化：芳香环数、共轭体系类型、杂原子数、供/受体取代、环数、分子量等会调节“谁更好”。不存在“UV 在所有子群都更好/更差”的单一结论。

> 这些是**方向性假设**，不是已经证实的结论。Line B 的输出应允许 H1–H4 被部分否定。

### 1.4 主要/次要/探索性终点

| 层级 | 终点 | 说明 |
|---|---|---|
| **主要** | 每个任务的 Macro AP（15/14 标签等权平均） | 阈值无关；对类别不平衡比 F1 稳健 |
| **主要** | 每个任务的 Macro AUROC | 阈值无关；与 AP 互补 |
| **次要** | Micro-F1 / Macro-F1 @ 0.5 | 统一阈值政策下的操作点指标 |
| **次要** | 逐标签 AP/AUROC/F1 + support | 解释模态差异来源 |
| **次要** | Validation-F1 阈值下的 F1 | 统一拟合流程，仅作敏感性 |
| **探索性** | 错误互补性、positive_or_fusion、真实 Oracle 正确性、McNemar | 相同 Test 样本 |
| **探索性** | 标签共现、结构子群、典型分子 | 解释性分析 |

---

## 2. 数据与标签来源

### 2.1 数据

| 项 | 路径 / 值 |
|---|---|
| IR 光谱 | `processed/qm9s/prepared/ir_float32.npy`，`(129817, 3501)`，500–4000 cm⁻¹ |
| Raman 光谱 | `processed/qm9s/prepared/raman_float32.npy`，`(129817, 3501)`，500–4000 cm⁻¹ |
| UV-Vis 光谱 | `processed/qm9s/prepared/uvvis_float32.npy`，`(129817, 701)`，1–15 eV |
| Shared14 标签 | `processed/qm9s/prepared/functional_group_labels.npy`，`(129817, 14)`，uint8 |
| Shared14 名称 | `processed/qm9s/prepared/functional_group_label_names.json` |
| UVSPEC15 标签 | `processed/qm9s/prepared/uvspec15_labels.npy`，`(129817, 15)`，float32 |
| UVSPEC15 名称 | `processed/qm9s/prepared/uvspec15_label_names.json`（**唯一权威顺序**） |
| UVSPEC15 provenance | `processed/qm9s/prepared/uvspec15_label_manifest.json` |
| 结构库 | `processed/qm9s/structures/qm9s_structure_library.csv` |
| 样本 manifest | `processed/qm9s/prepared/qm9s_manifest_prepared.csv` |

### 2.2 划分

**Shared14 评估**：

- IR：`ir_scaffold_split_valid.npz`，train 103855 / val 12981 / test 12981；
- Raman：`raman_scaffold_split_valid.npz`，train 103855 / val 12981 / test 12981；
- UV：`uvvis_scaffold_split_valid.npz`，train 103853 / val 12981 / test 12981。

**UVSPEC15 训练/评估**：

- 三模态统一使用 `uvvis_scaffold_split_valid.npz`（**103853 / 12981 / 12981**）；
- 这样三模态的训练行完全一致，避免“IR/Raman 多 2 条”成为模态差异的替代解释；
- 这 2 条是 `row_index=13998/14000` 的全零 UV 谱，只在 IR/Raman 的 train 中存在；排除它们不改变 val/test。

**共同事实**：

- 三模态 Validation/Test 行索引**逐元素相同、顺序相同**；
- scaffold 零重叠；
- 任何样本级对齐直接用行位置，不需要 CID/SMILES 匹配。

### 2.3 标签顺序的硬约束

- 14 类与 15 类标签都是**非互斥**多标签：同一分子可同时属于多个类别；
- 14 类：严格按 `functional_group_label_names.json`；
- 15 类：严格按 `uvspec15_label_names.json`；
- **禁止**按 `UVSPEC15_v1_final_decision.md` 第 1 节的字母序列表推断列顺序；
- 每个 checkpoint 保存 `label_names`；每个预测导出保存 `label_names`；每个阈值文件保存按名字索引的字典；
- 任何张量列索引都必须通过 `names.index(label)` 获得，不得硬编码叙事顺序。

### 2.4 不修改承诺

- 不重新生成 labels；
- 不改变 split；
- 不改变 UV 的 103853 口径；
- 不把 IR 的 103855 train 与 UV 的 103853 train 混用；
- 未来融合训练统一使用 `multimodal_complete_scaffold_split.npz`（129815）。

---

## 3. 实验矩阵

### 3.1 核心矩阵（primary）

统一配置：**BCE + Original BN**；主 F1 阈值 **0.5**；AP/AUROC 阈值无关；Validation-F1 阈值仅作次要敏感性。

| ID | 阶段 | 任务 | 模态 | 模型来源 | 新训练 | Seeds | 评估 |
|---|---|---|---|---|---:|---|---|
| A-01 | B1 / B4 | Shared14 | IR | `EXP-IR-001-*` | 0 | 42/123/2026 | B1 导出 validation；B4 冻结后导出一次 Test；AP/AUROC/F1 |
| A-02 | B1 / B4 | Shared14 | Raman | `EXP-RAMAN-001-*` | 0 | 42/123/2026 | 同上 |
| A-03 | B1 / B4 | Shared14 | UV-Vis | `EXP-UVVIS-001-*` | 0 | 42/123/2026 | 同上 |
| B-01 | B2 / B3 / B4 | UVSPEC15 | IR | 新建 `LINE-B-UVSPEC15-IR-SEED*` | **3** | 42/123/2026 | B2 训练；B3 导出 validation；B4 冻结后导出一次 Test；AP/AUROC/F1 |
| B-02 | B2 / B3 / B4 | UVSPEC15 | Raman | 新建 `LINE-B-UVSPEC15-RAMAN-SEED*` | **3** | 42/123/2026 | 同上 |
| B-03 | B3 / B4 | UVSPEC15 | UV-Vis | `EXP-UVVIS-003-UVSPEC15-*` | 0 | 42/123/2026 | B3 导出 validation；B4 冻结后导出一次 Test；AP/AUROC/F1 |

**规模**：

- 18 个模型（9 Shared14 + 9 UVSPEC15）；
- 每模型 2 个 split（validation/test）× 2 个阈值政策（0.5 / Validation-F1）；
- 阈值无关指标只算一次（同一概率数组）；
- **新训练 6 次**（IR 3 + Raman 3）；
- 记录规模：`18 × 2 × 2 = 72` 条 threshold-split 记录；阈值无关指标为 `18 × 2 = 36` 条。
- 矩阵维度：`2 任务 × 3 模态 × 3 seeds × 1 损失（BCE）× 1 BN（Original）× 2 阈值政策 = 36` 个 model-threshold 条件；再乘 2 个 split = 72 条记录。

### 3.2 次要矩阵（sensitivity）

| ID | 阶段 | 任务 | 内容 | 新训练 | 说明 |
|---|---|---|---|---:|---|
| S-01 | B3 | Shared14 | Validation-F1 阈值（9 个模型统一重拟合） | 0 | 只读概率；Validation only；不混用历史阈值 |
| S-02 | B3 | UVSPEC15 | Validation-F1 阈值（9 个模型统一重拟合） | 0 | 同上 |
| S-03 | B4 | Shared14 | WBCE × Original × 0.5 的描述性对照 | 0 | IR/Raman 三 Seed；UV 只有 Seed 42，不平衡，不作主结论 |
| S-04 | 延后 | Shared14 | BN Recalibrated 敏感性 | 3 份 UV shared BN 后处理 | 延后；IR/Raman 已有 Recalibrated，UV shared 缺；必须三模态同时做 |
| S-05 | 延后 | UVSPEC15 | BN Recalibrated 敏感性 | 6 份 IR/Raman BN 后处理 | 延后；UV 已有 Recalibrated；必须三模态同时做 |
| S-06 | 延后 | UVSPEC15 | WBCE 敏感性 | 6 次训练 | 延后；本轮不执行 |

### 3.3 明确不做的比较

- 不比较 Shared14 与 UVSPEC15 的绝对 Micro/Macro 指标（标签集不同）；
- 不把 IR v1 主方案（BCE/Original/Val-F1）与 Raman v1 主方案（WBCE/Recalibrated/Val-F1）直接相减；
- 不把 UVSPEC15 的 Recalibrated UV 与 Original IR/Raman 直接相减；
- 不用 Test 分数选择 Seed、阈值、BN 状态或损失；
- 不因为某个标签/子群结果不理想而事后删除标签或子群。

### 3.4 矩阵文件

完整行级矩阵见 `metadata/line_b_b0_experiment_matrix.csv`（含 `phase`（B1–B5）、`stage`、`new_training`、`n_models`、`output_root`、`notes`）。

---

## 4. 输入、输出、模型与超参数

### 4.1 输入/输出

**输入**：

- 单模态光谱：`[1, L]`，IR/Raman `L=3501`，UV `L=701`；
- 逐谱 max-abs 归一化；
- 不做全局标准化，不做跨模态拼接。

**输出**：

- 14 维 logits / sigmoid 概率（Shared14）；
- 15 维 logits / sigmoid 概率（UVSPEC15）；
- 每个样本保存 `row_index`、`y_true`、`prob`、`split`、`model_id`；
- 阈值化预测由后处理脚本派生，不写回 checkpoint。

### 4.2 模型

```python
from training.models.spectrum_cnn1d import SpectrumStructureModel

model = SpectrumStructureModel(
    num_labels=14,       # Shared14
    embedding_dim=256,
)
# 或
model = SpectrumStructureModel(
    num_labels=15,       # UVSPEC15
    embedding_dim=256,
)
```

- 复用现有 `SpectrumEncoder1D`；
- 12 层 BatchNorm1d；
- 不使用 Morgan fingerprint；
- 不修改模型结构；
- 新 IR/Raman UVSPEC15 **从头训练（随机初始化整个模型）**，不复用 Shared14 的 encoder 权重；输出头为 15 维。这样避免把“跨任务迁移”作为除模态之外的额外变量。

### 4.3 新训练超参数（UVSPEC15 IR/Raman）

| 项 | 值 |
|---|---|
| 损失 | `BCEWithLogitsLoss()`，`use_pos_weight=False` |
| 优化器 | AdamW，lr `1e-3`，weight decay `1e-4` |
| batch size | 512 |
| num_workers | 4 |
| max epochs | 50 |
| early stopping | patience 8 |
| 最优轮 | Validation mAP |
| AMP | 开启 |
| normalization | `"max"` |
| seeds | 42 / 123 / 2026 |
| train/val/test | 103853 / 12981 / 12981（UV split） |
| 输出头 | 15，顺序按 `uvspec15_label_names.json` |

### 4.4 训练脚本禁止事项

新训练脚本必须：

- 只加载 train/validation，**不加载 test**；
- 不计算 Test loss/Test 指标；
- 不读历史 Test 指标；
- 不覆盖已有目录（`out-dir` 必须不存在或为空）；
- 训练结束只保存 checkpoint + validation 记录 + run manifest；
- 不修改任何历史 run 目录；
- 固定 seed（Python/NumPy/PyTorch/CUDA）；
- 记录 git commit / 工作区状态 / 环境版本 / GPU 型号 / 训练时长；
- 训练日志包含每 epoch 的 train loss、val loss、val Micro-F1、val Macro-F1、val mAP、val Macro-AUROC、epoch seconds、peak GPU memory。

### 4.5 推理/导出统一口径

- `model.eval()`；
- **统一 FP32 推理**（关闭 autocast），避免 FP16 跨进程差异；
- `torch.no_grad()`；
- batch size 512，num_workers 4；
- 输出 `float32` 概率；
- 保存 `row_index`、`y_true`、`prob`、`split`、checkpoint SHA256、split SHA256、label names、normalization、BN state、code SHA256；
- 每个数组保存 SHA256；
- 同一 checkpoint 的 validation/test 各导出一次；
- 测试集导出必须在协议冻结后执行一次。

> 历史 v1 使用 AMP/FP16。Line B 统一 FP32 可能与个别历史数值有 1e-3 量级差异（Raman 已有记录，主要落在 `nitro`）。Line B 的所有模型都用同一 FP32 口径，因此跨模态比较是内部一致的；与历史 FP16 指标比较时需标注口径差异。

---

## 5. 样本与 checkpoint 对齐原则

### 5.1 样本对齐

1. 所有评估使用冻结 split 的 `row_index`；
2. Validation/Test 三模态逐元素相同；
3. 任何预测数组的第 `i` 行对应 `row_index[i]`；
4. 合并不同模型时，先按 `row_index` 排序/校验，不按文件名或目录顺序；
5. 任何缺失/重复 `row_index` 立即失败；
6. 不重新排序、不重新划分、不删除样本（除 UV 已冻结的 2 条全零谱）。

### 5.2 checkpoint 对齐

每个模型必须记录：

- checkpoint 路径与 SHA256；
- checkpoint 类型（original / bn_recalibrated）；
- seed；
- 损失；
- BN 状态；
- 训练集规模；
- 最优 epoch / best val mAP；
- 输出头维度；
- `label_names`；
- 训练代码版本 / git commit。

**禁止**：

- 把 Recalibrated checkpoint 与 Original 阈值配对；
- 把 14 类 checkpoint 与 15 类标签配对；
- 用不同 seed 的 checkpoint 互相覆盖；
- 用 Test 分数挑选 seed。

### 5.3 阈值对齐

- 阈值必须与 checkpoint、BN 状态、标签顺序、概率数组一一绑定；
- 每个阈值文件保存 `checkpoint_sha256`、`bn_state`、`label_names`、`source="validation split only"`、`frozen_before_test=true`；
- 重新拟合阈值时，必须使用同一代码路径和同一 Validation 概率数组；
- 不复制其他 seed / 其他 BN 状态的阈值；
- 阈值在 Test 导出前写盘并计算 SHA256。

### 5.4 BN 状态对齐

- 主比较：全部 **Original BN**；
- 不把 UV 的 Recalibrated 与 IR/Raman 的 Original 混比；
- BN 重估只能使用 Training split，且必须保存 manifest（源 checkpoint SHA256、重估后 SHA256、`num_batches_tracked`、batch 数、样本数）；
- BN 重估是后处理，不是新模型；不得用它来“提升”某个模态后再与未重估模态比较。

---

## 6. 评价指标

### 6.1 主要指标（threshold-free）

| 指标 | 定义 | 用途 |
|---|---|---|
| Per-label AP | 每个标签的 average precision | 逐标签排序质量 |
| Macro AP | 14/15 个标签 AP 的等权平均 | 主终点 1 |
| Per-label AUROC | 每个标签的 ROC AUC | 逐标签排序质量 |
| Macro AUROC | 14/15 个标签 AUROC 的等权平均 | 主终点 2 |

**AP 基线**：

- 每个标签的随机基线 ≈ 该标签在评估 split 上的正例率 `prevalence_l`；
- 报告 `AP_l`、`prevalence_l`、`AP_l - prevalence_l`；
- Macro 表同时报告 `mean(AP_l)` 与 `mean(prevalence_l)`；
- 不把 AP 与 F1 混为一谈；AP 不依赖阈值，F1 依赖阈值。

### 6.2 次要指标（threshold-dependent）

| 指标 | 定义 |
|---|---|
| Micro-F1 | 汇总所有标签的 TP/FP/FN 后计算 |
| Macro-F1 | 每个标签 F1 的等权平均 |
| Micro-Precision / Micro-Recall | 汇总 TP/FP/FN |
| Per-label F1 / Precision / Recall | 每个标签 |
| Support | 每个标签的正例数（val/test 分别报告） |

**阈值政策**：

- **主 F1**：固定 0.5，所有模态、所有标签一致；
- **次 F1**：Validation-F1 逐标签阈值，所有模态用同一代码路径独立拟合；
- 不把 0.5 与 Val-F1 的结果混在同一张“主表”里；
- 不按 Test 调整阈值；
- F1 表必须带 support 与阈值来源。

### 6.3 支持度分层

沿用 v1 的定义：

- `very_low`：`min(val positives, test positives) < 30`
- `low`：`min(...) < 150`
- `normal`：其余

每个逐标签表必须包含：

- val positives；
- test positives；
- test prevalence；
- support tier；
- AP / AUROC / F1；
- Bootstrap 95% CI（若样本量允许）。

**禁止**：

- 只报告表现好的标签；
- 用 `very_low` 标签单独支撑模态结论；
- 把 `nitro`（test 15）或 `pi_conjugated_imine`（test 148）的单个数值当作稳定结论。

**预定义敏感性（不是主表）**：

- 支持度加权 AP：`Σ_l n_pos_l · AP_l / Σ_l n_pos_l`；
- 排除 `very_low` 标签后的 Macro AP/AUROC；
- 两者都必须在看 Test 前写死，且不得用于选择模型/阈值/标签子集。

### 6.4 聚合与报告规则

- 所有主表按 **3 Seed Mean ± 样本 SD（n=3）** 报告；
- 同时给出每个 seed 的数值（附录/CSV）；
- **所有 Line B 指标都从统一 FP32 导出的概率数组重算**，不直接复制历史 FP16 JSON；历史表只作参考；
- Micro-F1 与 Macro-F1 分开解释：Micro 受高频标签主导，Macro 对稀有标签更敏感；
- AP/AUROC 与 F1 分开解释：前两者阈值无关，后者依赖 0.5 或 Val-F1；
- 不构造“最佳模态”单一总分；报告多指标 + 不确定性；
- 不使用“显著”措辞；Bootstrap CI 只作探索性描述。

---

## 7. 模态比较与错误互补性分析

### 7.1 预定义对比与两类“UV 证据”

**预定义模态对**（两个任务相同）：

- IR vs Raman；
- IR vs UV；
- Raman vs UV；
- 三模态 `positive_or_fusion` vs 最佳单模态（`best single modality` 必须由 Validation 预先指定；同时报告三个单模态，不得用 Test 事后选择）。

不把 6 个对比混成一个“总排名”；每个对比单独报告效应量与 CI。

#### A. 边际模态差异（marginal contrast；主分析）

- 分别定义 **UV−IR** 与 **UV−Raman** 的配对 AP/AUROC 差异；逐标签 + Macro；同一样本、同一 seed。
- 两个对比都必须报告；**不得根据 Test 结果只挑“更弱”或“更强”的振动模态作为对照**。
- 若需要一个“UV vs 最佳振动模态”的汇总，只允许在 Validation 上按 Macro AP 预先指定 `best_vibrational`，且同时保留 UV−IR、UV−Raman 两个原始对比；不得用 Test 选择。
- UV 的边际 AP/AUROC 低于 IR 或 Raman，**不等于** UV 没有增量信息；这只说明 UV 单模态排序质量较低。增量信息由 §7.3 的条件互补性和真实 Oracle 正确性回答。

#### B. 条件互补性（conditional complementarity；次要/探索性）

- 在固定阈值（主 0.5；次 Val-F1）下，定义：
  - `P(UV correct | IR wrong)`、`P(UV correct | Raman wrong)`；
  - `unique_UV_vs_IR = N(UV correct ∧ IR wrong)`，以及 IR、Raman 同时错误而 UV 正确的计数；
  - 同样报告 `P(UV correct)` 作为基线，避免把高准确率误当互补性。
- 条件互补性是 threshold-dependent；不能替代 AP/AUROC 的边际分析。
- 反过来，UV 的边际 AP 高也不自动证明互补性；必须看条件互补与真实 Oracle 正确性。

### 7.2 配对 scaffold 聚类 Bootstrap（探索性）

**Scaffold 来源与唯一性**：

- test scaffold 从 `processed/qm9s/prepared/qm9s_manifest_prepared.csv` 的 `murcko_scaffold` 列按 `row_index` 获取；
- 该列由 `scripts/prepare_qm9s_dataset.py` 的 RDKit `MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)` 生成；纯链状分子回退为 `ACYCLIC::<canonical>`；
- 已核验：129817/129817 非空；`row_index` 0..129816 唯一连续；`row_index → murcko_scaffold` 映射唯一；28,501 个唯一 scaffold；
- 若出现缺失/空白 scaffold，归入显式 `__MISSING__` 组并计数；不得静默补值或按行填充。

**流程**：

1. 按 `row_index` 合并 test split 与 manifest，断言一一对应；任何缺失/重复立即失败；
2. 以 scaffold 为聚类单位，有放回重采样 2,000 次；同一 scaffold 的样本必须整体进入/离开，不得按行独立重采样；
3. **所有模型与 3 个 Seed 使用同一组 scaffold 重采样索引**（保持配对）；
4. 对每个标签计算 AP/AUROC/F1 的差值，再对标签取 Macro；
5. 每个 Bootstrap 样本内对 3 个 Seed 的差值取平均；
6. 报告 95% percentile CI；
7. 固定 RNG seed（如 `20261117`）；
8. **undefined 处理**：重采样后某标签可能全正或全负，导致 AP/AUROC undefined；必须标记、计数、跳过该标签/重复，并在结果中报告 `n_undefined_ap`、`n_undefined_auroc`；不得把 undefined 当作 0 或 1；
9. 记录每个标签的 `n_valid_reps` 与总重复数。

**解释**：

- Bootstrap 以 scaffold 为聚类单位，描述的是**在固定三 Seed 模型与固定 Test 样本下，由 scaffold 抽样带来的不确定性**；它**不能替代**训练随机性（seed-to-seed）的估计。
- 3 Seed 的 Mean±SD 单独描述训练随机性；Bootstrap CI 与 Seed SD 不得互相替代，也不得把两者合并成“总不确定性”。
- CI 是否包含 0 只描述该重采样分布下差值的方向不确定性；**包含 0 不能被解释为“方向稳定”**，排除 0 也不能被解释为确认性显著。
- Macro AP/AUROC 只对 defined 标签求平均，并同时报告 `n_defined / n_total`；不得把 undefined 标签计入分母，也不得用 0/1 填补。
- 所有解释仍属 exploratory/post-hoc；`very_low` 标签的 Bootstrap 可能不稳定，需单独标注。

### 7.3 错误互补性、融合规则与真实 Oracle（相同 Test 样本）

对每个任务、每个模态对、每个 Seed：

#### A. 逐标签错误与融合规则（不是 Oracle）

- 混淆矩阵：TP/FP/FN/TN；
- 错误集合：FP 集合、FN 集合；
- `error_jaccard = |E_A ∩ E_B| / |E_A ∪ E_B|`，其中 `E = FP ∪ FN`（按样本-标签 pair）；若两个错误集合都为空，Jaccard 记为 1 并单独标注；
- `A_only_correct` / `B_only_correct` / `both_wrong`；
- McNemar 精确检验（逐标签、逐 Seed），Holm 校正；
- `positive_or_fusion`：任一模型判正即为正；`positive_and_fusion`：全部模型判正才为正；
- `fusion_gain = F1(positive_or_fusion) - max(F1_A, F1_B)`；该差值**可正可负**，不得假设 OR 融合一定提高 F1；pairwise 只与同一模态对的两个成员比较，三模态只与 Validation 预先指定的 best single 比较；
- 报告正增益/负增益的标签数，不只报平均值；
- **OR 融合不是 Oracle**：它只是固定规则，可能提高 Recall 但降低 Precision，不能作为互补性的上限。

#### B. 真实 Oracle 正确性（使用 `y_true`；不是可部署模型）

对每个样本-标签 pair，定义模态 `m` 的正确性：

```text
c_m = 1[pred_m == y_true]
```

报告：

- `any_correct = 1[Σ_m c_m ≥ 1]`：至少一个模态正确；
- `exactly_one_correct = 1[Σ_m c_m = 1]`：仅一个模态正确；
- `all_wrong = 1[Σ_m c_m = 0]`：全部模态错误；
- `all_correct = 1[Σ_m c_m = 3]`（可选）；
- 分模态的 `only_IR_correct` / `only_Raman_correct` / `only_UV_correct` 计数；
- 以上指标总体、逐标签、逐结构子群报告。

**边界**：真实 Oracle 正确性只表示“在给定三个模型中至少一个正确”的诊断上限，不是可部署模型；不得把 Oracle 的正确率当作融合模型 F1，也不得用它证明某个融合策略有效。

#### C. 逐样本

- `n_errors_i = Σ_l |pred_il - y_il|`；
- 各模态 `n_errors_i` 的分布（中位数、IQR）；
- 模态间 Spearman 相关；
- “hard sample” = 误差数 top 10%；
- hard sample 集合的 Jaccard；
- 分模态的独家正确样本数；
- 高置信度错误（`max prob` 高但错）比例；
- 条件互补性：`P(UV correct | IR wrong)`、`P(UV correct | Raman wrong)`，以及 IR、Raman 同时错误而 UV 正确的计数（见 §7.1B）。

#### D. 三模态

- 三模态 `positive_or_fusion`；
- 真实 Oracle 的“至少一个正确 / 仅一个正确 / 全部错误”分布；
- 三模态“全对/全错/部分对”的分类；
- 按结构子群拆分（见 §7.5）。

**报告方式**：

- 主表：3 Seed 的均值 ± SD；
- 附：每个 Seed 的 McNemar p 值与方向；
- 不把 3 个 Seed 的 p 值简单平均；
- 不把 McNemar 的“显著”当作确认性结论；
- 不把 `positive_or_fusion` 的增益当作真实 Oracle 或融合有效性的证明。

### 7.4 标签共现

**在 Training split 上计算**（避免 Test 驱动解释）：

- 共现计数矩阵 `C[a,b] = #(y_a=1 & y_b=1)`；
- 条件概率 `P(b=1 | a=1)`；
- phi 系数 / Cramér's V；
- 层级父子对的共现；
- 聚类/分组（conjugation、aromatic、carbonyl、nitrogen、donor-acceptor）。

**用途**：

- 解释 Macro 平均为什么被某些标签主导；
- 解释某些标签的 AP/F1 为什么高/低（标签冗余、父子关系）；
- 识别“一个标签正确但另一个标签错误”的结构原因。

**限制**：

- Test 共现只作描述性报告，不用于选阈值/选模型；
- 不因为共现高就删除标签；
- 不把共现当作因果。

### 7.5 结构子群

**特征来源**：`qm9s_structure_library.csv`。

**预定义分箱**（在看 Test 前写死）：

| 维度 | 分箱 |
|---|---|
| `ring_count` | 0 / 1 / 2 / ≥3 |
| `aromatic_ring_count` | 0 / 1 / ≥2 |
| 杂原子数（N+O+F） | 0 / 1 / 2 / ≥3 |
| `heavy_atom_count` | 训练集四分位：Q1–Q4 |
| `rotatable_bond_count` | 0 / 1 / ≥2 |
| `fraction_csp3` | <0.5 / ≥0.5 |
| 结构语义组 | `functional_group_types` 中含羰基 / 含芳香环 / 含腈 / 含炔 / 含胺 / 含羟基（来自结构库，不来自模型预测） |
| 二级描述组（可选） | 按 UVSPEC15 真实标签分组的共轭/芳香/供受体语境；仅用于解释，不用于模型/阈值选择 |

**报告规则**：

- 每个子群报告样本数、每个标签的正例数、三模态 AP/AUROC/F1；
- 只有 `≥30` 个正例且 `≥100` 个样本的子群才进入主表；
- 不足的子群合并或标注“support insufficient”；
- 子群差异用配对 scaffold Bootstrap（同一子群内）估计 CI；
- 子群分析是探索性的，不用于选模型/阈值。

**避免循环论证**：

- 主分组只使用结构库的**独立描述符**（环数、芳香环数、杂原子数、分子量、可旋转键、`fraction_csp3`、`functional_group_types`），不使用模型预测；
- 按 UVSPEC15 真实标签分组只作为**二级描述**，因为标签本身是预测目标，容易造成“用目标解释目标”的循环；
- 任何子群都不得用于选择模型、阈值、BN 状态、seed 或标签子集。

**“UV 边际优势”/“UV 互补价值”/“UV 是噪声”的操作定义**：

- **边际优势**：分别计算 `ΔAP_UV−IR = AP_UV - AP_IR` 与 `ΔAP_UV−Raman = AP_UV - AP_Raman`；不得用 Test 上的 `max(AP_IR, AP_Raman)` 事后选择对照模态。若需要 `best_vibrational` 汇总，只允许在 Validation 上按 Macro AP 预先指定，并同时保留两个原始对比。
- **有证据的 UV 边际优势**：对某个预定义对比，`ΔAP > 0` 且 Bootstrap 95% CI 下界 > 0 且 `ΔAP ≥ 0.02`；对另一个对比不做合并掩盖。
- **互补价值**：看 §7.1B/§7.3 的条件互补性，例如 `P(UV correct | IR wrong)`、`P(UV correct | Raman wrong)` 是否高于 `P(UV correct)` 基线，以及 `unique_UV` 计数；不能仅凭 UV 边际 AP 低于另一模态就断言“UV 没有增量信息”。
- **条件互补性的判定**：`rescue_rate = P(UV correct | other wrong)` 高于 `P(UV correct)`，且配对 Bootstrap CI 下界 > 0，才可写“有证据的条件互补”；否则写“无证据/不确定”。
- **“UV 是噪声”**：只有在两个预定义边际对比都不支持 UV、且条件互补性也没有显示可检测的 rescue 时，才可写“在该子群/该标签上，UV 未显示可检测的额外价值”；不使用绝对化措辞。
- 所有阈值（0.02、CI 方向、最小 support）必须在看 Test 前写死；不得根据 Test 结果调整。

### 7.6 典型分子分析

- 固定 **Seed 42** 作为定性示例（预先指定，不按 Test 分数挑 seed）；
- 每个结构子群选：
  - 1 个“正确且置信度位于子群中位数”的样本；
  - 1 个“最高置信度错误”的样本；
- 用 `scripts/qm9s_structure_tools.py render` 渲染 2D 结构；
- 附光谱、真实标签、三模态概率/预测；
- 只作定性说明，不进入统计表；
- 记录选取规则与样本 `row_index`，避免 cherry-picking。

### 7.7 结果解释的禁止项

- 不把“Micro-F1 更高”直接解释为“该模态化学信息更丰富”；
- 不把“某个标签 AP 高”直接解释为“该模态能定位该官能团”；
- 不把 `positive_or_fusion` 更高直接解释为“融合一定有效”；OR 融合不是 Oracle，也不保证 F1 提升，真实 Oracle 只是诊断上限；
- 不因 UV 的边际 AP/AUROC 低于 IR 或 Raman 就断言“UV 没有增量信息”；增量信息由条件互补性与真实 Oracle 正确性回答；
- 不把“UV 在 UVSPEC15 上更好”外推到“UV 在所有结构任务上更好”；
- 不把 QM9S 计算域结论外推到实验域。

---

## 8. Test 使用纪律与统计边界

### 8.1 Test 使用流程（硬门禁）

1. **B0（本轮）**：只读审计与协议修订；读取历史 Test 逐标签 CSV 与既有 Raman Test 概率用于一致性审计；**不执行新的 Test 推理，不根据 Test 重新选模型**。
2. **B1（预测导出基础设施 + Shared14 Validation）**：建立统一的导出/评估脚本；导出 Shared14 的 9 个模型 validation 概率并计算 validation 指标；不接触 Test，不写正式 Test 产物。
3. **B2（六个 IR/Raman UVSPEC15 BCE 训练）**：只用 train/validation；训练脚本不加载 Test；不计算 Test 指标。
4. **B3（全部 Validation、阈值与协议冻结）**：导出 UVSPEC15 的 9 个模型 validation 概率；计算全部 18 个模型的 validation 指标；用统一代码拟合全部 18 个模型的 Validation-F1 阈值；写 `FREEZE.json`。
5. **B4（Test 导出与完整分析）**：18 个模型各导出一次 Test 概率；写 `test_access_log.json`；执行指标、Bootstrap、互补性、共现、结构子群分析。
6. **B5（报告与归档）**：只读汇总；报告全部 18 个模型、全部标签、全部 Seed；完成验收与归档。

**`FREEZE.json` 必须包含**：

- 协议 SHA256；
- 分析代码 SHA256；
- 18 个 checkpoint SHA256；
- 18 个模型的 validation 概率数组 SHA256；
- 18 个阈值文件 SHA256；
- 标签顺序 SHA256；
- split SHA256；
- 冻结时间戳。

**Test 门禁**：B4 的 Test 导出必须检查 `FREEZE.json` 存在且哈希匹配；每个模型只导出一次；不得在 B4 修改阈值、模型或分析代码。

### 8.2 允许与禁止

| 允许 | 禁止 |
|---|---|
| Validation 用于 early stopping、阈值拟合、方法比较 | 用 Test 选 seed / 阈值 / BN / 损失 |
| Test 用于一次性最终评价 | 反复看 Test 后改分析计划 |
| 报告所有标签/seed | 只报显著/好看的标签 |
| 用 Bootstrap 给探索性 CI | 把探索性 CI 包装成确认性显著性 |
| 用外部域数据做确认 | 用同一 Test 做“独立确认” |

### 8.3 统计边界

- n=3 seeds：Mean±SD 是描述性的，不做显著性检验；
- 不声称“模态 A 显著优于 B”；
- 不声称“UV 在某个子群显著更好”；
- 不声称“positive_or_fusion 证明融合有效”；
- 不声称“真实 Oracle 正确率等于融合模型性能”；
- 不声称“AP 差异等于化学信息差异”；
- 所有结论标注 exploratory / post-hoc；
- 若要确认性结论：外部域（NIST/API-Raman）或新锁定评估集，另行立项。

### 8.4 多重比较

- 预定义对比：每个任务 3 个模态对 × 14/15 标签；不在看结果后增加对比；
- 若报告 McNemar p 值：每个模态对内 Holm 校正；报告校正前后；
- 若报告子群：子群数量预先写死；不因结果增加子群；
- 所有 p 值只作探索性排序，不作确认性推断。

---

## 9. 输出目录规划

### 9.1 新训练目录

```text
runs/
  LINE-B-UVSPEC15-IR-SEED42/
    config.json
    training_log.csv
    best_model.pt
    run_manifest.json
  LINE-B-UVSPEC15-IR-SEED123/
  LINE-B-UVSPEC15-IR-SEED2026/
  LINE-B-UVSPEC15-RAMAN-SEED42/
  LINE-B-UVSPEC15-RAMAN-SEED123/
  LINE-B-UVSPEC15-RAMAN-SEED2026/
```

### 9.2 预测导出目录

```text
runs/LINE-B-PREDICTIONS/
  Shared14/
    IR/EXP-IR-001-FULL/validation/
      probabilities.npy
      row_index.npy
      labels.npy
      prediction_manifest.json
    IR/EXP-IR-001-FULL/test/
    ...
  UVSPEC15/
    IR/LINE-B-UVSPEC15-IR-SEED42/validation/
    ...
```

### 9.3 阈值、指标、分析、最终报告

```text
runs/LINE-B-THRESHOLDS/
  Shared14/<run_name>/optimal_thresholds.json
  UVSPEC15/<run_name>/optimal_thresholds.json

runs/LINE-B-METRICS/
  Shared14/
    per_label_test.csv
    per_label_validation.csv
    split_summary.csv
  UVSPEC15/
    ...

runs/LINE-B-ANALYSIS/
  Shared14/
    bootstrap/
    complementarity/
    label_cooccurrence/
    structural_subgroups/
    typical_molecules/
  UVSPEC15/
    ...

runs/LINE-B-MANIFESTS/
  FREEZE.json
  test_access_log.json
  asset_manifest.csv

runs/LINE-B-FINAL/
  line_b_report.md
  tables/
  figures/
  README_summary.txt
```

### 9.4 命名规则

- 新训练 run：`LINE-B-UVSPEC15-{IR|RAMAN}-SEED{42|123|2026}`；
- 预测目录：`{task}/{modality}/{run_name}/{split}`；
- 阈值目录：`{task}/{run_name}`；
- 指标目录：`{task}/`；
- 分析目录：`{task}/{analysis_type}/`；
- 不使用 `EXP-` 前缀，避免与历史实验混淆；
- 不写入任何历史 run 目录。

---

## 10. 脚本与运行任务

### 10.1 需要新增的脚本

| 脚本 | 作用 | 关键约束 |
|---|---|---|
| `scripts/audit_line_b_phase_b0.py` | 本轮只读审计 | 已存在；不写历史目录 |
| `training/line_b/train_line_b_uvspec15.py` | 新 IR/Raman UVSPEC15 训练 | 参数化 modality/seed/out-dir；只加载 train/val；禁止 Test；禁止覆盖 |
| `training/line_b/export_line_b_predictions.py` | 统一概率/`row_index`/标签导出 | 支持 14/15 标签；FP32；保存 SHA256；Test 需 FREEZE |
| `training/line_b/calibrate_line_b_thresholds.py` | Validation-only 逐标签 F1 阈值 | 统一代码；不复制历史阈值；保存 manifest |
| `training/line_b/evaluate_line_b_multilabel.py` | AP/AUROC/F1/support、Macro/Micro | 只读数组；不加载模型 |
| `training/line_b/bootstrap_line_b_scaffold.py` | scaffold 聚类配对 Bootstrap | 固定 RNG；同一重采样跨模态/Seed；undefined 计数 |
| `training/line_b/analyze_line_b_complementarity.py` | 错误集合、McNemar、`positive_or_fusion`、真实 Oracle 正确性 | 相同 `row_index`；逐标签/逐样本 |
| `training/line_b/analyze_line_b_structure.py` | 标签共现 + 结构子群 + 典型分子清单 | 训练集共现；子群预定义；输出渲染清单 |
| `training/line_b/build_line_b_report.py` | 汇总表格/图/报告 | 只读；不重新训练 |
| `training/line_b/verify_line_b_manifests.py` | 哈希链与对齐校验 | 失败即停；不自动修复 |

### 10.2 运行任务（B1–B5，不在本轮执行）

```powershell
# B1：预测导出基础设施 + Shared14 Validation
# B1-1：Shared14 validation 概率导出（示例）
python training/line_b/export_line_b_predictions.py `
  --run-dir runs/EXP-IR-001-FULL --modality ir --label-set shared14 `
  --split validation `
  --out-dir runs/LINE-B-PREDICTIONS/Shared14/IR/EXP-IR-001-FULL/validation

# B1-2：Shared14 validation 指标（不做正式阈值）
python training/line_b/evaluate_line_b_multilabel.py --task Shared14 --split validation

# B2：六个 IR/Raman UVSPEC15 BCE 模型训练
# B2-1：新 IR UVSPEC15（3 seeds）
python training/line_b/train_line_b_uvspec15.py --modality ir --seed 42 `
  --out-dir runs/LINE-B-UVSPEC15-IR-SEED42
python training/line_b/train_line_b_uvspec15.py --modality ir --seed 123 `
  --out-dir runs/LINE-B-UVSPEC15-IR-SEED123
python training/line_b/train_line_b_uvspec15.py --modality ir --seed 2026 `
  --out-dir runs/LINE-B-UVSPEC15-IR-SEED2026

# B2-2：新 Raman UVSPEC15（3 seeds）
python training/line_b/train_line_b_uvspec15.py --modality raman --seed 42 `
  --out-dir runs/LINE-B-UVSPEC15-RAMAN-SEED42
python training/line_b/train_line_b_uvspec15.py --modality raman --seed 123 `
  --out-dir runs/LINE-B-UVSPEC15-RAMAN-SEED123
python training/line_b/train_line_b_uvspec15.py --modality raman --seed 2026 `
  --out-dir runs/LINE-B-UVSPEC15-RAMAN-SEED2026

# B3：全部 Validation、阈值与协议冻结
# B3-1：UVSPEC15 validation 概率导出（UV 3 + 新 IR/Raman 6）
python training/line_b/export_line_b_predictions.py `
  --run-dir runs/EXP-UVVIS-003-UVSPEC15-FULL --modality uvvis --label-set uvspec15 `
  --split validation `
  --out-dir runs/LINE-B-PREDICTIONS/UVSPEC15/UVVis/EXP-UVVIS-003-UVSPEC15-FULL/validation

# B3-2：18 个模型的 Validation-only 阈值拟合（示例）
python training/line_b/calibrate_line_b_thresholds.py `
  --predictions-dir runs/LINE-B-PREDICTIONS/Shared14/IR/EXP-IR-001-FULL/validation `
  --label-names processed/qm9s/prepared/functional_group_label_names.json `
  --out-dir runs/LINE-B-THRESHOLDS/Shared14/EXP-IR-001-FULL

# B3-3：冻结（哈希清单 + FREEZE.json）
python training/line_b/verify_line_b_manifests.py --freeze

# B4：Test 导出与完整分析
# B4-1：Test 概率导出（每模型一次；需 FREEZE）
python training/line_b/export_line_b_predictions.py `
  --run-dir runs/EXP-IR-001-FULL --modality ir --label-set shared14 `
  --split test --require-freeze `
  --out-dir runs/LINE-B-PREDICTIONS/Shared14/IR/EXP-IR-001-FULL/test

# B4-2：指标 / Bootstrap / 互补性 / 结构分析
python training/line_b/evaluate_line_b_multilabel.py --task Shared14 --split test
python training/line_b/bootstrap_line_b_scaffold.py --task Shared14 --reps 2000
python training/line_b/analyze_line_b_complementarity.py --task Shared14
python training/line_b/analyze_line_b_structure.py --task Shared14

# B5：报告与归档
python training/line_b/build_line_b_report.py
```

### 10.3 预计资源

基于历史训练日志：

| 任务 | 历史 epoch 时间 | 历史最优 epoch | 预计单 run |
|---|---:|---:|---:|
| IR Shared14 | 14–16 s/epoch | 21–30 | 6–10 min |
| Raman Shared14 | 15–17 s/epoch | 22–28 | 7–10 min |
| UVSPEC15 UV | 3–6 s/epoch | 5–10 | 1.5–2 min |
| 新 IR/Raman UVSPEC15 | 与 IR/Raman 同量级 | 预计 10–25 | 6–12 min/run |

- 6 次新训练：约 **1–2 小时 GPU**（含启动/评估开销）；
- 18 个模型 × 2 split 概率导出：约 **15–40 分钟**（GPU 推理；CPU 会更慢）；
- 阈值拟合/指标/Bootstrap：CPU，约 **30–90 分钟**；
- 总计 B1–B5：约 **2–4 小时**（B2 训练为主；B1/B3/B4 为导出、校准与分析）。

### 10.4 预计磁盘

- 6 份新 checkpoint：约 **74 MB**（每份 ~12.3 MB）；
- 18 个模型 × 2 split 概率/标签/行号：约 **30–50 MB**；
- 阈值/指标/分析表：< 100 MB；
- 图/渲染：< 100 MB；
- 总计 < 500 MB。历史 checkpoint 与光谱矩阵不在 Git 中，Line B 新产物也不应进 Git（除非是轻量 CSV/Markdown）。

---

## 11. 阶段验收检查清单

### 11.1 B0（本轮）

- [x] 只读审计脚本可重复运行；JSON/CSV 输出生成；
- [x] 40 份生产 checkpoint 的 head/BN/label 核验通过；
- [x] 三模态 val/test 逐元素一致；
- [x] 103855 vs 103853 定位到 13998/14000；
- [x] UVSPEC15 列顺序与源矩阵/checkpoint 一致；
- [x] 22 份逐标签 Test support 与标签矩阵一致；
- [x] scaffold 零重叠；
- [x] 未执行新的训练/推理；未用 GPU；B0 仅读取历史 Test CSV 与既有 Raman Test 概率用于一致性审计；
- [x] 未修改历史数据/checkpoint/指标/实验结论；仅按本轮要求做文档勘误；
- [x] 审计报告与协议 v0.2 修订完成；
- [ ] **人工审核并批准以下决策**：主比较配置、103853 训练集、首轮只 BCE、0.5 主/Val-F1 次、Test 解释边界。

### 11.2 B1（预测导出基础设施 + Shared14 Validation）

- [ ] 统一导出/评估脚本已实现，并在 1 个 Shared14 模型上完成 dry-run；
- [ ] 导出包含 `probabilities.npy`、`row_index.npy`、`labels.npy`、`prediction_manifest.json`；
- [ ] `row_index` 与冻结 split 完全一致；`y_true` 与标签矩阵完全一致；
- [ ] Shared14 的 9 个模型 validation 概率导出完成；
- [ ] Shared14 validation AP/AUROC/F1/support 可从数组复算；
- [ ] 每个数组 SHA256 与 manifest 一致；
- [ ] 本阶段未接触 Test、未拟合正式阈值。

### 11.3 B2（六个 IR/Raman UVSPEC15 BCE 训练）

- [ ] 6 个新 run 目录均为全新，未覆盖历史；
- [ ] 每个 run 的 `config.json` 记录 modality/seed/loss/BN/train_size/label_names；
- [ ] 输出头 15，`label_names` 与 `uvspec15_label_names.json` 完全一致；
- [ ] train/val/test = 103853/12981/12981；
- [ ] 最优轮按 Validation mAP；
- [ ] 训练日志无 NaN/Inf；
- [ ] 训练脚本从未加载 Test、从未计算 Test 指标；
- [ ] 6 份 checkpoint 保存 `run_manifest.json` 与 SHA256。

### 11.4 B3（全部 Validation、阈值与协议冻结）

- [ ] UVSPEC15 的 9 个模型（UV 3 + 新 IR/Raman 6）validation 概率导出完成；
- [ ] 18 个模型的 validation 指标全部可复算；
- [ ] 18 个模型的 Validation-only 逐标签 F1 阈值拟合完成；
- [ ] 阈值文件与 checkpoint、BN 状态、标签顺序、概率数组一一绑定；
- [ ] `FREEZE.json` 写入协议/代码/18 checkpoint/18 概率数组/18 阈值/标签/split 哈希；
- [ ] 本阶段未接触 Test。

### 11.5 B4（Test 导出与完整分析）

- [ ] Test 导出每模型仅一次，`test_access_log.json` 完整；
- [ ] Test 数组的 `row_index` 与冻结 split 完全一致；
- [ ] `y_true` 与标签矩阵完全一致；
- [ ] 概率有限且在 [0,1]；
- [ ] 阈值只来自 Validation；`frozen_before_test=true`；
- [ ] 逐标签 AP/AUROC/F1/support 可从数组复算；
- [ ] Macro/Micro 定义与本文档一致；
- [ ] 3 Seed Mean±SD 报告完整；
- [ ] Bootstrap 以 scaffold 为聚类单位；所有模型/Seed 使用同一重采样索引；
- [ ] Bootstrap 中 undefined AP/AUROC 已标记、计数、跳过；
- [ ] 错误互补性使用相同 `row_index`；
- [ ] `positive_or_fusion` 与真实 Oracle 正确性分开报告；
- [ ] 条件互补性（UV | IR wrong、UV | Raman wrong）与边际 AP/AUROC 分开报告；
- [ ] 标签共现只用 Training 拟合；
- [ ] 结构子群分箱在看 Test 前已写死；
- [ ] 所有 18 个模型、14/15 个标签、3 个 Seed 均报告；
- [ ] 报告明确标注 exploratory/post-hoc 与 Test 已开启；
- [ ] 未修改任何历史 checkpoint/manifest/阈值/报告。

### 11.6 B5（报告与归档）

- [ ] 每个任务给出三模态 Macro AP/AUROC + Bootstrap CI；
- [ ] 给出逐标签 AP/AUROC/F1/support 表；
- [ ] 给出 `positive_or_fusion`、真实 Oracle 正确性、条件互补性结果；
- [ ] 给出结构子群与典型分子分析；
- [ ] 明确列出支持/不支持 H1–H4 的证据；
- [ ] 给出“是否进入 Line C 融合”的建议；
- [ ] 不把探索性结果写成确认性结论；
- [ ] 完成文件清单、哈希与归档记录。

---

## 12. 需要人工决策的问题（汇总）

| ID | 问题 | 建议 | 影响 |
|---|---|---|---|
| D1 | 主比较是否统一用 BCE + Original + 0.5 | 是 | 决定整个 Line B 的可比性 |
| D2 | UVSPEC15 新 IR/Raman 是否用 103853 训练 | 是 | 决定三模态训练行是否完全一致 |
| D3 | 首轮是否只做 BCE、不做 WBCE/BN 重估 | 是 | 决定是 6 次还是 12+ 次训练 |
| D4 | 是否接受 0.5 主、Val-F1 次的阈值双轨 | 是 | 决定 F1 表的解释方式 |
| D5 | 是否接受 Test 仅作 exploratory/post-hoc | 是 | 决定论文措辞与确认性边界 |
| D6 | 低支持度标签是否保留在 Macro | 保留 + 预定义敏感性 | 决定是否事后删除标签 |
| D7 | 是否发布文档 errata（3,889→3,891；字母序说明） | 是，只改文档 | 决定文档一致性 |
| D8 | 结构子群的最小支持度与效应量门槛 | ≥30 正例/≥100 样本；abs(ΔAP)≥0.02 | 决定子群结论强度 |
| D9 | 典型分子选取规则 | 中位数正确 + 最高置信错误；Seed 42 | 决定定性示例的公平性 |
| D10 | 推理 dtype | 统一 FP32 | 决定与历史 FP16 的差异口径 |
| D11 | 是否允许 seed 平均概率 | 不作为新模型；Bootstrap 用配对 3 Seed | 决定互补性统计口径 |
| D12 | 是否在 B3 做 UV shared BN 重估 | 否，延后 | 决定是否增加 3 份 BN 后处理 |

---

## 13. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | 2026-11-17 | 首版 B0 协议草案：审计结论、两个任务、核心/次要矩阵、指标、Bootstrap、Test 纪律、目录与验收清单 |
| v0.2 | 2026-11-17 | 人工审核修订：① `oracle_union` 改为 `positive_or_fusion`，新增使用 `y_true` 的真实 Oracle 正确性分析；② 修正 Bootstrap CI 解释（scaffold 聚类、同一重采样、undefined 标记计数、不替代 seed 随机性）；③ 明确 B0 已读取历史 Test CSV/既有 Raman Test 概率但未做新 Test 推理；④ 拆分 UV 边际差异与条件互补性，禁止按 Test 事后选择对照模态；⑤ 阶段改为 B1–B5 并同步运行示例、FREEZE 门禁与验收表 |
