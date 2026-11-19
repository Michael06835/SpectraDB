# Line B · Phase B0 现有资产审计报告

> **状态**：B0 审计完成，待人工审核。本文档只做只读审计与设计；不执行新的训练/推理，不重估 BN，不校准阈值，不修改任何历史数据/checkpoint/指标/实验结论；仅按本轮要求做文档勘误。
>
> **日期**：2026-11-17（Asia/Shanghai）
>
> **审计脚本**：`scripts/audit_line_b_phase_b0.py`
>
> **机器可读输出**：
> - `metadata/line_b_b0_audit.json`（划分、标签、checkpoint、概率数组、阈值文件、哈希与逐项检查）
> - `metadata/line_b_b0_checkpoints.csv`（40 份生产 checkpoint 的头部维度、BN 层数、`num_batches_tracked`、seed/loss/train_size）
> - `metadata/line_b_b0_experiment_matrix.csv`（Line B 实验矩阵草案）
>
> **范围**：IR v1、Raman v1、UV-Vis shared（14 标签）、UVSPEC15 v1，以及三模态共享的数据/标签/划分资产。

---

## 0. 审计结论摘要

1. **三套单模态方法学实验均已完整冻结**：IR v1、Raman v1、UVSPEC15 v1 各有 6 个原始模型、6 份 Training-only BN 重估 checkpoint、12 次阈值校准和 24 条件消融汇总；UV-Vis shared 有 3 个 BCE 模型与 1 个 WBCE 模型，但没有 BN 重估 checkpoint。生产 checkpoint 共 **40 份**（22 原始 + 18 BN 重估），另有 3 份 sanity checkpoint。
2. **划分一致性良好**：IR/Raman 的 `103855/12981/12981` 与 UV 的 `103853/12981/12981` 差异**只有两行**——`row_index=13998` 与 `14000` 的全零 UV 谱。两行只出现在 IR/Raman 的 train 中，已被 UV 的 train/val/test 全部排除；三模态的 **Validation/Test 行索引逐元素相同且顺序相同**，因此 Line B 可以在完全相同的 val/test 样本上做配对比较。
3. **UVSPEC15 标签顺序已独立核验**：`uvspec15_label_names.json` 的 15 个名字、`uvspec15_labels.npy` 的 15 列、源标注矩阵的 15 个列索引、以及 UVSPEC15 checkpoint 的 `feature_head.weight`（`[15, 256]`）与 checkpoint 内的 `label_names` **完全一致**。`UVSPEC15_v1_final_decision.md` 第 1 节按字母序列举标签，**不是**输出头顺序；不得据此推断列顺序。
4. **现有资产的最大缺口是样本级概率/预测导出**：只有 Raman 保存了 24 份概率数组（2 种损失 × 2 种 BN × 3 seeds × 2 splits）；IR、UV-Vis shared 与 UVSPEC15 都没有保存概率数组，也没有任何样本级预测/`row_index` 导出。Line B 的错误互补性、scaffold Bootstrap、结构子群分析因此必须在 B1 新增统一的只读推理导出步骤。
5. **共享 14 标签任务可以直接复用冻结模型**：IR/Raman/UV 三模态各有 BCE + Original BN 的 3-seed checkpoint，正好构成同一损失、同一 BN、同一阈值政策的可比矩阵。**UVSPEC15 任务需要新训 IR/Raman 各 3 个 BCE 模型**（共 6 次训练）；UV 侧复用 `EXP-UVVIS-003-UVSPEC15-*` 的 3 个 Original-BN checkpoint。
6. **不能把各模态 v1 的“主方案”直接拿来做跨模态比较**：IR v1 主方案是 BCE + Original + Val-F1，Raman v1 主方案是 WBCE + Recalibrated + Val-F1，UVSPEC15 v1 主方案是 BCE + Recalibrated + Val-F1。三者损失/BN/阈值均不同，直接比较会把实验配置差异误读为模态差异。Line B 的主比较应统一使用三份 Final Decision 都明确保留的 **固定基线：BCE + Original BN + 0.5**，并以 AP/AUROC 为主证据，F1 作为同一阈值政策下的次要证据。
7. **Test 已不是“未开封”**：v1 阶段已经在 Test 上报告过指标并做过事后配置决策。B0 已读取历史 Test 逐标签 CSV 与既有 Raman Test 概率用于一致性审计，但**没有执行新的 Test 推理，也没有据此重选模型**。Line B 只能定位为**在已开启 Test 上的探索性/事后分析**；任何确认性主张都需要外部域数据或新的锁定评估集，不能包装成“盲法预注册确认性实验”。

---

## 1. 审计方法与只读声明

### 1.1 只读范围

本轮只执行了以下操作：

- 读取 `processed/qm9s/prepared/` 下的 `.npz` 划分、标签矩阵、名称 JSON、manifest 与报告；
- 以 `mmap` 方式读取 `uvvis_float32.npy`，只扫描全零行并核对 `13998/14000`；
- 以 `torch.load(..., map_location="cpu", weights_only=False)` 读取 checkpoint 的元数据与 state dict 形状，**没有 forward pass**；
- 读取 `runs/` 下的配置文件、阈值 JSON、逐标签 CSV、消融汇总与既有 Raman 概率数组；**包括 22 份历史 Test 逐标签 CSV 与既有 Raman Test 概率数组，仅用于 support/指标一致性审计**；
- 运行 `scripts/audit_line_b_phase_b0.py` 生成 JSON/CSV 审计记录。

未执行：

- 任何新的训练或推理（包括任何新的 forward pass）；B0 对 Test 的接触仅限读取历史 CSV/概率，**没有执行新的 Test 推理，也没有根据 Test 重新选择模型、阈值、BN 或 seed**；
- 任何 BN 重估、阈值校准；
- 任何 GPU 计算；
- 任何对 `runs/`、`processed/`、checkpoint、manifest、阈值、指标或历史实验结论的写入；仅按要求修订两份文档说明（不改数据/指标）；
- 任何 Git commit / push。

### 1.2 审计可复现性

```powershell
# 只读审计；输出到 metadata/
& D:\miniconda3\envs\spectra-db\python.exe scripts\audit_line_b_phase_b0.py `
    --json-out metadata\line_b_b0_audit.json
```

审计脚本中所有路径都相对于仓库根目录解析；不依赖当前工作目录。

---

## 2. 数据资产审计

### 2.1 光谱矩阵与谱轴

| 资产 | 路径 | 形状 / 范围 | 状态 | 说明 |
|---|---|---|---|---|
| IR 光谱 | `processed/qm9s/prepared/ir_float32.npy` | `(129817, 3501)`，float32 | 存在 | 500–4000 cm⁻¹，步长 1 cm⁻¹ |
| IR 轴 | `processed/qm9s/prepared/ir_axis.npy` | `(3501,)` | 存在 | 单调递增，无重复点 |
| Raman 光谱 | `processed/qm9s/prepared/raman_float32.npy` | `(129817, 3501)`，float32 | 存在 | 500–4000 cm⁻¹，步长 1 cm⁻¹ |
| Raman 轴 | `processed/qm9s/prepared/raman_axis.npy` | `(3501,)` | 存在 | 单调递增，无重复点 |
| UV-Vis 光谱 | `processed/qm9s/prepared/uvvis_float32.npy` | `(129817, 701)`，float32 | 存在 | 1–15 eV，步长约 0.02 eV |
| UV-Vis 轴 | `processed/qm9s/prepared/uvvis_axis.npy` | `(701,)` | 存在 | 单调递增 |
| 光谱 QC | `ir_qc_metrics.npz`、`raman_qc_metrics.npz`、`uvvis_qc_metrics.npz` | — | 存在 | 0 非有限值；UV 2 条全零谱；IR 1 条、Raman 128 条强度离群（保留） |

**归一化**：三个模态的 Dataset 都使用逐谱最大绝对值归一化（`normalization="max"`），在 `QM9SSpectrumDataset._normalize()` 内施加；不依赖任何跨 split 的均值/方差统计量，因此不构成跨样本泄漏。IR/Raman 输入长度 3501，UV 输入长度 701，这是模态本身的差异，不是实验配置差异。

### 2.2 标签资产

| 资产 | 路径 | 形状 / 内容 | 状态 | 说明 |
|---|---|---|---|---|
| 14 类官能团标签矩阵 | `processed/qm9s/prepared/functional_group_labels.npy` | `(129817, 14)`，uint8，取值 {0,1} | 存在 | 由结构库生成 |
| 14 类名称文件 | `processed/qm9s/prepared/functional_group_label_names.json` | 14 个名字，有序 | 存在 | 输出头顺序 |
| 14 类频数表 | `processed/qm9s/prepared/functional_group_label_counts.csv` | 14 行 | 存在 | 频数 |
| UVSPEC15 标签矩阵 | `processed/qm9s/prepared/uvspec15_labels.npy` | `(129817, 15)`，float32，取值 {0,1} | 存在 | 15 个 UV 专属语义标签 |
| UVSPEC15 名称文件 | `processed/qm9s/prepared/uvspec15_label_names.json` | 15 个名字，有序 | 存在 | **唯一权威列顺序** |
| UVSPEC15 provenance manifest | `processed/qm9s/prepared/uvspec15_label_manifest.json` | 源路径、列索引、源 SHA256、split 规模 | 存在 | 与源标注矩阵核验一致 |
| UV 源二值标注 | `processed/qm9s/uvvis_annotations_v1_0_1/structure_binary_labels.npy` | `(129817, 58)` | 存在 | builder v1.0.1，129817 全部成功 |
| UV 源标签名 | `processed/qm9s/uvvis_annotations_v1_0_1/binary_label_names.json` | 58 个名字 | 存在 | 与 manifest 列索引对应 |
| UV 标注统计 | `metadata/uvvis_annotation_statistics_v1_0_1.csv` | 58 二值 + 73 数值 | 存在 | 用于频数核对 |
| UV 标注 split 审计 | `metadata/uvvis_label_split_audit_v1_0_1.csv` | 按 split 统计 | 存在 | 用于稀疏性核对 |

**哈希核验**：

- 源二值矩阵 SHA256：`6de1918823ab0cd856c2c9bd339cd1641180fec262e286b50efb8a1dacc7dcec`
- `uvspec15_label_manifest.json` 中记录的源 SHA256：与上式 **完全一致**
- `uvspec15_labels.npy` SHA256：`f08170df62972828076b4cfdb1136b2890b86a4eec36f280466180f5c64f1705`
- `uvspec15_labels.npy` 的 15 列与 `structure_binary_labels.npy[:, column_indices_in_source]` **逐元素完全一致**

### 2.3 划分、valid mask 与行号映射

| 资产 | 路径 | 内容 | 状态 |
|---|---|---|---|
| 基础 scaffold split | `processed/qm9s/prepared/scaffold_split_80_10_10.npz` | train 103855 / val 12981 / test 12981 | 存在 |
| IR valid split | `processed/qm9s/prepared/ir_scaffold_split_valid.npz` | 103855 / 12981 / 12981 | 存在 |
| Raman valid split | `processed/qm9s/prepared/raman_scaffold_split_valid.npz` | 103855 / 12981 / 12981 | 存在 |
| UV valid split | `processed/qm9s/prepared/uvvis_scaffold_split_valid.npz` | 103853 / 12981 / 12981 | 存在 |
| 三模态完整 split | `processed/qm9s/prepared/multimodal_complete_scaffold_split.npz` | 103853 / 12981 / 12981，与 UV 完全相同 | 存在 |
| 模态有效 mask | `processed/qm9s/prepared/modality_valid_masks.npz` | `valid_ir=129817`、`valid_raman=129817`、`valid_uvvis=129815`、`valid_multimodal_complete=129815` | 存在 |
| 有效性报告 | `processed/qm9s/prepared/modality_validity_report.json` | 2 条 UV 全零谱被过滤 | 存在 |
| 样本 manifest | `processed/qm9s/prepared/qm9s_manifest_prepared.csv` | 129817 行，`row_index` 0..129816 连续 | 存在 |
| 带有效性 manifest | `processed/qm9s/prepared/qm9s_manifest_with_modality_validity.csv` | 129817 行 | 存在 |
| 结构库 | `processed/qm9s/structures/qm9s_structure_library.csv` | 26 列结构特征，129817 行 | 存在 |

**行号映射**：`row_index` 是 0-based、连续、与光谱矩阵/标签矩阵/结构库同一行位置。Dataset 的 `__getitem__` 直接返回 `row_index`，但历史上没有把 `row_index` 单独落盘；Line B 的新导出必须显式保存 `row_index.npy` 或等价的 manifest。

### 2.4 结构库可用字段

`processed/qm9s/structures/qm9s_structure_library.csv` 提供：

- `row_index`、`qm9_number`、`canonical_smiles`、`molecular_formula`、`exact_mol_wt`
- `heavy_atom_count`、`bond_count`、`carbon_count`、`nitrogen_count`、`oxygen_count`、`fluorine_count`
- `ring_count`、`aromatic_ring_count`、`rotatable_bond_count`
- `h_bond_donor_count`、`h_bond_acceptor_count`、`tpsa`、`fraction_csp3`
- `functional_group_types`、`functional_group_instance_count`
- 四个 `valid_*` 标记

这些字段足以支撑 Line B 的结构子群分析，无需重新计算 RDKit 描述符。结构渲染可复用 `scripts/qm9s_structure_tools.py render`。

---

## 3. 模型、checkpoint 与结果资产

### 3.1 生产 checkpoint 总量

| 类别 | 数量 | 说明 |
|---|---:|---|
| 原始 checkpoint（`best_model.pt`） | **22** | IR 6、Raman 6、UV shared 4、UVSPEC15 6 |
| BN 重估 checkpoint（`best_model_bn_recalibrated.pt`） | **18** | IR 6、Raman 6、UVSPEC15 6；UV shared 无 |
| 生产 checkpoint 合计 | **40** | 全部通过头部维度 / BN 层数 / 标签数核验 |
| Sanity checkpoint（`model.pt`） | 3 | IR/Raman/UV shared 各 1 个 sanity；不用于 Line B |

**全量核验结果**（`metadata/line_b_b0_checkpoints.csv`）：

- 所有 40 份 checkpoint 的 `feature_head.weight` 只有两种维度：`[14, 256]`（Shared14）与 `[15, 256]`（UVSPEC15）；
- 所有 checkpoint 都有 **12 层 BatchNorm1d**；
- 所有原始 checkpoint 的 `num_batches_tracked` 等于 `最优 epoch × 203`（203 = 103853 或 103855 条 train 在 batch 512 下的批数）；
- 所有 BN 重估 checkpoint 的 `num_batches_tracked` 均为 **203**，与“只用 Training split 重估一遍”的 manifest 一致；
- checkpoint 内 `label_names` 与对应名称 JSON 完全一致，`label_order_issues=[]`。

### 3.2 按实验组盘点

| 实验组 | run 目录 | 原始 ckpt | BN 重估 ckpt | 阈值校准 | 逐标签 Test | 概率数组 | 结论 |
|---|---|---:|---:|---|---|---|---|
| IR v1 · BCE | `EXP-IR-001-FULL`、`-SEED123`、`-SEED2026` | 3 | 3（在 `IR-BN-RECALIBRATED/`） | Original 3 + Recalibrated 3 | 3 | 无 | 完整 |
| IR v1 · WBCE | `EXP-IR-002-FULL`、`-SEED123`、`-SEED2026` | 3 | 3 | Original 3 + Recalibrated 3 | 3 | 无 | 完整 |
| Raman v1 · BCE | `EXP-RAMAN-001-FULL`、`-SEED123`、`-SEED2026` | 3 | 3（在 `RAMAN-BN-RECALIBRATED/`） | Original 3 + Recalibrated 3 | 3 | **12 npy**（3 seeds × 2 BN × 2 splits） | 完整 |
| Raman v1 · WBCE | `EXP-RAMAN-002-FULL`、`-SEED123`、`-SEED2026` | 3 | 3 | Original 3 + Recalibrated 3 | 3 | **12 npy**（3 seeds × 2 BN × 2 splits） | 完整 |
| UV-Vis shared · BCE | `EXP-UVVIS-001-FULL`、`-SEED123`、`-SEED2026` | 3 | 0 | 仅 Seed 42 Original | 3 | 无 | 完整（主线 0.5） |
| UV-Vis shared · WBCE | `EXP-UVVIS-002-FULL` | 1 | 0 | 无 | 1 | 无 | 仅 Seed 42，非三 Seed |
| UVSPEC15 · BCE | `EXP-UVVIS-003-UVSPEC15-FULL`、`-SEED123`、`-SEED2026` | 3 | 3（在 `UVSPEC15-BN-RECALIBRATED/`） | Original 3 + Recalibrated 3 | 3 | 无 | 完整 |
| UVSPEC15 · WBCE | `EXP-UVVIS-004-UVSPEC15-WBCE`、`-SEED123`、`-SEED2026` | 3 | 3 | Original 3 + Recalibrated 3 | 3 | 无 | 完整 |

**阈值文件总量**：`optimal_thresholds.json` 共 **39 份**，分布为：

- `IR-THRESHOLD-CALIBRATION` 6 + `IR-BN-THRESHOLD-CALIBRATION` 6 = 12；
- `RAMAN-THRESHOLD-CALIBRATION` 6 + `RAMAN-BN-THRESHOLD-CALIBRATION` 6 = 12；
- `UVSPEC15-BN-RECALIBRATED` 6 + UVSPEC15 训练 run 目录内 6 = 12；
- UV shared 训练 run 目录内 1（`EXP-UVVIS-001-FULL`）；`EXP-UVVIS-002-FULL` 没有阈值文件；
- IR/Raman 训练 run 目录内各 1（历史单阈值文件，已由独立校准目录取代）。

> **注意**：UV shared 的 `EXP-UVVIS-001-SEED123` / `EXP-UVVIS-001-SEED2026` **没有**逐标签阈值文件。若 Line B 要做“Validation-F1 阈值”的次要分析，必须用统一脚本在这两个 checkpoint 的 Validation 概率上重新拟合；不得从 Seed 42 复制阈值。

### 3.3 汇总与报告资产

| 资产 | 路径 | 状态 | 说明 |
|---|---|---|---|
| IR v1 汇总 | `runs/IR-v1-FINAL/` | 存在 | 3-seed 指标、逐标签 F1、图 |
| Raman v1 汇总 | `runs/Raman-v1-FINAL/` | 存在 | 3-seed 指标、逐标签 F1、图 |
| UV shared 汇总 | `runs/UVVIS-v1-FINAL/` | 存在 | final 3-seed 指标、逐标签 F1；旧版快照在 archive `superseded_uvvis_v1_final/` |
| IR 24 条件消融 | `runs/IR-ABLATION-SUMMARY/` | 存在 | 48 split-level、672 逐标签、审计表 |
| Raman 24 条件消融 | `runs/RAMAN-ABLATION-SUMMARY/` | 存在 | 48 split-level、672 逐标签、审计表 |
| UVSPEC15 24 条件消融 | `runs/UVSPEC15-ABLATION-SUMMARY/` | 存在 | 48 split-level、720 逐标签、审计表 |
| shared vs specific 对照 | `runs/UVSPEC15-ANALYSIS/shared_vs_specific_summary.md` | 存在 | UV 14 vs 15 标签 |
| IR Final Decision | `docs/experiments/IR_v1_final_decision.md` | 存在 | 冻结 |
| Raman Final Decision | `docs/experiments/Raman_v1_final_decision.md` | 存在 | 冻结 |
| UVSPEC15 Final Decision | `docs/experiments/UVSPEC15_v1_final_decision.md` | 存在 | 冻结 |
| IR/Raman/UVSPEC15 策略、登记表、复现协议 | `docs/experiments/*_experimental_strategy.md`、`*_ablation_registry.md`、`*_repro_protocol.md` | 存在 | 冻结 |

### 3.4 冻结的“固定基线”参考指标

下表只作为历史参考，**不是 Line B 的结论**。它对应三份 Final Decision 都明确保留的 **BCE + Original BN + 阈值 0.5** 条件；这正是 Line B 主比较要复用的同一配置家族。

| 任务 | 模态 | Micro-F1 | Macro-F1 | mAP（Macro AP） | Macro-AUROC | 来源 |
|---|---|---:|---:|---:|---:|---|
| Shared14 | IR | 0.9405 ± 0.0048 | 0.9193 ± 0.0050 | 0.9688 ± 0.0050 | 0.9954 ± 0.0007 | `IR-ABLATION-SUMMARY/README_summary.txt` |
| Shared14 | Raman | 0.9151 ± 0.0014 | 0.8885 ± 0.0079 | 0.9548 ± 0.0044 | 0.9911 ± 0.0004 | `RAMAN-ABLATION-SUMMARY/README_summary.txt` |
| Shared14 | UV-Vis | 0.5158 ± 0.0067 | 0.4130 ± 0.0232 | 0.5126 ± 0.0011 | 0.8526 ± 0.0016 | `UVVIS-v1-FINAL/uvvis_final_summary.json` |
| UVSPEC15 | UV-Vis | 0.6136 ± 0.0845 | 0.4397 ± 0.0744 | 0.5389 ± 0.0599 | 0.9259 ± 0.0204 | `UVSPEC15-ABLATION-SUMMARY/README_summary.txt` |
| UVSPEC15 | IR | **无** | **无** | **无** | **无** | 需新训练 |
| UVSPEC15 | Raman | **无** | **无** | **无** | **无** | 需新训练 |

> 均值 ± 样本 SD（n=3 seeds）。UV-Vis 行已于 2026-11-18 更正为真实 Seed2026 的三 Seed 结果；旧三 Run 均值（0.4892/0.5053/0.8501）包含 invalid Seed2026（实际 SEED=42），旧版快照在 `archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/superseded_uvvis_v1_final/`，仅作历史参考。这些数值说明：在共享 14 标签上，UV 的排序能力（AUROC 0.853）并不“归零”，但阈值 0.5 下的 F1 仍明显低于振动模态；在 UVSPEC15 上，UV 的 AP/AUROC 更高。Line B 需要回答的是“各模态各自擅长什么”，而不是重复这张表。

### 3.5 代码资产（Dataset / Model / 训练 / 评估）

| 类别 | 路径 | 状态 | 说明 |
|---|---|---|---|
| Dataset | `training/datasets/qm9s_spectrum_dataset.py` | 存在 | mmap 读取 NPY；逐谱 max 归一化；返回 `spectrum/labels/row_index` |
| Model | `training/models/spectrum_cnn1d.py` | 存在 | `SpectrumEncoder1D` + `SpectrumStructureModel(num_labels, embedding_dim)` |
| IR 训练 | `training/train_ir_full.py`、`train_ir_seed123.py`、`train_ir_seed2026.py`、`train_ir_weighted*.py` | 存在 | 各 seed/损失独立脚本；Validation mAP 选轮 |
| Raman 训练 | `training/train_raman_full.py`、`train_raman_seed123.py`、`train_raman_seed2026.py`、`train_raman_weighted*.py` | 存在 | 同上 |
| UV shared 训练 | `training/train_uvvis_full.py`、`train_uvvis_seed123.py`、`train_uvvis_seed2026.py`、`train_uvvis_weighted.py` | 存在 | 同上 |
| UVSPEC15 训练 | `training/train_uvvis_uvspec15.py`、`_seed123.py`、`_seed2026.py`、`_weighted*.py` | 存在 | 15 维头；Validation mAP 选轮 |
| 阈值校准 | `training/calibrate_ir_*.py`、`calibrate_raman_*.py`、`calibrate_uvvis_uvspec15*.py`、`calibrate_uvspec15_bn_thresholds.py` | 存在 | Validation-only；部分脚本会写历史 run 目录 |
| BN 诊断/重估 | `training/diagnose_*_batchnorm*.py`、`save_ir_bn_checkpoint.py`、`save_raman_bn_checkpoint.py`、`save_uvspec15_bn_checkpoint.py` | 存在 | 只读源 checkpoint；写入独立目录 |
| 汇总 | `training/summarize_ir_ablation.py`、`summarize_raman_ablation.py`、`summarize_uvspec15_ablation.py` | 存在 | 只读历史 JSON/CSV；写独立汇总目录 |
| 历史收尾 | `training/finalize_ir_results.py`、`finalize_uvvis_results.py` | 存在 | 写 `IR-v1-FINAL` / `UVVIS-v1-FINAL`；UV finalizer 已加非空目录防覆盖保护，**不得重跑** |
| 危险评估入口 | `training/evaluate_raman_checkpoint.py`、`evaluate_raman_seed123_checkpoint.py`、`evaluate_raman_seed2026_checkpoint.py`、`evaluate_raman_weighted_checkpoint.py` | 存在 | 以 `"w"` 模式**原地重写**对应 run 的 `test_metrics.json` / `test_per_label_metrics.csv`；**禁止直接运行** |
| 已废弃入口 | `training/calibrate_ir_thresholds.py`、`calibrate_raman_thresholds.py`、`finalize_raman_results.py` | 存在 | 直接运行立即非零退出；实现已归档 |
| Line B 新脚本 | `training/line_b/` | **缺失** | 需新增训练、导出、阈值、指标、Bootstrap、互补性、结构分析脚本 |
| 共享指标模块 | `training/utils/` | **空** | 历史脚本各自复制指标实现；Line B 应统一到一个模块，避免口径漂移 |

**结论**：Dataset 与 Model 可直接复用；训练/校准脚本可作为实现参考，但**不得直接重跑**历史脚本（会覆盖冻结目录）。Line B 需要一组新的、参数化的、带防覆盖与哈希 manifest 的脚本。

### 3.6 现有逐标签/概率/样本级资产

| 资产类型 | 现状 | 缺口 |
|---|---|---|
| 逐标签 Test 指标 | 22 份 `test_per_label_metrics.csv`，含 `positive_test_samples/negative_test_samples/precision/recall/f1/average_precision/auroc` | 覆盖全部原始模型；support 与冻结标签矩阵逐一核对，**22/22 全部一致** |
| 逐标签 Validation 指标 | IR/Raman/UVSPEC15 的 672/672/720 行消融汇总中有按条件聚合的逐标签记录 | UV shared 没有独立逐标签 Validation CSV；Line B 应统一重算 |
| 概率数组 | 仅 Raman：24 份 `*__{validation,test}__probabilities.npy`（`[12981, 14]`，float32），带 `probability_artifacts.json` 与 SHA256 | IR、UV shared、UVSPEC15 **全部缺失**；Raman 数组无显式 `row_index`，只能依赖冻结 split 顺序 |
| 阈值化预测 | 可由概率 + 阈值派生 | 没有保存为独立数组 |
| 样本级预测结果 | **不存在**（没有 `predictions.csv`、`row_index + y_true + prob` 的联合导出） | Line B 的错误互补性、McNemar、结构子群分析全部依赖它 |
| `row_index` 独立数组 | 不存在 | manifest 有 `row_index` 列；Dataset 返回但未落盘 |
| 概率数组的可复现性核验 | 已抽查 Raman Seed 42：用概率数组重算 14 类 AP/AUROC，与 `test_per_label_metrics.csv` 的最大绝对差 = **0** | 说明 Raman 概率数组与冻结指标完全对齐 |

---

## 4. 样本划分与标签一致性检查

### 4.1 103855 vs 103853：差异只有两行

| 比较 | train | val | test |
|---|---:|---:|---:|
| IR / Raman | 103855 | 12981 | 12981 |
| UV / multimodal complete | 103853 | 12981 | 12981 |
| 差集 | IR/Raman 多 2 行；UV 多 0 行 | 0 | 0 |
| 逐元素顺序 | 不同（UV 少两行） | **完全相同** | **完全相同** |

具体：

- `row_index=13998` 与 `row_index=14000` 的 UV 光谱 **全零**（`absmax=0`、`nonzero=0`）；
- 两者只出现在 IR/Raman 的 **train**；
- 两者在 UV 的 train/val/test 中 **全部不存在**；
- `valid_uvvis[13998]=False`、`valid_uvvis[14000]=False`；
- `valid_multimodal_complete` 同样为 `False`；
- 两者的 manifest `split` 列都是 `train`，但 UV valid split 在构建时已按 `valid_uvvis` 过滤；
- 因此 **Validation/Test 完全不受这两行影响**，Line B 的配对评估可以在 12981 个相同样本上做。

### 4.2 Validation/Test 行索引逐元素一致性

审计对以下五组划分做了 `np.array_equal` 与集合比较：

- `ir_valid` vs `raman_valid`：train/val/test 全部 `equal_order=True`；
- `ir_valid` vs `uvvis_valid`：train `equal_order=False`（只有 2 行差异），val/test `equal_order=True`；
- `raman_valid` vs `uvvis_valid`：同上；
- `uvvis_valid` vs `multimodal_complete`：train/val/test 全部 `equal_order=True`；
- `canonical_ir` vs `ir_valid`：全部 `equal_order=True`。

**结论**：三模态的 Validation/Test 是同一批样本、同一顺序。任何“相同样本的错误互补性”分析都可以直接用行位置对齐，不需要额外匹配 CID/SMILES。

### 4.3 Scaffold 与 manifest 对齐

| 检查 | 结果 |
|---|---|
| `manifest.row_index` 是否为 0-based 连续 | 是（0..129816） |
| `murcko_scaffold` 来源 | `qm9s_manifest_prepared.csv` 的 `murcko_scaffold` 列；由 `scripts/prepare_qm9s_dataset.py` 的 RDKit `MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)` 生成；纯链状分子回退为 `ACYCLIC::<canonical>` |
| 按 `row_index` 唯一获取 scaffold | 129817/129817 非空；`row_index` 唯一且连续；`row_index → murcko_scaffold` 映射唯一；28,501 个唯一 scaffold |
| UV npz vs manifest `split` 列 | val/test 完全一致；train 只差上述 2 行 |
| scaffold 总数 | 28,501 |
| train / val / test scaffold 数 | 22,796 / 2,852 / 2,853 |
| train–val / train–test / val–test scaffold 重叠 | 0 / 0 / 0 |
| 三集样本内部重叠 | 0 / 0 / 0 |
| 三集并集 | IR/Raman 129817；UV 129815 |

**结论**：scaffold 划分无重叠；UV 的两行移除没有引入新的 scaffold，也没有改变 val/test 的 scaffold 组成。

### 4.4 UVSPEC15 真实列顺序（唯一权威）

`uvspec15_label_names.json` 的顺序就是输出头顺序、标签矩阵列顺序、源矩阵列索引顺序：

| 列号 | 标签 | 源列索引 | Validation 正例 | Test 正例 | Test 正例率 | 支持度分层 |
|---:|---|---:|---:|---:|---:|---|
| 0 | `has_conjugation` | 0 | 3147 | 3000 | 0.2311 | normal |
| 1 | `has_nonaromatic_conjugation` | 1 | 1137 | 809 | 0.0623 | normal |
| 2 | `acyclic_conjugated_system` | 11 | 382 | 307 | 0.0236 | normal |
| 3 | `cyclic_conjugated_system` | 12 | 2765 | 2693 | 0.2075 | normal |
| 4 | `has_aromatic_system` | 17 | 2037 | 2208 | 0.1701 | normal |
| 5 | `heteroaromatic` | 24 | 2035 | 2198 | 0.1693 | normal |
| 6 | `mixed_aromatic_nonaromatic_conjugation` | 15 | 636 | 695 | 0.0535 | normal |
| 7 | `pi_conjugated_carbonyl` | 32 | 1174 | 1036 | 0.0798 | normal |
| 8 | `alpha_beta_unsaturated_carbonyl` | 33 | 392 | 308 | 0.0237 | normal |
| 9 | `aryl_carbonyl` | 35 | 155 | 178 | 0.0137 | normal（接近 low） |
| 10 | `pi_conjugated_imine` | 42 | 214 | 148 | 0.0114 | **low** |
| 11 | `pi_conjugated_nitrile` | 45 | 281 | 222 | 0.0171 | normal |
| 12 | `has_donor_substituted_pi_system` | 55 | 1945 | 2138 | 0.1647 | normal |
| 13 | `has_acceptor_substituted_pi_system` | 56 | 1380 | 1215 | 0.0936 | normal |
| 14 | `donor_acceptor_same_conjugated_system` | 57 | 318 | 370 | 0.0285 | normal |

**核验结果**：

- `uv15_matches_source_columns=True`；
- `uv15_matches_source_names=True`；
- 源矩阵 58 列中选出的 15 个列索引与 manifest 完全一致；
- 3 个 UVSPEC15 BCE checkpoint 与 3 个 WBCE checkpoint 的 `feature_head.weight` 全部是 `[15, 256]`；
- 6 个 checkpoint 内的 `label_names` 全部等于上述顺序；
- 6 个 `config.json` 的 `label_names` 全部等于上述顺序；
- `label_order_issues=[]`。

> **文档陷阱**：`docs/experiments/UVSPEC15_v1_final_decision.md` 第 1 节按**字母序**列出 15 个标签（`acyclic...`, `alpha_beta...`, `aryl...`, ...）。那不是列顺序，只是叙述顺序。任何张量索引、阈值映射、逐标签表都必须以 `uvspec15_label_names.json` 为准。`docs/data/uvspec15_label_guide.md` 的表格恰好是头顺序，但它也不是权威文件。

### 4.5 标签支持度与不确定性

**Shared14**：

| 列号 | 标签 | Validation 正例 | Test 正例 | Test 正例率 | 支持度分层 |
|---:|---|---:|---:|---:|---|
| 0 | aldehyde | 1447 | 1284 | 0.0989 | normal |
| 1 | alkene | 2914 | 1745 | 0.1344 | normal |
| 2 | alkyne | 1708 | 1514 | 0.1166 | normal |
| 3 | amide | 1297 | 1053 | 0.0811 | normal |
| 4 | amine | 3728 | 3491 | 0.2689 | normal |
| 5 | c_f_bond | 230 | 144 | 0.0111 | **low** |
| 6 | carbonyl | 4238 | 3898 | 0.3003 | normal |
| 7 | ester | 633 | 622 | 0.0479 | normal |
| 8 | ether | 5794 | 5260 | 0.4052 | normal |
| 9 | hydroxyl | 3949 | 4334 | 0.3339 | normal |
| 10 | imine | 1191 | 1295 | 0.0998 | normal |
| 11 | ketone | 1223 | 1272 | 0.0980 | normal |
| 12 | nitrile | 1697 | 1517 | 0.1169 | normal |
| 13 | nitro | 11 | 15 | 0.0012 | **very_low** |

- `nitro` 的 Validation 只有 11 个正例：阈值拟合与 F1 估计都极不稳定；任何涉及 `nitro` 的结论必须带 support，不能单独驱动模态排序。
- `c_f_bond` 的 Test 只有 144 个正例：AP/AUROC 的抽样波动也明显大于其他标签。
- Shared14 的 14 个标签都有 train 正例，没有 `NO_TRAIN_POSITIVES`。

**UVSPEC15**：

- 15 个标签都有 train 正例；
- 按 IR/UVSPEC15 既有分层（`min(val,test) < 30 = very_low`，`< 150 = low`），只有 `pi_conjugated_imine` 落入 `low`（val 214 / test 148），没有 `very_low`；
- 但绝对样本量明显小于 Shared14：`aryl_carbonyl` val 155 / test 178，`pi_conjugated_imine` val 214 / test 148，`pi_conjugated_nitrile` val 281 / test 222；这些标签的 per-label AP/F1 必须带 Bootstrap CI，且不能参与“最佳标签”排名。

### 4.6 标签层级与逻辑一致性

只检查 frozen schema 明确定义的父/子关系（`metadata/uvvis_annotation_schema_v1.0.json` 的 `hierarchies` 与 `consistency_rules`）：

| 规则 | 违反数 |
|---|---:|
| `has_nonaromatic_conjugation → has_conjugation` | 0 |
| `acyclic_conjugated_system → has_conjugation` | 0 |
| `cyclic_conjugated_system → has_conjugation` | 0 |
| `mixed_aromatic_nonaromatic_conjugation → has_conjugation` | 0 |
| `heteroaromatic → has_aromatic_system` | 0 |
| `alpha_beta_unsaturated_carbonyl → pi_conjugated_carbonyl` | 0 |
| `aryl_carbonyl → pi_conjugated_carbonyl` | 0 |
| `donor_acceptor_same_conjugated_system → has_donor_substituted_pi_system` | 0 |
| `donor_acceptor_same_conjugated_system → has_acceptor_substituted_pi_system` | 0 |

**不要**把以下“看似合理”的关系当作标签错误：

- `cyclic_conjugated_system → has_nonaromatic_conjugation`：不成立。环状共轭体系可以是芳香体系；schema 只声明 `cyclic → has_conjugation`。
- `mixed_aromatic_nonaromatic_conjugation → has_nonaromatic_conjugation`：schema 没有声明；不要据此判定标注错误。
- `has_donor_substituted_pi_system → has_conjugation`：schema 没有声明；供体可以连在孤立 π 体系上，不要求整个分子存在共轭。

### 4.7 文档计数差异

`docs/data/uvspec15_label_guide.md` 第 2 节把 `acyclic_conjugated_system` 的全样本正例写成 **3,889**；源矩阵与训练矩阵的实际值是 **3,891**（`structure_binary_labels.npy` 与 `uvspec15_labels.npy` 一致，统计 CSV 也是 3,891）。

- 这是**文档笔误**，不是标签矩阵不一致；
- 2026-11-17 B0 修订已把该文档更正为 **3,891**（只改文档，不改任何数据/checkpoint/指标）；
- 同时在 `UVSPEC15_v1_final_decision.md` 的标签列举处加入“按字母序、并非列顺序”的勘误说明。

### 4.8 输入归一化、谱轴、样本行号

| 检查 | 结果 |
|---|---|
| 归一化是否使用跨 split 统计量 | 否；逐谱 `max(abs(x))`，零谱不除 |
| 归一化是否在 Dataset 内 | 是；`QM9SSpectrumDataset._normalize()` |
| IR/Raman 谱轴 | 500–4000 cm⁻¹，3501 点，1 cm⁻¹ |
| UV 谱轴 | 1–15 eV，701 点，约 0.02 eV |
| 行号映射 | `row_index` 直接索引光谱/标签/结构库；manifest 连续 0..129816 |
| 是否存在排序/过滤导致行错位 | 没有；UV 只在 split 层排除 2 行，标签矩阵仍保留 129817 行原始行位置 |

### 4.9 数据泄漏与 Test 选择审计

**结论先行**：未发现 split、归一化、标签或 BN 重估导致的样本级数据泄漏；未发现“用 Test 分数选择 seed/阈值/BN/损失”的直接证据。但 Test 已在 v1 阶段被查看，Line B 不是盲法确认性实验。

**B0 的 Test 接触范围**：B0 读取了 22 份历史 `test_per_label_metrics.csv` 与既有 Raman `*__test__probabilities.npy`，用于 support 与逐标签指标的一致性审计；**没有执行新的 Test 推理，没有用 Test 重新选择模型、阈值、BN 或 seed**。

#### 未发现泄漏的证据

1. **scaffold 零重叠**：train/val/test 的 Bemis–Murcko scaffold 交集为 0；三集样本内部交集为 0。
2. **模态间 val/test 同集**：IR/Raman/UV 的 val/test 逐元素相同；因此跨模态比较不会把“不同样本”误当“不同模态能力”。
3. **UV 全零谱只在 train 侧被排除**：两行全零 UV 谱不在 val/test，不会造成 UV 评估的退化样本。
4. **归一化逐样本**：没有使用 train/val/test 的全局均值/方差，不存在“用 test 统计量归一化”的泄漏。
5. **标签来自结构而非光谱**：14 类官能团与 UVSPEC15 都由 RDKit 结构标注生成，光谱模型不能从输入直接看到标签。
6. **BN 重估只用 Training split**：UVSPEC15/IR/Raman 的 manifest 都写明 `train split only; reset running stats; no optimizer`；`num_batches_tracked=203` 与 103853/103855 条 train 的 203 批一致。
7. **阈值只在 Validation 拟合**：三份 Final Decision 都写明阈值在计算任何 Test 指标前落盘；Raman 的 `probability_artifacts.json` 还保存了概率数组 SHA256，可复核。
8. **模型选择用 Validation mAP**：IR/Raman/UVSPEC15 训练脚本都以 `val_metrics["mAP"]` 作为 best checkpoint 判据；不是 Test 指标。
9. **逐标签 support 一致**：22 份 `test_per_label_metrics.csv` 的 `positive_test_samples` 与冻结标签矩阵逐一比对，**22/22 全部一致**，说明历史评估没有错位。
10. **Raman 概率数组可复现指标**：用 `EXP-RAMAN-001-FULL__test__probabilities.npy` 重算 14 类 AP/AUROC，与历史 CSV 最大绝对差 = 0。

#### 仍存在的边界与风险

1. **Test 已经被人看过**：v1 的 Final Decision 是在 24 条件消融完成后写的，作者已经看过 Validation 与 Test 数值。Line B 不是盲法确认性实验，只能定位为探索性/事后分析。
2. **阈值校准的样本内乐观偏差**：逐标签阈值直接最大化 Validation F1，Validation 上的 F1 估计偏乐观；Test 才是独立评价。Line B 的次要 F1 分析必须明确这一点。
3. **Validation 同时用于 early stopping、方法比较与阈值拟合**：没有嵌套校准/独立校准集。Line B 不改变这一历史事实，但不应把 Validation-F1 的结果当作无偏泛化估计。
4. **UVSPEC15 标签冗余**：15 个标签有父子关系和强共现，Macro 平均把高度相关的标签等权相加；Micro-F1 会被高频标签（`has_conjugation`、`cyclic_conjugated_system`）主导。不是数据泄漏，但会影响指标解释。
5. **共享 14 标签的训练行差 2 条**：IR/Raman 的 train 比 UV 多两行。由于 val/test 相同，评估层面无影响；但若未来做融合训练，必须使用 `multimodal_complete_scaffold_split.npz` 的 103853，不能把 103855 与 103853 混用。
6. **UV shared 没有 BN 重估 checkpoint**：无法在 Shared14 上做“三模态 × 3 seeds × Recalibrated BN”的平衡敏感性分析；要做必须新生成 3 份 UV shared BN 重估 checkpoint（CPU 后处理），本轮不做。
7. **UV shared WBCE 只有 Seed 42**：不能把 UV 的 WBCE 结果当作三 Seed 对照；Shared14 的损失家族敏感性分析在 UV 侧不平衡。
8. **FP16 推理的非确定性**：Raman v1 已记录跨进程 FP16 的微小差异（主要落在极稀疏标签 `nitro` 上）。Line B 的概率导出应固定设备、dtype、batch size，并保存 SHA256；跨进程复核时优先用 FP32 或记录容差。

---

## 5. 可复用 vs 必须新训练

### 5.1 Shared14（三模态预测同一 14 类）

| 模态 | 复用 checkpoint | 数量 | 损失 | BN | 说明 |
|---|---|---:|---|---|---|
| IR | `runs/EXP-IR-001-FULL`、`-SEED123`、`-SEED2026` | 3 | BCE | Original | 冻结的 BCE 原始模型 |
| Raman | `runs/EXP-RAMAN-001-FULL`、`-SEED123`、`-SEED2026` | 3 | BCE | Original | 冻结的 BCE 原始模型 |
| UV-Vis | `runs/EXP-UVVIS-001-FULL`、`-SEED123`、`-SEED2026` | 3 | BCE | Original | 冻结的 shared 14 标签模型 |

**不需要新训练**。需要新生成的是：

- **B1**：统一的 Shared14 Validation 概率、`row_index`、`y_true` 导出（9 个模型）；
- **B3**：UVSPEC15 Validation 导出与全部 18 个模型的逐标签 AP/AUROC/F1/support 表、Validation-F1 阈值；
- **B4**：协议冻结后统一的 Shared14/UVSPEC15 Test 概率导出（每模型一次）；
- **B4**：错误互补性、标签共现、结构子群、scaffold Bootstrap。

### 5.2 UVSPEC15（三模态预测同一 15 类）

| 模态 | checkpoint 来源 | 数量 | 损失 | BN | 状态 |
|---|---|---:|---|---|---|
| UV-Vis | `runs/EXP-UVVIS-003-UVSPEC15-FULL`、`-SEED123`、`-SEED2026` | 3 | BCE | Original | **复用** |
| IR | 新建 `runs/LINE-B-UVSPEC15-IR-SEED42/123/2026` | 3 | BCE | Original | **必须新训练** |
| Raman | 新建 `runs/LINE-B-UVSPEC15-RAMAN-SEED42/123/2026` | 3 | BCE | Original | **必须新训练** |

**为什么不能复用 IR/Raman 的 Shared14 checkpoint**：输出头是 14 维，标签集是官能团，不是 UVSPEC15；换头会破坏冻结权重与标签语义，不能称为“复用”。

**为什么不能复用 UVSPEC15 的 Recalibrated checkpoint 作为主比较**：UVSPEC15 v1 主方案是 Recalibrated BN，而新 IR/Raman 首轮不重估 BN；若 UV 用 Recalibrated、IR/Raman 用 Original，就把 BN 后处理和模态差异混在一起。主比较统一用 Original；Recalibrated 只作为后续平衡敏感性分析。

### 5.3 新训练的超参数（与冻结 UVSPEC15 训练一致）

| 项 | 值 |
|---|---|
| 模型 | `training.models.spectrum_cnn1d.SpectrumStructureModel(num_labels=15, embedding_dim=256)` |
| 输入 | IR `ir_float32.npy` 或 Raman `raman_float32.npy`，`[N, 3501]` |
| 标签 | `uvspec15_labels.npy`，`[N, 15]` |
| 划分 | `uvvis_scaffold_split_valid.npz`，**train 103853 / val 12981 / test 12981** |
| 归一化 | `normalization="max"` |
| 损失 | `BCEWithLogitsLoss()`，`use_pos_weight=False` |
| 优化器 | AdamW，lr 1e-3，weight decay 1e-4 |
| batch size | 512 |
| num_workers | 4 |
| 最大 epoch | 50 |
| early stopping | patience 8 |
| 最优轮 | Validation mAP（阈值无关） |
| AMP | 开启（与历史一致） |
| seeds | 42 / 123 / 2026 |
| 输出头 | 15 维，顺序严格按 `uvspec15_label_names.json` |
| Test 使用 | **训练脚本不得计算 Test**；Test 由独立导出脚本在协议冻结后执行一次 |

### 5.4 为什么主比较用“固定基线”而不是各模态 v1 主方案

三份 Final Decision 的“主方案”分别是：

| 模态 | v1 主方案 | 损失 | BN | 阈值 |
|---|---|---|---|---|
| IR | BCE + Original + Val-F1 | BCE | Original | Validation-F1 |
| Raman | WBCE + Recalibrated + Val-F1 | WBCE | Recalibrated | Validation-F1 |
| UVSPEC15 | BCE + Recalibrated + Val-F1 | BCE | Recalibrated | Validation-F1 |

如果直接用这三套主方案做跨模态比较，差异同时包含：

- 损失函数（BCE vs WBCE）；
- BN 后处理（Original vs Recalibrated）；
- 阈值策略（0.5 vs Val-F1）；
- 模态本身。

这三者无法分离。Line B 的主比较必须统一到三份文档都保留的 **固定基线 BCE + Original + 0.5**；它也是三份文档明确指定的“融合实验单模态下限对照”。各模态主方案只作为“该模态自身最佳配置”的参考，不进入跨模态主结论。

---

## 6. 缺失资产与需新生成清单

| 优先级 | 资产 | 现状 | 处理阶段 | 说明 |
|---|---|---|---|---|
| P0 | 样本级概率/预测导出（Shared14 9 个模型） | 缺失 | B1（validation）/ B4（test） | 需要 `row_index`、`y_true`、`prob`、split、checkpoint hash |
| P0 | 样本级概率/预测导出（UVSPEC15 UV 3 个模型） | 缺失 | B3（validation）/ B4（test） | 同上 |
| P0 | 新 IR/Raman UVSPEC15 checkpoint（6 份） | 缺失 | B2 | BCE、Original、3 seeds |
| P0 | 新 IR/Raman UVSPEC15 训练脚本 | 缺失 | B1（脚本）/ B2（训练） | 参数化、禁止写 Test、禁止覆盖 |
| P0 | 统一预测导出脚本 | 缺失 | B1 | 支持 IR/Raman/UV、14/15 标签、Original/Recalibrated |
| P0 | 统一逐标签评估脚本 | 缺失 | B1/B3 | AP/AUROC/F1/support、Macro/Micro |
| P0 | Validation-only 阈值拟合脚本 | 缺失 | B3 | 统一代码路径，禁止复用历史阈值文件 |
| P1 | scaffold 聚类 Bootstrap 脚本 | 缺失 | B4 | 配对、按 scaffold 重采样、固定 RNG |
| P1 | 错误互补性脚本 | 缺失 | B4 | 相同样本、逐标签/逐样本、McNemar、真实 Oracle |
| P1 | 标签共现/结构子群脚本 | 缺失 | B4 | 训练集共现；测试集仅描述 |
| P1 | Line B 最终报告与图表 | 缺失 | B5 | 只读汇总 |
| P1 | Line B 资产 manifest（哈希链） | 缺失 | B1/B3/B4 | checkpoint、split、标签、预测、阈值、代码 |
| P2 | UV shared Seed 123/2026 阈值 | 缺失 | B3 | 仅在 Val-F1 次要分析需要；Validation-only |
| P2 | UV shared BN 重估 checkpoint（3 份） | 缺失 | 延后 | 仅在平衡 BN 敏感性分析需要 |
| P2 | 文档 errata：`uvspec15_label_guide.md` 3,889 → 3,891 | 本轮已修订 | B0 修订 | 只改文档，不改数据 |
| P2 | 典型分子渲染图 | 缺失 | B4/B5 | 复用 `scripts/qm9s_structure_tools.py render` |

---

## 7. 主要风险、歧义与需要人工决策的问题

### R1. 主比较配置：固定基线 vs 各模态主方案

- **歧义**：三份 Final Decision 各有自己的“主方案”，但三者损失/BN/阈值不同。
- **建议**：Line B 主比较统一用 BCE + Original + 0.5；各模态主方案只作为参考，不进入跨模态主结论。
- **需要决定**：是否批准这一“可比起见牺牲单模态最优配置”的口径。

### R2. UVSPEC15 新 IR/Raman 的训练划分

- **歧义**：IR/Raman 的官方 split 是 103855，UVSPEC15 是 103853。新 IR/Raman 应使用哪一个？
- **建议**：使用 **UVSPEC15 的 103853**，让三模态的训练行完全一致；代价是丢弃 2 条 IR/Raman 有效但 UV 全零的样本。评估不受影响（val/test 相同）。
- **需要决定**：是否批准 103853 作为 UVSPEC15 三模态统一训练集。

### R3. BN 状态

- **歧义**：UVSPEC15 v1 主方案用 Recalibrated；IR/Raman 首轮不重估。是否要为了“跟随主方案”而给新 IR/Raman 也做 BN 重估？
- **建议**：首轮全部 Original；BN 重估作为延后的平衡敏感性分析，必须三模态同时做，不能只给 UV 或只给 IR/Raman。
- **需要决定**：是否接受首轮不做 BN 重估；若要做，是否愿意承担 9 份额外 BN 后处理（UV shared 3 + UVSPEC15 IR/Raman 6）。

### R4. 阈值政策

- **歧义**：0.5 是“零后处理”但依赖概率校准；Val-F1 是“统一拟合”但有 Validation 乐观偏差。
- **建议**：0.5 为主，Val-F1 为次；两者都报告，且所有模态使用同一代码路径重新拟合，不混用历史阈值。
- **需要决定**：是否接受“0.5 主、Val-F1 次”的双轨政策；是否允许对 UV shared Seed 123/2026 新拟合阈值。

### R5. 新训练范围：只 BCE 三 Seed，还是也补 WBCE

- **歧义**：v1 的 Raman 主方案是 WBCE；UVSPEC15 的 WBCE 对照也存在。Line B 是否需要在 UVSPEC15 上补 WBCE？
- **建议**：首轮只 BCE（6 次训练）；WBCE 作为延后敏感性分析（再加 6 次训练）。理由是主比较要求同一损失家族，BCE 已经足够回答“模态差异”。
- **需要决定**：是否批准首轮只 BCE。

### R6. Test 的解释强度

- **歧义**：Test 已在 v1 被使用；Line B 又要在同一 Test 上做更细的分析。
- **建议**：明确标注为 **exploratory / post-hoc**；不做“盲法预注册确认性”表述；确认性结论留给外部域（NIST/API-Raman）或新的锁定评估集。
- **需要决定**：是否接受这一解释边界；是否需要在 Line B 报告中加入“Test 已开启”的显著声明。

### R7. 低支持度标签的处理

- **歧义**：Shared14 的 `nitro`（test 15）与 `c_f_bond`（test 144）；UVSPEC15 的 `pi_conjugated_imine`（test 148）等。是否从 Macro 中排除？
- **建议**：**不排除**；完整报告 14/15 类 Macro，同时预设备注“support-weighted Macro”或“排除 very_low 的 Macro”作为敏感性，且必须在看 Test 前写死规则。不得因为某个标签拉低某模态就事后删除。
- **需要决定**：是否接受“全标签主表 + 预定义敏感性表”。

### R8. 标签冗余与 Macro 解释

- **歧义**：UVSPEC15 的 15 个标签高度相关；Macro AP 对每个标签等权，可能被 5 个芳香/共轭标签主导。
- **建议**：同时报告 Macro 与 Micro；报告标签共现/层级；不构造“非冗余子集”作为主结论（避免事后挑选）。可以在补充材料中给出按标签簇分组的描述性结果。
- **需要决定**：是否需要预先指定“标签簇”分组（如 conjugation / aromatic / carbonyl / nitrogen / donor-acceptor）作为二级汇总。

### R9. 错误互补性的主定义

- **歧义**：“相同样本的错误互补性”可以指逐标签错误集合，也可以指逐样本“是否有任一标签错误”。
- **建议**：逐标签为主（与 AP/F1 对齐），逐样本为辅（hard-sample 分析）；两者都预定义。
- **需要决定**：是否接受这一主次。

### R10. 结构子群的分箱与“UV 是噪声”的判定

- **歧义**：子群分箱、最小支持度、“UV 是噪声”需要多大赛果才算；以及 UV 的边际劣势是否等于“无增量信息”。
- **建议**：分箱在看 Test 前写死；最小支持度 ≥30 正例且 ≥100 样本；分别对 UV−IR、UV−Raman 两个预定义对比报告 ΔAP 的 95% Bootstrap CI；只有某个对比的 CI 上界 < 0 且 `abs(ΔAP) ≥ 0.02` 才称“该对比下有证据的 UV 边际劣势”。不得用 Test 事后选择对照模态；也不得仅凭边际 AP 低就断言 UV 无增量信息，增量信息必须另看条件互补性与真实 Oracle 正确性。
- **需要决定**：是否接受 0.02 的效应量门槛、30/100 的最小支持度，以及“边际劣势 ≠ 无增量信息”的表述边界。

### R11. 典型分子挑选

- **歧义**：典型分子容易变成 cherry-picking。
- **建议**：预定义规则：每个子群选“正确且置信度中位数”1 个 + “最高置信度错误”1 个；固定 Seed 42；渲染图只作定性说明，不参与统计。
- **需要决定**：是否接受这一选取规则。

### R12. 文档与计数 errata

- **歧义**：`uvspec15_label_guide.md` 的 3,889 vs 矩阵 3,891；Final Decision 的字母序标签列表容易被误读为列顺序。
- **建议**：发布一个只改文档的 errata：修正 3,889→3,891；在 Final Decision 的标签列举处加一句“列顺序以 `uvspec15_label_names.json` 为准”。不改任何数据、checkpoint、指标。
- **需要决定**：是否批准文档 errata（本轮不自动改）。

### R13. UV shared WBCE 只有 1 个 Seed

- **歧义**：Shared14 的损失家族敏感性分析在 UV 侧只有 Seed 42，无法三 Seed 配对。
- **建议**：不在 Line B 主分析中做 Shared14 的 WBCE 三 Seed 比较；如要做，明确标注 UV 仅 1 Seed，不能与 IR/Raman 的三 Seed 均值直接配对。
- **需要决定**：是否把 WBCE 敏感性分析降级为“描述性补充”。

### R14. 概率导出的数值口径

- **歧义**：历史训练用 AMP/FP16；新导出用 FP16 还是 FP32？两者可能造成极稀疏标签的微小差异。
- **建议**：导出统一用 **FP32 推理**（`model.eval()` + 关闭 autocast），并保存 dtype/设备/batch size；同时记录与历史 FP16 指标的差异（Raman 已有容差经验）。若必须复现历史 FP16，则固定单进程、单设备并保存 SHA256。
- **需要决定**：是否接受 FP32 作为 Line B 的统一推理口径（这会与部分历史 FP16 数值有 1e-3 量级差异，但不影响排序结论）。

### R15. 是否允许 Seed 平均概率

- **歧义**：错误互补性/结构子群分析用单 Seed 还是 3 Seed 平均概率？
- **建议**：主表按 Seed 报告 Mean±SD；互补性与 Bootstrap 用“3 Seed 配对”的方式（每个 Bootstrap 样本同时使用 3 个 Seed，取 3 Seed 平均的 Δ），不构造新的 ensemble 模型。定性典型分子固定 Seed 42。
- **需要决定**：是否接受这一“配对 3 Seed”的统计口径。

---

## 8. B0 验收状态

| 检查项 | 状态 |
|---|---|
| 审计脚本可重复运行并生成 JSON | ✅ |
| 40 份生产 checkpoint 的 head/BN/label 核验 | ✅ 全部通过 |
| 三模态 val/test 逐元素一致性 | ✅ |
| 103855 vs 103853 差异定位到 13998/14000 | ✅ |
| UV 全零谱排除核查 | ✅ |
| UVSPEC15 列顺序与源矩阵/checkpoint 一致性 | ✅ |
| 22 份逐标签 Test support 与标签矩阵一致性 | ✅ 22/22 |
| Raman 概率数组复算 AP/AUROC | ✅ 最大绝对差 0 |
| 14/15 类标签 train 正例覆盖 | ✅ 无 `NO_TRAIN_POSITIVES` |
| scaffold 零重叠 | ✅ |
| 历史实验数据/checkpoint/指标未被修改 | ✅（`runs/`、`processed/` 无修改；仅按要求对 `uvspec15_label_guide.md` 与 `UVSPEC15_v1_final_decision.md` 做文档勘误） |
| 未执行新的训练/推理；未用 GPU | ✅（B0 只读取历史 Test CSV 与既有 Raman Test 概率用于一致性审计，无新 forward pass） |
| Line B 协议草案 | 见 `docs/experiments/line_b_protocol_draft.md` |
| 人工审核与配置批准 | ⏳ 等待 |

---

## 9. 一句话总结

**数据与标签是干净的，模型资产是完整的，缺的是“同一口径的样本级概率导出”和“UVSPEC15 的 IR/Raman 新基线”；Line B 的主比较必须统一到 BCE + Original + 0.5，以 AP/AUROC 为主证据，并把 Test 结果明确定位为已开启 Test 上的探索性分析。**
