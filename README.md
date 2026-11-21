# 最后更新时间：2026-11-21 22:09 (UTC+8)

# SpectraDB
## 项目简介
SpectraDB是一个面向有机小分子多模态光谱分析的数据库与模型建设项目。
项目目标是建立统一的多模态光谱数据库，并实现从光谱推断分子结构信息：以单模态1D-CNN模型（IR、Raman、UV-Vis）为基础，按各模态的化学证据分工进行chemistry-aware自适应融合，并逐步实现官能团/结构特征识别、完整分子结构候选推断、分子结构可视化及模型可解释性分析。

## 项目目标
### 第一阶段（数据体系与单模态模型）
包括：
* 建立QM9S多模态计算光谱数据
* 建立分子结构库
* 建立官能团、原子、键、环、共轭体系等结构标注
* 建立scaffold split
* 训练IR 1D-CNN
* 训练Raman 1D-CNN
* 训练UV-Vis 1D-CNN
* 比较不同光谱模态对结构信息的表征能力
### 第二阶段（多模态融合与完整结构推断）
包括：
* IR、Raman、UV-Vis特征融合
* 综合官能团与关键结构特征
* 完整分子结构候选推断
* Top-k候选排序
* 二维分子结构可视化
* Woodward等化学规则辅助验证和候选重排序
### 第三阶段（实验域、开放集与可解释性）
包括：
* NIST实验数据微调与验证
* API-Raman实验数据微调与验证
* calculated spectra → experimental spectra迁移
* 未知样品识别
* 开放集识别
* 置信度与拒识
* XAI
* 光谱区域与结构片段对应
* 最终结果界面与自然语言解释

## 数据源
* QM9S：IR、Raman、UV-Vis主要计算训练数据（约13万分子）
* GDB-9-Ex：备用外部域/方法迁移数据集（约9.7万分子，激发态能量与振子强度），当前不参与训练
* NIST：实验域IR、UV-Vis数据（微调与测试）
* API-Raman：实验Raman数据（约3500个化合物）

## 数据结构
### 每个分子对应唯一记录，主要包含：
* row_index（与光谱矩阵、结构标注对齐）
* QM9S编号
* SMILES（canonical）
* 分子式
* 分子量
* RDKit Descriptor
### 以及：
* 红外光谱（500–4000 cm⁻¹，3501点）
* 拉曼光谱（500–4000 cm⁻¹，3501点）
* 紫外-可见吸收光谱（1–15 eV，701点）
### 结构标注：
* 官能团多标签（14类）
* 原子/键/环/共轭体系标注
* UV专属化学语义标注（58二值 + 73数值特征）
* UVSPEC15（15类UV专属结构语义标签；从UV标注中选定，与Shared14为两个独立任务）

## 项目目录
* SpectraDB/
---------------------------
* README.md
* CHANGELOG.md
* TODO.md
* AGENTS.md
* scripts/
* training/
* runs/
* master/
* raw/
* processed/
* metadata/
* cache/
* logs/
* archive/
## 当前开发计划
1. 建立QM9S多模态数据集（已完成）
2. 生成官能团标签与结构库（已完成）
3. IR 1D-CNN单模态基线（已完成）
4. Raman 1D-CNN单模态基线（已完成）
5. UV-Vis 1D-CNN单模态基线（已完成）
6. UV专属共轭/发色团化学语义标签（已完成）
7. UV-specific单模态基线与shared/specific对照（已完成）
8. IR v1 完整方法学消融（6 模型 / 24 条件）与 Final Decision（已完成，2026-11-15）
9. UVSPEC15 v1 完整方法学消融（6 模型 / 24 条件）与 Final Decision（已完成，2026-11-15）
10. Raman 同构 BN 审计与 24 条件消融（已完成，2026-11-16）
11. 三模态信息画像与模态对比（Line B，B0–B5 已完成并归档）
12. Chemistry-aware多模态融合（Line C，下一阶段，尚未开始）
13. 完整结构候选与Top-k
14. 实验域数据（NIST、API-Raman）验证与微调
15. 开放集识别与可解释性分析
## 开发环境
### 已有：
* Python
* Conda
* RDKit
* Pandas
* NumPy
* PyTorch
* Scikit-learn
* Matplotlib
### 后续计划加入：
* HuggingFace
* Lightning
## 项目状态

> 最后同步 2026-11-21，Line B（B0–B5）已完成并通过独立验收，归档于 `archive/line-b-2026-11-20`；Line C 尚未开始。

### 单模态基线（IR/Raman 2026-08 冻结；UV-Vis corrected 2026-11-18；阈值 0.5，三随机种子 Mean）

| 模型 | 标签集 | Test Micro-F1 | Test Macro-AUROC |
|---|---|---|---|
| IR-v1 | 14 类官能团 | 0.9406 | 0.9954 |
| Raman-v1 | 14 类官能团 | 0.9151 | 0.9911 |
| UV-Vis-v1（corrected） | 14 类官能团（shared） | 0.5158 | 0.8526 |

> UV-Vis-v1 原三 Run 均值包含 invalid Seed2026（实际 SEED=42），旧版快照已移入 `archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/superseded_uvvis_v1_final/`；上表为 corrected 三真实 Seed 结果，见 `runs/UVVIS-v1-FINAL/`。

### 已冻结的方法学消融（各 6 模型 × 24 条件）

| 版本 | 主方案 | Validation Micro-F1 | Test Micro-F1 |
|---|---|---|---|
| **IR v1** | BCE + Original BN + Validation-F1 | 0.9505 ± 0.0025 | 0.9364 ± 0.0054 |
| **UVSPEC15 v1** | BCE + Recalibrated BN + Validation-F1 | 0.6283 ± 0.0059 | 0.6568 ± 0.0138 |
| **Raman v1** | WBCE + Recalibrated BN + Validation-F1 | 0.922034 ± 0.002103 | 0.914113 ± 0.006869 |

- IR v1 备选：BCE + Recalibrated BN + Validation-F1；固定基线：BCE + Original BN + 0.5（Test Micro-F1 0.9405 ± 0.0048）。记录见 `docs/experiments/IR_v1_final_decision.md`。
- UVSPEC15 v1 备选：Recalibrated + 0.5（Test Micro-F1 0.6580 ± 0.0154）与 WBCE + Recalibrated + Validation-F1；**0.6580 属备选配置，不是主方案指标**。记录见 `docs/experiments/UVSPEC15_v1_final_decision.md`。
- Raman v1 备选：WBCE + Original BN + Validation-F1（Validation 0.921640 ± 0.002583）；BCE 家族最佳：BCE + Recalibrated BN + Validation-F1（0.920907 ± 0.003517）；固定基线：BCE + Original BN + 0.5（Validation 0.910148 ± 0.002418 / Test 0.915080 ± 0.001382）。记录见 `docs/experiments/Raman_v1_final_decision.md`。
- 三者均含 48 条 split-level 记录；逐标签 IR 672 行、UVSPEC15 720 行、Raman 672 行。Raman 的 Test Micro-F1 在八组配置中排第 6（低于 Validation 第 1），属已记录现象，**未据此改选**。

### Line B 三模态信息画像（已完成）

Line B 的 B0–B5 已完成。B4.1 导出 18 个冻结模型的 Test 预测；B4.2 完成 Shared14/UVSPEC15 两任务的三模态评估；B4.3 完成 2,000 次 scaffold-clustered paired bootstrap；B4.4 完成错误互补性、Positive OR/AND、真实 Oracle、条件互补与结构子群分析；B5 汇总为正式报告并通过独立验证。

- 模型配置：2 Task × IR/Raman/UV-Vis × 3 seeds = 18 个冻结模型；Test n=12,981。
- 评估口径：主 F1 政策固定 0.5，次政策为冻结 Validation-F1；AP/AUROC 阈值无关；Bootstrap 描述 scaffold 重采样不确定性，不是训练随机性。
- Test 三 seed 均值（exploratory/post-hoc）：

| Task | Modality | Micro-F1 (0.5) | Macro-AP | Macro-AUROC |
|---|---|---|---|---|
| Shared14 | IR | 0.9405 | 0.9695 | 0.9955 |
| Shared14 | Raman | 0.9151 | 0.9551 | 0.9911 |
| Shared14 | UV-Vis | 0.5159 | 0.5125 | 0.8524 |
| UVSPEC15 | IR | 0.9512 | 0.9709 | 0.9977 |
| UVSPEC15 | Raman | 0.9538 | 0.9760 | 0.9982 |
| UVSPEC15 | UV-Vis | 0.6108 | 0.5380 | 0.9252 |

主要结论：

- IR/Raman 在当前两个任务上均优于 UV-Vis。
- UV-Vis 在 UVSPEC15 上未表现出预期的单模态优势；这不表示 UV 没有结构信息或潜在互补性。
- 不同模态的错误集合存在差异，但 Positive OR 未超过 best_single；True Oracle 是理论诊断上界，不是实际融合模型的性能。
- 最佳模态随任务与结构子群变化，因此仍有研究自适应融合的必要性；这还不是对 chemistry-aware routing 的验证。

正式判定：H1 **PARTIAL SUPPORT**；H2 **REJECTED**；H3 **PARTIAL SUPPORT**；H4 **DESCRIPTIVELY SUPPORTED**。

> Line B 的 Test 结果属于 exploratory/post-hoc，不是独立盲测，也不能作为 confirmatory evidence；结论仅限 QM9S 计算域和当前已评估任务，不能外推到 NIST、SDBS 或 API-Raman 实验域。

归档入口：归档分支 `archive/line-b-2026-11-20`，commit `ab1e85dd0274e803bae074fc19acce47f6feb142`。正式报告位于归档分支：[line_b_final_report.md](https://github.com/Michael06835/SpectraDB/blob/archive/line-b-2026-11-20/runs/LINE-B-FINAL/line_b_final_report.md)。该路径不在 `main`，阅读报告请使用归档分支链接。

### 下一步与尚未开展

- **Line C（下一阶段，尚未开始）**：naive concatenation / fixed-weight 三模态对照；IR+Raman 与 IR+Raman+UV 的对照；generic attention/gating；chemistry-aware routing；UV-specific chemical supervision 与关键消融；gate 可解释性与 scaffold-aware 不确定性。所有融合模型必须与 best_single 比较；negative transfer 仅作为待实验检验的现象，不是已确认结论。
- **Line D–I（后续计划）**：完整结构 Top-k、Woodward–Fieser 等化学规则 reranking、实验域迁移、开放集拒识、XAI 与自然语言解释；均尚未开展。
- **待落实**：12 份 IR checkpoint（约 140 MB）与 12 份 Raman checkpoint（约 140 MB）的仓库外独立备份（Git 仅保存轻量 JSON/CSV；Raman 备份已由项目决定跳过）。
