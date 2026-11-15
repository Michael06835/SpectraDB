# IR v1 实验策略：复用 UVSPEC15 v1 已验证流程

> **状态**：**已冻结归档**（2026-11-15 完成 24 条件全矩阵与最终决策）
> **最终决策**：见 [`IR_v1_final_decision.md`](IR_v1_final_decision.md)（主方案 BCE + Original BN + Validation-F1；备选 BCE + Recalibrated BN + Validation-F1；固定基线 BCE + Original BN + 0.5）
> **性质**：方法选择记录，非事先注册的实验方案
> **取代关系**：早期 IR 专属方案（Validation 内部 5×5 交叉拟合、P1–P5 定量资格判定、L1/L2′/L3 多层自动决策、协议哈希闸门）**已整体废止**，其记录文件已从仓库移除；本文件是 IR 唯一的现行策略来源
> **范围**：QM9S 计算域 IR → 14 类官能团多标签。**不代表**多模态融合或完整结构推断已完成。

## 1. 决策

IR 不再单独设计一套 BN 决策体系，**全面复用 UVSPEC15 v1 已实现并验证的四个程序**：

| 阶段 | 脚本（UVSPEC15 原件） | IR 对应 |
|---|---|---|
| Full BN 诊断 | `training/diagnose_uvspec15_batchnorm_multiseed.py` | `training/diagnose_ir_batchnorm_full.py` |
| BN 重估 checkpoint 保存 | `training/save_uvspec15_bn_checkpoint.py` | 待实现（阶段 1.5） |
| BN 后阈值校准 | `training/calibrate_uvspec15_bn_thresholds.py` | 待实现（阶段 2b） |
| 消融汇总 | `training/summarize_uvspec15_ablation.py` | 待实现（阶段 3） |

方法本身完全沿用：Full BN Diagnostic、Training-only BN Recalibration（独立副本、无参数更新）、独立 checkpoint 保存、Validation 逐标签 F1 阈值校准、只读消融汇总。**不做 Quick Diagnostic，不重新训练，不修改原始 checkpoint。**

## 2. 数据与评价协议

- 光谱：`processed/qm9s/prepared/ir_float32.npy`，`(129817, 3501)`，500–4000 cm⁻¹，逐谱 max 归一化。
- 标签：`functional_group_labels.npy`，`(129817, 14)`；顺序固定于 `functional_group_label_names.json`。
- 划分：`ir_scaffold_split_valid.npz`，**Train 103855 / Validation 12981 / Test 12981**（IR 自身的 scaffold split，包含 `row_index 13998/14000`；这两条只有 UV 谱无效，IR 谱有效）。
- 模型：`SpectrumStructureModel`，1D-CNN，embedding 256，14 logits，12 层 BatchNorm1d。
- 训练配置（既有，不得改动）：batch 512、num_workers 4、AdamW(1e-3, 1e-4)、max 50 epochs、patience 8、AMP、按 Validation mAP 选轮。
- 评价：Micro-F1、Macro-F1、mAP、Macro-AUROC，另记录 Precision/Recall 与逐标签结果。mAP / AUROC 阈值无关。

## 3. IR 相对 UVSPEC15 的必要差异

| 项 | UVSPEC15 | IR |
|---|---|---|
| 光谱长度 | 701 | **3501** |
| 标签数 | 15 | **14** |
| 划分规模 | 103853 / 12981 / 12981 | **103855** / 12981 / 12981 |
| 训练批次数 | 203 | 203（数值巧合，来源不同） |
| 实验映射 | `003`(BCE) / `004`(WBCE) × 3 seeds | `bce` × {42,123,2026}、`wbce` × {42,123,2026} |
| 输出根 | `runs/UVSPEC15-BN-DIAGNOSTICS/` | `runs/IR-BN-DIAGNOSTICS/` |
| Quick Diagnostic | 脚本支持 | **不实现** |

其余（BN 重估算子、deepcopy 隔离、参数不变断言、基线复现容差 0.002、防覆盖、独立落盘）与 UVSPEC15 完全一致。

## 4. 实验矩阵（24 组）

6 个独立训练模型 × 2 BN 状态 × 2 阈值策略 = 24 个消融评估条件（Validation 24 + Test 24 = 48 条 split-level 记录；逐标签 14 × 24 × 2 = 672 行）。

| 模型 | 损失 | Seed | 状态 |
|---|---|---|---|
| EXP-IR-001-FULL | BCE | 42 | canonical |
| EXP-IR-001-SEED123 | BCE | 123 | 复现用 |
| EXP-IR-001-SEED2026 | BCE | 2026 | 复现用 |
| EXP-IR-002-FULL | WBCE（sqrt(pos_weight), cap 20） | 42 | 损失函数消融 |
| EXP-IR-002-SEED123 | WBCE | 123 | 损失函数消融（2026-11-15 补齐） |
| EXP-IR-002-SEED2026 | WBCE | 2026 | 损失函数消融（2026-11-15 补齐） |

计划产出：6 次 Full BN Diagnostic、6 个独立保存的 BN 重估 checkpoint、12 次 Threshold Calibration（6 Original + 6 Recalibrated）。

**执行纪律（内存约束）**：本机曾接近系统内存提交上限，**所有 BN 诊断必须严格串行**，同一时刻只允许一个模型的前向任务在跑；禁止并行启动多个模型。其余阶段同样逐个执行。

**统计口径**：BCE 与 WBCE 各有三个 Seed，均报告 Mean ± SD 与同 Seed 配对变化。12 次阈值校准是独立执行与记录的任务，但其结果构成 24 条件矩阵中的 Validation-F1 分支；**不得**把"12 次校准 + 24 个条件"表述为 36 个互不重复的消融条件。

## 5. 方法选择原则（沿用 UVSPEC15）

- Training 用于 BN 统计量重估；Validation 用于方法比较与阈值优化；Test 仅用于性能评价、记录与一致性核验。
- 不依据 Test 选择 Seed、BN 方法或阈值策略；不根据最高 Test 分数挑选 Seed。
- **不预设 IR 必须采用 BN 重估**；是否采用由 Validation 结果与三 Seed 稳定性决定。
- 默认阈值 0.5 与 Validation-F1 阈值两套结果都保留。

**明确不采用**的 IR 专属附加设计（已废弃）：

- Validation 内部 5×5 交叉拟合；
- P1–P5 定量资格判定；
- L1 / L2 / L3 多层自动决策体系。

以上均不属于本次 IR 实验的必要步骤，相关执行规则与文件已一并废止，不得再作为判据或流程依据。

## 6. 当前进度与已知缺口

| 项 | 状态 |
|---|---|
| IR 六模型训练与冻结 | 完成（BCE ×3 已有；WBCE 123/2026 于 2026-11-15 补齐），六份 checkpoint 全部通过一致性核验 |
| Full BN Diagnostic | **完成 6/6**（2026-11-15，严格串行执行） |
| BN 重估 checkpoint | **完成 6/6**（BCE/42、BCE/123 由本会话保存；其余四个由用户手动串行保存并验收）。六份均通过仅-BN-buffer 变化、参数不变、回读一致与源文件保护检查 |
| Original 阈值校准 | **完成 6/6**（2026-11-15，新目录 `runs/IR-THRESHOLD-CALIBRATION/`；历史 BCE 42 产物保留未动） |
| Recalibrated 阈值校准 | **完成 6/6**（`runs/IR-BN-THRESHOLD-CALIBRATION/`，六组各 6 个文件） |
| 24 组条件 | **24/24 齐备**（6 组 Original+0.5、6 组 Original+Val-F1、6 组 Recalibrated+0.5、6 组 Recalibrated+Val-F1）。BCE 42 的历史校准（1024/0 推理）是同一 "Original+Val-F1" 条件的另一份记录，**不额外计数** |
| 消融汇总 | **已完成并冻结**（`runs/IR-ABLATION-SUMMARY/`，11 个文件：24+24 条件、16 组 Mean±SD、84 条配对、672 行逐标签、48 条来源登记、6 份审计，全部 audit 为 True） |
| 最终配置决策 | **已冻结**（`IR_v1_final_decision.md`：主方案 BCE + Original BN + Val-F1；备选 BCE + Recalibrated BN + Val-F1；固定基线 BCE + Original BN + 0.5） |

**最终汇总要点（事实记录）**：Test 上 `BCE/Original/0.5` 的 Micro-F1 为 0.9405±0.0048，为 8 个配置中最高；Validation 上阈值校准带来 +0.0035（BCE）/+0.0071（WBCE；Original BN）的 Micro-F1 增益，但同配置在 Test 上为 −0.0042/−0.0011，即**校准收益未稳定转化为 Test 收益**。BN 重估的效应在 Test 上为 −0.0030（BCE）/ +0.0019（WBCE），量级与大头均小于阈值校准。以上均为描述性统计（n=3），不构成显著性结论；按现行策略，配置选择只依据 Validation 多指标比较，不由本文件代为决定。

**Original Checkpoint 阈值校准结果（6/6，2026-11-15）**

脚本：`training/calibrate_ir_original_thresholds.py`（UVSPEC15 v1 原始阈值校准的参数化移植）。推理配置统一为 batch 512 / workers 4 / `normalization="max"`；阈值只在 Validation 拟合，并在**计算任何 Test 指标之前**落盘。

Validation：

| 损失 | Seed | Micro-F1 @0.5 → @Val-F1 | Δ | Macro-F1 @0.5 → @Val-F1 | Δ |
|---|---|---|---|---|---|
| BCE | 42 | 0.948142 → 0.950761 | +0.002619 | 0.924013 → 0.931290 | +0.007278 |
| BCE | 123 | 0.948013 → 0.952822 | +0.004809 | 0.911815 → 0.927022 | +0.015207 |
| BCE | 2026 | 0.944912 → 0.947871 | +0.002959 | 0.913708 → 0.922230 | +0.008522 |
| WBCE | 42 | 0.932173 → 0.938893 | +0.006720 | 0.899630 → 0.921222 | +0.021593 |
| WBCE | 123 | 0.945307 → 0.951569 | +0.006262 | 0.918647 → 0.931002 | +0.012355 |
| WBCE | 2026 | 0.936113 → 0.944491 | +0.008378 | 0.902705 → 0.914169 | +0.011465 |

Test（仅评价，未参与拟合）：

| 损失 | Seed | Micro-F1 @0.5 → @Val-F1 | Δ | Macro-F1 @0.5 → @Val-F1 | Δ |
|---|---|---|---|---|---|
| BCE | 42 | 0.938534 → 0.933249 | −0.005285 | 0.918515 → 0.920272 | +0.001757 |
| BCE | 123 | 0.946032 → 0.942617 | −0.003415 | 0.924611 → 0.927334 | +0.002723 |
| BCE | 2026 | 0.937072 → 0.933309 | −0.003763 | 0.914784 → 0.912175 | −0.002610 |
| WBCE | 42 | 0.925071 → 0.926707 | +0.001636 | 0.893458 → 0.909269 | +0.015810 |
| WBCE | 123 | 0.941964 → 0.940410 | −0.001554 | 0.926990 → 0.930453 | +0.003464 |
| WBCE | 2026 | 0.931305 → 0.927954 | −0.003351 | 0.906739 → 0.885174 | −0.021565 |

读数要点（仅事实）：

1. Validation 上六个模型、14 个标签全部不下降（WBCE/2026 为 13/14 提升、1 个持平），这是**同集合内优化的必然结果，含内生乐观偏差，不构成泛化证据**。
2. Test 上 Micro-F1 六模型中 4 降 2 升，Macro-F1 4 升 2 降；逐标签 Test 改善数分别为 6/4/3/10/5/8（共 14）。
3. 阈值量级差异明显：BCE 阈值多在 0.02–0.96，WBCE 同一标签的阈值普遍更高（如 aldehyde 0.81/0.96/0.72 vs 0.43/0.44/0.24），符合 `pos_weight` 改变概率标定的预期。
4. BCE 42 的历史校准（1024/0 推理）与新校准（512/4）对照：默认阈值下 Test Micro-F1 0.938496 → 0.938534（差 3.8e-05）；最大阈值差出现在 `hydroxyl`（0.5063 → 0.4431），但其 Validation F1 差仅 **7.1e-07**，属 F1 平台上的等价点；`amine` 差 0.0090 对应 F1 +0.0072，属真实数值差异。历史产物保留在 `runs/EXP-IR-001-FULL/`，未被覆盖。

**BN 诊断实测记录（六模型，仅事实，不作结论）**

各 Seed 的配对变化（Recalibrated − Original，阈值 0.5）：

| 损失 | Seed | ΔVal Micro-F1 | ΔVal Macro-F1 | ΔVal mAP | ΔVal AUROC | ΔTest Micro-F1 | ΔTest Macro-F1 | ΔTest mAP | ΔTest AUROC |
|---|---|---|---|---|---|---|---|---|---|
| BCE | 42 | −0.006303 | +0.003629 | +0.000207 | −0.000121 | −0.006547 | −0.002818 | −0.000246 | −0.000202 |
| BCE | 123 | +0.000024 | −0.000980 | −0.001505 | +0.003300 | −0.000432 | +0.005804 | +0.001370 | −0.000169 |
| BCE | 2026 | −0.003397 | +0.001639 | +0.002156 | +0.000169 | −0.001887 | −0.007431 | −0.002138 | +0.000102 |
| WBCE | 42 | +0.002140 | +0.006514 | +0.002521 | +0.000293 | +0.002632 | +0.004806 | +0.001162 | +0.000200 |
| WBCE | 123 | +0.002143 | −0.006652 | −0.000755 | +0.003132 | +0.001556 | −0.004679 | −0.000631 | −0.000019 |
| WBCE | 2026 | +0.001569 | −0.000389 | +0.000446 | +0.000025 | +0.001530 | +0.001804 | +0.001239 | +0.000271 |

三 Seed Mean ± 样本 SD（同一个 Seed 的 before/after/delta）：

| 损失 | 划分 | ΔMicro-F1 | ΔMacro-F1 | ΔmAP | ΔMacro-AUROC |
|---|---|---|---|---|---|
| BCE | Validation | −0.003225 ± 0.003167 | +0.001429 ± 0.002312 | +0.000286 ± 0.001832 | +0.001116 ± 0.001897 |
| BCE | Test | −0.002955 ± 0.003194 | −0.001482 ± 0.006718 | −0.000338 ± 0.001756 | −0.000090 ± 0.000166 |
| WBCE | Validation | +0.001951 ± 0.000331 | −0.000176 ± 0.006586 | +0.000737 ± 0.001657 | +0.001150 ± 0.001721 |
| WBCE | Test | +0.001906 ± 0.000629 | +0.000644 ± 0.004848 | +0.000590 ± 0.001058 | +0.000150 ± 0.000151 |

**观察（描述性，不作采用结论）**

1. 变化量级普遍在 1e-3 以内（mAP / AUROC），说明 BN 重估对**排序能力**影响很小；0.5 下的 Micro-F1 变化主要来自操作点位移（BCE 的 Precision 下降、Recall 上升）。
2. BCE 三 Seed 的 ΔMicro-F1 方向不一致（−0.0063 / +0.0000 / −0.0034），WBCE 三 Seed 则同向为正（+0.0021 / +0.0021 / +0.0016）。
3. 六个模型、各 12 层的**最小 `running_var`** 在重估前后的比值区间分别为 0.919–1.048 / 0.936–1.094 / 0.851–1.090 / 0.782–1.051 / 0.941–1.111 / 0.891–1.426，无数量级位移；`num_batches_tracked` 由 best_epoch×203 变为 203（预期）。
4. **以上仅为诊断记录**。是否采用 BN 重估须待 24 条件矩阵齐备后按 Validation 判断，且不依据 Test。

**BN 重估 checkpoint 保存（完成 6/6）**

脚本 `training/save_ir_bn_checkpoint.py`：BN 重估算子直接 `import` 自已批准的 `training/diagnose_ir_batchnorm_full.py`（`reset_recompute_bn`），未重写算法；配置与前向流程与 Full BN Diagnostic 完全一致（batch 512 / workers 4 / AMP / `max` 归一化）。

| 模型 | 源 checkpoint SHA256（前 16） | 重估 checkpoint SHA256（前 16） | 大小 | 变化 BN buffer | 与诊断记录的最大 Validation 指标差 |
|---|---|---|---|---:|---:|---:|
| BCE / 42 | `d60cc7b159891728` | `22db5c67ec82b896` | 12,261,303 B | 36/36 | 2.25e-07 |
| BCE / 123 | `d3856f25c0d1cbb9` | `666e9eed5469ed5e` | 12,261,303 B | 36/36 | 3.33e-05 |

下表为本会话保存的 BCE/42 与 BCE/123 两份明细；其余四份（BCE/2026、WBCE/42、WBCE/123、WBCE/2026）由研究者于 2026-11-15 16:50–17:03 手动串行保存，output 大小分别为 12,261,303 / 12,261,623 / 12,261,623 / 12,261,623 B。

两份均通过：仅 36 个 BN buffer 变化、非 BN 张量 0 变化、12 层 `num_batches_tracked` 均等于诊断记录的 203、全部 buffer 有限、回读一致、源 checkpoint SHA256/mtime 未变。六份的 manifest 哈希链（source/saved SHA256）均已由 `summarize_ir_ablation.py` 全量核验通过。

**执行资源（六次 Full BN Diagnostic 串行实测）**

| 模型 | 墙钟 (s) | Python 峰值工作集 (MB) | GPU 峰值 (MB) | 运行期最低可用物理内存 (MB) |
|---|---:|---:|---:|---:|
| BCE / 42 | 84.6 | 446.7 | 1992 | 3.2 |
| BCE / 123 | 85.3 | 434.3 | 2026 | 4.1 |
| BCE / 2026 | 80.2 | 435.3 | 2020 | 3.2 |
| WBCE / 42 | 78.7 | 449.5 | 1998 | 2.4 |
| WBCE / 123 | 88.8 | 433.2 | 2039 | 3.8 |
| WBCE / 2026 | 83.3 | 427.2 | 2014 | 4.3 |

单次墙钟 78.7–88.8 s，峰值工作集 ≤ 450 MB，GPU ≤ 2039 MB；六次全部串行、每次结束后无 Python 进程残留。运行期物理内存最低采样 2.4–4.3 MB 系 Windows 文件缓存占用（非提交上限），每次启动前空闲虚拟内存约 27 GB。**后续阶段仍须保持串行。**

**BN checkpoint 保存阶段的资源事件（2026-11-15，待裁决）**

- 保存 BCE/42：墙钟 59.4 s，峰值工作集 443.6 MB，GPU 1842 MB；运行期"系统已提交内存"采样峰值 **98%**，结束后回到 58%。
- 保存 BCE/123：墙钟 58.9 s，峰值工作集 434.8 MB，GPU 2009 MB；提交内存采样峰值 **100%（65,378 MB / 限制 64,817 MB）**。产物本身完整有效。
- 依"提交内存接近上限即停止"的规则，**其余 4 个模型未启动**。
- 取证中的**指标矛盾**（需进一步确认）：`C:\pagefile.sys` 分配 32,768 MB、**PeakUsage 仅 10,052 MB**；空载 45 s 提交占用稳定在 59%；三处 WMI 宿主进程私有内存 ≤ 52 MB；Python 主进程工作集 ≤ 450 MB。若真实提交曾达 64 GB，页面文件峰值应显著更高，故 98%/100% 可能部分来自 `Win32_OperatingSystem.FreeVirtualMemory` 的瞬时读数特性，而非真实提交耗尽。**在解决该矛盾前不恢复保存任务。**

已知风险（执行时须注意）：

1. 旧的单 Seed 校准脚本已**废弃归档**：实现迁至 `archive/2026-11-15/calibrate_ir_thresholds.py`，原路径 `training/calibrate_ir_thresholds.py` 仅保留弃用入口（直接运行立即非零退出，不加载数据、不写文件）。归档版本以 `"w"` 模式写盘且无 seed 守卫，**不得重跑**，否则覆盖 Seed 42 既有校准产物；请使用 `calibrate_ir_original_thresholds.py` / `calibrate_ir_recalibrated_thresholds.py`。
2. 该脚本推理用 `batch_size=1024 / workers=0`，与训练 512/4 不同，造成 Test Micro-F1 有 2.1e-5 级差异；补做校准时应统一为 512/4。
3. `runs/IR-v1-FINAL/ir_ablation_summary.csv` 三行均为 Seed 42，不可当作三 Seed 消融。

## 7. 结果布局与复现入口

- 诊断：`runs/IR-BN-DIAGNOSTICS/<run_name>/batchnorm_full_diagnostic.json`
- 重估 checkpoint：`runs/IR-BN-RECALIBRATED/<run_name>/`（待建）
- 汇总：`runs/IR-ABLATION-SUMMARY/`（待建）

```powershell
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 42
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 123
python training\diagnose_ir_batchnorm_full.py --loss bce  --seed 2026
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 42
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 123
python training\diagnose_ir_batchnorm_full.py --loss wbce --seed 2026
```

## 8. 下一步

**IR v1 已完成冻结归档**，本文件后续不再更新（如需变更请以新实验编号登记）。

1. ~~六个模型的 Full BN Diagnostic~~ **已完成 6/6**。
2. ~~六个 Original checkpoint 的阈值校准~~ **已完成 6/6**（新目录 `runs/IR-THRESHOLD-CALIBRATION/`）。
3. ~~保存六份 Training-only BN 重估 checkpoint~~ **完成 6/6**。
4. ~~六个 Recalibrated checkpoint 的独立阈值校准~~ **完成 6/6**，24 条件矩阵齐备。
5. ~~生成 24 条件消融汇总与审计表~~ **已完成**（`runs/IR-ABLATION-SUMMARY/`）。
6. ~~最终配置决策~~ **已完成并冻结**（`IR_v1_final_decision.md`）。

**后续重点**：Raman 的同构 BN 审计与 24 条件消融（可复用 IR 的四个脚本与统计口径）；三模态信息画像（Line B）与 chemistry-aware 融合（Line C）。
