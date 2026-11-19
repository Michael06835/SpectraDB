# Line B · Phase B1 报告：统一 FP32 预测导出与 Shared14 Validation

> **状态**：B1 执行完成，并完成 UV Shared14 Seed 2026 纠错；待人工审核。
>
> **日期**：2026-11-18（Asia/Shanghai）
>
> **范围**：Shared14、三模态、9 个有效 Validation 预测；另含经授权的 UV Shared14 Seed 2026 纠错重训与历史格式 FINAL TEST。未训练其他模型、未访问/推理 Line B Test、未重估 BN、未拟合阈值。

---

## 1. 实验设计

- 任务：QM9S UV-Vis / IR / Raman → 14 个 shared 官能团标签，多标签预测。
- 模型：IR、Raman、UV-Vis 各 3 个已有 run，seed 为 42/123/2026；主比较固定 `Original BN + 0.5`，不重估 BN、不拟合阈值。
- 评价：Shared14 Validation，12,981 行；只使用 Validation，不读取 Line B Test。
- 推理：`model.eval()`、`torch.no_grad()`、FP32、autocast 关闭、batch 512、`num_workers=0`。
- 产物：统一 `probabilities.npy`、`row_index.npy`、`labels.npy`、`prediction_manifest.json`。

核心产物：

- `runs/LINE-B-PREDICTIONS/Shared14/`
- `runs/LINE-B-METRICS/Shared14/`
- `runs/LINE-B-MANIFESTS/B1/`

---

## 2. 执行摘要

1. 实现统一导出/评估/审计脚本及 6 个 CPU 单元测试；合成数据 **6/6 通过**，之后才启动 GPU Validation 推理。
2. 9 个模型全部导出：IR 3、Raman 3、UV-Vis 3；每个 **12,981 行**。
3. 9/9 通过行数、`row_index`、`labels`、标签顺序、概率有限/[0,1]、checkpoint/BN 不可变性检查；全部 `test_split_accessed=False`。
4. **FP32 重复推理全部 bitwise 一致**：同进程二次前向 `max_abs_diff=0.0`（9/9）；IR Seed 42 跨进程复跑亦 bitwise 一致。
5. 九模型 Validation 结果与三模态 Mean±SD 见 §4；IR/Raman 明显强于 UV-Vis shared。
6. UV Shared14 Seed 2026 历史错误已纠正：旧实验实际为 Seed 42，已归档弃用；修正后沿用原编号重训并重导 B1，详见 §8。

---

## 3. 环境与资源

| 项 | 值 |
|---|---|
| Python / PyTorch | 3.12.13 / 2.11.0+cu128 |
| CUDA / GPU | 12.8 / NVIDIA RTX 5070 Laptop GPU，7.96 GB |
| 推理 dtype | FP32，autocast 关闭 |
| batch / workers | 512 / 0（全部 9 次） |
| 设备 | `cuda` |

资源摘要（完整表见 `runs/LINE-B-MANIFESTS/B1/resource_summary.csv`）：

- 进程峰值 RSS 最大 **1,192.1 MB**；系统可用内存最低 **11,209.5 MB**；
- GPU 峰值 allocated：IR/Raman **788.8 MB**，UV **168.5 MB**；reserved：IR/Raman **904.0 MB**，UV **202.0 MB**；
- 单模型首轮推理 0.5–3.3 s，重复前向 0.37–1.4 s；
- 无 CUDA OOM、无系统 RAM 耗尽、无异常退出。

CPU 测试：

- 命令：`python -m unittest discover -s training\line_b\tests -v`
- 结果：**6/6 OK**；覆盖多标签 AP/AUROC/F1、undefined 标签、阈值不影响 AP/AUROC、合成 checkpoint 导出契约、Test split 拒绝、9 组汇总逻辑。
- 日志：`runs/LINE-B-MANIFESTS/B1/cpu_tests.log`。

---

## 4. Validation 指标

### 4.1 九个模型

阈值固定 0.5；AP/AUROC 与阈值无关。

| Run | Modality | Seed | Micro-F1 | Macro-F1 | Macro-AP | Macro-AUROC |
|---|---|---:|---:|---:|---:|---:|
| EXP-IR-001-FULL | IR | 42 | 0.948232 | 0.924119 | 0.968554 | 0.996046 |
| EXP-IR-001-SEED123 | IR | 123 | 0.948078 | 0.912054 | 0.958667 | 0.995980 |
| EXP-IR-001-SEED2026 | IR | 2026 | 0.945086 | 0.913892 | 0.956822 | 0.995589 |
| EXP-RAMAN-001-FULL | Raman | 42 | 0.912398 | 0.868277 | 0.946494 | 0.988341 |
| EXP-RAMAN-001-SEED123 | Raman | 123 | 0.907559 | 0.888036 | 0.950560 | 0.988243 |
| EXP-RAMAN-001-SEED2026 | Raman | 2026 | 0.910424 | 0.866581 | 0.945650 | 0.988381 |
| EXP-UVVIS-001-FULL | UV-Vis | 42 | 0.497110 | 0.400555 | 0.513869 | 0.849718 |
| EXP-UVVIS-001-SEED123 | UV-Vis | 123 | 0.506181 | 0.378201 | 0.515776 | 0.848074 |
| EXP-UVVIS-001-SEED2026 | UV-Vis | 2026 | 0.503842 | 0.405086 | 0.509748 | 0.848893 |

完整逐标签（9 模型 × 14 标签）：`runs/LINE-B-METRICS/Shared14/per_label_all_models.csv`。

### 4.2 按模态汇总

| Modality | Micro-F1 | Macro-F1 | Macro-AP | Macro-AUROC | Seeds |
|---|---:|---:|---:|---:|---|
| IR | 0.947132 ± 0.001773 | 0.916688 ± 0.006500 | 0.961348 ± 0.006308 | 0.995872 ± 0.000247 | 42/123/2026 |
| Raman | 0.910127 ± 0.002433 | 0.874298 ± 0.011928 | 0.947568 ± 0.002625 | 0.988322 ± 0.000071 | 42/123/2026 |
| UV-Vis | 0.502378 ± 0.004709 | 0.394614 ± 0.014393 | 0.513131 ± 0.003081 | 0.848895 ± 0.000822 | 42/123/2026 |

> UV 均值±SD 基于三个真实、唯一的 config seeds。IR/Raman 在 shared 14 类上显著强于 UV-Vis；UV 的排序能力仍明显高于随机，但 0.5 阈值下 F1 很低。

### 4.3 重点逐标签结果

完整 3-seed 逐标签均值/SD：`runs/LINE-B-METRICS/Shared14/per_label_mean_sd.csv`。下表只列代表性标签：

| Label | Support | IR AP | Raman AP | UV AP | UV F1 |
|---|---:|---:|---:|---:|---:|
| carbonyl | 4238 | 0.9978 | 0.9968 | 0.9098 | 0.8146 |
| aldehyde | 1447 | 0.9977 | 0.9961 | 0.7203 | 0.6091 |
| ether | 5794 | 0.9873 | 0.9190 | 0.6986 | 0.5774 |
| amine | 3728 | 0.9540 | 0.9205 | 0.5488 | 0.4818 |
| hydroxyl | 3949 | 0.9999 | 0.9995 | 0.4370 | 0.1995 |
| nitrile | 1697 | 0.9960 | 0.9992 | 0.2369 | 0.0308 |
| nitro | 11 | 0.7669 | 0.8524 | 0.0323 | 0.0000 |

- UV 在 carbonyl、aldehyde 上仍有较强排序能力，但对 hydroxyl、nitrile 明显弱；
- `nitro` 只有 11 个 Validation 正例，任何数值都不应单独支撑结论；
- 三模态的标签排序差异比总体均值更重要，B3 应按标签/结构子群分析。

---

## 5. 预测产物与对齐核验

9 组预测位于：

`runs/LINE-B-PREDICTIONS/Shared14/{IR,Raman,UVVis}/<run>/validation/`

每个目录包含 `probabilities.npy`（float32，12981×14）、`row_index.npy`（int64，12981）、`labels.npy`（uint8，12981×14）、`prediction_manifest.json`。完整 inventory 与哈希见 `runs/LINE-B-MANIFESTS/B1/prediction_inventory.csv`、`b1_export_audit.json`。

| 检查 | 结果 |
|---|---|
| 9 组预测目录 | 9/9 存在 |
| 每个数组行数 | 12981/12981（9/9） |
| `row_index` 与冻结 Validation split | 9/9 逐元素一致 |
| 三模态 `row_index` 互相对齐 | 9/9 同一数组 |
| `labels` 与冻结 14 类矩阵 | 9/9 逐元素一致 |
| 标签顺序 | 9/9 与 `functional_group_label_names.json` 一致 |
| checkpoint 输出头 | 9/9 为 14 维，embedding 256，12 层 BN |
| 概率有限 / [0,1] | 9/9 |
| 每个数组 SHA256 | 9/9 与 manifest 一致 |
| checkpoint 推理前后 SHA256 | 9/9 不变 |
| 模型 state dict / BN buffers | 9/9 未改变 |
| `test_split_accessed` | 9/9 False |

> IR/Raman 与 UV 的 split 文件哈希不同，是因为 UV train 少 2 条全零谱；三个模态的 Validation `row_index` 逐元素相同。

---

## 6. FP32 数值稳定性

- 每个模型导出时执行第二次完整 Validation 前向：9/9 `max_abs_diff=0.0`，bitwise equal，`allclose(atol=1e-6)` True。
- IR Seed 42 额外在独立 Python 进程中重新导出：`max_abs_diff=0.0`，bitwise equal，概率 SHA256 与主产物完全相同。
- 结论：固定 GPU、batch/worker、FP32、eval 模式下，本轮 Validation 推理可复现；未观察到 Raman v1 中 FP16 跨进程的 1e-3 量级漂移。

---

## 7. 历史 FP16 与新 FP32 的差异

历史参考值来自 checkpoint 内 `val_metrics`（训练时的 Validation，AMP/FP16）；新值为本轮统一 FP32 前向重算。新 UV Seed 2026 的 FINAL TEST 是授权纠错评价，记录在 `runs/EXP-UVVIS-001-SEED2026/test_metrics.json`，不与其他模型的历史 Test 指标比较。

| Modality | 最大 abs Δ | 对应指标 / run | Macro-AUROC 最大 abs Δ |
|---|---:|---|---:|
| IR | 0.003667 | macro_AUROC / SEED123 | 0.003667 |
| Raman | 0.000673 | mAP / Seed42 | 0.000097 |
| UV-Vis | 0.005323 | Micro-F1 / EXP-UVVIS-001-FULL | 0.001240 |

- IR/Raman 差异 ≤0.004，符合 FP32/AMP 数值预期；
- UV 差异略大（最大 0.0053），与 UV 的 AP 低、部分标签极稀疏、F1 对分数敏感一致；
- 所有新指标以 FP32 数组单独记录，不与历史 FP16 数值拼接。完整逐指标差异：`runs/LINE-B-METRICS/Shared14/fp16_reference_delta.csv`。

---

## 8. UV Shared14 Seed 2026 纠错

原 `EXP-UVVIS-001-SEED2026` 实际使用 Seed 42；旧实验已完整移入 `archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/`，旧指标全部弃用（详见该目录 `INVALIDATION.md`）。已修正 `training/train_uvvis_seed2026.py` 为 `SEED = 2026`，沿用原编号完成重训与历史格式评价；新 `config.json` 与 checkpoint 均核实为 Seed 2026，新 checkpoint SHA256 为 `9e2b0dd81cb5319e5c4f1aa6f2bb550bf23f6b6a3045e478ac1aa2f97f88e01d`。新结果已替代旧结果进入 B1 三 Seed 统计；历史 Test 纠错评价与 Line B B4 的统一 FP32 Test 评价相互独立，未用于选择模型、损失、阈值、BN 或超参数，也不替代 B4。更正后的 UVVIS-v1 Test 汇总见 `runs/UVVIS-v1-FINAL/`。

---

## 9. 关键风险与局限

- UV-Vis shared 14 类的阈值 0.5 F1 很低，不能把 UV 的总体 AUROC 解读为可直接用于官能团判定；B3 应按标签/结构子群分析。
- `nitro` 仅 11 个 Validation 正例，`c_f_bond` 仅 230 个；稀疏标签的 AP/F1 不稳定，不应单独支撑结论。
- 九模型均为已有 checkpoint 的 Validation 重导，不是重新训练；不能把本轮结果当作新训练配方或超参数搜索结果。
- B1 只覆盖 Validation；Test 评价、条件互补性和 Oracle 分析必须在 B3 FREEZE → B4 中统一执行。

---

## 10. B3 前必须完成

- **A. 条件互补性拆分**：FN rescue / FP correction / new errors，不以 `rescue_rate > P(UV correct)` 作为必要判据。
- **B. WBCE S-03 边界**：仅作历史描述性结果，不进入 B4 的 18 模型冻结矩阵。
- **C. 分析代码**：Bootstrap、互补性、结构子群、报告指标代码在 B3 FREEZE 前实现并完成合成测试。
- **D. Seed 2026 追溯**：archive 与新 checkpoint SHA 已登记，B3 只需引用，不再重复决策。

---

## 11. 追溯与结论

关键追溯路径：

- 预测与 manifest：`runs/LINE-B-PREDICTIONS/Shared14/`
- 指标：`runs/LINE-B-METRICS/Shared14/`
- 导出审计与 inventory：`runs/LINE-B-MANIFESTS/B1/`
- 旧 Seed 2026 archive：`archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/`
- 更正后的 UVVIS-v1 Test 汇总：`runs/UVVIS-v1-FINAL/`

**结论**：B1 的九模型 FP32 Validation 导出与对齐检查通过；IR/Raman 在 shared 14 类上显著强于 UV-Vis，UV 的标签级差异是后续多模态分析的核心信息。B1 可作为 B2/B3 的 Validation 基础设施，但不得把本轮 Validation 结果当作 Test 结论。
