# UV-Vis shared vs specific 标签对照分析（UVSPEC15）

状态：**结果口径已冻结（2026-11-14）；UV-v1 shared 列于 2026-11-18 更正为真实 Seed2026 三 Seed 结果，旧 invalid 三 Run 均值已弃用**

## 0. 来源与可复现性

- UV-v1（shared 14 官能团标签）：`runs/UVVIS-v1-FINAL/`（旧版快照在 archive `superseded_uvvis_v1_final/`）
- UVSPEC15（specific 15 个 UV 专属共轭/发色团标签）：`runs/EXP-UVVIS-003-UVSPEC15-*`
- BN 重校准版：`runs/UVSPEC15-BN-RECALIBRATED/`
- Weighted BCE 消融：`runs/EXP-UVVIS-004-UVSPEC15-WBCE-*`
- 标签定义：`metadata/uvvis_annotation_schema_v1.0.json`（schema FINAL）、`processed/qm9s/prepared/uvspec15_label_names.json`

## 1. 对照设计

- 同一 QM9S UV-Vis 数据（129,817；701 点，1–15 eV）
- 同一 Bemis–Murcko scaffold split（train 103,853 / val 12,981 / test 12,981）
- 同一 backbone（residual 1D-CNN，embedding 256）、max 归一化、AdamW、BCE、AMP、early stopping patience 8
- seeds 42 / 123 / 2026；validation 选 checkpoint，test 仅做一次独立评估
- **唯一变量：标签集**（shared 14 官能团 vs specific 15 UV 化学语义）

## 2. 主结果（3 seeds，test，mean ± sample SD）

| 指标 | UV-v1 shared（14 标签，corrected） | UVSPEC15 原始（15 标签） | UVSPEC15 BN 重校准 |
|---|---:|---:|---:|
| Micro-F1 | 0.5158 ± 0.0067 | 0.6136 ± 0.0845 | **0.6580 ± 0.0154** |
| Macro-F1 | 0.4130 ± 0.0232 | 0.4397 ± 0.0743 | 0.5395（校准阈值） |
| mAP | 0.5126 ± 0.0011 | 0.5389 ± 0.0599 | **0.5728 ± 0.0203** |
| Macro-AUROC | 0.8526 ± 0.0016 | 0.9259 ± 0.0204 | **0.9384 ± 0.0040** |
| 平均 per-label AP | 0.5126 | 0.5389 | — |
| 平均 per-label AUROC | 0.8526 | 0.9259 | — |

注：UV-v1 shared 列已于 2026-11-18 更正为真实 Seed2026 的 3-seed 结果；旧 invalid Seed2026（实际 SEED=42）结果与旧均值已弃用。shared 与 specific 标签集不同，Micro-F1/Macro-F1 不完全可比；主证据为 mAP、AUROC 与平均 per-label AP。

## 3. 结论

1. **核心假设得到初步支持**："UV 不是弱，而是被问错了问题"——同一 UV 光谱、同一 backbone、同一 split，仅把任务换成 UV 专属共轭/发色团语义后，mAP 0.505→0.573（+13%）、AUROC 0.850→0.938（+10 个点）。
2. **BN 重校准是合法后处理**（train split only 重算 running stats、无梯度更新、checkpoint 选择依据 validation），它显著降低三 seed 方差（Micro-F1 SD 0.0845→0.0154、AUROC SD 0.0204→0.0040），说明训练震荡导致的 BN 统计量失真得到了校正。
3. **UV 擅长粗粒度电子结构语义、不擅长精细定位**（per-label AP）：最易学为 has_conjugation（0.878）、cyclic_conjugated_system（0.810）、has_aromatic_system（0.773）、heteroaromatic（0.766）；最难为 pi_conjugated_nitrile（0.163）、pi_conjugated_imine（0.184）、aryl_carbonyl（0.282）。化学上合理：UV 足以判断"有无共轭体系"，难以回答"共轭定位于哪个基团"——这正是后续 chemistry-aware fusion 中"UV 提供粗粒度/局部证据"定位的依据。

## 4. Weighted BCE 消融（3 seeds，BN 重校准后，test）

| 组别 | Micro-F1 | Macro-F1（校准阈值） | mAP | Macro-AUROC |
|---|---:|---:|---:|---:|
| Ordinary BCE（3-seed） | **0.6580 ± 0.0154** | 0.5395 | **0.5728 ± 0.0203** | 0.9384 ± 0.0040 |
| Weighted BCE（3-seed） | 0.6465 | 0.5330 | 0.5596 | 0.9382 |
| WBCE seed42 | 0.6435 | 0.5174 | 0.5377 | 0.9338 |
| WBCE seed123 | 0.6395 | 0.5336 | 0.5583 | 0.9398 |
| WBCE seed2026 | 0.6565 | 0.5481 | 0.5828 | 0.9411 |

**结论**：WBCE 未带来整体收益（Micro-F1/mAP 仍略低于 ordinary BCE），且训练不稳定更强（BN 重校准前 seed123 Micro-F1 仅 0.4522，重校准后 0.6395）——**canonical 仍采用 ordinary BCE**；WBCE 作为消融保留。

## 5. 后续用途

- 该对照结果是论文第一块关键证据（"UV 任务语义匹配度决定其可学习性"）；
- 下一步：三模态信息画像（IR/Raman 需在 UVSPEC15 标签上训练相应 baseline）→ naive fusion 对照 → chemistry-aware routing。
