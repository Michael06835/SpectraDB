# UVSPEC15 标签定义与生成规则

## 1. 来源

- 标注 schema：`metadata/uvvis_annotation_schema_v1.0.json`（v1.0 FINAL：8 层 166 features、48 条一致性规则、`uv_pi_system_graph` 定义）。
- 自动标注产物：`processed/qm9s/uvvis_annotations_v1_0_1/`（builder v1.0.1，129,817 样本全部成功）。
- 训练标签矩阵：`processed/qm9s/prepared/uvspec15_labels.npy`（129,817×15，float32）；生成脚本 `scripts/prepare_uvspec15_labels.py`；provenance 见 `uvspec15_label_manifest.json`（含源 label 的 SHA256 与列索引）。

## 2. 15 个标签（输出 head 固定顺序）

| # | 标签 | 来源层 | 化学含义（简述） | 生成规则（摘要） | 全样本正样本数 / 占比 |
|---|---|---|---|---|---|
| 0 | has_conjugation | B1 | 存在共轭 π 体系 | uv_pi_system_graph 中至少一个共轭/离域体系 | 31,266 / 24.1% |
| 1 | has_nonaromatic_conjugation | B1 | 存在非芳香共轭 | 至少一个共轭体系含非芳香显式 π 单元 | 9,282 / 7.2% |
| 2 | acyclic_conjugated_system | B1 | 无环共轭体系 | 至少一个共轭体系不含环键 | 3,889 / 3.0% |
| 3 | cyclic_conjugated_system | B1 | 环状共轭体系 | 至少一个共轭体系含环键 | 27,375 / 21.1% |
| 4 | has_aromatic_system | B2 | 存在芳香体系 | 芳香子图中至少一个芳香原子 | 22,186 / 17.1% |
| 5 | heteroaromatic | B2 | 杂芳香体系 | 至少一个芳香体系含非碳原子 | 21,916 / 16.9% |
| 6 | mixed_aromatic_nonaromatic_conjugation | B1 | 芳香-非芳香混合共轭 | 同一共轭体系同时含芳香成分与非芳香显式 π | 7,227 / 5.6% |
| 7 | pi_conjugated_carbonyl | B3 | π 共轭羰基 | 羰基 C=O 参与含外部 π/芳香单元的共轭体系 | 11,084 / 8.5% |
| 8 | alpha_beta_unsaturated_carbonyl | B3 | α,β-不饱和羰基 | 存在 O=C–C=C 共轭片段 | 2,686 / 2.1% |
| 9 | aryl_carbonyl | B3 | 芳基羰基 | 羰基与芳香体系直接共轭相连 | 2,027 / 1.6% |
| 10 | pi_conjugated_imine | B4 | π 共轭亚胺 | C=N 参与含外部 π/芳香单元的共轭体系 | 2,544 / 2.0% |
| 11 | pi_conjugated_nitrile | B4 | π 共轭腈 | 腈碳通过共轭与 π/芳香体系相连 | 2,477 / 1.9% |
| 12 | has_donor_substituted_pi_system | B5 | 供体取代 π 体系 | hydroxy/alkoxy/lone-pair-donor N 直接连 π 体系 | 25,006 / 19.3% |
| 13 | has_acceptor_substituted_pi_system | B5 | 受体取代 π 体系 | 外部 π 共轭的 carbonyl/nitrile/nitro 连 π 体系 | 13,012 / 10.0% |
| 14 | donor_acceptor_same_conjugated_system | B5 | 同一体系供-受体 | 同一共轭连通分量同时含供体与受体附着（heuristic） | 4,207 / 3.2% |

> 正样本数/占比来自 `metadata/uvvis_annotation_statistics_v1_0_1.csv`（全样本 129,817）；按 split 的 train/val/test 分布见 `metadata/uvvis_label_split_audit_v1_0_1.csv`。

## 3. 关键生成规则

- **uv_pi_system_graph**：芳香环系压缩为一个 delocalized 节点；非芳香 π 单元（C=C / C#C / C=O / C=N / N=N / C#N）仅在参与耦合共轭时进入语义图；sp3 中断处不连边；biaryl 单键连接的芳环不合并；累积二烯（C=C=C）记 cumulated_unsaturation，不计入普通线性共轭。
- **供体/受体**：donor = hydroxy / alkoxy / lone-pair-donor N（排除酰氧与酰胺 N 等）；acceptor = 外部 π 共轭的 carbonyl / nitrile / nitro。
- 完整节点/边规则、层级与一致性约束见 `metadata/uvvis_annotation_schema_v1.0.json` 的 `graph_definitions`、`hierarchies`、`consistency_rules`。

## 4. 使用与审计状态

- 训练使用固定的 15 个二值标签；不增删。如需新增标签，须发布 schema v1.1 并重新生成。
- 频数与 split 审计已完成；极稀疏标签（如 has_nitro 相关、部分共轭精细标签）在训练中保留，但需注意正样本数不足带来的异常（详见 `docs/experiments/UVSPEC15_v1_experimental_strategy.md` 与消融汇总）。
- 标签生成与训练目标选择分离：标注可用 ≠ 必须作为训练目标。
