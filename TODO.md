# SpectraDB TODO

## 当前阶段

**第一阶段（数据体系与单模态模型）已完成：IR、Raman、UV-Vis 三个单模态 1D-CNN 基线全部冻结；UV专属化学语义标注 v1.0.1 已完成；IR v1、UVSPEC15 v1 与 **Raman v1** 的完整方法学消融（各 6 模型 × 24 条件）与 Final Decision 均已冻结归档（2026-11-15 / 2026-11-16）。**

注意（2026-11-16 同步）：**IR v1 已冻结归档并入库**——6/6 独立模型训练、6/6 Full BN Diagnostic、6/6 BN 重估 checkpoint、12/12 Threshold Calibration、24/24 消融条件、48 条 split-level 与 672 条逐标签记录全部完成；主方案 BCE + Original BN + Validation-F1，备选 BCE + Recalibrated BN + Validation-F1，固定基线 BCE + Original BN + 0.5（`docs/experiments/IR_v1_final_decision.md`）。**UVSPEC15 v1 亦已完成 24 条件消融与独立 Final Decision**（主方案 BCE + Recalibrated BN + Validation-F1，`docs/experiments/UVSPEC15_v1_final_decision.md`）。上述成果已提交 Git（`bbcdfa4`）并推送 GitHub main。**Raman v1 亦已完成同构审计并冻结归档（2026-11-16）**：6/6 模型训练、6/6 Full BN Diagnostic、6/6 BN 重估 checkpoint（gate policy v5：Gate A 10 项 + Gate B 六指标 1e-3）、12/12 阈值校准、24/24 消融条件、48 条 split-level 与 672 条逐标签记录；主方案 **WBCE + Recalibrated BN + Validation-F1**（Validation Micro-F1 0.922034±0.002103 / Test 0.914113±0.006869），备选 WBCE + Original BN + Validation-F1，固定基线 BCE + Original BN + 0.5；文档见 `docs/experiments/Raman_v1_final_decision.md`、`raman_ablation_registry.md`、`raman_repro_protocol.md`。**当前下一步是三模态信息画像（Line B）**；多模态融合尚未开始，shared vs specific 对照已成立。

核心研究问题：不是把三条谱简单拼起来，而是让模型学会像化学家一样，知道什么时候该听谁——IR/Raman 承担振动结构证据，UV-Vis 承担共轭/发色团电子结构证据，通过化学证据驱动的自适应融合改善完整结构 Top-k 推断。

当前路线：

- 单模态均使用 1D-CNN。
- Morgan fingerprint 不作为光谱模型的输入。
- 官能团预测是中间结构信息，不是最终目标。
- 最终目标：从光谱推断完整分子结构候选，输出官能团/结构特征、结构候选、Top-k 排序与可视化。

---

## 已完成

- [x] 下载 QM9S 数据集（`qm9s.pt` 及 IR、Raman、UV-Vis 计算光谱 CSV）。
- [x] 构建 QM9S manifest（129817 个样本，无缺失 SMILES、无重复编号）。
- [x] IR、Raman、UV-Vis 光谱转为 float32 NPY（IR/Raman 129817×3501，UV-Vis 129817×701）。
- [x] 光谱质量检查（0 非有限值、2 条全零 UV-Vis、129 条强度离群）。
- [x] 生成 Bemis–Murcko scaffold split（80/10/10，28501 个 scaffold，跨分区零重叠）。
- [x] 生成模态有效性 mask，三模态完整配对 129815 个样本。
- [x] 建立 QM9S 结构库（原子、键、官能团、环、共轭体系标注）。
- [x] 生成官能团多标签矩阵（129817×14）。
- [x] 训练 IR 1D-CNN 基线并冻结 IR-v1（test Micro-F1 0.9406±0.0048，Macro-AUROC 0.9954±0.0007）。
- [x] 完成 IR 类别不平衡消融实验与阈值校准实验（均保留 IR-v1 默认设置）。
- [x] 训练 Raman 1D-CNN 基线并冻结 Raman-v1（test Micro-F1 0.9151±0.0014，Macro-AUROC 0.9911±0.0004）。
- [x] 完成 Raman 类别不平衡消融实验与阈值校准实验（均保留 Raman-v1 默认设置）。
- [x] 训练 UV-Vis 1D-CNN 基线并冻结 UVVIS-v1（corrected test Micro-F1 0.5158±0.0067，Macro-AUROC 0.8526±0.0016；旧 invalid Seed2026 三 Run 均值已弃用）。
- [x] 完成 UV-Vis 类别不平衡、阈值校准与 BatchNorm 重校准诊断（均保留 UVVIS-v1 默认设置）。
- [x] 弃用 GDB-9-Ex 为主训练数据，降级为备用外部域/方法迁移数据集。
- [x] 冻结 UV-Vis 化学语义标注 schema v1.0 FINAL（8 层 166 features、48 条一致性规则、uv_pi_system_graph 定义）。
- [x] 编写并运行 `scripts/build_qm9s_uvvis_annotations.py`（builder v1.0.1）：129817 样本全部标注成功（0 失败，172 警告），输出 58 个二值标签 + 73 个数值特征。
- [x] 完成标注统计（`metadata/uvvis_annotation_statistics_v1_0_1.csv`）与按 split 审计（`metadata/uvvis_label_split_audit_v1_0_1.csv`）。
- [x] 完成 UVSPEC15 单模态基线三随机种子训练（seed 42/123/2026）。
- [x] 完成 UVSPEC15 阈值校准、完整 BN 诊断、BN 重校准模型保存与 BN 后阈值校准（普通 BCE 与 Weighted BCE 各 3 seeds）。
- [x] 完成 Weighted BCE 三随机种子对照实验（确认不采用）。
- [x] 完成 shared vs specific 正式对照分析（`runs/UVSPEC15-ANALYSIS/shared_vs_specific_summary.md`），确认 UV 专属语义任务下 UV 显著变强。
- [x] 更新 `README.md`、`CHANGELOG.md`、`AGENTS.md`、`TODO.md`。
- [x] 更新 Git 忽略规则与环境锁定文件。
- [x] 完成 UVSPEC15 v1 正式实验策略归档（`docs/experiments/UVSPEC15_v1_experimental_strategy.md`：明确 Training/Validation/Test 分工与域差异归因）。
- [x] 完成 UVSPEC15 实验数据一致性审计（checkpoint-seed 对应、原始/BN 重估配套、阈值配套、样本 SD、mAP/AUROC 阈值无关、audit 三项全过）。
- [x] 新增 `docs/experiments/uvspec15_repro_protocol.md`、`docs/data/uvspec15_label_guide.md`、`docs/experiments/uvspec15_ablation_registry.md`。
- [x] 修复训练/汇总脚本行尾空格。
- [x] 完成 UVSPEC15 v1 独立 Final Decision 文档（`docs/experiments/UVSPEC15_v1_final_decision.md`，2026-11-15；主方案 BCE + Recalibrated BN + Validation-F1，Test Micro-F1 0.6568±0.0138）。
- [x] 完成 IR v1 完整方法学消融（6 模型 × 2 BN × 2 阈值 = 24 条件；48 条 split-level、672 条逐标签）与 Final Decision / Ablation Registry / Repro Protocol。
- [x] 归档废弃脚本 `training/calibrate_ir_thresholds.py`（迁入 `archive/2026-11-15/`，原路径改为立即非零退出的弃用入口）。
- [x] IR/UVSPEC15 归档成果提交 Git（`bbcdfa4`）并推送 GitHub `main`（2026-11-16）。
- [x] 补齐 Raman WBCE Seed 123 / 2026 训练与最终 Test 评估，Raman 达到 6 个独立模型（BCE/WBCE × 3 seeds）。
- [x] Raman Full BN Diagnostic 6 次（严格串行）：6/6 完成，六份 JSON 通过基线复现、参数不变性、BN 统计量与文件保护检查。
- [x] 保存 6 份 Training-only BN 重估 checkpoint（gate policy v5：Gate A 10 项 + Gate B 六指标 1e-3 双门）：6/6 `ARTIFACT_PASS__NUMERIC_PASS`、`ablation_release=ELIGIBLE`。
- [x] Raman Original / Recalibrated 阈值校准各 6 次：12/12 完成，阈值在 Test 指标前落盘，两个阈值条件共用同一组概率并落盘 SHA256（`probability_artifacts.json`）。
- [x] 执行 `training/summarize_raman_ablation.py` 生成 24 条件汇总（24+24 条件、48 条 split-level、672 行逐标签、11 个文件、6 份 audit 全 True）：`runs/RAMAN-ABLATION-SUMMARY/`。
- [x] 完成 Raman v1 Final Decision / Ablation Registry / Repro Protocol 三份文档，并完成两轮审核修订与 FP16 元数据勘误（`raman_ablation_registry.md` §9.5）。
- [x] Raman 归档成果提交 Git 并推送 GitHub `main`（2026-11-16）。
- [x] 完成 Line B Phase B0 只读资产审计、样本/标签一致性检查与实验协议草案（2026-11-17；未执行新的训练/推理、未改历史数据/checkpoint/指标），并完成 v0.2 方法学修订（`positive_or_fusion`/真实 Oracle 拆分、Bootstrap 解释、B0 Test 接触声明、UV 边际/互补拆分、B1–B5 阶段编号、文档 errata）；等待人工审核。
- [x] 完成 Line B Phase B1 统一 FP32 Validation 导出与 Shared14 九模型评估（2026-11-18；6 个 CPU 测试通过，9/9 对齐/哈希/不可变性检查通过，FP32 重复推理 max_abs_diff=0）；并按授权完成 UV `EXP-UVVIS-001-SEED2026` 历史错误归档（实际 SEED=42）、脚本修正、真实 Seed 2026 重训、历史格式 FINAL TEST 与 B1 重导；报告见 `docs/experiments/line_b_b1_shared14_validation_report.md`。

---

## 当前最高优先级

### Priority A：UV-specific 标签冻结与单模态基线（Line A，已完成）

- [x] 冻结 UV 标注 schema v1.0 FINAL，自动生成标签（builder v1.0.1，129817 全部成功）。
- [x] 完成频数统计与 split 审计（COUNT_SCREEN_PASS / SPARSE / NO_TRAIN_POSITIVES 分类）。
- [x] 基于审计选定 UVSPEC15（15 个 UV 专属标签）并生成标签矩阵与 provenance manifest。
- [ ] 人工抽样核查关键标签（共轭体系、发色团、芳香 N、供受体）与化学合理性（可选复核）。
- [x] UVSPEC15 单模态基线（3 seeds + 阈值校准 + BN 诊断 + BN 重校准 + WBCE 对照）。
- [x] shared vs specific 对照，验证"UV 被问对问题后显著变强"（见 `runs/UVSPEC15-ANALYSIS/`）。

### Priority B：三模态信息画像（Line B）

**B0：只读资产审计、协议设计与 v0.2 方法学修订（2026-11-17 完成，已通过人工验收）**

- [x] 盘点 IR v1、Raman v1、UV-Vis shared、UVSPEC15 v1 的 checkpoint、阈值、逐标签结果、概率数组与汇总文件。
- [x] 核查 IR/Raman 103855/12981/12981 与 UV 103853/12981/12981 的两行差异（row_index 13998/14000 全零 UV 谱），确认只在 IR/Raman train 中、val/test 逐元素相同。
- [x] 以 `uvspec15_label_names.json` 为唯一权威核验 15 类列顺序、源矩阵列索引、checkpoint 输出头与 `label_names`。
- [x] 核查归一化、谱轴、row_index、scaffold 零重叠、Test 使用纪律与泄漏风险。
- [x] 生成审计/协议/矩阵文件；完成 v0.2 修订：`positive_or_fusion` 与真实 Oracle 拆分、Bootstrap 解释更正、B0 Test 接触声明、UV 边际/互补拆分、B1–B5 阶段编号、文档 errata。
- [x] **人工审核 B0 v0.2 修订稿**：主比较统一 BCE + Original + 0.5；UVSPEC15 三模态统一 103853；首轮只 BCE；0.5 主 / Val-F1 次；Test 仅作 exploratory/post-hoc。

**B1：预测导出基础设施 + Shared14 Validation（2026-11-18 完成，含 UV Seed 2026 纠错，待审核）**

- [x] 实现统一导出/评估/审计脚本与 6 个 CPU 单元测试；合成数据 6/6 通过。
- [x] 导出 Shared14 的 9 个模型 validation 概率、`row_index`、`y_true`、checkpoint/split/label/code 哈希。
- [x] 计算 Shared14 validation AP/AUROC/F1/support；输出 `runs/LINE-B-METRICS/Shared14/` 与 `runs/LINE-B-MANIFESTS/B1/`。
- [x] 9/9 行数、row_index、labels、标签顺序、概率范围、checkpoint/BN 不变性与 `test_split_accessed=False` 全部通过；FP32 重复推理 max_abs_diff=0。
- [x] 归档并废弃 invalid `EXP-UVVIS-001-SEED2026`（实际 SEED=42），修正脚本为 SEED=2026，完整重训与历史格式 FINAL TEST，重导 B1 Validation，恢复 UV 三真实 Seed；详见 `docs/experiments/line_b_b1_shared14_validation_report.md`。
- [ ] **人工审核 B1 结果与 UV Seed 2026 纠错记录**。

**B2：六个 IR/Raman UVSPEC15 BCE 模型训练（未开始）**

- [ ] 新训 IR UVSPEC15 3 seeds（seed 42/123/2026，BCE，Original BN，15 维头，UV split 103853）。
- [ ] 新训 Raman UVSPEC15 3 seeds（同上）。
- [ ] 保存 6 份 checkpoint、run manifest 与 SHA256；不加载 Test。

**B3：全部 Validation、阈值与协议冻结（未开始）**

- [ ] 导出 UVSPEC15 的 9 个模型 validation 概率（UV 3 + 新 IR/Raman 6）。
- [ ] 计算全部 18 个模型的 validation 指标。
- [ ] 统一 Validation-only 逐标签 F1 阈值拟合（不复制历史阈值）。
- [ ] 写 `FREEZE.json` 并校验哈希链；本阶段不接触 Test。
- [ ] **B3-A**：条件互补性拆分 FN rescue / FP correction / new errors，不以 rescue_rate > UV 总体准确率作为必要判据。
- [ ] **B3-B**：Shared14 WBCE S-03 仅作历史描述性结果，不得在 B4 的 18 模型矩阵外临时新增 Test 推理。
- [ ] **B3-C**：Bootstrap、互补性、结构子群与报告指标代码在 FREEZE 前实现并完成合成数据测试。
- [x] **B3-D**：UV `EXP-UVVIS-001-SEED2026` 历史问题已解决——原实验已归档废弃，修正脚本后以 Seed 2026 完整重训，新 checkpoint/config seed=2026，B1 已重导并恢复三真实 Seed（42/123/2026）。

**B4：Test 导出与完整分析（未开始）**

- [ ] 18 个模型各导出一次 Test 概率（需 FREEZE；写 test_access_log）。
- [ ] 逐标签 AP/AUROC/F1/support、Macro/Micro、3 Seed Mean±SD。
- [ ] scaffold 聚类配对 Bootstrap（同一重采样索引；undefined 计数）。
- [ ] 相同样本的错误互补性、`positive_or_fusion`、真实 Oracle 正确性、McNemar。
- [ ] 标签共现、结构子群、典型分子分析。
- [ ] `runs/LINE-B-FINAL/` 报告与图表。

**B5：报告与归档（未开始）**

- [ ] 形成三模态信息画像结论，明确支持/不支持 H1–H4 的证据。
- [ ] 给出是否进入 chemistry-aware 融合（Line C）的建议。
- [ ] 完成文件清单、哈希与归档记录。
- [ ] 禁止把已开启 Test 的探索性结果写成确认性结论。

### Priority C：Chemistry-aware 融合（Line C，论文核心方法）

- [ ] naive concatenation / fixed-weight 三模态对照（显式复现 IR+Raman vs IR+Raman+UV 的 negative transfer）。
- [ ] generic attention/gating 对照（无化学监督）。
- [ ] proposed chemistry-aware routing（modality embeddings + shared evidence + UV-specific chemical evidence + sample-dependent gate）。
- [ ] 关键消融：无化学监督 / 普通 attention/MoE / 加 UV 专属化学监督 / 去掉 UV 标签监督。
- [ ] gate 权重与化学标签/证据相关性分析。

---

## 后续任务

### IR/Raman BN 审计同步（IR 与 Raman 均已完成并冻结）

IR 已对齐 UVSPEC15 v1 的实验规模（6 模型 × 2 BN × 2 阈值 = 24 条件），策略记录见 `docs/experiments/IR_v1_experimental_strategy.md`：

- [x] 补齐 WBCE Seed 123 / 2026 训练，IR 达到 6 个独立模型（BCE/WBCE × 3 seeds），六份 checkpoint 通过一致性核验。
- [x] IR Full BN Diagnostic 6 次（严格串行）：6/6 完成（2026-11-15），六份 JSON 通过基线复现、参数不变性、BN 统计量与文件保护检查。
- [x] 六个 Original checkpoint 的 Validation 逐标签 F1 阈值校准：6/6 完成，输出新目录 `runs/IR-THRESHOLD-CALIBRATION/`，历史 BCE 42 产物保留未动。
- [x] 保存 6 份 Training-only BN 重估 checkpoint（Original 全部保留）：**6/6 完成**。
- [x] 6 个 Recalibrated checkpoint 的独立阈值校准：**6/6 完成**，24 条件矩阵齐备。
- [x] 执行 `training/summarize_ir_ablation.py` 生成 24 条件汇总（48 条 split-level、672 行逐标签、审计表）：`runs/IR-ABLATION-SUMMARY/`，6 份 audit 全 True。
- [x] 依据 Validation 多指标比较确定 IR v1 最终配置并冻结归档（`docs/experiments/IR_v1_final_decision.md`）：主方案 BCE + Original BN + Validation-F1；备选 BCE + Recalibrated BN + Validation-F1；固定基线 BCE + Original BN + 0.5。
- [x] 归档成果提交 Git `bbcdfa4` 并推送 GitHub `main`（2026-11-16）。
- [x] **Raman 资产审计与同构实验：已完成并冻结（2026-11-16）**——6/6 模型训练、6/6 Full BN Diagnostic、6/6 BN 重估 checkpoint、12/12 阈值校准、24/24 消融条件、48 条 split-level 与 672 条逐标签记录齐备；主方案 WBCE + Recalibrated BN + Validation-F1（Validation Micro-F1 0.922034±0.002103 / Test 0.914113±0.006869），备选 WBCE + Original BN + Validation-F1，固定基线 BCE + Original BN + 0.5。决策见 `docs/experiments/Raman_v1_final_decision.md`。
- [ ] checkpoint 与原始数据的仓库外独立备份（12 份 IR checkpoint 约 140 MB + 12 份 Raman checkpoint 约 140 MB + 光谱/标签/划分矩阵；以 manifest 的 `saved_sha256` 作为校验依据）。**Raman 备份用户已决定跳过（2026-11-16），IR 待定。**
- [x] 废弃脚本已归档（`archive/2026-11-15/calibrate_ir_thresholds.py`），原路径改为弃用入口；**不得运行归档版本**（`"w"` 模式覆盖写盘 + 硬编码 Seed 42 + 1024/0 推理）。

### 完整结构推断与 Top-k（Line D）

- [ ] 从融合后的谱图表征生成/检索完整结构候选。
- [ ] 评估 Recall@1/5/10、MRR、candidate ranking quality 与 scaffold/难度分层分析。
- [ ] 接入结构渲染（`qm9s_structure_tools.py render`）实现可视化输出。

### 化学规则 reranking（Line E）

- [ ] Woodward–Fieser 等规则或近似电子吸收一致性作为 consistency score。
- [ ] 规则用于 reranking/evidence，不硬编码最终答案。

### 缺失/噪声鲁棒性（Line F）

- [ ] modality dropout、missing modality 实验。
- [ ] 强度噪声/基线漂移/峰展宽/波数扰动实验。
- [ ] 单模态低质量时 gate 自动降权验证。

### 实验域验证（Line G）

- [ ] 将 NIST IR `.jdx` 解析并重采样到统一网格，用 master SMILES 计算官能团标签。
- [ ] 清洗 API-Raman 数据（3,510 条）并与 SMILES 匹配、对齐波数轴。
- [ ] zero-shot / fine-tune / domain adaptation 对比，明确区分计算域与实验域结果。

### 开放集与可解释性（Line H/I）

- [ ] 置信度/不确定性、reject option、OOD/open-set 指标。
- [ ] XAI：模态贡献、关键谱区/峰、结构证据、规则支持/反对、2D 结构图标注。

---

## 暂停任务

以下任务当前暂停：

- [ ] GDB-9-Ex 数据接入（备用外部域/方法迁移数据集，当前不参与训练，不删除原始数据）。
- [ ] QMe14S 数据接入（已退出当前方案）。
- [ ] SDBS 采集（已暂停，恢复需先小样本页面结构调试）。
- [ ] NIST 大规模采集扩展。
- [ ] 大规模随机采样数据获取。
- [ ] 大规模随机采样 master 的使用。
- [ ] Morgan fingerprint 作为光谱模型输入（已明确不使用）。

---

## 推荐执行顺序

### Step 1：单模态基线

- [x] IR-v1、Raman-v1、UV-Vis-v1 全部冻结。

### Step 2：UV专属化学语义标签（Line A）——已完成

- [x] taxonomy 设计（schema v1.0 FINAL）→ 自动生成（builder v1.0.1）→ 统计与 split 审计。
- [x] 基于审计选定 UVSPEC15（15 标签）并冻结标签矩阵。
- [x] UVSPEC15 单模态基线（3 seeds + BN 重校准）。
- [x] shared vs specific 对照（`runs/UVSPEC15-ANALYSIS/shared_vs_specific_summary.md`）。
- [x] UVSPEC15 v1 完整消融（6 模型 × 24 条件）与 Final Decision（2026-11-15）。

### Step 3：IR / UVSPEC15 方法学消融冻结——已完成

- [x] IR v1：6 模型 × 24 条件、12 次阈值校准、6 份 BN 重估 checkpoint、Ablation Registry 与 Repro Protocol（2026-11-15）。
- [x] Git 提交 `bbcdfa4` 并推送 GitHub `main`（2026-11-16）。

### Step 4：Raman 同构审计——已完成

- [x] Raman 现有资产审计（4 个 run：BCE ×3 + WBCE ×1），并补齐 WBCE Seed 123 / 2026。
- [x] 6 次 Full BN Diagnostic → 6 份 BN 重估 checkpoint → 12 次阈值校准 → 24 条件消融汇总（2026-11-16）。
- [x] 依据 Validation 确定 Raman v1 配置并冻结归档（主方案 WBCE + Recalibrated BN + Validation-F1）。

### Step 5：模态对比与信息画像（Line B）

- [ ] 输出三模态表征能力对比总结。

### Step 6：chemistry-aware 融合（Line C）

- [ ] naive 对照 → generic gating → proposed routing → 消融。

### Step 7：完整结构 Top-k（Line D）

- [ ] 融合证据成立后进入完整结构推断。

### Step 8：鲁棒性与实验域（Line F/G）

### Step 9：开放集与可解释（Line H/I）

进入完整结构阶段（Line D）前必须满足：

- [ ] UV-specific 监督生效（shared vs specific 对照成立）。
- [ ] chemistry-aware fusion 证据成立（优于 naive，gate 可解释）。
- [ ] 用户明确确认。
