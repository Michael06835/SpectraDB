# SpectraDB TODO

## 当前阶段

**第一阶段（数据体系与单模态模型）已完成：IR、Raman、UV-Vis 三个单模态 1D-CNN 基线全部冻结；UV专属化学语义标注 v1.0.1 与 UVSPEC15 单模态基线（3 seeds + BN 重校准）已完成。**

shared vs specific 对照已成立（UV 在专属语义任务上显著变强）；当前进入三模态信息画像（Line B）与 chemistry-aware 多模态融合（Line C）。

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
- [x] 训练 UV-Vis 1D-CNN 基线并冻结 UVVIS-v1（test Micro-F1 0.4892±0.0406，Macro-AUROC 0.8501±0.0044）。
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

- [ ] 共享 14 标签上三模态 per-label AP/F1 对比。
- [ ] UV-specific 标签上三模态表现对比。
- [ ] 标签共现/混淆与样本级 difficulty 分析。
- [ ] 识别 UV 有额外价值与 UV 是噪声的结构子群。

### Priority C：Chemistry-aware 融合（Line C，论文核心方法）

- [ ] naive concatenation / fixed-weight 三模态对照（显式复现 IR+Raman vs IR+Raman+UV 的 negative transfer）。
- [ ] generic attention/gating 对照（无化学监督）。
- [ ] proposed chemistry-aware routing（modality embeddings + shared evidence + UV-specific chemical evidence + sample-dependent gate）。
- [ ] 关键消融：无化学监督 / 普通 attention/MoE / 加 UV 专属化学监督 / 去掉 UV 标签监督。
- [ ] gate 权重与化学标签/证据相关性分析。

---

## 后续任务

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

### Step 3：模态对比与信息画像（Line B）

- [ ] 输出三模态表征能力对比总结。

### Step 4：chemistry-aware 融合（Line C）

- [ ] naive 对照 → generic gating → proposed routing → 消融。

### Step 5：完整结构 Top-k（Line D）

- [ ] 融合证据成立后进入完整结构推断。

### Step 6：鲁棒性与实验域（Line F/G）

### Step 7：开放集与可解释（Line H/I）

进入完整结构阶段（Line D）前必须满足：

- [ ] UV-specific 监督生效（shared vs specific 对照成立）。
- [ ] chemistry-aware fusion 证据成立（优于 naive，gate 可解释）。
- [ ] 用户明确确认。
