# SpectraDB TODO

## 当前阶段

**第一阶段（数据体系与单模态模型）已完成：IR、Raman、UV-Vis 三个单模态 1D-CNN 基线全部冻结。**

当前进入第二阶段准备：UV专属化学语义监督与 chemistry-aware 多模态融合。

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
- [x] 更新 `README.md`、`CHANGELOG.md`、`AGENTS.md`、`TODO.md`。
- [x] 更新 Git 忽略规则与环境锁定文件。

---

## 当前最高优先级

### Priority A：UV专属化学语义标签（Line A，论文核心第一步）

- [ ] 设计 UV-specific label taxonomy v0.1（共轭体系有无、共轭路径长度/π-system size、独立共轭系统数量、芳香性/融合芳环、发色团类别与组合、α,β-不饱和羰基、chromophore count/type 等），逐标签说明化学定义、RDKit 判定逻辑、预期频数风险、与 UV 吸收的物理意义。
- [ ] 用户确认 taxonomy 后，编写 `scripts/build_uvvis_specific_labels.py` 在 QM9S 结构上自动生成标签。
- [ ] 标签统计核查：positive count / prevalence / 共现 / 稀有类 / 相关类 + 人工抽样核查。
- [ ] 冻结 UV-specific labels v1。
- [ ] UV-specific 单模态基线（复用冻结 1D-CNN backbone + 专属 head，sanity→full→3 seeds）。
- [ ] shared vs specific 对照，验证"UV 被问对问题后是否显著变强"。

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

### Step 2：UV专属化学语义标签（Line A）

- [ ] taxonomy 设计 → 用户确认 → 自动生成 → 统计核查 → 冻结 v1。
- [ ] UV-specific 单模态基线。
- [ ] shared vs specific 对照。

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
