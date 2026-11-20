# SpectraDB TODO

## 当前阶段

**第一阶段（数据体系与单模态模型）已完成：QM9S数据、结构库、scaffold split与IR/Raman/UV-Vis三个1D-CNN基线均已冻结；UV专属化学语义标签v1.0.1与UVSPEC15 v1已完成；IR v1、UVSPEC15 v1、Raman v1的24条件消融与Final Decision均已归档。**

**Line A（UV专属语义）与Line B（三模态信息画像）已完成。Line B的B0–B5全部结束，B5正式报告通过独立验收，并已归档至本地分支`archive/line-b-2026-11-20`（commit `ab1e85dd0274e803bae074fc19acce47f6feb142`，已推送）。当前没有进行中的正式训练或实验；下一阶段为Line C。**

核心研究问题：不是把三条谱简单拼起来，而是让模型学会像化学家一样，知道什么时候该听谁——IR/Raman承担振动结构证据，UV-Vis承担共轭/发色团电子结构证据，通过化学证据驱动的自适应融合改善完整结构Top-k推断。

当前路线：

- 单模态均使用1D-CNN。
- Morgan fingerprint不作为光谱模型的输入。
- 官能团预测是中间结构信息，不是最终目标。
- 最终目标：从光谱推断完整分子结构候选，输出官能团/结构特征、结构候选、Top-k排序与可视化。

---

## 已完成

### 数据与结构基础

- [x] 下载QM9S数据集（`qm9s.pt`及IR、Raman、UV-Vis计算光谱CSV）。
- [x] 构建QM9S manifest（129817个样本，无缺失SMILES、无重复编号）。
- [x] IR、Raman、UV-Vis光谱转为float32 NPY（IR/Raman 129817×3501，UV-Vis 129817×701）。
- [x] 光谱质量检查（0非有限值、2条全零UV-Vis、129条强度离群）。
- [x] 生成Bemis–Murcko scaffold split（80/10/10，28501个scaffold，跨分区零重叠）。
- [x] 生成模态有效性mask，三模态完整配对129815个样本。
- [x] 建立QM9S结构库（原子、键、官能团、环、共轭体系标注），生成129817×14官能团标签矩阵。

### 单模态基线

- [x] 训练IR 1D-CNN并冻结IR-v1（Test Micro-F1 0.9406±0.0048，Macro-AUROC 0.9954±0.0007）。
- [x] 训练Raman 1D-CNN并冻结Raman-v1（Test Micro-F1 0.9151±0.0014，Macro-AUROC 0.9911±0.0004）。
- [x] 训练UV-Vis 1D-CNN并冻结UVVIS-v1（修正后Test Micro-F1 0.5158±0.0067，Macro-AUROC 0.8526±0.0016；旧invalid Seed2026结果已弃用）。
- [x] 完成三类单模态的类别不平衡、阈值校准与BatchNorm重校准诊断；默认配置均保留。
- [x] 弃用GDB-9-Ex为主训练数据，降级为备用外部域/方法迁移数据集。

### Line A：UV专属语义与基线

- [x] 冻结UV-Vis化学语义标注schema v1.0 FINAL（8层166 features、48条一致性规则、`uv_pi_system_graph`定义）。
- [x] 完成builder v1.0.1全量标注：129817样本成功，输出58个二值标签与73个数值特征；完成统计与split审计。
- [x] 基于审计选定UVSPEC15（15个UV专属共轭/发色团标签），生成129817×15标签矩阵与provenance manifest。
- [x] 完成UVSPEC15三随机种子基线、阈值校准、Full BN诊断、BN重校准与Weighted BCE对照；Weighted BCE不采用。
- [x] 完成shared vs specific正式对照：UVSPEC15主方案BCE + Recalibrated BN + Validation-F1，Test Micro-F1 0.6568±0.0138。
- [x] 完成UVSPEC15 v1的Final Decision、Ablation Registry与Repro Protocol归档。

### Line B：三模态信息画像（B0–B5）

- [x] **B0**：完成只读资产审计与协议v0.2，拆分Positive OR与真实Oracle，明确scaffold聚类Bootstrap、undefined处理与Test接触边界；B0 v0.2 FINAL SHA256 `713fa92f…`。
- [x] **B1**：完成Shared14九模型FP32 Validation导出与评估；发现并纠正`EXP-UVVIS-001-SEED2026`实际Seed 42的问题，恢复UV三真实Seed（42/123/2026）。
- [x] **B2**：完成IR/Raman × seed 42/123/2026共6个UVSPEC15模型训练与资产核验；固定UV 103853 split、15类标签、BCE、Original BN、0.5、Validation mAP选模。
- [x] **B3**：完成18模型FP32 Validation导出、261条Validation-only逐标签F1阈值校准（126+135）、B3-A互补性正式输出、B3.5-A代码溯源/B3.5-B manifest校验，以及C1/C2/D1/D2的Validation-only基础设施与分析。
- [x] **B3 FREEZE**：完成B0 v0.2 FINAL定稿与`FREEZE.json`；18/18模型、261条阈值、212项资产、B3-A 6+6输出；预检与写入后复验`PASS 10 / FAILED 0 / ACCEPTED 4`。
- [x] **B4.1**：18个冻结模型Test导出（每模型12981行，72文件/18,969,279 B）；FREEZE哈希门禁、模型身份校验、重复推理bitwise一致；独立验证PASS 466/466。
- [x] **B4.2**：18模型Test性能评估与三模态比较（17文件/1,841,098 B）；主F1政策0.5，次政策Validation-F1；独立验证PASS 8,374/8,374。
- [x] **B4.3**：2,000次Test scaffold-clustered paired bootstrap（2853个Test scaffold，RNG 20261117，18模型/3 seed/全部标签共用draw matrix）；独立验证PASS 1,144/1,144。
- [x] **B4.4**：完成错误互补性、McNemar/Holm、Positive OR/AND、真实Oracle、条件互补、sample-level、hard sample与结构子群分析；独立验证PASS 1,127/1,127。
- [x] **B4.4 Supplement**：按B0 §7.6完成88个典型分子案例（每Task 22 correct + 22 error）；原生RDKit PNG/SVG、真实光谱轴与完整metadata；独立验证PASS 263/263。
- [x] **B5**：完成Phase 0预检、16节最终报告、19表、11图、88案例归档（388文件/12,643,790 B）；独立验证PASS 67/67。最终结论：H1 PARTIAL、H2 REJECTED、H3 PARTIAL、H4 DESCRIPTIVELY SUPPORTED。
- [x] **Git归档**：创建`archive/line-b-2026-11-20`分支并完成本地commit `ab1e85dd…`；1173新增文件/143,591,799 B，加6个已跟踪未修改文件后共1,179路径/143,780,373 B；已推送。

### 环境与工程

- [x] 修复原始`spectra-db`环境的RDKit原生绘图故障：根因为`libglib`/`libintl`共享`Library/bin/intl-8.dll`；仅更新`libintl`至`0.22.5 h5728263_4`后恢复，`rdMolDraw2D`、Cairo PNG、SVG与项目样本渲染全部通过。
- [x] 记录环境约束：禁止对`libglib`执行force-reinstall/install/update；禁止未经评估升级`libffi`或切换Python构建；相关conda操作前必须dry-run并复核`intl-8.dll` SHA256。
- [x] 完成工作区外历史资产迁移与重复运行归档：1739文件/190,543,006 B迁入工作区；803文件/46,747,421 B归档至`archive/2026-11-19/line-b-relocation-duplicates/`。
- [x] 完成B3.5-D2 R2工作区原生化与路径安全补丁：R1保持不可变，R2保持669文件，Builder/Verifier支持工作区内输出、中文路径与历史路径解析。

---

## 当前最高优先级

### Priority A：UV专属语义与单模态基线（Line A，已完成）

- [x] 标注schema、UVSPEC15标签、三seed基线、shared vs specific对照与Final Decision均已完成。
- [ ] 可选人工复核：UVSPEC15关键标签（共轭体系、发色团、芳香N、供受体）的化学合理性抽样检查；不阻塞Line C。

### Priority B：三模态信息画像（Line B，B0–B5，已完成）

- [x] B0–B5全部完成并通过独立验收；Test结果按exploratory/post-hoc定位，不作为confirmatory evidence。
- [x] 三模态结论：IR/Raman在Shared14与UVSPEC15标签上均强于UV-Vis；UV-Vis在UVSPEC15上未表现出预期优势；错误集合存在差异，但条件互补与Positive OR不足以支持“UV带来稳定增量”。
- [x] 结构子群与典型分子分析已完成；典型分子仅为定性案例，不是XAI或统计证据。

### Priority C：Chemistry-aware融合（Line C，下一步，尚未开始）

- [ ] naive concatenation / fixed-weight三模态对照，显式复现IR+Raman vs IR+Raman+UV的negative transfer。
- [ ] generic attention/gating对照（无化学监督）。
- [ ] proposed chemistry-aware routing（modality embeddings + shared evidence + UV-specific chemical evidence + sample-dependent gate）。
- [ ] 关键消融：无化学监督 / 普通attention/MoE / 加UV专属化学监督 / 去掉UV标签监督。
- [ ] gate权重与化学标签/证据相关性分析。
- [ ] 融合结果必须与best_single比较，并使用scaffold-aware不确定性；不得仅以Positive OR作为融合优于单模态的依据。

---

## 后续任务

### 完整结构推断与Top-k（Line D）

- [ ] 从融合后的谱图表征生成/检索完整结构候选。
- [ ] 评估Recall@1/5/10、MRR、candidate ranking quality与scaffold/难度分层分析。
- [ ] 接入结构渲染（`qm9s_structure_tools.py render`）实现可视化输出。

### 化学规则reranking（Line E）

- [ ] Woodward–Fieser等规则或近似电子吸收一致性作为consistency score。
- [ ] 规则用于reranking/evidence，不硬编码最终答案。

### 缺失/噪声鲁棒性（Line F）

- [ ] modality dropout、missing modality实验。
- [ ] 强度噪声/基线漂移/峰展宽/波数扰动实验。
- [ ] 单模态低质量时gate自动降权验证。

### 实验域验证（Line G）

- [ ] 将NIST IR `.jdx`解析并重采样到统一网格，用master SMILES计算官能团标签。
- [ ] 清洗API-Raman数据（3,510条）并与SMILES匹配、对齐波数轴。
- [ ] zero-shot / fine-tune / domain adaptation对比，明确区分计算域与实验域结果。

### 开放集与可解释性（Line H/I）

- [ ] 置信度/不确定性、reject option、OOD/open-set指标。
- [ ] XAI：模态贡献、关键谱区/峰、结构证据、规则支持/反对、2D结构图标注。

### 归档与运维

- [ ] IR checkpoint仓库外独立备份（12份约140 MB）；Raman备份已决定跳过。
- [x] Line B归档分支`archive/line-b-2026-11-20`已推送（已完成）。
- [ ] 决定B5-PREFLIGHT（5文件）与`training/line_b/tests`（4文件）是否补入后续维护commit；本次研究归档未包含。
- [ ] B3.5-D2全量、C2/D1未选定大表与备份目录保留本地，不纳入研究归档。

---

## 暂停任务

以下任务当前暂停：

- [ ] GDB-9-Ex数据接入（备用外部域/方法迁移数据集，当前不参与训练，不删除原始数据）。
- [ ] QMe14S数据接入（已退出当前方案）。
- [ ] SDBS采集（已暂停，恢复需先小样本页面结构调试）。
- [ ] NIST大规模采集扩展。
- [ ] 大规模随机采样数据获取。
- [ ] 大规模随机采样master的使用。
- [ ] Morgan fingerprint作为光谱模型输入（已明确不使用）。

---

## 推荐执行顺序

### Step 1：Line C（当前重点）

- [ ] naive/fixed-weight对照 → generic gating → chemistry-aware routing → 消融。
- [ ] 以best_single为基线，补充scaffold-aware不确定性与gate可解释性检查。

### Step 2：Line D（融合证据成立后）

- [ ] 完整结构候选生成/检索、Top-k评估与结构渲染。

### Step 3：Line F/G

- [ ] 缺失/噪声鲁棒性；NIST/API-Raman实验域验证。

### Step 4：Line H/I

- [ ] 开放集识别、不确定性与可解释性。

进入Line D前必须满足：

- [ ] Line C的融合证据成立（相对best_single有稳定改进，gate可解释）。
- [ ] 用户明确确认。

---

## 环境状态（2026-11-19）

### RDKit原生绘图

- [x] 原始`spectra-db`环境已原地修复；`rdMolDraw2D`、Cairo PNG、SVG、Pillow解码与项目四个样本渲染全部通过。
- [x] repair环境与成功参照环境渲染结果逐字节一致；原始科研栈Python/RDKit/PyTorch/CUDA/PyG版本未变。
- 报告：`docs/environment/2026-11-19_rdkit_original_env_repair.md`。

### 硬性约束

- 禁止对`libglib`执行`--force-reinstall` / `install` / `update`：会把`g_libintl_*`版本的`intl-8.dll`写回，RDKit绘图立即退回WinError 127。
- 禁止未经评估升级`libffi`：conda-forge `libffi`无`ffi.dll`，会破坏`ctypes`/`torch`/`pandas`。
- 禁止未经评估把`python`切换到conda-forge构建：会级联`liblzma`/`xz`等。
- 任何涉及上述包的conda操作前必须dry-run，操作后复核`Library\bin\intl-8.dll` SHA256 = `19B198C14E5D54F3DD7B966F09458CC31566D9F2406C64FFC13B4386CE92EE6E`。
- 已知残留：`glib/gobject/gio/cairo-gobject`系列DLL不可加载（项目与RDKit绘图链均不使用）；`conda check`仍报告`libglib`/`liblzma` altered-file（共享路径固有，修复前即存在）。
- 回滚保护：`D:\Miniconda3\envs\spectra-db-pre-rdkit-fix`（修复前状态，5.83 GB）。
