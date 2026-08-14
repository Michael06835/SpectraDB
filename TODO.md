# SpectraDB TODO

## 当前阶段

**第一阶段：数据体系与单模态模型。**

当前目标不是追求数据规模，而是建立稳定的 QM9S 多模态数据集，完成 IR、Raman、UV-Vis 三个单模态 1D-CNN 基线，并比较不同光谱模态对结构信息的表征能力。

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
- [x] 完成类别不平衡消融实验与阈值校准实验（均保留 IR-v1 默认设置）。
- [x] 更新 `README.md`、`CHANGELOG.md`、`AGENTS.md`、`TODO.md`。
- [x] 更新 Git 忽略规则与环境锁定文件。

---

## 当前最高优先级

### Priority A：Raman 1D-CNN 单模态基线

- [ ] 使用 `raman_float32.npy` + `raman_scaffold_split_valid.npz` + `functional_group_labels.npy` 训练。
- [ ] 沿用 IR-v1 约定：seed 42（canonical）+ 123/2026、max 归一化、阈值 0.5、BCE、早停。
- [ ] 完成三随机种子测试。
- [ ] 冻结 Raman-v1，输出 `runs/Raman-v1-FINAL/` 汇总（指标、图表、Markdown 总结）。

### Priority B：UV-Vis 1D-CNN 单模态基线

- [ ] 使用 `uvvis_float32.npy`（701 点）+ `uvvis_scaffold_split_valid.npz` 训练。
- [ ] 注意 UV-Vis 轴域（1–15 eV）与 IR/Raman（cm⁻¹）不同，输入长度走配置。
- [ ] 完成三随机种子测试。
- [ ] 冻结 UV-Vis-v1，输出 `runs/UVVIS-v1-FINAL/` 汇总。

### Priority C：模态对比

- [ ] 比较 IR、Raman、UV-Vis 三模态在相同标签体系下的 Micro-F1 / Macro-F1 / mAP / Macro-AUROC。
- [ ] 分析各模态的弱类别差异。
- [ ] 输出模态对比总结。

---

## 后续任务

### 光谱→结构候选

- [ ] 用 `SpectrumEncoder1D` embedding + 结构库建立 Top-k 结构候选检索原型。
- [ ] 评估检索命中率（Top-1/5/10）。
- [ ] 接入结构渲染（`qm9s_structure_tools.py render`）实现可视化输出。

### 实验域数据准备

- [ ] 将 NIST IR `.jdx` 解析并重采样到与 QM9S 一致的统一网格。
- [ ] 用 master SMILES 计算 NIST 数据的官能团标签。
- [ ] 清洗 API-Raman 数据（3,510 条）并与 SMILES 匹配、对齐波数轴。
- [ ] 建立实验域测试/微调集划分协议。

### GDB-9-Ex

- [ ] 将 96,731 个分子的激发态数据（ex1–50/prob1–50）合成为 UV-Vis 吸收光谱。
- [ ] 对齐到统一 eV 网格（注意激发态最高到约 28 eV，需确定窗口）。
- [ ] 核对与 QM9S 的分子重叠并去重。
- [ ] 明确其用途（UV-Vis 增强训练或实验域分析）。

### 多模态融合与可解释性

- [ ] IR、Raman、UV-Vis 特征融合（单模态全部稳定后进行）。
- [ ] 综合官能团与关键结构特征。
- [ ] 完整分子结构候选推断与 Top-k 排序。
- [ ] 二维分子结构可视化。
- [ ] Woodward 等化学规则辅助验证和候选重排序。
- [ ] XAI 与光谱区域-结构片段对应分析。
- [ ] 开放集识别、置信度与拒识。

---

## 暂停任务

以下任务当前暂停：

- [ ] QMe14S 数据接入（已退出当前方案）。
- [ ] SDBS 采集（已暂停，恢复需先小样本页面结构调试）。
- [ ] NIST 大规模采集扩展。
- [ ] 大规模随机采样数据获取。
- [ ] 大规模随机采样 master 的使用。
- [ ] Morgan fingerprint 作为光谱模型输入（已明确不使用）。

---

## 推荐执行顺序

### Step 1：文档与规则收尾

- [x] 更新 `AGENTS.md`、`TODO.md`，明确当前阶段策略。

### Step 2：Raman 单模态基线

- [ ] 训练并冻结 Raman-v1。

### Step 3：UV-Vis 单模态基线

- [ ] 训练并冻结 UV-Vis-v1。

### Step 4：模态对比

- [ ] 输出三模态表征能力对比总结。

### Step 5：光谱→结构候选

- [ ] 建立 Top-k 结构候选检索原型与可视化。

### Step 6：实验域验证

- [ ] NIST、API-Raman 数据准备与验证。

### Step 7：决定是否进入第二阶段

进入第二阶段（多模态融合与完整结构推断）前必须满足：

- [ ] IR、Raman、UV-Vis 三个单模态基线全部冻结。
- [ ] 模态对比分析完成。
- [ ] 结构候选检索原型可用。
- [ ] 用户明确确认。
