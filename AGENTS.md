# SpectraDB Agent 工作说明 最近修改：18:55 8/16/2026

## 1. 项目定位

SpectraDB 是一个面向有机小分子多模态光谱分析与分子结构推断的数据库和模型建设项目。

项目目标是建立统一的多模态光谱数据库，并实现从光谱推断分子结构信息：以单模态 1D-CNN 模型（IR、Raman、UV-Vis）为基础，按各模态的化学证据分工进行 chemistry-aware 自适应融合，逐步输出官能团/结构特征、完整分子结构候选与分子结构可视化，最终实现模型可解释性分析。

官能团预测只是中间结构信息，不是最终目标。

核心关联标识包括：

- CID
- SMILES
- InChIKey
- CAS
- 分子式
- 数据源内部 ID

当前重点光谱模态包括：

- IR
- Raman
- UV-Vis

当前数据源包括：

- QM9S：IR、Raman、UV-Vis 主要计算训练数据（约 13 万分子）
- GDB-9-Ex：备用外部域/方法迁移数据集（约 9.7 万分子，激发态能量与振子强度），当前不参与训练
- NIST：实验域 IR、UV-Vis 数据，用于后续微调与测试
- API-Raman：实验 Raman 数据（约 3500 个化合物）

当前不使用：

- GDB-9-Ex 不参与当前主线训练（保留为备用外部域/方法迁移数据集）
- QMe14S（已退出当前方案）
- SDBS（已暂停）
- 大规模随机采样数据

---

## 2. 当前项目阶段

当前处于：

**第一阶段（数据体系与单模态模型）已完成；当前进入第二阶段准备（UV专属化学语义监督与 chemistry-aware 多模态融合）。**

第一阶段包括：

1. 建立 QM9S 多模态计算光谱数据。
2. 建立分子结构库。
3. 建立官能团、原子、键、环、共轭体系等结构标注。
4. 建立 scaffold split。
5. 训练 IR 1D-CNN。
6. 训练 Raman 1D-CNN。
7. 训练 UV-Vis 1D-CNN。
8. 比较不同光谱模态对结构信息的表征能力。

当前进度：

- QM9S 数据准备：已完成。
- 结构标注：已完成。
- scaffold split：已完成。
- 官能团多标签（14 类）：已完成。
- IR 1D-CNN：已完成，IR-v1 已冻结（test Micro-F1 0.9406，Macro-AUROC 0.9954）。
- Raman 1D-CNN：已完成，Raman-v1 已冻结（test Micro-F1 0.9151，Macro-AUROC 0.9911）。
- UV-Vis 1D-CNN：已完成，UV-Vis-v1 已冻结（test Micro-F1 0.4892，Macro-AUROC 0.8501）。
- 模态对比：待输出（三模态信息画像）。
- UV-specific 化学语义标签：下一步（当前最高优先级）。

---

## 3. 当前数据策略

当前默认采用：

**QM9S 计算光谱作为主要训练数据。**

当前训练数据组织：

1. 光谱矩阵：`processed/qm9s/prepared/{ir,raman,uvvis}_float32.npy`（float32，可 mmap）。
2. 数据划分：`scaffold_split_80_10_10.npz`（train/val/test，scaffold 零重叠）。
3. 模态有效性：`modality_valid_masks.npz` 及分模态 valid split。
4. 监督标签：`functional_group_labels.npy`（129817×14，官能团多标签）。
5. 结构库：`processed/qm9s/structures/`（原子、键、官能团、环、共轭体系标注）。

当前不使用：

- Morgan fingerprint 不作为光谱模型的输入。
- QMe14S 数据。
- 大规模随机采样数据。

实验域数据（NIST、API-Raman）用于第三阶段的微调、验证与迁移分析，当前不进入 QM9S 训练。

GDB-9-Ex 保留为备用外部域/方法迁移数据集，当前不参与主线训练，不删除原始数据。

---

## 4. 数据真实性原则

数据真实性优先级高于数据规模。

必须遵守：

1. 不允许把 HTML 页面、错误页、免责声明页面、空页面保存为 `.jdx` 光谱文件。
2. 不允许生成占位光谱文件并计为成功。
3. 不允许为了提高成功率而降低数据真实性标准。
4. 不确定是否成功时，应按失败处理。
5. 宁可少采，也不能采集伪数据。
6. 所有失败样本必须记录原因。
7. 所有跳过样本必须记录原因。
8. 所有成功样本必须可追溯到原始数据源。

对于 `.jdx` 文件，必须满足：

1. 文件非空。
2. 文件不是 HTML。
3. 文件至少包含以下 JCAMP-DX 标记之一：

   - `##JCAMP-DX=`
   - `##TITLE=`
   - `##DATA TYPE=`

4. 如果内容包含 `<html`、`<!DOCTYPE html>`、`<body` 等 HTML 标记，应判定为失败。
5. HTML 页面只能保存为 `.html`，不能保存为 `.jdx`。

---

## 5. 采集脚本通用要求

所有采集脚本必须支持：

1. 断点续跑。
2. 日志记录。
3. 失败记录。
4. 跳过记录。
5. Ctrl+C 安全退出。
6. 重试机制。
7. 命令行参数。
8. 小样本测试。
9. 输出统计汇总。
10. 每处理一个样本后立即保存进度。

所有采集脚本必须避免：

1. 未校验即保存文件。
2. 未命中却记为成功。
3. 把空文件记为成功。
4. 把错误页面记为成功。
5. 大规模运行前没有小样本验证。
6. 页面结构异常时继续盲目重试。
7. 使用不可复现的手动操作作为默认流程。

---

## 6. 调试原则

对于任何采集脚本，如果出现以下情况：

- 成功率异常低；
- 大量 skipped；
- 命中数与成功数严重不匹配；
- 页面解析失败；
- 输出文件格式不可信；
- 运行时间异常；
- 网站返回空白页、错误页或免责声明页；

必须立即停止扩大运行。

正确流程是：

1. 保存失败样本。
2. 保存页面 HTML。
3. 保存截图。
4. 记录当前 URL、页面标题、页面类型。
5. 分析真实页面结构。
6. 修复解析逻辑。
7. 用少量样本重新测试。
8. 成功后再扩大到 100、1000 或更大规模。

禁止在原因未明时继续扩大运行。

---

## 7. NIST 当前规则

NIST 当前定位：**实验域数据，用于后续微调与测试。**

当前状态：

- NIST Round 1 采集已完成并归档（`archive/nist_round1_2026-07-14/`）。
- 覆盖唯一 CID 6534 个，保存真实光谱 7340 条（IR 6154、UV-Vis 1186、Raman 0）。
- `raw/nist/nist_metadata.csv` 已生成。

NIST 成功标准：

1. `.jdx` 文件通过 JCAMP-DX 标记校验。
2. 每个成功文件都有对应 metadata。
3. 失败和跳过均有明确原因。

NIST 禁止事项：

1. 不允许把 NIST 普通 HTML 页面保存成 `.jdx`。
2. 不允许把页面锚点误当作下载链接。
3. 不允许跳过校验直接保存。
4. 不允许清理旧伪 `.jdx` 文件，除非用户明确确认。

后续工作（进入第三阶段前）：

- 将 NIST `.jdx` 解析并重采样到与 QM9S 一致的统一网格。
- 用 master SMILES 计算官能团标签，作为实验域测试/微调集。

---

## 8. SDBS 当前规则

SDBS 当前已暂停，不作为数据源。

- 大规模随机测试中覆盖率和成功率极低。
- 当前不应继续扩大运行。

如果未来需要恢复 SDBS，必须：

1. 只针对少量已命中的 CAS 样本调试页面结构。
2. 能区分 disclaimer、home、search、result list、compound detail、zero hit、error。
3. 能识别真实 IR / Raman / UV-Vis 入口。
4. 不能把 0 hit 页面计为 result page。
5. 不能把占位文件计为成功。
6. 禁止直接跑 1000 或全量。

---

## 9. CAS 缓存规则

NIST 等数据源可能依赖 CAS。

应建立或复用共享缓存：

- `cache/cid_cas.csv`

建议字段：

- `cid`
- `cas`
- `source`
- `status`
- `updated_at`
- `message`

规则：

1. 已查询过的 CID 不应重复请求 PubChem。
2. 无 CAS 的 CID 也应缓存。
3. 查询失败和 CAS 不存在必须区分。
4. 缓存写入必须兼容 Windows。
5. 写入时应避免文件锁导致程序崩溃。
6. `os.replace` 失败时应自动重试。
7. 单个缓存写入失败不应导致整个采集任务崩溃。
8. 如果缓存被 Excel、VS Code 或其他进程占用，应记录错误并安全处理。

---

## 10. 项目目录说明

主要目录：

- `scripts/`  
  数据获取与准备脚本。

- `training/`  
  模型训练代码（模型、数据集、训练与汇总脚本）。

- `runs/`  
  训练实验与结果（指标、图表、Markdown 总结）。

- `master/`  
  分子索引表。默认保留在项目中。

- `raw/`  
  原始数据。默认不扫描，不提交 Git。

- `processed/`  
  处理后数据集（NPY 矩阵、split、结构库）。默认不扫描，不提交 Git。

- `paired/`  
  配对数据。当前为空。

- `metadata/`  
  数据检查记录与报告。

- `cache/`  
  缓存与断点。默认不扫描，不提交 Git。

- `logs/`  
  运行日志、失败日志。默认不扫描，不提交 Git。

- `archive/`  
  旧测试产物、历史输出。默认不扫描，不提交 Git。

重要文件：

- `AGENTS.md`  
  Agent 和开发者的项目规则。

- `TODO.md`  
  当前任务列表。

- `CHANGELOG.md`  
  项目修改记录。

- `README.md`  
  项目对外说明。

- `.gitignore`  
  Git 忽略规则。

- `.codexignore`  
  Codex 忽略规则。

---

## 11. 重要脚本说明

### 数据准备脚本

### `scripts/build_master_in_order.py`

用于按顺序从 PubChem 构建 compound master。

### `scripts/clean_master.py`

用于清洗 master，主要去除：

- 同位素记录
- 多组分记录

### `scripts/harvest_nist.py`

用于从 NIST WebBook 获取实验光谱数据。

当前定位：实验域数据来源，Round 1 已完成。

### `scripts/build_qm9s_manifest.py`

从 `raw/qm9s/qm9s.pt` 与光谱 CSV 构建 QM9S manifest（129817 个样本）。

### `scripts/prepare_qm9s_dataset.py`

将 IR、Raman、UV-Vis 光谱 CSV 转为 float32 NPY，完成质量检查，并生成 Bemis–Murcko scaffold split。

### `scripts/build_qm9s_valid_splits.py`

生成模态有效性 mask 与分模态有效 split。

### `scripts/qm9s_structure_tools.py`

建立 QM9S 结构库（原子、键、官能团、环、共轭体系标注），提供结构渲染命令。

### `scripts/prepare_functional_group_labels.py`

从结构库生成官能团多标签矩阵（129817×14）。

### 训练脚本

### `training/models/spectrum_cnn1d.py`

可复用 1D-CNN 光谱 encoder（`SpectrumEncoder1D`）与结构特征训练头。

### `training/datasets/qm9s_spectrum_dataset.py`

QM9S 单模态光谱 Dataset（mmap 读取 NPY，支持归一化）。

### `training/train_ir_*.py`

IR 1D-CNN 训练脚本（每个随机种子一份脚本）。

### `training/train_raman_*.py`

Raman 1D-CNN 训练脚本（每个随机种子一份脚本）。

### `training/train_uvvis_*.py`

UV-Vis 1D-CNN 训练脚本（每个随机种子一份脚本）。

### `training/calibrate_ir_thresholds.py` / `training/finalize_ir_results.py`

阈值校准与结果汇总脚本。

---

## 12. Codex 工作方式

请把自己当成 SpectraDB 项目的开发者，而不是代码生成器。

每次接受任务时，应遵循：

1. 先阅读 `AGENTS.md`。
2. 再阅读 `TODO.md`。
3. 再阅读 `CHANGELOG.md`。
4. 只阅读相关代码。
5. 不扫描大数据目录。
6. 先分析问题。
7. 给出最小修改方案。
8. 修改代码。
9. 运行小规模测试。
10. 如果测试失败，自行定位并继续修复。
11. 测试通过后再建议扩大规模。
12. 更新 `CHANGELOG.md`。
13. 更新 `TODO.md`。
14. 输出中文总结。

不要频繁等待用户确认，除非涉及：

- 删除数据；
- 覆盖已有 raw 数据；
- 改变项目方向；
- 大规模重构；
- 新增重量级外部依赖；
- 长时间全量运行；
- 运行可能消耗大量时间或网络资源的任务。

---

## 13. 禁止默认扫描的目录和文件

除非用户明确要求，不要扫描或全文读取：

- `raw/`
- `processed/`
- `data/`
- `downloads/`
- `logs/`
- `cache/`
- `archive/`
- `runs/` 中的大文件
- 大型 `.csv`
- 大型 `.npy` / `.npz`
- 模型 checkpoint（`*.pt` / `*.pth`）
- 大量 `.html`
- 大量 `.jdx`
- 图片文件
- 压缩包
- `__pycache__/`
- `.venv/`
- Conda 环境目录

可以读取这些目录的文件列表，但不要批量读取文件内容。

---

## 14. Git 与归档规则

Git 中保留：

- 代码
- 配置
- 实验指标
- 图表
- Markdown 总结

Git 中不保存：

- `raw/` 大型数据
- `processed/` 大型数据
- `.npy` / `.npz`
- 模型 checkpoint（`*.pt` / `*.pth` / `*.ckpt`）
- 大型 master CSV
- 大型 inspection 文件
- `cache/`、`logs/`、`archive/`

`master/` 默认保留在项目中（当前 master CSV 由 `/master/*.csv` 忽略规则管理，按实际需要调整）。

归档旧数据时：

1. 不删除文件。
2. 移动到 `archive/YYYY-MM-DD/description/`。
3. 更新 `.gitignore`。
4. 更新 `.codexignore`。
5. 输出移动清单。
6. 不移动 `master/`、`scripts/`、`README.md`、`AGENTS.md`、`TODO.md`、`CHANGELOG.md`。

---

## 15. 推荐测试流程

任何脚本修改后，默认测试顺序：

1. 静态检查。
2. 小样本测试（采集脚本前 10 个样本；训练脚本先 sanity 小规模）。
3. 检查输出文件真实性。
4. 检查日志。
5. 检查 checkpoint。
6. 汇总成功、失败、跳过数量。
7. 分析失败原因。
8. 决定是否扩大规模。

训练脚本禁止跳过 sanity 测试直接全量训练。

---

## 16. 当前最高优先级

当前最高优先级如下：

1. 更新 `AGENTS.md` 和 `TODO.md`，明确当前阶段策略。
2. 设计并冻结 UV-specific label taxonomy，编写 `scripts/build_uvvis_specific_labels.py` 自动生成标签（Line A）。
3. UV-specific 单模态基线 + shared vs specific 对照，验证"UV 被问对问题后是否显著变强"（Line A）。
4. 三模态信息画像，识别各模态擅长与不擅长的结构子群（Line B）。
5. chemistry-aware 融合：naive 对照 → generic gating → proposed routing + 消融（Line C，论文核心方法）。
6. 完整结构 Top-k：融合证据成立后进入（Line D）。
7. 鲁棒性与实验域：缺失/噪声模态、NIST/API-Raman 验证（Line F/G）。
8. 开放集与可解释性分析（Line H/I）。

---

## 17. 长期路线图

### 第一阶段：数据体系与单模态模型

目标：

- 建立 QM9S 多模态计算光谱数据。
- 建立分子结构库与结构标注。
- 建立 scaffold split。
- 训练 IR、Raman、UV-Vis 三个单模态 1D-CNN。
- 比较不同光谱模态对结构信息的表征能力。

### 第二阶段：多模态融合与完整结构推断

目标：

- UV-Vis 专属共轭/发色团化学语义监督（UV-specific labels）。
- IR、Raman、UV-Vis 化学证据分工的 chemistry-aware 融合。
- 综合官能团与关键结构特征。
- 完整分子结构候选推断与 Top-k 排序。
- 二维分子结构可视化。
- Woodward 等化学规则辅助验证和候选重排序。

完整结构推断的具体实现方式不写死，可概括为：

- candidate retrieval
- molecular graph generation
- SMILES / SELFIES generation
- retrieval + generation

### 第三阶段：实验域、开放集与可解释性

目标：

- NIST、API-Raman 实验数据微调与验证。
- calculated spectra → experimental spectra 迁移。
- 未知样品识别、开放集识别、置信度与拒识。
- XAI、光谱区域与结构片段对应。
- 最终结果界面与自然语言解释。

---

## 18. 完成任务后的默认行为

完成任何开发任务后，应自动：

1. 仿照之前的语言用中文更新 `CHANGELOG.md`。
2. 用中文更新 `TODO.md`。
3. 输出修改摘要。
4. 输出测试结果。
5. 输出失败原因。
6. 输出下一步建议。

除非用户明确要求，否则无需再次询问是否更新文档。
