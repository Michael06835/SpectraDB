# Line B B3 — Pre-FREEZE Readiness Audit

> **审计类型**：只读审计 + 既有 CPU/合成证据核对  
> **审计日期**：2026-11-18  
> **审计依据**：`docs/experiments/line_b_protocol_draft.md` v0.2、B1 报告、当前 B3 prediction/threshold/manifest/B3-A 产物  
> **Test 状态**：未访问 Test split，未读取 Test 索引，未构造 Test DataLoader，未执行 Test 推理  
> **修改状态**：本轮未修改任何预测、阈值、checkpoint、manifest、历史指标或分析结果；仅生成本报告  
> **FREEZE 状态**：未生成 `FREEZE.json`

---

## 1. 总览

| 审计范围 | 状态 | 结论 |
|---|---|---|
| 18 个 Validation prediction 资产 | **PASS** | 9 Shared14 + 9 UVSPEC15 全部通过 |
| 18 个 threshold/calibration 资产 | **PASS** | Shared14 126 + UVSPEC15 135 条记录 |
| B3-A 正式产物 | **NEEDS_REVIEW** | 6 CSV + summary 全部存在、哈希通过；缺分析代码 SHA |
| B4 分析代码 | **MISSING / NEEDS_REVIEW** | 5 个脚本中 4 个缺失；complementarity 仅部分实现 B4 |
| FREEZE 资产 | **MISSING** | 无 `FREEZE.json`、无 `verify_line_b_manifests.py` |
| 数据/阈值哈希失败 | **FAILED = 0** | 未发现实际数据哈希不一致 |
| 历史 B1 code hash | **NEEDS_REVIEW** | 历史 B1 manifest 的 exporter/common hash 与当前代码不同，属于历史版本差异 |

**当前结论：B3 Validation 数据与阈值输入完整；B3-A 结果本身哈希完整，但分析代码版本未记录；B4 分析代码与 FREEZE 工具链尚未就绪。**

---

## 2. Validation 资产审计

### 2.1 覆盖

| Task | Modalities | Models | Labels | Seeds |
|---|---|---:|---:|---|
| Shared14 | IR / Raman / UV-Vis | 9 | 14 | 42 / 123 / 2026 |
| UVSPEC15 | IR / Raman / UV-Vis | 9 | 15 | 42 / 123 / 2026 |

### 2.2 逐模型核验结果

18/18 通过：

- checkpoint 文件实际 SHA256 与 prediction manifest 一致；
- `config.seed` 存在且与注册 run ID 推导的 Seed 一致；
- probabilities / row_index / labels 实际 SHA256 与 manifest 一致；
- `row_index` 与权威 Validation split 逐元素一致；
- labels 与权威标签矩阵按 `row_index` 取值一致；
- label_names 与权威标签名称文件一致；
- prediction manifest 的 `split=validation`；
- prediction manifest 的 `test_split_accessed=false`；
- checkpoint 1 份、prediction 1 份、threshold 1 份、calibration manifest 1 份均存在；
- 全部模型共享同一 Validation 行索引；
- 全部模型使用 Original BN、普通 BCE、threshold 0.5 主评价。

### 2.3 Shared14 UV Seed2026 纠错核查

**PASS。**

- run：`EXP-UVVIS-001-SEED2026`
- checkpoint SHA256：
  `9e2b0dd81cb5319e5c4f1aa6f2bb550bf23f6b6a3045e478ac1aa2f97f88e01d`
- checkpoint `config.seed` = **2026**
- 不是归档的旧 invalid Seed42 checkpoint（`087bf...`）。

### 2.4 Threshold 资产

| Task | Threshold 记录 | 状态 |
|---|---:|---|
| Shared14 | 126 | PASS |
| UVSPEC15 | 135 | PASS |
| **合计** | **261** | **PASS** |

18/18 模型通过：

- `optimal_thresholds.json` 实际 SHA256 与 `calibration_manifest.json` 的 `outputs.optimal_thresholds` 一致；
- 与 `LINE-B-THRESHOLDS/<Task>/summary.json` 的 `thresholds_sha256` 一致；
- threshold JSON 的 probabilities/row_index/labels/checkpoint SHA256 与实际输入一致；
- threshold 标签顺序与权威标签顺序一致；
- `test_split_accessed=false`；
- calibration manifest 的 `all_checks_pass=true`；
- 261 条阈值记录、126/135 分任务计数正确；
- 单类/零分母标签保留 `None`/空统计，不写成 0。

**Validation / threshold 资产未发现 FAILED。**

---

## 3. B3-A 正式产物审计

### 3.1 文件完整性

每个任务存在 7 个文件：

```text
runs/LINE-B-ANALYSIS/Shared14/complementarity/
  model_metrics.csv
  pairwise_metrics.csv
  three_modality_metrics.csv
  model_mean_sd.csv
  pairwise_mean_sd.csv
  three_modality_mean_sd.csv
  summary.json
```

UVSPEC15 同结构。共 **14 个正式文件**。

### 3.2 行数与覆盖

| Task | model_metrics | pairwise_metrics | three_modality_metrics | model_mean_sd | pairwise_mean_sd | three_mean_sd |
|---|---:|---:|---:|---:|---:|---:|
| Shared14 | 252 | 504 | 84 | 588 | 3360 | 1232 |
| UVSPEC15 | 270 | 540 | 90 | 630 | 3600 | 1320 |

覆盖：

- 3 Seeds：42/123/2026；
- 2 threshold policies：`fixed_0.5`、`validation_f1`；
- 3 modalities；
- 6 个有序模态方向；
- 全部 14/15 标签；
- `n_error_jaccard_undefined = 0`；
- `n_undefined_seeds` 字段存在，单 Seed SD 为空而不是 0；
- 实际 undefined 统计：Shared14 的 `three_modality_mean_sd.csv` 有 5 行、UVSPEC15 有 2 行包含 undefined Seed；对应 mean/SD 使用空值或有效 Seed 的 sample SD，没有把 undefined 写成 0；
- 全部 12 个输出 CSV 的 SHA256 与 summary 记录一致。

### 3.3 输出 hash

summary 的 `outputs` 记录 6 个 CSV SHA256；实际重新计算全部一致：

- Shared14：6/6 PASS；
- UVSPEC15：6/6 PASS。

`summary.json` 本身未自哈希；其 outputs 覆盖六个 CSV。

### 3.4 输入 provenance

B3-A summary 记录的：

- 18 个 prediction manifest SHA256；
- 18 个 probabilities/row_index/labels 数组 SHA256；
- 18 个 threshold JSON SHA256；
- 18 个 checkpoint SHA256；
- frozen B3 metrics CSV SHA256；

均与当前实际文件一致。18/18 条模型的 `row_index`、labels、label_names 跨模型一致。

### 3.5 分析代码版本

**MISSING / NEEDS_REVIEW。**

- 两个 B3-A `summary.json` 都没有 `code` 或 `code_sha256` 字段；
- 当前 `training/line_b/analyze_line_b_complementarity.py`：
  - 大小：44,305 bytes；
  - SHA256：`f0ff92d8f7f5e67df24b7b03178adbdda9d92eb99e885f3b1b574d5c84bedb4c`
- 无法从正式产物中证明其确实由该版本代码生成；
- FREEZE 要求“分析代码 SHA256”，因此这是冻结前的 **MISSING** 项，不是数据 FAILED。

当前关键代码文件实际 SHA256：

| 文件 | SHA256 |
|---|---|
| `training/line_b/analyze_line_b_complementarity.py` | `f0ff92d8f7f5e67df24b7b03178adbdda9d92eb99e885f3b1b574d5c84bedb4c` |
| `training/line_b/calibrate_line_b_thresholds.py` | `2706d0713f43467356834f9173c41f8a436472bdb3dfcae826ed3f5b0adb8c74` |
| `training/line_b/export_line_b_predictions.py` | `c98245edbd1859664f5e684db5d677e96b4af52e6c7ca36ef50736fec762dec7` |
| `training/line_b/common.py` | `5738a057d916ad374e010153b7008e48032f0339add7865a27f5f2a3ba1665e8` |
| `training/line_b/evaluate_line_b_multilabel.py` | `c0589397e39f123e476b1d11c59c06877c66c69b7517aa88cd556ee595e59cfd` |
| `training/line_b/summarize_b1_exports.py` | `44b877fef9ff4ddbfb05621301e39954192205fe5d4e90aec63113caa07fa55e` |

这些仅为当前工作树哈希，不是 B3-A/threshold 正式产物中的历史执行版本记录。

**B3-A 结果本身：PASS（文件、哈希、覆盖、数学字段）；代码版本：NEEDS_REVIEW。**

---

## 4. B4 分析代码就绪度

### 4.1 逐脚本状态

| 脚本 | 状态 | 已实现 | 缺失 | 输入/输出 | Test 隔离 | CPU 合成测试 |
|---|---|---|---|---|---|---|
| `bootstrap_line_b_scaffold.py` | **MISSING** | 无文件 | paired scaffold 2,000 次重采样、同一重采样索引、3 Seed 配对、95% percentile CI、AP/AUROC undefined 计数与跳过 | 输入应为 prediction/row_index/labels/split；输出应为 Bootstrap CI/undefined 汇总 | 未实现，无 Test 门禁 | 0 |
| `analyze_line_b_complementarity.py` | **PARTIAL / NEEDS_REVIEW** | TP/TN/FP/FN、错误交集、FN rescue、FP correction、new FP/FN、positive OR、真实 Oracle、条件率、双/三模态 Oracle 差值、三 Seed 汇总 | McNemar 精确检验、Holm 校正、逐样本误差分布、Spearman、hard sample top 10%、高置信错误、结构子群、典型分子、B4 Test 门禁 | 输入 Validation prediction/threshold/metrics；输出 B3-A CSV/summary | 仅 Validation；无 `--require-freeze` | 开发期外部合成检查，无仓库测试 |
| `analyze_line_b_structure.py` | **MISSING** | 无文件 | 训练集标签共现、预定义结构子群、子群最小支持度、典型分子清单与渲染规则 | 输入训练标签/结构 metadata；输出共现/子群/典型分子清单 | 未实现，无 Test 门禁 | 0 |
| `build_line_b_report.py` | **MISSING** | 无文件 | 18 模型表格、三模态指标、Bootstrap、互补性、结构子群、图表与最终报告 | 输入 B3/B4 汇总；输出报告/表格/图 | 未实现，无 Test 门禁 | 0 |
| `verify_line_b_manifests.py` | **MISSING** | 无文件 | 18 checkpoint/概率/阈值/标签/split 哈希链、FREEZE 校验、Test access log 门禁、失败即停 | 输入全部冻结资产；输出校验/退出码 | 未实现，无 Test 门禁 | 0 |

### 4.2 B4 Test 隔离与门禁

- `analyze_line_b_complementarity.py` 仅读取 Validation：
  - 不访问 Test；
  - 不加载模型；
  - 不重拟合阈值；
  - 但没有任何 `--require-freeze` 或 Test 门禁。
- `export_line_b_predictions.py` 当前 `--split` 只允许 `validation`；**尚未实现 B4 Test 导出**：
  - 无 `--split test`；
  - 无 `--require-freeze`；
  - 无 `test_access_log.json` 生成；
  - 不能作为 B4 Test 导出入口。
- 当前不存在 `FREEZE.json`，因此 B4 Test 前置门禁未建立。

### 4.3 CPU 合成测试覆盖

- 没有 `tests/` 目录或独立测试文件；
- `analyze_line_b_complementarity.py` 在开发阶段有合成数学检查（包括此前 100 组随机合成恒等关系），但这些检查没有被保存为可重复的仓库测试；
- 其余四个 B4 脚本完全缺失，CPU 合成测试覆盖率为 **0**；
- 因此 B4 代码测试覆盖状态为 **NEEDS_REVIEW / MISSING**，不能仅凭历史临时检查认定可冻结。

### 4.4 B4 关键协议项对照

| B0 v0.2 要求 | 当前状态 |
|---|---|
| scaffold 聚类配对重采样 2,000 次 | MISSING |
| 所有模型/Seed 使用同一重采样索引 | MISSING |
| Bootstrap undefined AP/AUROC 标记、计数、跳过 | MISSING |
| 逐标签 McNemar 精确检验 | MISSING |
| Holm 校正与校正前后 p 值 | MISSING |
| 逐样本误差分布、median/IQR、Spearman | MISSING |
| hard sample top 10% 与 Jaccard | MISSING |
| 高置信错误分析 | MISSING |
| 结构子群预定义与最小支持度 | MISSING |
| 典型分子选择规则 | MISSING |
| B4 Test 导出 FREEZE 门禁 | MISSING |
| `test_access_log.json` | MISSING |

---

## 5. FREEZE 清单

B0 要求 `FREEZE.json` 包含：

| 项目 | 当前状态 |
|---|---|
| 协议 SHA256 | 协议文件存在，但尚未写入 FREEZE |
| 分析代码 SHA256 | **MISSING**：B3-A/threshold 产物未记录代码 SHA |
| 18 checkpoint SHA256 | 资产存在且已核验 |
| 18 validation probabilities SHA256 | 资产存在且已核验 |
| 18 threshold SHA256 | 资产存在且已核验（261 条） |
| 标签顺序 SHA256 | 标签文件存在且已核验 |
| split SHA256 | split 存在且已核验 |
| 冻结时间戳 | **MISSING**：未生成 FREEZE |
| `FREEZE.json` | **MISSING** |
| `verify_line_b_manifests.py` | **MISSING** |

### 5.1 其他哈希/版本问题

1. **B3-A 分析代码未记录 SHA**：两个 B3-A summary 无 `code` 字段。
2. **Threshold calibration 未记录代码 SHA**：threshold summary 与 calibration manifest 均无 `code` 字段。
3. **B1 Shared14 历史 prediction manifest 的 code hash 与当前代码不同**：
   - `exporter_sha256`、`common_sha256` 为 B1 历史版本；
   - dataset/model hash 仍一致；
   - 不应重写历史 manifest；FREEZE 必须显式区分“历史执行代码版本”和“当前分析代码版本”。
4. **UVSPEC15 prediction manifests 的 code hashes 当前一致**；
5. **无 `test_access_log.json`**：B4 尚未开始，不是 B3 数据错误，但 FREEZE/B4 门禁未建立。

---

## 6. 状态分类汇总

### PASS

- 18/18 Validation checkpoint / config.seed / probabilities / row_index / labels / manifest；
- 18/18 label order / split alignment / Test isolation；
- Shared14 UV Seed2026 使用修正的 Seed2026 checkpoint；
- 261/261 threshold records；
- 18/18 threshold SHA 与 calibration manifest/summary 对齐；
- B3-A 14/14 正式文件存在；
- B3-A 12/12 输出 CSV 哈希一致；
- B3-A provenance 输入哈希全部一致；
- B3-A seed/policy/label 覆盖完整；
- 未发现数据或阈值哈希 FAILED。

### MISSING

- `training/line_b/bootstrap_line_b_scaffold.py`
- `training/line_b/analyze_line_b_structure.py`
- `training/line_b/build_line_b_report.py`
- `training/line_b/verify_line_b_manifests.py`
- `FREEZE.json`
- B4 Test 导出入口与 `test_access_log.json`
- B3-A / threshold calibration 分析代码 SHA 记录
- 可重复保存的 B4 CPU 合成测试

### FAILED

- **无数据/阈值/预测哈希失败。**
- 未发现 B3-A 输出哈希失败。
- 未发现 Test 访问。

### NEEDS_REVIEW

- B3-A 正式输出是否由当前 `analyze_line_b_complementarity.py` 版本生成：**无法证明**（summary 无 code SHA）。
- B1 历史 manifest code hash 与当前代码不同：历史版本差异，需在 FREEZE 中明确声明，不得静默替换。
- `analyze_line_b_complementarity.py` 是否满足 B4 的 McNemar/Holm/逐样本/结构子群要求：**不满足**，仅部分实现。
- B4 缺失脚本的输入输出与测试覆盖：**未定义/未实现**。

---

## 7. 最小必要修改清单

按优先级：

1. **补代码 provenance**
   - 为 B3-A 两个 `summary.json` 增加当前 `analyze_line_b_complementarity.py` SHA256；
   - 为 threshold summary / calibration manifest 增加 `calibrate_line_b_thresholds.py` SHA256；
   - 或在 FREEZE 顶层单独记录当前代码版本，但必须能追溯到 B3-A/threshold 产物。

2. **实现 `verify_line_b_manifests.py`**
   - 校验 18 checkpoint、18 probabilities、18 threshold、标签、split、B3-A 输出；
   - 校验 B1 历史 code hash 的例外声明；
   - 失败即停，不自动修复；
   - 提供 `--freeze` 的 dry-run / 生成前校验路径。

3. **实现 B4 Bootstrap**
   - `bootstrap_line_b_scaffold.py`；
   - 2,000 次 scaffold 聚类配对重采样；
   - 同一重采样索引跨模型/Seed；
   - AP/AUROC undefined 计数与跳过；
   - 95% percentile CI；
   - 固定 RNG。

4. **补齐 complementarity 的 B4 部分**
   - McNemar 精确检验与 Holm 校正；
   - 逐样本误差数分布、median/IQR、Spearman；
   - hard sample top 10%；
   - 高置信错误；
   - 结构子群与典型分子入口。

5. **实现 `analyze_line_b_structure.py`**
   - 训练集标签共现；
   - 预定义结构子群；
   - 最小支持度；
   - 典型分子选择规则与渲染清单。

6. **实现 `build_line_b_report.py`**
   - 只读汇总；
   - 表格、图、最终报告；
   - 不重新训练/推理。

7. **B4 Test 门禁**
   - 扩展 `export_line_b_predictions.py` 的 Test 模式；
   - 强制 `FREEZE.json` 哈希匹配；
   - 每模型仅一次；
   - 写 `test_access_log.json`。

8. **CPU 合成测试**
   - 覆盖 Bootstrap 配对/undefined、McNemar/Holm、逐样本 hard sample、结构子群、报告输入、FREEZE 失败门禁；
   - 不新增独立实验文档，只保留必要测试与代码。

---

## 8. 建议执行顺序

```text
1. 补 B3-A / threshold 代码 SHA 与 FREEZE 代码版本声明
2. 实现并运行 verify_line_b_manifests.py 的只读/dry-run 校验
3. 实现 Bootstrap + McNemar/Holm + 逐样本/结构/报告脚本
4. 完成 B4 CPU 合成测试
5. 重新核验 B3-A 输出（不重算、不覆盖）
6. 生成 FREEZE.json
7. 才允许进入 B4 Test 导出
```

---

## 9. 最终判断

**B3 Validation 数据与阈值资产：PASS。**  
**B3-A 正式结果：文件/哈希/覆盖 PASS；代码版本 provenance MISSING / NEEDS_REVIEW。**  
**FREEZE：未就绪，MISSING `FREEZE.json`、`verify_line_b_manifests.py` 和完整代码 SHA 链。**  
**B4：未就绪，4/5 指定脚本缺失，complementarity 仅部分实现；不得进入 B4。**

本轮未访问 Test、未训练、未重新推理、未重拟合阈值、未修改任何冻结资产、未生成 FREEZE、未 commit/push。
