# CHANGELOG

## 2026-06-13
### 完成内容
* 初始化SpectraDB项目目录
* 配置Python环境
* 配置Conda环境
* 创建`spectra-db`虚拟环境
* 安装RDKit
* 完成开发环境测试
### 当前进度
* 项目框架已建立
* 数据库设计方案已确定
### 下一步计划
* 测试`compound_master.csv`分子信息文档能否建立
* 获取2000–5000个有机小分子基础信息以进行测试
* 编写与测试数据库构建程序
* 完成`Master Database`数据库第一版构建

## 2026-06-18
### 完成内容
* 编写与测试`Master Database`第一版构建程序
* 完成基于PubChem API的顺序取样数据库构建流程验证
* 测试100、5000、50000、100000条顺序取样数据库的生成
* 验证RDKit分子标准化流程
* 验证分子结构标识符及分子量等属性可以自动生成
* 完成数据库质量检测流程设计与测试并确定清洗策略
* 针对顺序取样的CID偏倚性问题设计分层随机取样数据库构建方案并编写代码
### 当前进度
* `Master Database`构建流程已验证可行
* 顺序取样流程已通过验证
* 分层随机取样流程等待验证
* 已确定数据清洗规则为删除同位素与多组分记录
### 下一步计划
* 完成100000条分层随机取样数据库生成并建立`compound_master_100000_random.csv`
* 完成正式数据清洗
* 收集光谱数据

## 2026-06-19
### 完成内容
* 对分层取样流程进行验证
* 添加检查点模块
### 当前进度
* 发现程序容易因为无法kekulize分子而崩溃，添加try...except...模块与检查点模块
* 正在运行分层随机取样代码
### 下一步计划
* 完成100000条分层随机取样数据库生成并建立`compound_master_100000_random.csv`
* 完成正式数据清洗
* 收集光谱数据

## 2026-06-20
### 完成内容
* 完成分层取样流程验证
* 生成索引数据库
* 完成数据库清洗
### 当前进度
* 索引数据库已搭建
* 数据已清洗
### 下一步计划
* 根据索引数据库收集单光谱数据

## 2026-07-01
### 完成内容
* 编写NIST光谱下载器
* 编写SDBS光谱下载器
* 以十万个随机取样分子为索引从下载光谱
### 当前进度
* 正在下载NIST上可获取光谱
### 下一步计划
* 从SDBS等渠道继续下载光谱

## 2026-07-02
### 完成内容
* 重写`scripts/harvest_nist.py`最小可靠版，修复真实JCAMP-DX下载链路
* 完成NIST前10个顺序样本测试，保存的`.jdx`均含真实JCAMP-DX标记
* 完成20个样本稳定性测试
* 更新`AGENTS.md`，明确Phase 1策略与数据真实性规则
### 当前进度
* NIST真实JCAMP-DX下载链路已打通
* 旧测试产物已归档到`archive/2026-07-02/`
### 下一步计划
* 运行100、1000个顺序样本测试

## 2026-07-03
### 完成内容
* 完成NIST 100个顺序样本测试
* 完成NIST 1000个顺序样本稳定性测试
* 更新`TODO.md`
### 当前进度
* NIST小样本测试全部通过
### 下一步计划
* 进行NIST全量采集

## 2026-07-14
### 完成内容
* 完成NIST Round 1全量采集
* 处理master化合物86295个，覆盖唯一CID 6534个（覆盖率7.57%）
* 保存真实光谱7340条：IR 6154、UV-Vis 1186、Raman 0
* 生成`raw/nist/nist_metadata.csv`（7340行）
### 当前进度
* NIST实验IR、UV-Vis数据就绪；NIST无Raman数据
### 下一步计划
* 归档Round 1产物
* 评估其他Raman数据源

## 2026-07-28
### 完成内容
* 归档NIST Round 1与旧测试产物到`archive/`
* 更新Git忽略规则，收敛仓库
* 导出环境锁定文件（`environment.yml`、`explicit-win-64.txt`、`requirements-pip-exact.txt`）
### 当前进度
* 仓库状态收敛，Phase 1采集基建完成
### 下一步计划
* 建立QM9S计算光谱训练数据

## 2026-07-30
### 完成内容
* 下载QM9S数据集（`qm9s.pt`及IR、Raman、UV-Vis计算光谱CSV）
* 下载GDB-9-Ex数据集（`gdb9_ex.csv`）
* 下载实验Raman数据（`raman_spectra_api_compounds.csv`等）
* 检查QM9S数据文件结构
### 当前进度
* 训练数据源就绪
### 下一步计划
* 构建QM9S manifest

## 2026-07-31
### 完成内容
* 编写并运行`scripts/build_qm9s_manifest.py`
* 生成QM9S manifest（129817个样本，无缺失SMILES、无重复编号）
* 验证IR、Raman、UV-Vis三个光谱CSV行数与manifest一致
### 当前进度
* QM9S索引表建立
### 下一步计划
* 将光谱CSV转换为NPY矩阵

## 2026-08-01
### 完成内容
* 编写并运行`scripts/prepare_qm9s_dataset.py`
* IR、Raman转为float32 NPY（129817×3501，500–4000 cm⁻¹）
* UV-Vis转为float32 NPY（129817×701，1–15 eV）
* 完成光谱质量检查（0非有限值、2条全零UV-Vis、129条强度离群）
* 生成Bemis–Murcko scaffold split（80/10/10，28501个scaffold，跨分区零重叠）
### 当前进度
* QM9S光谱矩阵与数据划分就绪
### 下一步计划
* 生成模态有效性mask

## 2026-08-02
### 完成内容
* 编写并运行`scripts/build_qm9s_valid_splits.py`
* 生成模态有效性mask，三模态完整配对129815个样本
* 编写并运行`scripts/build_qm9s_morgan_fingerprints.py`
* 生成Morgan指纹（radius=2，2048bit，全部有效）
* 修复RDKit环境，保留修复前后环境快照
### 当前进度
* 多模态配对数据就绪
### 下一步计划
* 建立分子结构库

## 2026-08-09
### 完成内容
* 编写并运行`scripts/qm9s_structure_tools.py`
* 建立QM9S结构库（129817个有效结构）
* 生成原子、键、官能团、环、共轭体系标注
* 提供结构渲染命令（带编号与官能团高亮）
### 当前进度
* 结构标注数据就绪
### 下一步计划
* 生成官能团多标签矩阵

## 2026-08-13
### 完成内容
* 编写并运行`scripts/prepare_functional_group_labels.py`
* 生成官能团多标签矩阵（129817×14）
### 当前进度
* IR基线训练标签就绪
### 下一步计划
* 训练IR 1D-CNN基线

## 2026-08-14
### 完成内容
* 编写IR 1D-CNN训练管线（Residual 1D-CNN encoder + 14标签头）
* 完成sanity测试与正式训练（seed 42/123/2026）
* 完成类别不平衡消融实验（sqrt pos_weight不采用）
* 完成阈值校准实验（保留统一阈值0.5）
* 汇总三随机种子结果
### 当前进度
* IR-v1冻结：test Micro-F1 0.9406±0.0048、mAP 0.9689±0.0050、Macro-AUROC 0.9954±0.0007
### 下一步计划
* 提交基线并冻结IR-v1

## 2026-08-15
### 完成内容
* 提交QM9S IR 1D-CNN基线v1（commit 9e4b743）
* 更新`.gitignore`（忽略raw/、processed/、模型checkpoint、master CSV等大型生成文件）
* 更新环境锁定文件
* 更新`README.md`、`CHANGELOG.md`
* 编写Raman 1D-CNN训练管线（复用IR管线模式）
* 完成sanity测试与正式训练（seed 42/123/2026）
* 完成类别不平衡消融实验（sqrt pos_weight不采用）
* 完成阈值校准实验（保留统一阈值0.5）
* 汇总三随机种子结果并冻结Raman-v1
### 当前进度
* IR单模态基线完成并冻结
* Raman-v1冻结：test Micro-F1 0.9151±0.0014、mAP 0.9548±0.0044、Macro-AUROC 0.9911±0.0004
### 下一步计划
* UV-Vis 1D-CNN单模态基线
* IR、Raman、UV-Vis模态对比分析
* IR、Raman、UV-Vis三模态融合
* 完整分子结构候选推断与可视化
* 实验域数据（NIST、API-Raman）验证与微调
* 开放集识别与可解释性分析

## 2026-08-16
### 完成内容
* 编写UV-Vis 1D-CNN训练管线（复用IR/Raman管线模式）
* 完成sanity测试与正式训练（seed 42/123/2026）
* 完成类别不平衡消融实验（sqrt pos_weight不采用）
* 完成阈值校准实验（canonical阈值保留0.5，校准结果作为辅助）
* 完成BatchNorm重校准诊断（非低mAP主因）
* 汇总三随机种子结果并冻结UVVIS-v1
* 弃用GDB-9-Ex为主训练数据，降级为备用外部域/方法迁移数据集
* 新增`task/`（任务说明书）与`competitor/`（文献竞争态势周检）目录
### 当前进度
* IR、Raman、UV-Vis三个单模态基线全部完成并冻结
* UV-Vis在共享14类标签上明显弱于振动光谱（test Micro-F1 0.4892±0.0406），定位为研究问题observation
* 当前进入UV专属共轭/发色团化学语义标签设计（Line A）
### 下一步计划
* UV-specific label taxonomy设计与生成
* 三模态信息画像与模态对比
* chemistry-aware多模态融合
* 完整分子结构候选与Top-k
* 实验域数据（NIST、API-Raman）验证与微调
* 开放集识别与可解释性分析

## 2026-10-24
### 完成内容
* 冻结UV-Vis化学语义标注schema v1.0 FINAL（8层166 features、48条一致性规则、uv_pi_system_graph定义）
* 编写并运行`scripts/build_qm9s_uvvis_annotations.py`（builder v1.0.1）
* 129817个样本全部标注成功（0失败，172警告），输出`processed/qm9s/uvvis_annotations_v1_0_1/`（58二值+73数值特征）
* 完成标注统计（`metadata/uvvis_annotation_statistics_v1_0_1.csv`）与按split审计（`metadata/uvvis_label_split_audit_v1_0_1.csv`）
### 当前进度
* UV专属化学语义标注v1.0.1完成；"UV不是弱，而是shared generic标签不匹配"的假设进入可验证阶段
* QM9S化学空间事实：cumulated_unsaturation=0、N=N=0、羧酸/酰氟/酸酐=0、多独立芳香体系=0、biaryl=0；nitro仅130例（SPARSE）
* 大部分标签通过计数筛查（COUNT_SCREEN_PASS），少数为SPARSE或NO_TRAIN_POSITIVES
### 下一步计划
* 基于审计冻结UV-specific labels v1（active/auxiliary/reserved判定 + 人工抽样核查）
* UV-specific单模态基线
* shared vs specific对照
* 三模态信息画像与模态对比
* chemistry-aware多模态融合

## 2026-11-14
### 完成内容
* 冻结UVSPEC15标签集（15个UV专属共轭/发色团标签），生成`processed/qm9s/prepared/uvspec15_labels.npy`（129817×15）与provenance manifest
* 完成UVSPEC15单模态基线三随机种子训练（seed 42/123/2026）
* 完成阈值校准、完整BN诊断、BN重校准模型保存与BN后阈值校准（普通BCE与Weighted BCE各3 seeds）
* 完成Weighted BCE三随机种子对照实验
* 完成shared vs specific正式对照分析（`runs/UVSPEC15-ANALYSIS/shared_vs_specific_summary.md`）
* 完成UVSPEC15 v1正式实验策略归档（`docs/experiments/UVSPEC15_v1_experimental_strategy.md`：明确Training/Validation/Test分工与计算-实验域差异归因）
* 完成实验数据一致性审计（checkpoint-seed对应、原始/BN重估配套、阈值配套、三seed样本SD、mAP/AUROC阈值无关、audit全过）
* 新增`docs/experiments/uvspec15_repro_protocol.md`、`docs/data/uvspec15_label_guide.md`、`docs/experiments/uvspec15_ablation_registry.md`
* 修复训练/汇总脚本行尾空格
### 当前进度
* shared vs specific对照成立：同一UV光谱、同一backbone、同一split，仅换标签集，UVSPEC15（BN重校准）达test Micro-F1 0.6580±0.0154、mAP 0.5728±0.0203、Macro-AUROC 0.9384±0.0040，显著优于UV-v1 shared（0.4892/0.5053/0.8501）
* 结论：UV不是弱，而是shared generic标签不匹配；BN重校准显著降低三seed方差（训练震荡所致）
* Weighted BCE三seed对照确认不采用（整体指标仍略低）
* UVSPEC15 v1已归档；IR/Raman的BN审计尚未同步，多模态融合尚未开始
### 下一步计划
* 三模态信息画像（Line B）：IR/Raman在UVSPEC15标签上的baseline
* naive fusion对照 → chemistry-aware routing（Line C）

## 2026-11-15
### 完成内容
* 补齐 WBCE Seed 123 / 2026 训练，IR 达到 6 个独立模型（BCE/WBCE × seed 42/123/2026）
* 新增 `training/diagnose_ir_batchnorm_full.py`（UVSPEC15 Full BN 诊断的 IR 移植，仅 Full，不含 Quick）
* 完成 6 次 Full BN Diagnostic（严格串行），6 份 JSON 通过基线复现、参数不变性、BN 统计量与源文件保护检查
* 新增 `training/save_ir_bn_checkpoint.py`，保存 6 份 Training-only BN 重估 checkpoint 与 manifest（源/重估 SHA256、配置摘要、数据规模、与诊断记录的一致性核验）
* 新增 `training/calibrate_ir_original_thresholds.py` 与 `training/calibrate_ir_recalibrated_thresholds.py`，完成 12 次 Validation 逐标签 F1 阈值校准（6 Original + 6 Recalibrated），阈值在计算任何 Test 指标前落盘，Test 不参与拟合
* 新增 `training/summarize_ir_ablation.py`，生成 24 条件消融汇总（24+24 条件、16 组 Mean±SD、84 条配对、672 行逐标签、48 条来源登记、6 份审计），全部只读、含 manifest 哈希链核验
* 新增 `docs/experiments/IR_v1_experimental_strategy.md` 与 `docs/experiments/IR_v1_final_decision.md`
* 新增 `docs/experiments/ir_ablation_registry.md` 与 `docs/experiments/ir_repro_protocol.md`（与 UVSPEC15 对齐的溯源登记表与复现协议）
* 新增 `docs/experiments/UVSPEC15_v1_final_decision.md`（补齐 UVSPEC15 缺失的独立 Final Decision 文档；章节与统计口径同 IR，结论按 UVSPEC15 自身数据得出，正式配置维持 BCE + Recalibrated BN + Validation-F1）
* 废弃脚本归档：`training/calibrate_ir_thresholds.py` 实现迁至 `archive/2026-11-15/calibrate_ir_thresholds.py`（内容未改，992 行）；原路径改为弃用入口，直接运行立即非零退出且不加载数据/不写文件；同步更新 `ir_repro_protocol.md`、`IR_v1_experimental_strategy.md`、`TODO.md`、`AGENTS.md` 中的引用
* 废止并移除早期 IR 专属复杂方案（5×5 交叉拟合、P1–P5、L1/L2′/L3），IR 全面对齐 UVSPEC15 v1 已验证流程
### 当前进度
* **IR v1 冻结归档**：6/6 模型训练、6/6 Full BN Diagnostic、6/6 BN 重估 checkpoint、12/12 阈值校准、24/24 消融条件、48 条 split-level 与 672 条逐标签记录全部齐备
* 最终配置：主方案 BCE + Original BN + Validation-F1（Validation Micro-F1 0.9505±0.0025；Test 0.9364±0.0054）；备选 BCE + Recalibrated BN + Validation-F1（Validation Macro-F1 0.9299±0.0035，mAP 0.9607±0.0067）；固定基线 BCE + Original BN + 0.5（Test Micro-F1 0.9405±0.0048）
* 观察到阈值校准的 Validation 增益（+0.0035～+0.0071 Micro-F1）未稳定迁移至 Test（−0.0011～−0.0042），已如实记录并讨论，不以 Test 反向调整决策
* 历史 `runs/IR-v1-FINAL/` 保持 2026-08-14 原样未改写
### 下一步计划
* **Raman 同构实验（下一步重点）**：Full BN Diagnostic → BN 重估 checkpoint → 12 次阈值校准 → 24 条件汇总，复用 IR 的四个脚本
* 三模态信息画像（Line B）与 chemistry-aware 融合（Line C）
* IR v1 归档提交 → **已完成，见 2026-11-16 条目（`bbcdfa4`，已推送 GitHub main）**

## 2026-11-16
### 完成内容
* IR v1 与 UVSPEC15 v1 的归档成果提交 Git（commit `bbcdfa4192580a9df6384cd8d24489cdfce24515`，提交时间 2026-11-16 00:16 UTC+8）并推送 GitHub `main`（推送完成日期 2026-11-16；远端 `refs/heads/main` 已指向该 commit）
* 废弃脚本归档：`training/calibrate_ir_thresholds.py` 的 992 行实现迁入 `archive/2026-11-15/calibrate_ir_thresholds.py`（内容未改，blob `e93d3bb8`）；原路径改为弃用入口，直接运行立即非零退出且不加载数据、不写文件
* 补齐 UVSPEC15 独立 Final Decision 文档（`docs/experiments/UVSPEC15_v1_final_decision.md`，6 模型 × 24 条件，主方案 BCE + Recalibrated BN + Validation-F1，Test Micro-F1 0.6568±0.0138）
* 同步 `README.md` / `TODO.md` / `AGENTS.md` / `CHANGELOG.md`：修正首行时间戳（原为 2026-11-14 20:35）、更新里程碑与 IR/UVSPEC15 主方案指标口径、清理重复的 Raman 待办、补充 checkpoint 外部备份待办
### 当前进度
* IR v1 与 UVSPEC15 v1 均已冻结归档：各 6 模型、24 条件、48 条 split-level；逐标签 IR 672 行 / UVSPEC15 720 行
* IR 历史基线（Test Micro-F1 0.9406）与新主方案（0.9364）口径已区分；UVSPEC15 历史"0.6580"实为 Recalibrated + 0.5 备选配置，主方案为 0.6568
* Raman 仅有历史单模态基线；同构审计与 24 条件消融尚未开始
### 下一步计划
* Raman 资产审计（现有 4 个 run）→ 按需补齐 WBCE Seed 123/2026 → Full BN Diagnostic → BN 重估 checkpoint → 12 次阈值校准 → 24 条件消融
* checkpoint 与原始光谱数据的仓库外独立备份
* 三模态信息画像（Line B）与 chemistry-aware 融合（Line C）

## 2026-11-16 · Raman v1 冻结归档
### 完成内容
* 补齐 WBCE Seed 123 / 2026 训练与最终 Test 评估，Raman 达到 6 个独立模型（BCE / WBCE × seed 42 / 123 / 2026）
* 完成 6 次 Full BN Diagnostic（`training/diagnose_raman_batchnorm_full.py`），6 份 JSON 通过基线复现、参数不变性、BN 统计量与源文件保护检查
* 新增 `training/save_raman_bn_checkpoint.py`（gate policy v5）：Gate A 制品完整性 10 项确定性检查 + Gate B 官方 FP16 六指标 1e-3 数值复现，二者独立判定；保存 6 份 Training-only BN 重估 checkpoint 与 provenance manifest（源/重估 SHA256、配置摘要、数据规模、与诊断记录的一致性核验）；支持 `--finalize-existing` 复用既有制品、`--approve-release` 写入 sha 绑定的人工放行记录
* 新增 `training/raman_numeric_audit.py`：数值审计判据与放行状态的单一来源（三分类 `PASS` / `MAP_ONLY_NUMERIC_MISMATCH` / `OTHER_ANOMALY`、首次观测即定论、断点续跑、冻结概率数组）
* 完成 12 次 Validation 逐标签 F1 阈值校准（6 Original + 6 Recalibrated），阈值在计算任何 Test 指标前落盘，Test 不参与拟合；每个 (模型, BN) 只推理一次，两个阈值条件共用同一组概率，概率数组与 SHA256 落盘（`probability_artifacts.json`）
* 新增 `training/summarize_raman_ablation.py`，生成 24 条件消融汇总（24 + 24 条件、16 组 Mean±SD、84 条配对、672 行逐标签、48 条来源登记、6 份审计），含 manifest 哈希链核验与 0.0025 交叉容差
* 新增 `docs/experiments/Raman_v1_final_decision.md`、`docs/experiments/raman_ablation_registry.md`、`docs/experiments/raman_repro_protocol.md`（与 IR / UVSPEC15 同构）
* 新增数值复现支撑脚本与测试：`training/diagnose_raman_validation_repro.py`（T1/T2 跨进程探测）、`training/verify_raman_bn_reproducibility.py`（非门控确定性复现路径）、`training/test_raman_bn_gate_policy.py` / `test_raman_bn_reload_integrity.py`（CPU 回归测试）
* 修复 Gate A 第 09 项（`checkpoint_reload_integrity`）执行顺序缺陷：写盘前门禁曾评估仍处于 pending 的第 09 项，导致保存路径必然中止（BCE seed 123 首次暴露）；改为写盘前只排除 09、写盘后以完整清单判定，09 占位仍为 FAIL（fail-closed）
* FP16 历史读数算术勘误：`GATE_POLICY["measured"]` 中 T1/T2 跨进程差值由误写的 `1.086e-3` 更正为 `1.080e-3`；`1.086e-3` 对应最初 Gate B 失败的另一对读数 `[0.945208, 0.946294]`。该字段仅为描述性元数据，不参与 Gate A/B、容差、分类或放行判断
* 六份已冻结的历史 manifest **原样保留、不作事后覆盖**；勘误记录见 `docs/experiments/raman_ablation_registry.md` §9.5
* 废弃脚本归档：`training/calibrate_raman_thresholds.py`、`training/finalize_raman_results.py` 实现迁入 `archive/2026-11-16/`，原路径改为立即非零退出的弃用入口
* Raman v1 归档成果分两次提交并推送 GitHub `main`：代码、门禁与实验文档提交 `5588b81af93abd72219174926fc047a9fac754dc`（`feat: freeze Raman v1 code, gates and experiment documentation`）；轻量实验记录提交 `484f6af9cfd9bead43cb5e1f7e36063b28e1e9cd`（`feat: archive Raman v1 experiment records (24 conditions, 672 per-label rows)`）。两次提交均已推送至 GitHub `main`
### 当前进度
* **Raman v1 冻结归档**：6/6 模型训练、6/6 Full BN Diagnostic、6/6 BN 重估 checkpoint、12/12 阈值校准、24/24 消融条件、48 条 split-level 与 672 条逐标签记录全部齐备
* 最终配置（纯 Validation 决定）：主方案 **WBCE + Recalibrated BN + Validation-F1**（Validation Micro-F1 0.922034±0.002103；Test 0.914113±0.006869）；备选 WBCE + Original BN + Validation-F1（0.921640±0.002583）；BCE 家族最佳 BCE + Recalibrated BN + Validation-F1（0.920907±0.003517）；固定基线 BCE + Original BN + 0.5（Validation 0.910148±0.002418 / Test 0.915080±0.001382）
* 主方案相对备选的 Validation 领先幅度（+0.0004 ~ +0.0018）低于跨 Seed SD，**不作显著性宣称**；Test Micro-F1 主方案排 6/8，**未据此改选**
* 三个模态的单模态基线与方法学消融全部冻结（IR v1 / UVSPEC15 v1 / Raman v1）；三模态融合、完整结构推断、开放集与解释模块尚未开始
* 12 份 Raman checkpoint（约 140 MB）与原始光谱矩阵不进 Git；仓库外备份已由项目决定跳过
### 下一步计划
* 三模态信息画像（Line B）：IR/Raman 在 UVSPEC15 标签上的 baseline 与逐标签对比
* chemistry-aware 融合（Line C）：naive 对照 → generic gating → proposed routing + 消融
* 完整结构 Top-k（Line D）、实验域验证（Line G）、开放集与可解释性（Line H/I）