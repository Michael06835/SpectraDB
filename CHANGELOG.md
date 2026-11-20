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
* UV-Vis在共享14类标签上明显弱于振动光谱（当时 test Micro-F1 0.4892±0.0406），定位为研究问题observation
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
* 完成UVSPEC15单模态基线三随机种子训练（seed 42/123/2026）、阈值校准、完整BN诊断、BN重校准与BN后阈值校准（普通BCE与Weighted BCE各3 seeds）
* 完成Weighted BCE三随机种子对照，确认不采用
* 完成shared vs specific正式对照：同一UV光谱、同一backbone、同一split，仅替换标签集，UVSPEC15显著优于UV-v1 shared
* 归档`UVSPEC15_v1_experimental_strategy.md`、`uvspec15_repro_protocol.md`、`uvspec15_ablation_registry.md`、`uvspec15_label_guide.md`
### 当前进度
* UVSPEC15主方案BCE + Recalibrated BN + Validation-F1：Test Micro-F1 0.6568±0.0138、mAP 0.5728±0.0203、Macro-AUROC 0.9384±0.0040；shared vs specific对照成立
* 结论：UV-Vis在共享14类标签上的弱势主要来自标签集不匹配，而不是UV光谱本身缺乏结构信息；BN重校准显著降低三seed方差
* IR/Raman的BN审计尚未同步，多模态融合尚未开始
### 下一步计划
* 完成IR v1同构审计与冻结
* 完成Raman v1同构审计与冻结
* 启动Line B三模态信息画像

## 2026-11-15
### 完成内容
* 完成IR v1完整方法学消融：6个独立模型（BCE/WBCE × seed 42/123/2026）、6次Full BN Diagnostic、6份Training-only BN重估checkpoint、12次Validation逐标签F1阈值校准、24条件汇总（48条split-level、672条逐标签）
* IR主方案确定为BCE + Original BN + Validation-F1；备选BCE + Recalibrated BN + Validation-F1；固定基线BCE + Original BN + 0.5
* 补齐`UVSPEC15_v1_final_decision.md`，统一IR与UVSPEC15的Final Decision、Ablation Registry与Repro Protocol结构
* 归档废弃的IR专用复杂方案与旧阈值脚本；原路径改为立即非零退出的弃用入口
### 当前进度
* IR v1冻结：Validation Micro-F1 0.9505±0.0025，Test Micro-F1 0.9364±0.0054；24条件、48条split-level、672条逐标签记录齐备
* 观察到阈值校准的Validation增益未稳定迁移至Test（Test −0.0011～−0.0042），如实记录，不以Test反向调整配置
* UVSPEC15 v1 Final Decision同步完成（Test Micro-F1 0.6568±0.0138）
### 下一步计划
* 完成Raman v1同构审计
* 提交IR/UVSPEC15归档
* 启动Line B

## 2026-11-16
### 完成内容
* 完成Raman v1同构审计：6个独立模型、6次Full BN Diagnostic、6份Training-only BN重估checkpoint、12次阈值校准、24条件汇总（48条split-level、672条逐标签）
* 落地Raman数值审计与gate policy v5（Gate A 10项 + Gate B六项1e-3）；修复Gate A第09项写盘顺序缺陷，完成FP16历史读数1.086e-3 → 1.080e-3的算术勘误，冻结manifest保持原样
* 新增`Raman_v1_final_decision.md`、`raman_ablation_registry.md`、`raman_repro_protocol.md`
### 当前进度
* IR v1、UVSPEC15 v1、Raman v1全部冻结：各6模型、24条件、48条split-level；逐标签记录IR 672行、UVSPEC15 720行、Raman 672行
* IR主方案Test Micro-F1 0.9364±0.0054；UVSPEC15主方案Test 0.6568±0.0138；Raman主方案WBCE + Recalibrated BN + Validation-F1：Validation Micro-F1 0.922034±0.002103、Test 0.914113±0.006869
* Raman主方案相对备选的Validation领先低于跨seed SD，不作显著性宣称
### 下一步计划
* 启动Line B：先做B0资产审计与协议设计，再进入B1 Validation导出

## 2026-11-17
### 完成内容
* 完成Line B B0只读资产审计：核查40份checkpoint、22份逐标签Test CSV、三模态split/row_index/标签一致性、UVSPEC15列顺序与support；输出`metadata/line_b_b0_*`与`line_b_b0_asset_audit.md`
* 完成协议v0.2：拆分`positive_or_fusion`与真实Oracle，明确scaffold聚类Bootstrap与undefined处理，记录B0历史Test接触边界，拆分UV边际/互补，阶段统一为B1–B5
* 完成B1基础设施与6个CPU合成测试；Shared14九模型FP32 Validation导出（各12981行），9/9对齐/哈希/checkpoint-BN不可变性检查通过，重复前向bitwise一致
* 输出B1九模型指标、逐标签与模态汇总、prediction inventory/resource/audit/code hashes；新增`line_b_b1_shared14_validation_report.md`
### 当前进度
* B0审计与协议v0.2完成；B1初版Validation导出完成，但发现`EXP-UVVIS-001-SEED2026`实际使用Seed 42，UV唯一config seeds为{42,123}，留待审核处理
* 未训练新模型、未访问/推理Line B Test、未重估BN、未拟合阈值；B2/B3未启动
### 下一步计划
* 人工审核B0 v0.2与B1初版结果，决定UV Seed2026处理方式
* 审核通过后按B1→B2→B3→B4→B5推进

## 2026-11-18
### 完成内容
* 纠正UV Shared14 Seed2026配置错误：旧实验实际`SEED=42`，完整归档至`archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/`并弃用旧指标；修正训练脚本为`SEED=2026`，沿用原编号完成重训、历史格式FINAL TEST与B1 FP32 Validation重导
* 重建UVVIS-v1 Test三seed汇总：Micro-F1 0.5158±0.0067、mAP 0.5126±0.0011、Macro-AUROC 0.8526±0.0016；修正版成为唯一`runs/UVVIS-v1-FINAL/`
* 完成B2训练代码与正式训练：共享Validation-only训练模块，IR/Raman × seed 42/123/2026共6个模型；固定UV 103853 split、15类UVSPEC15、BCE、Original BN、0.5、Validation mAP选模；36/36资产、manifest、best epoch、early stopping与Test隔离检查通过
* 完成B3统一接口与门禁：`training/line_b/common.py`支持modality × label_set；导出/评价/汇总支持14/15维；强制注册run ID与seed、数组实际SHA256、权威row_index/labels/label_names、重复推理0.0/bitwise、checkpoint/BN不变性；完成负向sanity
* 完成B3 18模型FP32 Validation导出与261条Validation-only逐标签F1阈值校准（Shared14 126 + UVSPEC15 135）；固定0.5为主评价，Validation-F1为同集次评价
* 完成B3-A正式互补性分析：TP/TN/FP/FN、错误交集、FN rescue、FP correction、new FP/FN、条件正确率、positive OR、真实Oracle与三模态独特纠错空间；18模型对齐、双阈值政策、14/15标签全覆盖
* 完成B3.5-A代码溯源与非破坏性复现：261条阈值与12个B3-A CSV在临时目录复现，正式产物树哈希前后不变；新增`B3/code_provenance.json`，历史执行代码SHA明确为unknown
* 完成B3.5-B统一只读校验：18/18模型、261/261阈值、B3-A 14文件、B1/B3 audit、code provenance、原始prepared/结构资产全部通过；输出`b3_5b_manifest_verification.json`与`_v2.json`
* 完成B3.5-C1/C2 Validation-only分析：C1实现scaffold簇配对bootstrap基础设施，20次/任务sanity通过（Test正式2000次留待B4）；C2完成McNemar/Holm、六方向条件互补、逐分子错误分布、Spearman、hard sample/Jaccard、分子exact-match、sample-label Oracle、OR增益与高置信度FP/FN诊断
* 完成B3.5-D1结构子群基础设施：Training共现、hierarchy/语义组、22个非空子群定义与互斥覆盖断言、支持度门禁、子群AP/AUROC/F1、条件互补/Oracle/OR三seed汇总、逐标签bootstrap；每子群20次scaffold bootstrap sanity通过，未写正式Test产物
* 完成B3.5-D2 Validation典型分子报告工具：22子群×2选择位置、原生RDKit PNG/SVG、每分子独立目录与metadata、三模态光谱、统计表、图与完整报告；Smoke/Full及独立验证通过；明确为非B4/B5正式报告
### 当前进度
* B1–B3完成：18模型Validation资产、261条阈值、B3-A正式输出、B2六模型与统一门禁均已归档
* B3.5 A/B/C1/C2/D1/D2完成：代码溯源、manifest校验、Bootstrap基础设施、扩展互补性、结构子群与Validation报告工具均通过；Test未被访问
* 正式Test导出、FREEZE、2000次Test Bootstrap与B4/B5尚未开始
### 下一步计划
* 完成B3 FREEZE与B0 v0.2 FINAL定稿
* 启动B4.1 Test预测导出，随后按B4.2→B4.3→B4.4→B4.4 Supplement→B5推进

## 2026-11-19
### 完成内容
* 修复原始科研环境`spectra-db`的RDKit原生绘图故障：根因为`libglib`与`libintl`共享`Library/bin/intl-8.dll`，clone对齐`libglib 2.90.1`时该文件被删除，导致`fontconfig → cairo → RDKitMolDraw2D`依赖链断裂
* 仅执行`conda install libintl=0.22.5=h5728263_4`，`intl-8.dll`恢复为conda-forge版本（224256 B，SHA256 `19B198C1…`）；`rdMolDraw2D`导入、Cairo PNG、SVG、Pillow解码与项目四个样本渲染全部通过，Python/RDKit/PyTorch/CUDA/PyG/Pillow版本未变
* 记录修复后的硬性约束：禁止对`libglib`执行force-reinstall/install/update；禁止未经评估升级`libffi`；禁止未经评估切换Python构建；任何相关conda操作前必须dry-run，操作后复核`intl-8.dll` SHA256
* 完成工作区外历史资产迁移：1739文件、190,543,006 bytes迁入`runs/LINE-B-ANALYSIS/validation_development/`与`runs/environment_validation/rdkit/20261119/`，逐文件SHA256一致；重复/被取代运行803文件、46,747,421 bytes归档至`archive/2026-11-19/line-b-relocation-duplicates/`
* 完成B3.5-D2 R2工作区原生化与路径安全补丁：Builder/Verifier支持工作区内输出、中文路径、历史路径relocation解析、MOL/SDF独立解析、metadata与Manifest outputs边界校验；R1保持不可变，R2保持669文件
### 当前进度
* 原始科研环境已恢复RDKit原生绘图，可继续B3.5-D2与B4.4 Supplement结构渲染；回滚环境`spectra-db-pre-rdkit-fix`（5.83 GB）保留
* B3.5-D2 R1/R2报告与图像一致；D2仅保留R1 final为活跃历史资产，R2保持669文件
* Line B正式Test仍未开始
### 下一步计划
* 完成B3 FREEZE与B0 v0.2 FINAL定稿
* 启动B4.1 Test导出与B4.2–B4.4分析

## 2026-11-20
### 完成内容
* 完成B3 FREEZE：修复`verify_line_b_manifests.py`对`runs/LINE-B-ANALYSIS`的整树摘要误判，改为对14个权威B3-A文件逐文件校验并重建历史摘要；`validation_development/`（C1 8 + C2 36 + D1 52 + D2 8931 = 9027文件）明确识别为快照后追加资产
* 定稿B0 v0.2 FINAL（SHA256 `713fa92f…`），生成`FREEZE.json`（SHA256 `65a84959…`，status=FROZEN、phase=B3）：18/18模型、261条阈值、B3-A 6+6输出、212项资产路径/大小/SHA256；预检与写入后复验`PASS 10 / FAILED 0 / ACCEPTED 4`，18/18、261、正式根前后不变、Test零访问
* 完成B4.1 Test预测导出：18个冻结模型各导出一次，每模型12981行，72文件/18,969,279 B；FREEZE哈希门禁、模型身份校验、重复推理bitwise一致、checkpoint/BN不变；`test_access_log.json`18条，`blind_test_claim=false`、`result_classification=exploratory/post-hoc`；独立验证PASS 466/466
* 完成B4.2 Test性能评估：18模型×2 Task，主F1政策固定0.5、次政策冻结Validation-F1，AP/AUROC阈值无关；Shared14 IR/Raman/UV-Vis Micro-F1 0.9405/0.9151/0.5159、mAP 0.9695/0.9551/0.5125；UVSPEC15 IR/Raman/UV-Vis Micro-F1 0.9512/0.9538/0.6108、mAP 0.9709/0.9760/0.5380；17文件/1,841,098 B，独立验证PASS 8,374/8,374
* 完成B4.3正式Test scaffold-clustered paired bootstrap：2853个Test scaffold、2000次重采样、固定RNG 20261117，18模型/3 seed/全部标签/3组比较共用同一draw matrix；AP/AUROC按draw count加权，undefined不填补，F1两政策，95% percentile CI，Seed SD ddof=1；Shared14 UV−IR AP −0.4570 [−0.4871,−0.3857]，UVSPEC15 UV−IR AP −0.4329 [−0.4910,−0.3931]；独立验证PASS 1,144/1,144
* 完成B4.4 Test错误互补性与结构子群：pairwise TP/FP/FN/TN、错误Jaccard、A/B-only、FN rescue、FP correction、new FP/FN、McNemar exact + Holm、Positive OR/AND、真实Oracle、六方向条件互补（2000次bootstrap，复用B4.3 draw）、sample-level、hard 10%、高置信度错误与结构子群（D1定义，子群内500次bootstrap）
* B4.4主要结果：macro错误Jaccard 0.049–0.101；Oracle any-correct 0.9968/0.9989、exactly-one 1.49%/0.55%、all-wrong 0.32%/0.11%；IR/Raman同时错误而UV正确的pair为509/205，但UV rescue 0.469/0.498低于边际0.880/0.925，delta −0.411/−0.426，95% CI全负；三模态OR F1比best_single低0.06–0.23；Shared14 19/21支持子群以IR最优，UVSPEC15 16/21以Raman最优；全部84个fixed-0.5子群UV比较CI上界<0；独立验证PASS 1,127/1,127
* 完成B4.4 Supplement典型分子分析：固定Seed 42/0.5，按B0 §7.6选择每个非空子群的1个median-confidence correct与1个最高-confidence error；两Task各22个非空子群、44案例，共88案例；原生RDKit PNG/SVG、prepared真实光谱轴、完整metadata与输入SHA；358文件/9,356,623 B，独立验证PASS 263/263；B4.4树SHA `d3e69052…`未变
* 完成B5 Phase 0预检与正式报告：388文件/12,643,790 B（19表、11图、88案例）；修复三个WARN——明确B4.3 CI为配对差值CI而非绝对性能CI、明确H3的`only_UV/only_IR/only_Raman × 结构子群`富集未直接检验、补充AP/AUROC≠化学信息总量与QM9S计算域≠NIST/SDBS/API-Raman实验域等边界
* B5最终结论：H1 PARTIAL SUPPORT（IR/Raman>UV的边际优势成立，但振动标签差距未更大）；H2 REJECTED（UVSPEC15 UV未优于IR/Raman）；H3 PARTIAL SUPPORT（错误集合互补，但Positive OR未超过best_single、UV条件rescue低于边际、子群独家正确富集未检验）；H4 DESCRIPTIVELY SUPPORTED（最佳模态随Task与结构子群变化）
* B5独立验证PASS 67/67；报告SHA `398c0481…`、verification `70eae468…`、asset_manifest `a7e5910b…`；所有Test结果标记exploratory/post-hoc，不作为confirmatory evidence
* 完成Line B Git归档：分支`archive/line-b-2026-11-20`，commit `ab1e85dd0274e803bae074fc19acce47f6feb142`（parent `f0395ad`）；1173个新增文件、内容143,591,799 B，加6个已跟踪未修改文件后共1,179路径/143,780,373 B；9处既有尾随空格作为格式例外保留；D类`CHANGELOG.md`/`TODO.md`/`.codexignore`未纳入，已推送至远端分支 archive/line-b-2026-11-20
### 当前进度
* Line B B0–B5全部完成并归档；B5正式报告通过独立验收；Test结果仅为exploratory/post-hoc
* 三模态信息画像的核心结论已形成：IR/Raman在Shared14与UVSPEC15标签上均强于UV-Vis；UV-Vis在UVSPEC15上未表现出预期优势；错误集合存在差异，但条件互补与Positive OR均不足以支持“UV带来稳定增量”的结论
* QM9S为计算光谱域，尚未进行NIST/SDBS/API-Raman实验域验证；典型分子分析仅为定性案例，不是XAI或统计证据
### 下一步计划
* Line C：先做naive concatenation/fixed-weight对照，再做generic attention/gating，最后实现chemistry-aware routing与消融；所有融合必须与best_single比较，并使用scaffold-aware不确定性
* Line D：在融合证据成立后进入完整结构候选与Top-k
* Line F/G：缺失/噪声鲁棒性与实验域验证
* Line H/I：开放集识别与可解释性
