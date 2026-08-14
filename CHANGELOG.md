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
### 当前进度
* IR单模态基线完成并冻结
### 下一步计划
* Raman 1D-CNN单模态基线
* UV-Vis 1D-CNN单模态基线
* IR、Raman、UV-Vis三模态融合
* 完整分子结构候选推断与可视化
* 实验域数据（NIST、API-Raman）验证与微调
* 开放集识别与可解释性分析