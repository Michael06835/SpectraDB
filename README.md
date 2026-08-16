# 最后更新时间：18:50(UTC+8) 08/16/2026

# SpectraDB
## 项目简介
SpectraDB是一个面向有机小分子多模态光谱分析的数据库与模型建设项目。
项目目标是建立统一的多模态光谱数据库，并实现从光谱推断分子结构信息：以单模态1D-CNN模型（IR、Raman、UV-Vis）为基础，按各模态的化学证据分工进行chemistry-aware自适应融合，并逐步实现官能团/结构特征识别、完整分子结构候选推断、分子结构可视化及模型可解释性分析。

## 项目目标
### 第一阶段（数据体系与单模态模型）
包括：
* 建立QM9S多模态计算光谱数据
* 建立分子结构库
* 建立官能团、原子、键、环、共轭体系等结构标注
* 建立scaffold split
* 训练IR 1D-CNN
* 训练Raman 1D-CNN
* 训练UV-Vis 1D-CNN
* 比较不同光谱模态对结构信息的表征能力
### 第二阶段（多模态融合与完整结构推断）
包括：
* IR、Raman、UV-Vis特征融合
* 综合官能团与关键结构特征
* 完整分子结构候选推断
* Top-k候选排序
* 二维分子结构可视化
* Woodward等化学规则辅助验证和候选重排序
### 第三阶段（实验域、开放集与可解释性）
包括：
* NIST实验数据微调与验证
* API-Raman实验数据微调与验证
* calculated spectra → experimental spectra迁移
* 未知样品识别
* 开放集识别
* 置信度与拒识
* XAI
* 光谱区域与结构片段对应
* 最终结果界面与自然语言解释

## 数据源
* QM9S：IR、Raman、UV-Vis主要计算训练数据（约13万分子）
* GDB-9-Ex：备用外部域/方法迁移数据集（约9.7万分子，激发态能量与振子强度），当前不参与训练
* NIST：实验域IR、UV-Vis数据（微调与测试）
* API-Raman：实验Raman数据（约3500个化合物）

## 数据结构
### 每个分子对应唯一记录，主要包含：
* row_index（与光谱矩阵、结构标注对齐）
* QM9S编号
* SMILES（canonical）
* 分子式
* 分子量
* RDKit Descriptor
### 以及：
* 红外光谱（500–4000 cm⁻¹，3501点）
* 拉曼光谱（500–4000 cm⁻¹，3501点）
* 紫外-可见吸收光谱（1–15 eV，701点）
### 结构标注：
* 官能团多标签（14类）
* 原子/键/环/共轭体系标注

## 项目目录
* SpectraDB/
---------------------------
* README.md
* CHANGELOG.md
* TODO.md
* AGENTS.md
* scripts/
* training/
* runs/
* master/
* raw/
* processed/
* metadata/
* cache/
* logs/
* archive/
## 当前开发计划
1. 建立QM9S多模态数据集（已完成）
2. 生成官能团标签与结构库（已完成）
3. IR 1D-CNN单模态基线（已完成）
4. Raman 1D-CNN单模态基线（已完成）
5. UV-Vis 1D-CNN单模态基线（已完成）
6. UV专属共轭/发色团化学语义标签
7. 三模态信息画像与模态对比
8. Chemistry-aware多模态融合
9. 完整结构候选与Top-k
10. 实验域数据（NIST、API-Raman）验证与微调
11. 开放集识别与可解释性分析
## 开发环境
### 已有：
* Python
* Conda
* RDKit
* Pandas
* NumPy
* PyTorch
* Scikit-learn
* Matplotlib
### 后续计划加入：
* HuggingFace
* Lightning
## 项目状态
IR、Raman、UV-Vis三个单模态基线已完成并冻结（IR-v1：test Micro-F1 0.9406，Macro-AUROC 0.9954；Raman-v1：0.9151，0.9911；UV-Vis-v1：0.4892，0.8501）；当前进入UV专属化学语义标签与chemistry-aware融合阶段。
