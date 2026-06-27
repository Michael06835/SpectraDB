# 最后更新时间：17:16(UTC+8) 06/27/2026

# SpectraDB
## 项目简介
SpectraDB是一个面向人工智能多模态光谱分析研究的数据库构建项目。
项目旨在建立统一的有机小分子多模态光谱数据库，将紫外-可见吸收光谱（UV-Vis）、红外光谱（IR）和拉曼光谱（Raman）等不同模态的光谱信息通过统一分子标识（SMILES、InChIKey、CID 等）进行关联，为后续多模态融合模型训练和未知化合物识别提供数据基础。
## 项目目标
### 第一阶段（可行性验证）
* 建立2000–5000个有机小分子数据库
* 实现 UV-Vis、IR、Raman 光谱自动配对
* 验证数据库构建流程可行性
### 第二阶段（正式数据库）
* 扩展至约100000个有机小分子
* 建立统一多模态光谱数据库
* 支持 AI 多模态光谱融合研究
## 数据结构
每个分子对应唯一记录，主要包含：
* Compound ID
* PubChem CID
* SMILES
* InChIKey
* 分子式
* 分子量
* RDKit Descriptor
以及：
* 紫外-可见吸收光谱
* 红外光谱
* 拉曼光谱
后续可扩展：
* NMR
* MS
* 分子描述符
## 项目目录
SpectraDB/

README.md
CHANGELOG.md
TODO.md
scripts/
master/
raw/
paired/
processed/
logs/
cache/
.vscode/
## 当前开发计划
1. 建立compound_master.csv作为基础数据表
2. 获取有机小分子基础信息
3. 获取IR光谱
4. 获取Raman光谱
5. 获取UV-Vis光谱
6. 自动配对多模态数据
7. 数据预处理
8. AI数据集生成
9. 多模态模型训练
## 开发环境
已有：
·Python
·Conda
·RDKit
·Pandas
·NumPy
后续计划加入：
·PyTorch
·Scikit-learn
·HuggingFace
·Lightning
## 项目状态
当前处于光谱收集阶段
