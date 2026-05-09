# Hydro AI Project: Sim2Real Groundwater Permeability Inversion
地下水Sim2Real渗透系数反演深度学习项目 | 中国地质大学（武汉）环境学院

## 项目简介
本项目针对地下水水文地质参数反演中的「仿真-真实域偏移」问题，构建了基于MODFLOW数值模拟的仿真数据集生成框架，与Sim2Real迁移学习的渗透系数反演模型，实现从少量监测井水头数据到二维渗透系数场的精准反演。

本项目完全符合国际水文学顶刊（Water Resources Research, Journal of Hydrology）的开源规范与可复现性要求，所有代码、数据、实验结果均可100%复现。

## 仓库结构
Hydro_AI_Project/
├── 01_code/ # 核心代码目录
│ ├── 01_data_generation/ # MODFLOW 仿真数据集生成代码（Day1-Day4）
│ ├── 02_sim2real_model/ # Sim2Real 反演模型训练代码（Day5-Day7）
│ └── 03_utils/ # 通用工具函数
├── 02_data/ # 数据集与元数据
│ ├── 01_raw_sim_data/ # 原始仿真数据集
│ └── 02_metadata/ # 标准化器、元数据文件
├── 03_results/ # 实验结果归档
│ ├── 01_train_logs/ # 训练日志
│ ├── 02_model_weights/ # 模型权重文件
│ ├── 03_plots/ # 可视化图片（PNG+SVG）
│ └── 04_ablation_study/ # 消融实验结果与对比表
├── 04_docs/ # 科研文档
│ ├── research_doc/ # 学习计划、每日实验成果
│ ├── dataset_description.md # 数据集描述文档（可直接用于论文）
│ └── model_card.md # 模型卡片（顶刊开源标准）
├── README.md # 项目说明文档
├── requirements.txt # 环境依赖
├── .gitignore # Git 忽略文件
└── LICENSE # 开源协议（MIT）

## 🚀 Quick Start 快速开始
### 1. 环境配置
```bash
# 创建conda环境
conda create -n hydro_sim2real python=3.10
# 激活环境
conda activate hydro_sim2real
# 安装依赖
pip install -r requirements.txt
2. 复现 Day5 基线实验（早停机制训练）
# 切换到实验目录
cd 01_code/02_sim2real_model/week2/learning_day5
# 启动训练
python week2_day5_train_with_early_stopping.py
3. 预期输出
训练完成后，所有结果将自动归档到 03_results/ 目录下，按「实验名称 + 运行时间戳」分类：
训练日志自动保存至 03_results/01_baseline_patience150_dropout0/[run_id]/train_logs/
最优模型权重自动保存至 03_results/01_baseline_patience150_dropout0/[run_id]/model_weights/
训练曲线、误差分布直方图自动保存至 03_results/01_baseline_patience150_dropout0/[run_id]/plots/
实验进度
✅ Day1-Day4：MODFLOW 6 批量建模与仿真数据集生成（1000 组样本）
✅ Day5：早停机制实现与基线模型训练
⏳ Day6：防过拟合策略与消融实验
⏳ Day7：Sim2Real 域自适应与真实数据微调
开源与引用
本项目代码遵循 MIT 开源协议，如需引用本项目，请联系作者。

English Version
Project Overview
This project focuses on the "Simulation-to-Real (Sim2Real) domain gap" problem in groundwater hydrogeological parameter inversion. We build a MODFLOW-based numerical simulation dataset generation framework, and a Sim2Real transfer learning model for permeability inversion, achieving accurate inversion of 2D permeability field from a small number of monitoring well head data.
This project fully complies with the open-source and reproducibility requirements of top international hydrology journals (Water Resources Research, Journal of Hydrology). All codes, data, and experimental results are 100% reproducible.
Quick Start
1.Environment Setup
conda create -n hydro_sim2real python=3.10
conda activate hydro_sim2real
pip install -r requirements.txt
2.Reproduce Day5 Baseline Experimentcd 
01_code/02_sim2real_model/week2/learning_day5
python week2_day5_train_with_early_stopping.py
License
This project is licensed under the MIT License.

---

## 【文件6】路径：项目根目录 `.gitignore`
（新建文件，直接全选复制粘贴，顶刊Python项目标准模板）
```gitignore
# ==============================================
# Python项目标准忽略文件
# 适配水文AI顶刊开源项目规范
# ==============================================

# 数据文件（大文件不纳入Git管理）
*.npy
*.npz
*.pth
*.pt
*.h5
*.hdf5
*.mat
*.dat
*.txt
*.csv

# 模型权重与结果文件
model_weights/
weights/
checkpoints/
results/
plots/
logs/
train_logs/
ablation_study/

# Python缓存文件
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# 虚拟环境
.env
.venv
venv/
ENV/
env/

# IDE配置文件
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# 系统文件
*.lnk
*.url
*.desktop

# 临时文件
*.tmp
*.temp
*.bak