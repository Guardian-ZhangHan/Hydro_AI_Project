# Hydro AI Project - 水力人工智能项目

## 项目简介
本项目针对缺资料农业盆地地下水超采治理，构建物理约束迁移学习（PINN-Transfer Learning）模型，融合水文地质物理约束与深度学习，实现地下水水位高精度预测与超采治理政策优化。

## 核心技术栈
- 深度学习框架：PyTorch、DeepXDE（PINN）
- 水文数据处理：xarray、rasterio、netCDF4、flopy
- 模型评估：hydroeval（NSE/R²）
- 环境管理：Anaconda、hydro_ai
- 版本控制：Git + GitHub

## 项目结构
D:\Hydro_AI_Project
├── .ipynb_checkpoints/
├── ai_environment.py
├── main_research.py
├── pm2.5/
├── WESAD/
├── hydro_ai_environment_cleaned.yml
├── hydro_ai_model.pth
├── hydro_ai_prediction_result.png
├── Hydro_AI_一键启动环境.bat
├── scaler_X.pkl
├── scaler_y.pkl
├── README.md
├── .gitignore
└── LICENSE

## 环境配置
### 1. 一键启动（推荐）
双击 Hydro_AI_一键启动环境.bat

### 2. 手动配置
conda activate hydro_ai
conda install pytorch cpuonly -c pytorch -y
pip install --upgrade numpy pandas matplotlib xarray rasterio netCDF4 scikit-learn deepxde hydroeval scipy jupyter flopy

## 模型核心创新点
1. 物理约束迁移学习
2. 高精度预测 R² ≥ 0.99
3. 政策情景模拟
4. 100% 可复现

## 作者
张翰 (Guardian-ZhangHan)
邮箱：z18178909532@gmail.com

## 许可证
MIT License