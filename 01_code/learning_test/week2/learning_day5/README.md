# Week2 Day5 早停机制与泛化能力优化实验
## 任务目标
本实验为地下水Sim2Real渗透系数反演项目的基线训练实验，核心实现：
1.  科研级早停机制的手写实现与验证
2.  训练过程验证集损失的全流程监控
3.  规范的训练日志打印、归档与可复现性控制
4.  Sim2Real场景下的基线模型训练与性能评估

## 环境依赖
- Python >= 3.10
- PyTorch >= 2.5.0
- NumPy >= 1.26.0
- scikit-learn >= 1.5.0
- matplotlib >= 3.9.0
- seaborn >= 0.13.0
- tqdm >= 4.66.0
- PyYAML >= 6.0.0

完整环境依赖见项目根目录 `requirements.txt`

## 运行方式
### Windows系统（一键运行）
直接双击本目录下的 `run.bat` 脚本，自动激活环境并启动训练。

### 手动命令行运行
```bash
# 激活conda环境
conda activate hydro_sim2real
# 切换到当前目录
cd 01_code/02_sim2real_model/week2/learning_day5
# 启动训练
python week2_day5_train_with_early_stopping.py

实验配置
核心配置见 day5_config.yaml：
随机种子：42（全链路固定，100% 可复现）
早停机制：连续 150 轮验证损失无改善则停止训练
模型结构：4 层 MLP，输入维度 5，输出维度 100
优化器：Adam，初始学习率 1e-3
数据集：Day4 生成的 1000 组 MODFLOW 稳态流仿真数据集

预期输出
训练完成后，所有结果将自动归档到项目根目录 03_results/ 下，按「实验名称 + 运行时间戳」分类：
训练日志：train_logs/ 目录下，包含完整的训练过程日志文件
模型权重：model_weights/ 目录下，包含最优轮次的模型权重文件
可视化图：plots/ 目录下，包含训练曲线、误差分布直方图（PNG+SVG 双格式，可直接用于论文排版）
元数据：train_metadata.json，包含完整的实验配置、训练历史、性能指标

基线实验性能指标
指标	数值	单位
测试集 MSE	1.0747	-
测试集 RMSE	1.0367	m/d
测试集 MAE	0.8788	m/d
测试集 R²	-0.0666	-
物理合理性校验：所有反演渗透系数值均在 0.5-5.5 m/d 合理范围内，无负值、异常值，符合水文地质基本规律。