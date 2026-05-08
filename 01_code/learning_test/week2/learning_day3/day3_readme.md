# 地下水参数反演深度学习数据集构建（顶刊版）
## 研究背景
本数据集与代码为地下水非均质渗透系数场反演的Sim2Real深度学习建模提供标准化数据处理流程，所有操作完全符合水文水资源顶刊的可复现性与数据质量要求。

## 数据来源
- 原始数据：MODFLOW 6三维地下水数值模拟生成的稳态水头观测值与对应渗透系数场标签
- 模拟软件：USGS MODFLOW 6.4.1
- 模拟软件Python接口：flopy 3.4.1

## 数据处理流程
1. **数据质量校验**：完成文件完整性、键完整性、水文物理合理性校验，处理异常值与无效值
2. **格式转换**：将NumPy数组转换为PyTorch张量，自动适配CPU/GPU设备
3. **数据集划分**：
   - 样本量≥10：采用8:1:1分层随机划分，固定随机种子保证可复现
   - 样本量<10：采用留一法交叉验证，符合水文小样本建模规范
4. **DataLoader构建**：训练集打乱，验证/测试集固定顺序，严格避免数据泄露

## 文件结构
learning_day3/
├── week2_day3_modflow_batch.py # MODFLOW 6 批量建模代码（顶刊规范版）
├── week2_day3_dataset.py # 顶刊级数据集构建代码
├── modflow_batch_runs/ # 批量生成的 MODFLOW 模型文件
├── dataset_processing.log # 数据处理全流程日志
├── dataset_metadata.npy # 数据集元数据（可复现性归档）
└── day3_readme.md # 本说明文档

## 可复现性说明
- 所有随机源固定：Python、NumPy、PyTorch全链路随机种子固定为42
- 运行环境：Python 3.10+, PyTorch 2.0+, flopy 3.4.1+, NumPy 1.24+
- 所有代码在CPU与GPU环境下均可完全复现

## 数据可用性声明
所有原始模拟数据、处理代码、元数据均已归档至GitHub仓库：https://github.com/Guardian-ZhangHan/Hydro_AI_Project，符合FAIR科研数据原则，可免费获取与复用。