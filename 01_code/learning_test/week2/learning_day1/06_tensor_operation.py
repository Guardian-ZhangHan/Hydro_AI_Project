# ==============================================
# Week2-Day1 优化版 | MODFLOW水文数据预处理
# 功能：地下水水头数据 → PyTorch张量标准化处理
# 适配：论文代码规范 | 可复现 | 高健壮性
# 路径：D:\Hydro_AI_Project\01_code\learning_test\week2\learning_day1
# ==============================================
import numpy as np
import torch
import os

# ===================== 1. 全局配置（论文可直接修改参数） =====================
# 数据路径（项目绝对路径，避免路径错误）
DATA_PATH = r"D:\Hydro_AI_Project\data\sim_dataset\full_dataset.npz"
# 输出文件名
SAVE_NAME = "day1_final_tensor.pt"
# 目标数据键名（MODFLOW输出的原始水头数据）
TARGET_KEY = "head_raw"

# ===================== 2. 数据加载函数（模块化，论文易读） =====================
def load_hydro_data(data_path, target_key):
    """加载MODFLOW模拟的地下水水头数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    dataset = np.load(data_path)
    if target_key not in dataset.files:
        raise KeyError(f"数据键不存在！可用键：{dataset.files}")
    data = dataset[target_key]
    # 处理无效值（水文数据通用预处理）
    data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data_clean, dataset.files

# ===================== 3. 张量预处理主流程 =====================
if __name__ == "__main__":
    # 加载并清洗数据
    head_np, keys = load_hydro_data(DATA_PATH, TARGET_KEY)
    
    # Numpy转PyTorch张量（CPU张量，符合学习要求）
    head_tensor = torch.from_numpy(head_np).float()
    
    # 维度变换：适配AI模型输入格式 [批次, 通道, 特征长度]
    model_input = head_tensor.unsqueeze(0)
    
    # 数据归一化（0~1标准化，深度学习标准预处理）
    data_min, data_max = model_input.min(), model_input.max()
    normalized_tensor = (model_input - data_min) / (data_max - data_min) if data_max != data_min else torch.zeros_like(model_input)

    # ===================== 4. 科研级格式化输出 =====================
    print("="*80)
    print(f"📊 MODFLOW数据集键名：{keys}")
    print(f"✅ 原始水头数据形状：{head_np.shape} | 设备：CPU")
    print(f"✅ 模型输入张量形状：{model_input.shape}")
    print(f"✅ 归一化范围：[{normalized_tensor.min().item():.4f}, {normalized_tensor.max().item():.4f}]")
    print("="*80)
    print("🎉 Week2-Day1 水文数据预处理完成 | 张量已标准化")
    
    # 保存结果
    torch.save(normalized_tensor, SAVE_NAME)
    print(f"✅ 结果保存至：{os.path.abspath(SAVE_NAME)}")