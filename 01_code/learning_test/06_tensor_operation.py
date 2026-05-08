# ==============================================
# Week2-Day1 最终强制版
# 功能：MODFLOW水头数据 → PyTorch张量转换
# 路径：D:\Hydro_AI_Project\01_code\learning_test\week2\learning_day1\06_tensor_operation.py
# 说明：直接复制覆盖整个文件，无需任何修改
# ==============================================
import numpy as np
import torch
import os

# 读取MODFLOW生成的完整数据集
dataset_path = r"D:\Hydro_AI_Project\data\sim_dataset\full_dataset.npz"
dataset = np.load(dataset_path)

# 打印数据集中包含的键
print("✅ 数据集中包含的键：", list(dataset.keys()))

# 提取水头数据
head_np = dataset['head']
print("\n✅ 水头数据读取成功！")
print("数据形状：", head_np.shape)
print("数据类型：", head_np.dtype)
print("数值范围：最小值={:.2f}, 最大值={:.2f}".format(np.nanmin(head_np), np.nanmax(head_np)))

# 清理无效数据（NaN）
head_np_clean = np.nan_to_num(head_np, nan=0.0)

# NumPy → PyTorch CPU张量转换
head_tensor = torch.from_numpy(head_np_clean).float()
print("\n✅ 转换为PyTorch张量成功！")
print("张量形状：", head_tensor.shape)
print("运行设备：", head_tensor.device)

# 张量切片：提取目标含水层
target_layer = head_tensor[0, 0, :, :]
print("\n✅ 切片后的目标含水层张量信息：")
print("张量形状：", target_layer.shape)

# 维度变换：匹配AI模型输入格式
model_input = target_layer.unsqueeze(0).unsqueeze(0)
print("\n✅ 维度变换完成！")
print("张量形状：", model_input.shape)

# 归一化运算：缩放到0-1之间
min_val = model_input.min()
max_val = model_input.max()
normalized_input = (model_input - min_val) / (max_val - min_val)

# 输出最终结果
print("\n======================================")
print("🎉 Week2-Day1 所有任务已完成！")
print("最终张量形状：", normalized_input.shape)
print("运行设备：", normalized_input.device)
print("数值范围：最小值={:.4f}, 最大值={:.4f}".format(normalized_input.min().item(), normalized_input.max().item()))
print("======================================")

# 保存最终张量到当前文件夹
torch.save(normalized_input, "day1_final_tensor.pt")
print("✅ 最终张量已保存到当前文件夹！")