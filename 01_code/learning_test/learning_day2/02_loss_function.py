# ===================== 1. 固定随机种子，与Day1完全一致，保证可复现 =====================
import torch
import numpy as np

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== 2. 手写MSE损失函数原生实现 =====================
def custom_mse_loss(y_pred, y_true):
    """
    手写原生MSE均方误差损失函数，与PyTorch官方nn.MSELoss()完全对齐
    适配水文参数反演场景：输入为预测/真实K值场（shape=[batch, 100]）
    """
    # 计算误差：真实值-预测值
    error = y_true - y_pred
    # 平方误差：消除正负号，放大大误差
    squared_error = error ** 2
    # 均值误差：对所有网格的误差取平均
    mse_loss = torch.mean(squared_error)
    return mse_loss

# ===================== 3. 生成水文场景测试数据 =====================
# 模拟真实K值：1组10×10=100个网格的真实渗透系数
y_true = torch.randn(1, 100)
# 模拟预测K值：加入噪声，模拟模型预测结果
y_pred = y_true + 0.2 * torch.randn(1, 100)

# ===================== 4. 与PyTorch官方实现对照验证 =====================
official_mse = torch.nn.MSELoss()
loss_custom = custom_mse_loss(y_pred, y_true)
loss_official = official_mse(y_pred, y_true)

# ===================== 5. 输出验证结果 =====================
print("="*70)
print("MSE损失函数 原生实现 vs 官方实现 对照验证")
print("="*70)
print(f"手写实现MSE：{loss_custom.item():.8f}")
print(f"官方实现MSE：{loss_official.item():.8f}")
print(f"绝对误差：{torch.abs(loss_custom - loss_official).item():.10f}")
print("="*70)

if torch.abs(loss_custom - loss_official) < 1e-6:
    print("✅ 验证通过：手写实现与官方实现结果完全一致")
else:
    print("❌ 验证失败：结果存在差异，请检查代码")

# 保存结果到证据链
np.savetxt("research_doc/Day2_MSE_对比结果.csv",
           np.array([[loss_custom.item(), loss_official.item()]]),
           delimiter=",", header="手写MSE,官方MSE", comments="")
print("结果已保存到research_doc文件夹，可作为论文补充材料")