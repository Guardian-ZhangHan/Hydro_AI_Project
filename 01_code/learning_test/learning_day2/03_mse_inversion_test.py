# ===================== 1. 导入依赖，复用Day1模型骨架 =====================
import torch
import torch.nn as nn
import numpy as np

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 100%复用Day1的水文反演神经网络骨架
class HydroInversionNN(nn.Module):
    def __init__(self):
        super(HydroInversionNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 100)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# 复用手写MSE损失函数
def custom_mse_loss(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)

# ===================== 2. 模拟水文反演场景数据 =====================
# 输入：1组5口监测井的水头值
x_monitor = torch.randn(1, 5)
# 标签：1组10×10网格的真实K值（后续替换为MODFLOW数据）
y_true_k = torch.randn(1, 100)

# ===================== 3. 模型前向传播+损失计算 =====================
model = HydroInversionNN()
model.eval()

with torch.no_grad():
    y_pred_k = model(x_monitor)

loss = custom_mse_loss(y_pred_k, y_true_k)

# ===================== 4. 输出场景化分析结果 =====================
print("="*70)
print("水文参数反演场景下MSE损失计算测试")
print("="*70)
print(f"输入：5口监测井水头值：{x_monitor.numpy().flatten()}")
print(f"真实K值场均值：{y_true_k.mean().item():.4f}，标准差：{y_true_k.std().item():.4f}")
print(f"预测K值场均值：{y_pred_k.mean().item():.4f}，标准差：{y_pred_k.std().item():.4f}")
print(f"MSE损失值：{loss.item():.6f}")
print("="*70)
print("📌 物理意义解读：")
print("1. MSE损失值代表预测K值场与真实水文条件的差异程度")
print("2. 模型训练的目标就是通过反向传播不断降低这个损失值")
print("3. 当损失值收敛到足够小时，预测的K场即可用于后续流场模拟")

# 保存测试结果
np.savetxt("research_doc/Day2_MSE_反演测试结果.csv",
           np.array([[loss.item(), y_true_k.mean().item(), y_pred_k.mean().item()]]),
           delimiter=",", header="MSE损失,真实K均值,预测K均值", comments="")
print("测试结果已保存到research_doc文件夹")