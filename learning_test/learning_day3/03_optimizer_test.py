# ===================== 1. 固定全量随机种子，与Day1/2完全一致，保证学术可复现性 =====================
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 必须与Day1/2的SEED完全一致，保证模型初始化、数据生成完全相同，严格控制变量
SEED = 42
random_seed = SEED
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 2. 100%复用Day1的水文反演神经网络骨架，严格控制变量 =====================
class HydroInversionNN(nn.Module):
    def __init__(self):
        super(HydroInversionNN, self).__init__()
        # 与Day1完全一致的网络结构，唯一变量仅为学习率
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 100)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# ===================== 3. 复用Day2的MSE损失函数，保证实验一致性 =====================
def mse_loss(y_pred, y_true):
    return torch.mean((y_true - y_pred) ** 2)

# ===================== 4. 实验核心配置：严格单因素控制变量 =====================
# 唯一变量：3组对照学习率
lr_list = [1e-2, 1e-3, 1e-4]
# 固定参数：所有实验完全一致
epochs = 100  # 100轮训练，符合你的要求
input_dim = 5  # 5口监测井
output_dim = 100  # 10×10网格K值

# 固定模拟数据集：所有实验组使用完全相同的输入和标签，严格控制变量
# 输入：100组5口监测井的水头数据（批量训练，更贴合真实场景）
x_train = torch.randn(100, input_dim)
# 标签：100组对应的10×10网格真实K值（后续替换为MODFLOW真实数据）
y_train = torch.randn(100, output_dim)

# ===================== 5. 对照实验执行 =====================
# 用于记录每一轮的损失值，后续画损失下降曲线
loss_history = {}

for lr in lr_list:
    print(f"\n===================== 开始训练：学习率lr={lr} =====================")
    # 每次实验重新初始化模型，保证初始权重完全一致（SEED固定）
    model = HydroInversionNN()
    # 初始化Adam优化器，仅学习率不同，其余参数完全固定
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    # 初始化当前学习率的损失记录
    loss_history[lr] = []

    # 100轮训练循环
    for epoch in range(epochs):
        # 前向传播：得到预测K值
        y_pred = model(x_train)
        # 计算MSE损失
        loss = mse_loss(y_pred, y_train)

        # 反向传播→参数更新 核心三步（固定流程，学术标准写法）
        optimizer.zero_grad()  # 清空上一轮的梯度，避免梯度累积
        loss.backward()        # 反向传播，计算所有参数的梯度
        optimizer.step()       # 基于梯度和学习率，更新模型参数

        # 记录当前轮的损失值
        loss_value = loss.item()
        loss_history[lr].append(loss_value)

        # 每10轮打印一次训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_value:.6f}")

    print(f"===================== 训练完成：学习率lr={lr}，最终损失={loss_value:.6f} =====================")

# ===================== 6. 实验结果保存与可视化 =====================
# 1. 保存损失数据到research_doc，用于论文补充材料
for lr in lr_list:
    np.savetxt(f"research_doc/Day3_lr_{lr}_loss_history.csv", 
               loss_history[lr], 
               delimiter=",", 
               header=f"lr={lr}_loss",
               comments="")
print("\n✅ 所有实验组的损失数据已保存到research_doc文件夹")

# 2. 绘制损失下降曲线，直接用于论文图表
plt.figure(figsize=(10, 6), dpi=300)
for lr in lr_list:
    plt.plot(range(1, epochs+1), loss_history[lr], label=f"lr={lr}", linewidth=1.5)

# 图表格式规范，符合学术论文要求
plt.xlabel("Training Epochs", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Influence of Learning Rate on Model Convergence (Hydrogeological Parameter Inversion)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存高清图片，用于论文/答辩
plt.savefig("research_doc/Day3_learning_rate_loss_curve.png", dpi=300, bbox_inches="tight")
print("✅ 损失下降曲线已保存到research_doc文件夹，可直接用于论文")

# ===================== 7. 实验结果量化分析 =====================
print("\n" + "="*80)
print("3组学习率对照实验 量化分析结果")
print("="*80)
for lr in lr_list:
    final_loss = loss_history[lr][-1]
    min_loss = min(loss_history[lr])
    # 计算收敛速度：损失下降到最终损失90%所需的轮数
    converge_epoch = np.where(np.array(loss_history[lr]) <= final_loss * 1.1)[0][0] + 1
    
    print(f"学习率 lr={lr}:")
    print(f"  最终损失值：{final_loss:.6f}")
    print(f"  训练过程最小损失：{min_loss:.6f}")
    print(f"  收敛所需轮数：{converge_epoch} 轮")
    print("-"*50)
print("="*80)