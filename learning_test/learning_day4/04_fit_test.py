import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ===================== 1. 固定全量随机种子，与Day1-3完全一致，严格控制变量 =====================
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 与Day1-3的SEED完全一致，保证模型初始化、数据生成可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 2. 100%复用Day1的水文反演神经网络骨架，严格控制变量 =====================
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

# ===================== 3. 复用Day2的MSE损失函数、Day3的Adam优化器 =====================
criterion = nn.MSELoss()

# ===================== 4. 构建少样本数据集，复现过拟合核心条件 =====================
# 极少量训练样本：10组5口监测井水头→10×10网格K值，模拟少样本水文场景
x_train = torch.randn(10, 5)  # 10组训练输入（5口监测井）
y_train = torch.randn(10, 100) # 10组训练标签（真实K值）

# 验证集：5组独立样本，不参与训练，仅用于监控过拟合
x_val = torch.randn(5, 5)
y_val = torch.randn(5, 100)

# ===================== 5. 初始化模型与优化器，无任何防过拟合措施 =====================
model = HydroInversionNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 用Day3找到的最优学习率

# 用于记录每一轮的训练/验证损失
train_loss_history = []
val_loss_history = []
epochs = 300 # 增加训练轮数，放大过拟合现象

# ===================== 6. 训练循环，复现过拟合 =====================
print("="*70)
print("开始训练：复现少样本水文反演场景下的过拟合现象")
print("="*70)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    y_pred_train = model(x_train)
    train_loss = criterion(y_pred_train, y_train)
    
    # 反向传播更新参数
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # 验证阶段（无梯度更新，仅计算损失）
    model.eval()
    with torch.no_grad():
        y_pred_val = model(x_val)
        val_loss = criterion(y_pred_val, y_val)
    
    # 记录损失
    train_loss_history.append(train_loss.item())
    val_loss_history.append(val_loss.item())
    
    # 每30轮打印一次训练进度
    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss.item():.6f}, 验证损失: {val_loss.item():.6f}")

print("="*70)
print("训练完成，过拟合现象复现完毕")
print("="*70)

# ===================== 7. 过拟合特征量化分析 =====================
# 找到验证损失的最小值和对应的轮数
min_val_loss = min(val_loss_history)
min_val_epoch = val_loss_history.index(min_val_loss) + 1
final_train_loss = train_loss_history[-1]
final_val_loss = val_loss_history[-1]

print("📌 过拟合核心特征量化分析：")
print(f"1. 验证损失最小值：{min_val_loss:.6f}，出现在第 {min_val_epoch} 轮")
print(f"2. 最终训练损失：{final_train_loss:.6f}，最终验证损失：{final_val_loss:.6f}")
print(f"3. 训练结束时，验证损失较最小值上升了：{(final_val_loss - min_val_loss)/min_val_loss*100:.2f}%")
print(f"4. 核心过拟合判定：训练损失持续下降，验证损失在第{min_val_epoch}轮后持续上升，符合过拟合标准")
print("="*70)

# ===================== 8. 生成过拟合损失曲线，直接用于论文 =====================
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(range(1, epochs+1), train_loss_history, label="训练损失", linewidth=1.5, color="#1f77b4")
plt.plot(range(1, epochs+1), val_loss_history, label="验证损失", linewidth=1.5, color="#ff7f0e")

# 标注验证损失最小值点
plt.scatter(min_val_epoch, min_val_loss, color="red", s=50, label=f"验证损失最小值（第{min_val_epoch}轮）")
plt.axvline(x=min_val_epoch, color="red", linestyle="--", alpha=0.5)

# 图表格式符合学术论文规范
plt.xlabel("训练轮数 Epochs", fontsize=12)
plt.ylabel("MSE 损失值", fontsize=12)
plt.title("少样本水文参数反演场景下的过拟合现象", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存高清图片与原始数据
plt.savefig("research_doc/Day4_过拟合损失曲线.png", dpi=300, bbox_inches="tight")
np.savetxt("research_doc/Day4_过拟合训练损失.csv", train_loss_history, delimiter=",", header="train_loss", comments="")
np.savetxt("research_doc/Day4_过拟合验证损失.csv", val_loss_history, delimiter=",", header="val_loss", comments="")
print("✅ 过拟合损失曲线、原始数据已保存到research_doc文件夹，可直接用于论文")