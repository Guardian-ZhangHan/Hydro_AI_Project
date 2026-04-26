# ===================== 1. 固定全量随机种子，保证实验100%可复现 =====================
# 学术规范：所有随机数相关库全部固定，与后续对照实验种子完全一致
import random
import numpy as np
import torch
import torch.nn as nn

# 学术通用固定种子，可修改但必须全程固定
SEED = 42
random.seed(SEED)                # 固定Python内置随机数
np.random.seed(SEED)             # 固定numpy随机数
torch.manual_seed(SEED)          # 固定PyTorch CPU随机数

# GPU环境固定（如有）
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 关闭cudnn非确定性算子，保证GPU计算结果完全一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 2. 定义水文反演专用全连接神经网络 =====================
# 物理意义：拟合 5口监测井水头 → 10×10网格K值 的非线性物理映射
class HydroInversionNN(nn.Module):
    def __init__(self):
        super(HydroInversionNN, self).__init__()
        # 输入层→隐藏层1：输入5维（5口监测井），输出32维（低阶特征提取）
        # 水文意义：从监测井水头中提取水力梯度、水头差等低阶水文特征
        self.fc1 = nn.Linear(in_features=5, out_features=32)
        # 隐藏层1→隐藏层2：输入32维，输出64维（高阶特征提取）
        # 水文意义：从低阶特征中提取与K直接相关的高阶非线性水文特征
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        # 隐藏层2→输出层：输入64维，输出100维（10×10网格K值）
        # 水文意义：将高阶特征映射为反演目标——K的空间分布
        self.fc3 = nn.Linear(in_features=64, out_features=100)
        # 激活函数：Tanh，实现非线性映射，拟合地下水动力学非线性方程
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 前向传播，严格对应水文特征提取流程
        x = self.tanh(self.fc1(x))  # 低阶特征提取+非线性激活
        x = self.tanh(self.fc2(x))  # 高阶特征提取+非线性激活
        x = self.fc3(x)              # 输出层，无激活函数（红线要求，保证物理意义）
        return x

# ===================== 3. 模型初始化与结构验证 =====================
if __name__ == "__main__":
    # 初始化模型
    model = HydroInversionNN()
    # 打印模型结构，验证是否符合要求
    print("="*50)
    print("水文地质参数反演神经网络结构：")
    print(model)
    print("="*50)

    # 构造测试输入：1组5口监测井的水头数据，维度[1,5]
    test_input = torch.randn(1, 5)
    # 测试前向传播，无梯度计算（不修改权重）
    with torch.no_grad():
        test_output = model(test_input)
    
    # 验证输入输出维度是否符合要求
    print(f"测试输入维度：{test_input.shape} （1组5口监测井水头）")
    print(f"测试输出维度：{test_output.shape} （1组10×10网格K值）")
    print("="*50)

    # 验证输出物理合理性：无离谱数值，初始状态符合K值基本特征
    print(f"输出K值最小值：{test_output.min().item():.4f}")
    print(f"输出K值最大值：{test_output.max().item():.4f}")
    print("="*50)