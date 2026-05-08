# ===================== 1. 固定全量随机种子，与任务3完全一致，保证控制变量 =====================
import random
import numpy as np
import torch
import torch.nn as nn

# 必须与01_nn_structure.py的SEED完全一致，保证权重初始化100%相同
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 2. 定义两个模型，唯一差异为激活函数，其余结构100%一致 =====================
# 模型1：Tanh激活函数，与任务3模型完全一致
class TanhHydroNN(nn.Module):
    def __init__(self):
        super(TanhHydroNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 100)
        self.activation = nn.Tanh()  # 唯一变量：Tanh激活

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型2：ReLU激活函数，其余结构与模型1完全一致
class ReLUHydroNN(nn.Module):
    def __init__(self):
        super(ReLUHydroNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 100)
        self.activation = nn.ReLU()  # 唯一变量：ReLU激活

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# ===================== 3. 对照实验执行，所有条件完全固定 =====================
if __name__ == "__main__":
    # 初始化两个模型，种子固定，权重初始化100%一致
    model_tanh = TanhHydroNN()
    model_relu = ReLUHydroNN()

    # 固定输入数据，与任务3完全一致，唯一变量仅为激活函数
    test_input = torch.randn(1, 5)
    print("="*60)
    print(f"固定测试输入（5口监测井水头）：{test_input.numpy().flatten()}")
    print("="*60)

    # 前向传播，无梯度计算，不修改权重
    with torch.no_grad():
        output_tanh = model_tanh(test_input)
        output_relu = model_relu(test_input)

    # 展平输出，用于统计分析
    output_tanh_np = output_tanh.numpy().flatten()
    output_relu_np = output_relu.numpy().flatten()

    # ===================== 4. 量化统计差异，学术实验必须量化 =====================
    print("Tanh激活函数输出K值统计特征：")
    print(f"均值：{np.mean(output_tanh_np):.4f}，标准差：{np.std(output_tanh_np):.4f}")
    print(f"最小值：{np.min(output_tanh_np):.4f}，最大值：{np.max(output_tanh_np):.4f}")
    print("-"*60)
    print("ReLU激活函数输出K值统计特征：")
    print(f"均值：{np.mean(output_relu_np):.4f}，标准差：{np.std(output_relu_np):.4f}")
    print(f"最小值：{np.min(output_relu_np):.4f}，最大值：{np.max(output_relu_np):.4f}")
    print("-"*60)

    # 计算两个输出的绝对差异
    abs_diff = np.abs(output_tanh_np - output_relu_np)
    print("Tanh与ReLU输出的绝对差异统计：")
    print(f"平均绝对差异：{np.mean(abs_diff):.4f}，最大绝对差异：{np.max(abs_diff):.4f}")
    print(f"差异大于0.1的网格占比：{np.sum(abs_diff > 0.1)/len(abs_diff)*100:.2f}%")
    print("="*60)

    # ===================== 5. 原始数据保存，用于论文补充材料 =====================
    np.savetxt("research_doc/tanh_output.csv", output_tanh_np, delimiter=",")
    np.savetxt("research_doc/relu_output.csv", output_relu_np, delimiter=",")
    np.savetxt("research_doc/activation_abs_diff.csv", abs_diff, delimiter=",")
    print("实验原始数据已保存到research_doc文件夹，可用于论文补充材料")