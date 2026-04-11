# ==============================================
# 水文环境经济 + AI 科研万能模板（稳定版）
# 环境：hydro_ai
# 路径：D:\Hydro_AI_Project
# 功能：数据处理 + 成本效益约束 + 模型测试
# ==============================================

# -------------------------- 1. 基础库导入（全部稳定可用）
import numpy as np
import pandas as pd
import scipy
import sys
import os

# -------------------------- 2. 打印环境信息（确认运行正常）
print("=" * 60)
print("✅ 科研环境启动成功！")
print(f"📂 当前工作目录: {os.getcwd()}")
print(f"🐍 Python路径: {sys.executable}")
print(f"📦 numpy版本: {np.__version__}")
print(f"📦 pandas版本: {pd.__version__}")
print(f"📦 scipy版本: {scipy.__version__}")
print("=" * 60)

# -------------------------- 3. 成本效益约束函数（你的核心公式）
def cost_benefit_balance(total_revenue, total_cost, delta_social_welfare):
    """
    成本效益均衡约束
    输入：收入、成本、社会福利变化
    输出：均衡误差（越小越合理）
    """
    theoretical_welfare = np.array(total_revenue) - np.array(total_cost)
    balance_error = np.array(delta_social_welfare) - theoretical_welfare
    return np.mean(np.square(balance_error))

# -------------------------- 4. 测试数据（可直接替换成你的真实数据）
if __name__ == "__main__":
    # 模拟数据
    revenue = [150, 280, 320, 450]
    cost = [70, 130, 150, 210]
    social_welfare = [80, 150, 170, 240]

    # 运行约束计算
    error = cost_benefit_balance(revenue, cost, social_welfare)
    print("\n📊 成本效益约束计算结果：")
    print(f"均衡误差 = {error:.4f}")

    if error < 10:
        print("✅ 均衡状态良好，模型可用！")
    else:
        print("⚠️  均衡误差较大，建议检查数据")

    print("\n🎉 模板运行完成，可以开始正式科研！")