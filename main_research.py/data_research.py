# ==============================================
# 水文环境经济 + AI 科研模板（Excel 数据版）
# 功能：读取 Excel 数据 → 计算成本效益约束 → 输出结果
# ==============================================

# -------------------------- 1. 导入库（新增 openpyxl 用于读 Excel）
import numpy as np
import pandas as pd
import scipy
import sys
import os

# -------------------------- 2. 打印环境信息
print("=" * 60)
print("✅ 科研环境启动成功！")
print(f"📂 当前工作目录: {os.getcwd()}")
print(f"🐍 Python路径: {sys.executable}")
print(f"📦 numpy版本: {np.__version__}")
print(f"📦 pandas版本: {pd.__version__}")
print(f"📦 scipy版本: {scipy.__version__}")
print("=" * 60)

# -------------------------- 3. 成本效益约束函数（和之前一样）
def cost_benefit_balance(total_revenue, total_cost, delta_social_welfare):
    """
    成本效益均衡约束
    输入：收入、成本、社会福利变化（可以是列表或 Excel 列）
    输出：均衡误差（越小越合理）
    """
    theoretical_welfare = np.array(total_revenue) - np.array(total_cost)
    balance_error = np.array(delta_social_welfare) - theoretical_welfare
    return np.mean(np.square(balance_error))

# -------------------------- 4. 读取 Excel 数据（核心新功能）
def load_data_from_excel(file_path):
    """
    从 Excel 文件加载数据
    要求 Excel 里有三列：'revenue'（收入）、'cost'（成本）、'welfare'（社会福利）
    """
    try:
        df = pd.read_excel(file_path)
        print(f"✅ 成功读取数据：{file_path}")
        print(f"📊 数据行数: {len(df)}")
        print(df.head())  # 打印前5行预览
        return df['revenue'], df['cost'], df['welfare']
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print("⚠️  请确保 Excel 文件路径正确，且列名是 'revenue', 'cost', 'welfare'")
        return None, None, None

# -------------------------- 5. 主程序：选择用模拟数据还是 Excel 数据
if __name__ == "__main__":
    # 选项 A：用模拟数据（和之前一样，方便测试）
    use_mock_data = False  # 改成 False 就会读 Excel
    
    if use_mock_data:
        print("\n📌 使用模拟数据测试...")
        revenue = [150, 280, 320, 450]
        cost = [70, 130, 150, 210]
        social_welfare = [80, 150, 170, 240]
    else:
        # 选项 B：读你的 Excel 文件（把路径改成你自己的）
        excel_file = "D:\\Hydro_AI_Project\\hydro_data.xlsx"  # 注意双反斜杠
        revenue, cost, social_welfare = load_data_from_excel(excel_file)
        if revenue is None:
            exit()  # 读失败就退出

    # 计算约束
    error = cost_benefit_balance(revenue, cost, social_welfare)
    print("\n📊 成本效益约束计算结果：")
    print(f"均衡误差 = {error:.4f}")

    if error < 10:
        print("✅ 均衡状态良好，模型可用！")
    else:
        print("⚠️  均衡误差较大，建议检查数据")

    print("\n🎉 模板运行完成！")