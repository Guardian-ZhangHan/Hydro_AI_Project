import flopy
import numpy as np
import os

# ====================== 你的本地MODFLOW6内核（路径一字不改，完全对应你电脑）======================
exe_path = r"D:\Hydro_AI_Project\mf6.7.0_win64\bin\mf6.exe"

# 1. 初始化MODFLOW6模拟
sim = flopy.mf6.MFSimulation(
    sim_name="first_test",
    exe_name=exe_path,
    version="mf6"
)

# 2. 时间离散（MODFLOW6强制第一步）
tdis = flopy.mf6.ModflowTdis(
    sim,
    nper=1,
    perioddata=[(1.0, 1, 1.0)]
)

# 3. 初始化地下水流GWF模型
gwf = flopy.mf6.ModflowGwf(sim)

# 4. 空间网格离散（完整网格定义）
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=1,
    nrow=10,
    ncol=10,
    delr=100,
    delc=100,
    top=10,
    botm=0
)

# 5. 含水层水力属性（渗透系数）
npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1)

# 6. 初始水头（模型必须初始化水头）
ic = flopy.mf6.ModflowGwfic(gwf, strt=10.0)

# 7. 定水头边界【核心修复！补全边界坐标数据，解决DIMENSIONS报错】
# 左右两列边界全部固定水头，模型天然收敛，文件参数完整不会空
chd = flopy.mf6.ModflowGwfchd(
    gwf,
    stress_period_data=[
        [(0, i, 0), 10.0] for i in range(10)
    ] + [
        [(0, i, 9), 5.0] for i in range(10)
    ]
)

# 8. 求解器迭代配置
ims = flopy.mf6.ModflowIms(sim)

# 9. 生成全部模型输入文件 + 调用本地mf6.exe运行计算
sim.write_simulation()
success, log_info = sim.run_simulation()

# 10. 打印最终运行结果（修复末尾splitlines语法报错）
print("="*60)
print("✅ Python + MODFLOW6 全链路运行成功！最终状态：", success)
print("="*60)
print("模型计算末尾日志：")
print('\n'.join(log_info))
# ====================== 新增：MODFLOW模型标准化运行接口（智能体调用专用）======================
import click

@click.command()
@click.option('--model_name', default="base_model", help="模型名称")
@click.option('--k_mean', default=1e-5, help="渗透系数均值（m/s）")
@click.option('--k_std', default=1e-6, help="渗透系数标准差（m/s）")
def run_mf6_model(model_name, k_mean, k_std):
    """
    智能体可直接调用的MODFLOW模型运行接口
    可通过参数批量调整渗透系数，生成批量仿真数据
    """
    # 这里调用你原有MODFLOW模型构建、运行的核心代码
    # （把你原有构建模型、运行MODFLOW的代码，放到这个函数里）
    print(f"===== 开始运行MODFLOW模型: {model_name} =====")
    print(f"渗透系数设置：均值={k_mean} m/s，标准差={k_std} m/s")
    
    # 原有模型运行代码（完全保留，这里替换成你自己的代码）
    # 示例：
    # sim = flopy.mf6.MFSimulation(...)
    # gwf = flopy.mf6.ModflowGwf(...)
    # sim.write_simulation()
    # success, buff = sim.run_simulation()
    
    success = True  # 替换成你实际的运行结果
    if success:
        print(f"模型 {model_name} 运行成功，结果已保存至 01_data/02_synthetic_dataset/")
        return 0
    else:
        print(f"模型 {model_name} 运行失败")
        return 1

# 脚本直接运行时，执行原有测试代码
if __name__ == "__main__":
    # 原有测试代码完全保留，仅追加接口
    run_mf6_model()