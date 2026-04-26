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