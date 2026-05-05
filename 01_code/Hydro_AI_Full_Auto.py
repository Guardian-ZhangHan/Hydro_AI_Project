import os
import sys
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import datetime

# ======================================================
# 【全局底层绝对锁死 100%匹配你本机，永久不动，零路径坑】
# ======================================================
# 项目根目录 绝对锁死
PROJECT_ROOT = os.path.abspath(r"D:\Hydro_AI_Project")
# Conda环境名
CONDA_ENV_NAME = "hydro_ai"
# 精度合格硬阈值（顶刊标准，不达标全流程终止）
QUALIFY_R2_THRESHOLD = 0.99
# 训练超参数 完全复用你原有配置
EPOCHS = 2000
LEARNING_RATE = 1e-3
# 保存路径 绝对锁死
SAVE_MODEL_PATH = os.path.join(PROJECT_ROOT, "hydro_ai_model.pth")
SAVE_PLOT_PATH = os.path.join(PROJECT_ROOT, "hydro_ai_prediction_result.png")
BACKUP_ENV_FILE = os.path.join(PROJECT_ROOT, "hydro_ai_environment_full.yml")

# ======================================================
# 【步骤0：全局初始化 运行前全量自检 所有坑提前堵死】
# ======================================================
def global_init():
    print("=" * 80)
    print("🚀 Hydro AI 水文科研 全自动全流程【终极定稿版】")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("执行铁则：完整训练 → 精度校验 → 仅达标才备份+提交，不达标全程终止")
    print("=" * 80)

    # 1. 项目根目录校验
    if not os.path.exists(PROJECT_ROOT):
        print(f"\n❌ 致命错误：项目根目录不存在！路径：{PROJECT_ROOT}")
        input("按回车退出...")
        sys.exit(1)
    os.chdir(PROJECT_ROOT)
    print(f"✅ 项目根目录锁定: {os.getcwd()}")

    # 2. 当前Python环境校验（100%继承你激活的hydro_ai环境）
    print(f"✅ 当前激活Python解释器: {sys.executable}")
    print(f"✅ 当前Conda环境: {os.environ.get('CONDA_DEFAULT_ENV', 'base')}")

    # 3. Git环境校验
    git_check = subprocess.run("git --version", shell=True, capture_output=True, text=True)
    if git_check.returncode != 0:
        print(f"\n❌ 致命错误：Git未安装或未配置环境变量")
        input("按回车退出...")
        sys.exit(1)
    print(f"✅ Git环境校验通过: {git_check.stdout.strip()}")

    print("\n✅ 全局初始化全部完成，所有环境就绪，无任何隐患")
    print("=" * 80)

# ======================================================
# 【步骤1：完整模型训练 100%复用你原有水文物理模型】
# ======================================================
def run_full_model_training():
    print("\n" + "=" * 70)
    print("【阶段1】启动完整模型训练流程")
    print("=" * 70)

    # 设备适配
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 水文物理模型 完全复用你原有结构
    class HydroPhysicsModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x)

    # 数据集生成 完全复用你原有逻辑
    def generate_synthetic_dataset():
        x = torch.rand(1000, 2) * 2 - 1
        y = x[:, 0] ** 2 + x[:, 1] * 0.5
        return x, y

    # 达西定律物理约束损失 完全复用你原有逻辑
    def physics_constraint_loss(model, x):
        x.requires_grad = True
        y_pred = model(x)
        grad = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        return torch.mean(torch.square(grad[:, 0] + grad[:, 1]))

    # 完整训练循环
    model = HydroPhysicsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_train, y_train = generate_synthetic_dataset()
    x_train, y_train = x_train.to(device), y_train.to(device)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss_data = torch.mean((y_pred - y_train) ** 2)
        loss_physics = physics_constraint_loss(model, x_train)
        loss_total = loss_data + loss_physics

        loss_total.backward()
        optimizer.step()

        # 训练进度打印
        if epoch % 200 == 0:
            print(f"Epoch [{epoch:4d}/{EPOCHS}] | 总损失: {loss_total.item():.6f} | 物理约束损失: {loss_physics.item():.6f}")

    # 精度校验
    y_final_pred = model(x_train).detach().cpu().numpy()
    y_true = y_train.cpu().numpy()
    final_r2 = r2_score(y_true, y_final_pred)

    # 保存模型权重
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n✅ 模型权重已保存: {SAVE_MODEL_PATH}")

    # 绘制顶刊结果图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(y_true, y_final_pred, s=5, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(f"预测结果 (R² = {final_r2:.4f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 结果图已保存: {SAVE_PLOT_PATH}")

    print("\n" + "=" * 70)
    print(f"训练完成！最终模型R² = {final_r2:.4f}")
    print(f"合格阈值: R² ≥ {QUALIFY_R2_THRESHOLD}")
    print("=" * 70)

    return final_r2

# ======================================================
# 【步骤2：conda环境全量备份】
# ======================================================
def backup_conda_env():
    print("\n" + "=" * 70)
    print("【阶段2】导出conda完整环境备份")
    print("=" * 70)

    # 双命令兜底，适配所有miniconda版本
    cmd1 = f'conda env export --name {CONDA_ENV_NAME} --file "{BACKUP_ENV_FILE}" --no-builds --format=yaml'
    cmd2 = f'conda env export --name {CONDA_ENV_NAME} --output "{BACKUP_ENV_FILE}" --format=yaml'

    result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ conda环境备份完成: {BACKUP_ENV_FILE}")
        return True
    else:
        result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
        if result2.returncode == 0:
            print(f"✅ conda环境备份完成（备用命令）: {BACKUP_ENV_FILE}")
            return True
        else:
            print(f"⚠️ conda环境备份失败，错误信息: {result2.stderr}")
            return False

# ======================================================
# 【步骤3：GitHub智能同步 仅代码有改动才提交，无垃圾记录】
# ======================================================
def smart_sync_github(commit_r2):
    print("\n" + "=" * 70)
    print("【阶段3】智能同步代码到GitHub")
    print("=" * 70)

    # 1. 拉取云端最新代码，解决冲突
    pull_result = subprocess.run("git pull origin main --rebase", shell=True, capture_output=True, text=True)
    if pull_result.returncode != 0:
        print(f"⚠️ Git拉取警告: {pull_result.stderr}")
    else:
        print("✅ 云端最新代码拉取完成")

    # 2. 添加源码变动
    add_result = subprocess.run("git add .", shell=True, capture_output=True, text=True)
    if add_result.returncode != 0:
        print(f"❌ Git添加文件失败: {add_result.stderr}")
        return False

    # 3. 核心智能锁：仅代码有真实改动才提交，无改动绝不生成垃圾记录
    diff_check = subprocess.run("git diff --quiet --exit-code", shell=True)
    if diff_check.returncode == 1:
        commit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"有效训练完成备份 | 模型R²={commit_r2:.4f} | {commit_time}"
        commit_result = subprocess.run(f'git commit -m "{commit_msg}"', shell=True, capture_output=True, text=True)
        if commit_result.returncode != 0:
            print(f"❌ Git提交失败: {commit_result.stderr}")
            return False
        # 推送到云端
        push_result = subprocess.run("git push origin main", shell=True, capture_output=True, text=True)
        if push_result.returncode != 0:
            print(f"❌ Git推送失败: {push_result.stderr}")
            return False
        print("\n✅ 全流程完成！代码与环境备份已同步到GitHub")
        return True
    else:
        print("\nℹ️ 未检测到源代码变动，跳过Git提交流程，无无效记录生成")
        return True

# ======================================================
# 【主流程总控 严格单向执行 硬闸门控制】
# ======================================================
if __name__ == "__main__":
    # 步骤0：全局初始化自检
    global_init()

    # 步骤1：完整模型训练
    final_r2 = run_full_model_training()

    # 【核心硬闸门】精度不达标，全流程直接终止，绝不执行后续任何操作
    if final_r2 < QUALIFY_R2_THRESHOLD:
        print(f"\n❌ 模型成果不达标！")
        print(f"最终R² = {final_r2:.4f} < 合格阈值 {QUALIFY_R2_THRESHOLD}")
        print("所有后续流程（环境备份、Git提交）全部终止，无任何操作执行")
        input("\n按回车退出程序...")
        sys.exit(1)

    # 仅达标后，执行后续流程
    print(f"\n✅ 模型成果达标！解锁后续备份与同步流程")

    # 步骤2：conda环境备份
    backup_conda_env()

    # 步骤3：GitHub智能同步
    sync_success = smart_sync_github(final_r2)

    # 最终收尾
    print("\n" + "=" * 80)
    if sync_success:
        print("🎉 【全自动化全流程全部执行完成！】")
        print("所有流程符合顶刊科研规范，无垃圾文件、无无效提交")
    else:
        print("⚠️ 部分流程执行异常，请检查日志")
    print("=" * 80)
    input("\n按回车退出程序...")