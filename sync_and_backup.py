import os
import sys
import subprocess
from datetime import datetime

def run_command(cmd, desc):
    """运行命令并打印输出"""
    print(f"\n{desc}...")
    try:
        # 使用 shell=True 以支持 conda 命令
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"✅ {desc} 完成！")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} 失败！错误码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"⚠️  {desc} 出现未知错误: {e}")
        sys.exit(1)

def main():
    print("="*50)
    print(f"HYDRO AI DAILY BACKUP + GITHUB SYNC")
    print(f"TIME: {datetime.now()}")
    print("="*50)

    # 1. 切换目录 (确保路径正确)
    project_dir = r"D:\Hydro_AI_Project"
    print(f"\n📂 正在切换目录至: {project_dir}")
    os.chdir(project_dir)
    print("✅ 目录切换成功")

    # 2. 激活环境并执行后续命令 (通过 conda run)
    # 这是最稳定的方式，替代 activate
    print(f"\n🔧 正在激活环境: hydro_ai")
    # 测试环境是否存在
    subprocess.run("conda env list | findstr hydro_ai", shell=True, check=True)
    print("✅ 环境已找到")

    # 3. 备份 Conda 环境 (核心修复：用 python 调用 conda，避免中文/卡死)
    print(f"\n💾 正在备份 Conda 环境...")
    # 导出完整环境
    full_env_cmd = f"conda env export -n hydro_ai > hydro_ai_environment_full.yml"
    subprocess.run(full_env_cmd, shell=True, check=True)
    
    # 导出精简环境
    clean_env_cmd = f"conda env export -n hydro_ai --from-history > hydro_ai_environment_cleaned.yml"
    subprocess.run(clean_env_cmd, shell=True, check=True)
    
    print("✅ Conda 环境备份完成！")

    # 4. 锁定 Pip 依赖
    print(f"\n📦 正在锁定 Pip 依赖...")
    subprocess.run("pip freeze > requirements.txt", shell=True, check=True)
    print("✅ Pip 依赖锁定完成！")

    # 5. GitHub 同步
    print(f"\n🔄 正在同步到 GitHub...")
    subprocess.run("git add .", shell=True, check=True)
    
    commit_msg = f"Auto-sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    subprocess.run(f'git commit -m "{commit_msg}"', shell=True, check=True)
    
    subprocess.run("git push", shell=True, check=True)
    print("✅ GitHub 同步成功！")

    print("\n" + "="*50)
    print("🎉 所有任务完成！数据已安全备份。")
    print("="*50)
    input("按任意键退出...") # 防止窗口一闪而过

if __name__ == "__main__":
    main()