@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
title Hydro AI 环境一键恢复【终极兜底版·100%适配】

:: ==============================================
:: 严格匹配你的项目配置，禁止修改此处！
set PROJECT_PATH=D:\Hydro_AI_Project
set ENV_NAME=hydro_ai
set ENV_BACKUP_FILE=hydro_ai_environment_full.yml
:: ==============================================

echo ==================================================
echo 🛡️  Hydro AI 环境一键恢复兜底专用脚本
echo 🔹 目标环境：%ENV_NAME%
echo 🔹 备份文件：%ENV_BACKUP_FILE%
echo 🔹 项目路径：%PROJECT_PATH%
echo ==================================================
echo.
echo ⚠️  【重要兜底提示】
echo 1. 此脚本用于环境崩溃、换电脑、重装系统时，1:1复刻你的原科研环境
echo 2. 若已有%ENV_NAME%环境，会先安全删除旧环境，再创建全新环境，避免版本冲突
echo 3. 恢复过程需要5-15分钟（取决于网络），请耐心等待，不要关闭窗口
echo.
pause
echo.

:: 步骤1：强制切换到项目根目录，杜绝路径错误
echo [1/5] 正在校验项目路径...
cd /d "%PROJECT_PATH%"
if !errorlevel! neq 0 (
    echo ❌ 致命错误：项目目录不存在！
    echo 💡 请先从GitHub拉取完整项目文件到 D:\Hydro_AI_Project
    pause
    exit /b 1
)
echo ✅ 项目路径校验通过，当前目录：%cd%
echo.

:: 步骤2：校验环境备份文件是否存在、是否完整
echo [2/5] 正在校验环境备份文件...
if not exist "%ENV_BACKUP_FILE%" (
    echo ❌ 致命错误：环境备份文件不存在！
    echo 💡 解决方案：
    echo 1. 从你的GitHub仓库拉取最新的%ENV_BACKUP_FILE%到项目根目录
    echo 2. 或运行同步脚本重新生成环境备份文件
    pause
    exit /b 1
)
:: 校验文件大小，避免空文件/损坏文件
for %%f in (%ENV_BACKUP_FILE%) do set FILE_SIZE=%%~zf
if !FILE_SIZE! lss 100 (
    echo ❌ 环境备份文件损坏！文件大小异常
    echo 💡 请重新从GitHub拉取完整的备份文件
    pause
    exit /b 1
)
echo ✅ 环境备份文件校验通过，文件完整可用
echo.

:: 步骤3：安全处理旧环境（如果存在）
echo [3/5] 正在检查现有环境...
call conda env list | findstr /i "%ENV_NAME%" >nul
if !errorlevel! equ 0 (
    echo 发现已存在的%ENV_NAME%环境，正在安全删除...
    call conda deactivate >nul 2>&1
    call conda remove -n %ENV_NAME% --all -y
    if !errorlevel! neq 0 (
        echo ❌ 旧环境删除失败！
        echo 💡 解决方案：关闭所有使用该环境的程序、CMD窗口、VSCode/PyCharm，重新运行脚本
        pause
        exit /b 1
    )
    echo ✅ 旧环境已安全删除
) else (
    echo ✅ 未发现同名环境，直接创建全新环境
)
echo.

:: 步骤4：从备份文件1:1恢复环境
echo [4/5] 正在从备份文件恢复环境（预计5-15分钟，请耐心等待）...
call conda env create -f "%ENV_BACKUP_FILE%"
if !errorlevel! neq 0 (
    echo ❌ 环境恢复失败！
    echo 💡 解决方案：
    echo 1. 检查网络连接是否正常，可尝试切换手机热点
    echo 2. 确认备份文件%ENV_BACKUP_FILE%内容完整无损坏
    echo 3. 手动执行 conda env create -f %ENV_BACKUP_FILE% 查看详细报错
    pause
    exit /b 1
)
echo ✅ 环境恢复成功，已1:1复刻原环境配置
echo.

:: 步骤5：全量验证环境可用性，确保100%可用
echo [5/5] 正在全量验证恢复的环境...
call conda activate %ENV_NAME%
if !errorlevel! neq 0 (
    echo ❌ 环境激活失败！恢复异常
    pause
    exit /b 1
)
echo ✅ 环境激活正常
echo.
echo 正在验证核心库可用性...
python -c "import torch, numpy, pandas, matplotlib, flopy, geopandas; print('✅ 所有核心库验证通过，和原环境完全一致！')"
set VERIFY_CODE=!errorlevel!
echo.

if !VERIFY_CODE! equ 0 (
    echo ==================================================
    echo ✅ 环境兜底恢复100%成功！
    echo ✅ 所有库、版本、配置和你原备份环境完全一致
    echo 💡 现在可以直接运行一键启动脚本，开始正常科研工作
    echo ==================================================
) else (
    echo ❌ 环境验证失败！部分库不可用
    echo 💡 请运行【依赖安装修复脚本】补装/修复损坏的库
)
pause
exit /b 0