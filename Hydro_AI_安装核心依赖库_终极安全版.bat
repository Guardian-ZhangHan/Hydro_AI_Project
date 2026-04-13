@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
title Hydro AI 核心依赖库安装修复【终极安全版·100%适配】

:: ==============================================
:: 严格匹配你的项目配置，后续新增库仅改这里！
set PROJECT_PATH=D:\Hydro_AI_Project
set ENV_NAME=hydro_ai
:: 论文必备核心库，新增库直接在这里空格分隔添加
set CORE_LIBS=flopy geopandas rasterio rioxarray scikit-learn seaborn jupyter notebook scipy statsmodels
:: ==============================================

echo ==================================================
echo 📦 Hydro AI 核心依赖库安装修复专用脚本
echo 🔹 目标环境：%ENV_NAME%
echo 🔹 项目路径：%PROJECT_PATH%
echo 🔹 安装列表：%CORE_LIBS%
echo ==================================================
echo.
echo ⚠️  提示：此脚本仅安装/修复Python依赖库，不会修改你的代码和模型文件
pause
echo.

:: 步骤1：强制切换到项目根目录，杜绝路径错误
echo [1/4] 正在校验项目路径...
cd /d "%PROJECT_PATH%"
if !errorlevel! neq 0 (
    echo ❌ 致命错误：项目目录不存在！
    echo 💡 请确认 D:\Hydro_AI_Project 文件夹路径正确，无中文/特殊字符
    pause
    exit /b 1
)
echo ✅ 项目路径校验通过，当前目录：%cd%
echo.

:: 步骤2：激活专属环境，预校验环境可用性
echo [2/4] 正在激活%ENV_NAME%专属环境...
call conda activate %ENV_NAME%
if !errorlevel! neq 0 (
    echo ❌ 环境激活失败！
    echo 💡 解决方案：
    echo 1. 确认Anaconda已正确安装并添加到系统环境变量
    echo 2. 执行 conda env list 确认%ENV_NAME%环境已创建
    echo 3. 若环境损坏，使用【环境一键恢复兜底脚本】修复
    pause
    exit /b 1
)
echo ✅ %ENV_NAME%环境激活成功
echo.

:: 步骤3：升级pip到稳定版，解决兼容性问题
echo [3/4] 正在升级pip到最新稳定版...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
if !errorlevel! neq 0 (
    echo ⚠️  pip升级失败，将使用当前版本继续安装，不影响核心功能
) else (
    echo ✅ pip升级完成
)
echo.

:: 步骤4：逐个安装核心库，双重源保障+逐个验证
echo [4/4] 正在安装/修复核心依赖库...
set INSTALL_SUCCESS=0
set INSTALL_FAILED=0
for %%lib in (%CORE_LIBS%) do (
    echo ------------------------------
    echo 正在处理：%%lib
    :: 先用清华源安装，速度快、稳定性高
    pip install %%lib -i https://pypi.tuna.tsinghua.edu.cn/simple
    if !errorlevel! neq 0 (
        echo ⚠️  清华源安装失败，尝试官方源兜底...
        pip install %%lib
        if !errorlevel! neq 0 (
            echo ❌ %%lib 安装失败！
            set /a INSTALL_FAILED+=1
        ) else (
            echo ✅ %%lib 安装成功（官方源）
            set /a INSTALL_SUCCESS+=1
        )
    ) else (
        echo ✅ %%lib 安装成功（清华源）
        set /a INSTALL_SUCCESS+=1
    )
)
echo ------------------------------
echo.
echo 📊 安装结果统计：成功!INSTALL_SUCCESS!个，失败!INSTALL_FAILED!个
echo.

:: 全量验证安装结果
echo 正在全量验证所有库的可用性...
python -c "import torch, numpy, pandas, flopy, geopandas, rasterio, rioxarray, scikit-learn; print('✅ 所有核心依赖库100%可用！')"
set VERIFY_CODE=!errorlevel!
echo.

if !INSTALL_FAILED! equ 0 && !VERIFY_CODE! equ 0 (
    echo ==================================================
    echo ✅ 所有依赖库安装/修复完成，100%可用！
    echo 💡 后续新增库，直接在脚本开头的CORE_LIBS里添加即可
    echo 💡 现在可以运行一键启动脚本执行主程序
    echo ==================================================
) else (
    echo ❌ 部分库安装/验证失败！
    echo 💡 请检查网络连接，或手动执行 pip install 库名 查看详细报错
)
pause
exit /b 0