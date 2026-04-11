@echo off
chcp 65001 >nul
title Hydro AI — 一键启动环境
setlocal enabledelayedexpansion

:: ==============================================
:: 🔧 核心配置（已适配你的路径，无需修改）
:: ==============================================
set "PROJECT_PATH=D:\Hydro_AI_Project"
set "ENV_NAME=hydro_ai"

:: ==============================================
:: 1. 自动定位并激活conda环境（兼容Anaconda/Miniconda）
:: ==============================================
echo.
echo ==================================================
echo           Hydro AI 环境一键启动脚本
echo ==================================================
echo.
echo 正在激活 %ENV_NAME% 环境...

:: 自动查找conda安装路径
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
    set "CONDA_ROOT=%USERPROFILE%\miniconda3"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
    set "CONDA_ROOT=%USERPROFILE%\anaconda3"
) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=C:\ProgramData\miniconda3\Scripts\activate.bat"
    set "CONDA_ROOT=C:\ProgramData\miniconda3"
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_ACTIVATE=C:\ProgramData\anaconda3\Scripts\activate.bat"
    set "CONDA_ROOT=C:\ProgramData\anaconda3"
) else (
    echo ❌ 错误：未找到conda环境！请检查Anaconda/Miniconda是否正确安装。
    pause
    exit /b 1
)

:: 激活conda基础环境
call "!CONDA_ACTIVATE!" "!CONDA_ROOT!"
if !errorlevel! neq 0 (
    echo ❌ 错误：conda基础环境激活失败！
    pause
    exit /b 1
)

:: 激活hydro_ai环境
call conda activate %ENV_NAME%
if !errorlevel! neq 0 (
    echo ❌ 错误：%ENV_NAME% 环境激活失败！请执行 conda env list 确认环境存在。
    pause
    exit /b 1
)
echo ✅ 环境激活成功！当前环境：%ENV_NAME%

:: ==============================================
:: 2. 切换到你的项目目录（已适配你的D盘路径）
:: ==============================================
echo.
echo 正在切换到项目目录...
cd /d "%PROJECT_PATH%"
if !errorlevel! neq 0 (
    echo ❌ 错误：项目目录 %PROJECT_PATH% 不存在！
    pause
    exit /b 1
)
echo ✅ 已切换到项目目录：%cd%

:: ==============================================
:: 3. 异步启动VS Code（彻底解决卡死问题）
:: ==============================================
echo.
echo 正在打开 VS Code 项目目录...

:: 优先使用全局code命令
where code >nul 2>&1
if !errorlevel! equ 0 (
    start "" code .
    echo ✅ VS Code 已通过全局命令启动！
) else (
    :: 兜底：自动匹配常见VS Code安装路径
    if exist "%USERPROFILE%\AppData\Local\Programs\Microsoft VS Code\Code.exe" (
        start "" "%USERPROFILE%\AppData\Local\Programs\Microsoft VS Code\Code.exe" .
        echo ✅ VS Code 已通过用户安装路径启动！
    ) else if exist "C:\Program Files\Microsoft VS Code\Code.exe" (
        start "" "C:\Program Files\Microsoft VS Code\Code.exe" .
        echo ✅ VS Code 已通过系统安装路径启动！
    ) else (
        echo ❌ 错误：未找到VS Code程序！请检查VS Code是否正确安装。
        pause
        exit /b 1
    )
)

:: ==============================================
:: 4. 脚本自动退出，永不卡死
:: ==============================================
echo.
echo ==================================================
echo ✅ Hydro AI 环境启动完成！3秒后自动关闭窗口...
echo ==================================================
timeout /t 3 /nobreak >nul
endlocal
exit /b 0