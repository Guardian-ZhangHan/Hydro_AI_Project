@echo off
echo ==============================================
echo  水文环境经济项目 一键备份脚本
echo ==============================================

:: 1. 备份 Conda 环境
echo [1/3] 正在备份 Conda 环境...
conda env export > hydro_ai_environment.yml
echo ? Conda 环境备份完成

:: 2. 压缩项目文件夹（带时间戳，避免覆盖）
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set backup_name=Hydro_AI_Project_backup_%timestamp%.zip
echo [2/3] 正在压缩项目文件夹为 %backup_name%...
powershell -command "Compress-Archive -Path '*' -DestinationPath '%userprofile%\Desktop\%backup_name%' -Force"
echo ? 项目压缩包已保存到桌面

:: 3. 可选：复制到 OneDrive/百度网盘（把下面路径改成你自己的）
:: echo [3/3] 正在复制到云盘备份...
:: copy "%userprofile%\Desktop\%backup_name%" "D:\BaiduNetdiskWorkspace\Hydro_AI_Backup\"
:: echo ? 云盘备份完成

echo.
echo ==============================================
echo ?? 全部备份完成！
echo ?? 备份文件：
echo   - 环境配置：hydro_ai_environment.yml
echo   - 项目压缩包：%backup_name%（在桌面）
echo ==============================================
pause