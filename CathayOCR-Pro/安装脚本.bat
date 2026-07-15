@echo off
chcp 65001 >nul
:: ============================================================
::  CathayOCR Pro 安装脚本
::  功能：创建桌面快捷方式 + 开始菜单快捷方式
::  运行环境：Windows 10/11，无需管理员权限
:: ============================================================

setlocal enabledelayedexpansion

:: 获取脚本所在目录（支持文件夹整体移动）
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: 获取文件夹名称
for %%i in ("%SCRIPT_DIR%") do set "FOLDER_NAME=%%~nxi"

:: 确认插件目录结构
set "EXPECTED_PLUGIN_DIR=%SCRIPT_DIR%"

:: 查找启动脚本（支持不同命名）
set "BAT_PATH=%SCRIPT_DIR%\启动 CathayOCR Pro.bat"
if not exist "%BAT_PATH%" set "BAT_PATH=%SCRIPT_DIR%\开启 CathayOCR Pro.bat"
if not exist "%BAT_PATH%" (
    echo [错误] 找不到启动 .bat 文件：
    echo   期望: 启动 CathayOCR Pro.bat
    echo   或:   开启 CathayOCR Pro.bat
    echo 请确认 CathayOCR Pro 已正确安装到插件目录。
    pause
    exit /b 1
)

if not exist "%EXPECTED_BAT%" (
    echo [错误] 找不到启动文件：
    echo %EXPECTED_BAT%
    echo 请确认 CathayOCR Pro 已正确安装到插件目录。
    pause
    exit /b 1
)

:: 检测 Umi-OCR 主目录（向上查找）
set "UMI_BASE="
for %%i in ("%SCRIPT_DIR%") do set "LEVEL1=%%~dpi"
for %%i in ("%LEVEL1:~0,-1%") do set "LEVEL2=%%~dpi"
for %%i in ("%LEVEL2:~0,-1%") do set "LEVEL3=%%~dpi"

if exist "%LEVEL3%Umi-OCR_Paddle_v2.1.5\UmiOCR-data\plugins\%FOLDER_NAME%" (
    set "UMI_BASE=%LEVEL3%Umi-OCR_Paddle_v2.1.5"
) else if exist "%LEVEL2%Umi-OCR_Paddle_v2.1.5\UmiOCR-data\plugins\%FOLDER_NAME%" (
    set "UMI_BASE=%LEVEL2%Umi-OCR_Paddle_v2.1.5"
) else if exist "%LEVEL1%Umi-OCR_Paddle_v2.1.5\UmiOCR-data\plugins\%FOLDER_NAME%" (
    set "UMI_BASE=%LEVEL1%Umi-OCR_Paddle_v2.1.5"
) else if exist "%SCRIPT_DIR%\..\..\Umi-OCR_Paddle_v2.1.5" (
    for %%i in ("%SCRIPT_DIR%\..\..") do set "UMI_BASE=%%~fi"
)

:: 获取桌面路径
for /f "tokens=2*" %%i in (
    'reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v Desktop'
) do set "DESKTOP=%%j"

:: 获取开始菜单程序目录
for /f "tokens=2*" %%i in (
    'reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders" /v "Start Menu"'
) do set "STARTMENU=%%j"

set "SHORTCUT_NAME=CathayOCR Pro.lnk"

:: ============================================================
:: 创建桌面快捷方式（使用 PowerShell）
:: ============================================================
echo.
echo [1/2] 正在创建桌面快捷方式...

set "DESKTOP_LINK=%DESKTOP%\%SHORTCUT_NAME%"

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$s = $ws.CreateShortcut('%DESKTOP_LINK%'); " ^
    "$s.TargetPath = 'cmd.exe'; " ^
    "$s.Arguments = '/c chcp 65001 ^&^& \"%BAT_PATH%\"'; " ^
    "$s.WorkingDirectory = '%SCRIPT_DIR%'; " ^
    "$s.Description = 'CathayOCR Pro - GPU加速PDF处理器'; " ^
    "$s.IconLocation = 'fitz.dll,0'; " ^
    "$s.Save()"

if exist "%DESKTOP_LINK%" (
    echo     ✓ 桌面快捷方式已创建: %SHORTCUT_NAME%
) else (
    echo     [警告] 桌面快捷方式创建失败，跳过
)

:: ============================================================
:: 创建开始菜单快捷方式
:: ============================================================
echo.
echo [2/2] 正在创建开始菜单快捷方式...

set "STARTMENU_DIR=%STARTMENU%\CathayOCR Pro"
if not exist "%STARTMENU_DIR%" mkdir "%STARTMENU_DIR%"

set "STARTMENU_LINK=%STARTMENU_DIR%\%SHORTCUT_NAME%"

powershell -NoProfile -Command ^
    "$ws = New-Object -ComObject WScript.Shell; " ^
    "$s = $ws.CreateShortcut('%STARTMENU_LINK%'); " ^
    "$s.TargetPath = 'cmd.exe'; " ^
    "$s.Arguments = '/c chcp 65001 ^&^& \"%BAT_PATH%\"'; " ^
    "$s.WorkingDirectory = '%SCRIPT_DIR%'; " ^
    "$s.Description = 'CathayOCR Pro - GPU加速PDF处理器'; " ^
    "$s.IconLocation = 'fitz.dll,0'; " ^
    "$s.Save()"

if exist "%STARTMENU_LINK%" (
    echo     ✓ 开始菜单快捷方式已创建
) else (
    echo     [警告] 开始菜单快捷方式创建失败，跳过
)

:: ============================================================
:: 完成
:: ============================================================
echo.
echo ============================================================
echo  安装完成！
echo ============================================================
echo.
echo  安装位置: %SCRIPT_DIR%
echo  快捷方式: %DESKTOP%\%SHORTCUT_NAME%
echo.
echo  运行前请确保：
echo    1. CathayOCR Paddle v2.1.5 已安装
echo    2. umi_plugin_v6 插件已安装
echo    3. NVIDIA CUDA 驱动已安装
echo.
echo  首次使用建议点击主界面右上角的 [帮助] 按钮查看说明。
echo ============================================================
echo.
pause
