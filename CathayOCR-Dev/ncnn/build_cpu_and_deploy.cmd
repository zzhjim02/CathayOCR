@echo off
:: ==================================================
:: PaddleOCR-ncnn-CPP CPU Build and Deploy Script
:: Usage: Run in "x64 Native Tools Command Prompt for VS 2022"
:: Note: No Vulkan SDK required
:: ==================================================

set PROJECT_ROOT=%~dp0

:: ================= Dependency Paths =================
set OPENCV_DIR=%PROJECT_ROOT%libs\opencv-4.11.0-windows-vs2022-x64-md\x64\vc17\staticlib
set NCNN_DIR=%PROJECT_ROOT%libs\ncnn\x64\lib\cmake\ncnn

:: CPU Plugin Directory
set CPU_PLUGIN_DIR=%PROJECT_ROOT%PPOCR-ncnn-CPU

:: ================= Build Parameters =================
set BUILD_TYPE=Release
set CMAKE_GENERATOR=Visual Studio 17 2022
set PLATFORM=-A x64

:: ================= Enter Build Directory =================
cd /d "%PROJECT_ROOT%"
if not exist build mkdir build
cd build

if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

echo ==========================================
echo [INFO] Configuring CMake (CPU Build)...
echo [INFO] OpenCV_DIR : %OPENCV_DIR%
echo [INFO] ncnn_DIR   : %NCNN_DIR%
echo [INFO] Generator  : %CMAKE_GENERATOR%
echo ==========================================

cmake .. -G "%CMAKE_GENERATOR%" %PLATFORM% ^
    -DOpenCV_DIR="%OPENCV_DIR%" ^
    -Dncnn_DIR="%NCNN_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DENABLE_VULKAN=OFF

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [INFO] Building ppocr_ocr_cpu.exe ...
echo.

cmake --build . --config %BUILD_TYPE% --target ppocr_ocr_cpu

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [INFO] Copying exe to plugin directory...
echo.

if not exist "%CPU_PLUGIN_DIR%" mkdir "%CPU_PLUGIN_DIR%"
copy /Y "Release\ppocr_ocr_cpu.exe" "%CPU_PLUGIN_DIR%\ppocr_ocr_cpu.exe"

echo.
echo ==========================================
echo [SUCCESS] CPU build and deploy completed!
echo ==========================================
echo.
echo   CPU Program    : build\Release\ppocr_ocr_cpu.exe
echo   Plugin Dir     : %CPU_PLUGIN_DIR%
echo.
echo   Contents Deployed:
echo     - ppocr_ocr_cpu.exe
echo.
echo   Note: No Vulkan SDK required for CPU version
echo.

pause
