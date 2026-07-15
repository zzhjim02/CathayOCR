@echo off
:: ==================================================
:: PaddleOCR-ncnn-CPP Vulkan Build and Deploy Script
:: Usage: Run in "x64 Native Tools Command Prompt for VS 2022"
:: Note: Vulkan SDK must be installed
:: ==================================================

set PROJECT_ROOT=%~dp0

:: ================= Dependency Paths =================
set OPENCV_DIR=%PROJECT_ROOT%libs\opencv-4.11.0-windows-vs2022-x64-md\x64\vc17\staticlib
set NCNN_DIR=%PROJECT_ROOT%libs\ncnn\x64\lib\cmake\ncnn

:: Vulkan SDK Path (modify if needed)
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0

:: Vulkan Plugin Directory
set VULKAN_PLUGIN_DIR=%PROJECT_ROOT%PPOCR-ncnn-Vulkan

:: ================= Build Parameters =================
set BUILD_TYPE=Release
set CMAKE_GENERATOR=Visual Studio 17 2022
set PLATFORM=-A x64

:: ================= Enter Build Directory =================
cd /d "%PROJECT_ROOT%"
if not exist build_vulkan mkdir build_vulkan
cd build_vulkan

if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

echo ==========================================
echo [INFO] Configuring CMake (Vulkan Build)...
echo [INFO] OpenCV_DIR : %OPENCV_DIR%
echo [INFO] ncnn_DIR   : %NCNN_DIR%
echo [INFO] VULKAN_SDK : %VULKAN_SDK%
echo [INFO] Generator  : %CMAKE_GENERATOR%
echo ==========================================

cmake .. -G "%CMAKE_GENERATOR%" %PLATFORM% ^
    -DOpenCV_DIR="%OPENCV_DIR%" ^
    -Dncnn_DIR="%NCNN_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DENABLE_VULKAN=ON ^
    -DVULKAN_SDK=%VULKAN_SDK% ^
    -DCMAKE_INCLUDE_PATH="%PROJECT_ROOT%libs\ncnn\x64\include"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CMake configuration failed! Please ensure Vulkan SDK is installed.
    pause
    exit /b %errorlevel%
)

echo.
echo [INFO] Building ppocr_ocr_vulkan.exe ...
echo.

cmake --build . --config %BUILD_TYPE% --target ppocr_ocr_vulkan

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [INFO] Copying exe to plugin directory...
echo.

if not exist "%VULKAN_PLUGIN_DIR%" mkdir "%VULKAN_PLUGIN_DIR%"
copy /Y "Release\ppocr_ocr_vulkan.exe" "%VULKAN_PLUGIN_DIR%\ppocr_ocr_vulkan.exe"

echo.
echo ==========================================
echo [SUCCESS] Vulkan build and deploy completed!
echo ==========================================
echo.
echo   Vulkan Program : build_vulkan\Release\ppocr_ocr_vulkan.exe
echo   Plugin Dir    : %VULKAN_PLUGIN_DIR%
echo.
echo   Contents Deployed:
echo     - ppocr_ocr_vulkan.exe
echo.
echo   Note: Ensure Vulkan SDK DLLs are in system PATH
echo   (usually in %VULKAN_SDK%\Bin)
echo.

pause
