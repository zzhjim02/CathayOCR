@echo off
title CathayOCR Dev - Launcher
cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
set "PORTA=%~dp0..\portapython"

REM -- Find Python --
set "PYTHON="
if exist "%PORTA%\python.exe" (
    set "PYTHON=%PORTA%\python.exe"
) else if exist "%SCRIPT_DIR%..\portapython\python.exe" (
    set "PYTHON=%SCRIPT_DIR%..\portapython\python.exe"
) else if exist "%SCRIPT_DIR%..\ppocr_v6\ppocr_v6_env\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%..\ppocr_v6\ppocr_v6_env\Scripts\python.exe"
) else (
    echo [Error] Python not found. Please confirm directory structure is complete.
    pause
    exit /b 1
)

echo ===== CathayOCR Dev Portable =====
echo Starting UI...
echo.

"%PYTHON%" "%SCRIPT_DIR%umi_ocr_pdf_processor_ui.py"

pause

