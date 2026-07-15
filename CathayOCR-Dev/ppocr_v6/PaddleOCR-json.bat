@echo off
REM PP-OCRv6 ONNX Runtime Plugin Launcher (Portable)
REM Python search order:
REM   1. Portable Python (portapython, for portable deployment)
REM   2. Plugin-local venv (ppocr_v6_env, for dev)
REM   3. Source repo venv (../../ppocr_v6_env, for dev)
REM   4. System PATH python
set "SCRIPT_DIR=%~dp0"
set "PYTHON="

if exist "%SCRIPT_DIR%..\portapython\python.exe" (
    set "PYTHON=%SCRIPT_DIR%..\portapython\python.exe"
) else if exist "%SCRIPT_DIR%ppocr_v6_env\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%ppocr_v6_env\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%..\..\ppocr_v6_env\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%..\..\ppocr_v6_env\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "%SCRIPT_DIR%ppocr_v6_server.py" %*
