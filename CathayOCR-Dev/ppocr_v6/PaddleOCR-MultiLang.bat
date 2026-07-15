@echo off
REM PP-OCR Multi-Language Plugin Launcher (v6 + v5 ONNX)
REM Supports all PaddleOCR languages through onnxruntime
REM --language parameter is mapped by UI to language code
set "SCRIPT_DIR=%~dp0"
set "PPOCR_DIR=%SCRIPT_DIR%ppocr_v6_env"
set "PYTHON="

if exist "%PPOCR_DIR%\Scripts\python.exe" (
    set "PYTHON=%PPOCR_DIR%\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "%SCRIPT_DIR%ppocr_v6_server.py" %*
