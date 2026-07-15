@echo off
title CathayOCR Lite - Vulkan-Only Edition (Launcher)
cd /d "%~dp0"
start "" "..\portapython\python.exe" "umi_ocr_pdf_processor_ui.py" %*
exit /b 0
