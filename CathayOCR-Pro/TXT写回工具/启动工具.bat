@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" "%~dp0..\portapython\python.exe" "%~dp0gui_main.py"
