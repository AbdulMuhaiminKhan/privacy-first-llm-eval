@echo off
title Private Document Q^&A
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Using: %PY%
"%PY%" scripts\launch_app.py
echo.
echo If something went wrong, the details are in app_log.txt
pause
