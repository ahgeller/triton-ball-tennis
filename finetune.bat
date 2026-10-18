@echo off
rem Double-click this file (or run `finetune` in cmd) to open the fine-tuning menu.
rem With arguments it behaves like finetune.ps1:  finetune status, finetune label video10, ...
rem Any python will do here: ft.py finds one with torch + cv2 and re-runs itself under it.
setlocal
set "PY="
if defined TENNIS_FINETUNE_PYTHON if exist "%TENNIS_FINETUNE_PYTHON%" set "PY=%TENNIS_FINETUNE_PYTHON%"
rem The interpreter ft.py found last time, so it does not have to be searched for again.
if not defined PY if exist "%~dp0finetune\.training-python" set /p PY=<"%~dp0finetune\.training-python"
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    where py >nul 2>&1 && set "PY=py"
)
if not defined PY (
    echo No python found. Install one, or set TENNIS_FINETUNE_PYTHON to a python with torch + cv2.
    pause
    exit /b 1
)
"%PY%" "%~dp0finetune\ft.py" %*
if errorlevel 1 pause
