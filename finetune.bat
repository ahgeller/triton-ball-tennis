@echo off
rem Double-click this file (or run `finetune` in cmd) to open the fine-tuning menu.
rem With arguments it behaves like finetune.ps1:  finetune status, finetune label video10, ...
setlocal
set "PY=C:\Users\Andrew\Desktop\gridtracknet_finetuning\.venv\Scripts\python.exe"
if defined TENNIS_FINETUNE_PYTHON if exist "%TENNIS_FINETUNE_PYTHON%" set "PY=%TENNIS_FINETUNE_PYTHON%"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0finetune\ft.py" %*
if errorlevel 1 pause
