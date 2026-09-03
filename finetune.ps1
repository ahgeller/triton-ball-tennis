# Front door for the fine-tuning workspace. Runs finetune/ft.py with the training venv.
#   .\finetune.ps1                       interactive menu (or double-click finetune.bat)
#   .\finetune.ps1 status
#   .\finetune.ps1 import all
#   .\finetune.ps1 label video10
#   .\finetune.ps1 train --val-clips video10
$ErrorActionPreference = "Stop"
$candidates = @(
    $env:TENNIS_FINETUNE_PYTHON,
    "C:\Users\Andrew\Desktop\gridtracknet_finetuning\.venv\Scripts\python.exe",
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$python = if ($candidates) { @($candidates)[0] } else { "python" }
& $python (Join-Path $PSScriptRoot "finetune\ft.py") @args
exit $LASTEXITCODE
