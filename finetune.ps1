# Front door for the fine-tuning workspace. Runs finetune/ft.py, which re-runs itself
# under an interpreter that has torch + cv2, so any python is enough to start it.
#   .\finetune.ps1                       interactive menu (or double-click finetune.bat)
#   .\finetune.ps1 status
#   .\finetune.ps1 import all
#   .\finetune.ps1 label video10
#   .\finetune.ps1 train --val-clips video10
$ErrorActionPreference = "Stop"
$candidates = @(
    $env:TENNIS_FINETUNE_PYTHON,
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Get-Command python -ErrorAction SilentlyContinue).Source,
    (Get-Command py -ErrorAction SilentlyContinue).Source
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $candidates) {
    throw "No python found. Install one, or set TENNIS_FINETUNE_PYTHON to a python with torch + cv2."
}
$python = @($candidates)[0]
& $python (Join-Path $PSScriptRoot "finetune\ft.py") @args
exit $LASTEXITCODE
