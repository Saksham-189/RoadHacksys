$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv-win\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found: $Python"
}

& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& $Python -m mvp.preflight --config (Join-Path $Root "configs\mvp.yaml")

Write-Host ""
Write-Host "MVP setup complete." -ForegroundColor Green
Write-Host "Start it with: .\run_mvp.ps1"
