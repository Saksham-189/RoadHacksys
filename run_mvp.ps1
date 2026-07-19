$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv-win\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment not found: $Python"
}

Push-Location $Root
try {
    & $Python -m mvp.preflight --config "configs\mvp.yaml"
    if ($LASTEXITCODE -ne 0) {
        throw "MVP preflight failed."
    }

    $Port = 8765
    while ($Port -le 8775) {
        $Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $Listener) { break }
        $Port += 1
    }
    if ($Port -gt 8775) {
        throw "No free port found between 8765 and 8775."
    }

    $Url = "http://127.0.0.1:$Port"
    Write-Host ""
    Write-Host "Route Resilience MVP" -ForegroundColor Cyan
    Write-Host "Open: $Url" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the server."
    & $Python -m uvicorn mvp.app:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
