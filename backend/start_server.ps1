# Start MediLens Backend Server
Write-Host "Starting MediLens Backend Server on port 8000..." -ForegroundColor Green
Write-Host ""

Set-Location $PSScriptRoot
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
