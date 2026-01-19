# =============================================
# MediLens - Development Server Script
# Runs Frontend and Backend concurrently
# =============================================

Write-Host ""
Write-Host "  __  __          _ _ _                    " -ForegroundColor Cyan
Write-Host " |  \/  | ___  __| (_) |    ___ _ __  ___ " -ForegroundColor Cyan
Write-Host " | |\/| |/ _ \/ _' | | |   / _ \ '_ \/ __|" -ForegroundColor Cyan
Write-Host " | |  | |  __/ (_| | | |__|  __/ | | \__ \" -ForegroundColor Cyan
Write-Host " |_|  |_|\___|\__,_|_|_____\___|_| |_|___/" -ForegroundColor Cyan
Write-Host ""
Write-Host " Starting Development Servers..." -ForegroundColor Yellow
Write-Host ""

$scriptDir = $PSScriptRoot
$rootDir = Split-Path -Parent $scriptDir
$backendDir = Join-Path $rootDir "backend"
$frontendDir = Join-Path $rootDir "frontend"

# Check if directories exist
if (-not (Test-Path $backendDir)) {
    Write-Host "[ERROR] Backend directory not found: $backendDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $frontendDir)) {
    Write-Host "[ERROR] Frontend directory not found: $frontendDir" -ForegroundColor Red
    exit 1
}

# Check if Python venv exists
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Python venv not found. Run: cd backend && python -m venv .venv" -ForegroundColor Red
    exit 1
}

# Check if bun is installed
$bunPath = Get-Command bun -ErrorAction SilentlyContinue
if (-not $bunPath) {
    Write-Host "[ERROR] Bun not found. Install from: https://bun.sh" -ForegroundColor Red
    exit 1
}

Write-Host "[*] Backend: $backendDir" -ForegroundColor DarkGray
Write-Host "[*] Frontend: $frontendDir" -ForegroundColor DarkGray
Write-Host ""

# Start Backend
Write-Host "[BACKEND] Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    param($dir, $python)
    Set-Location $dir
    & $python -m uvicorn app.main:app --reload --port 8000 2>&1
} -ArgumentList $backendDir, $venvPython

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "[FRONTEND] Starting Next.js on http://localhost:3000" -ForegroundColor Blue
$frontendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    bun run dev 2>&1
} -ArgumentList $frontendDir

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host " Servers Running:" -ForegroundColor White
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor Blue
Write-Host "   Backend:  http://localhost:8000" -ForegroundColor Green
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor DarkGray
Write-Host ""
Write-Host " Press Ctrl+C to stop all servers" -ForegroundColor DarkGray
Write-Host ""

# Function to cleanup jobs on exit
function Stop-AllServers {
    Write-Host ""
    Write-Host "[*] Stopping servers..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $frontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "[*] All servers stopped." -ForegroundColor Green
}

# Register cleanup on exit
Register-EngineEvent PowerShell.Exiting -Action { Stop-AllServers } | Out-Null

# Stream output from both jobs
try {
    while ($true) {
        # Get backend output
        $backendOutput = Receive-Job -Job $backendJob -ErrorAction SilentlyContinue
        if ($backendOutput) {
            $backendOutput | ForEach-Object {
                Write-Host "[API] $_" -ForegroundColor Green
            }
        }

        # Get frontend output
        $frontendOutput = Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue
        if ($frontendOutput) {
            $frontendOutput | ForEach-Object {
                Write-Host "[WEB] $_" -ForegroundColor Blue
            }
        }

        # Check if jobs are still running
        if ($backendJob.State -eq "Failed" -or $frontendJob.State -eq "Failed") {
            Write-Host "[ERROR] A server crashed. Stopping..." -ForegroundColor Red
            break
        }

        Start-Sleep -Milliseconds 100
    }
}
finally {
    Stop-AllServers
}
