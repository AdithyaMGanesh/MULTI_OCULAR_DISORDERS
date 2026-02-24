# Start Backend (Dual-Model API on port 8001)
# ============================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multi-Ocular Dual-Model Backend Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Dual-Model API Server..." -ForegroundColor Yellow
Write-Host "Backend will run on: http://localhost:8001" -ForegroundColor Green
Write-Host ""

$scriptPath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Push-Location $scriptPath

# Activate virtual environment if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Start backend on port 8001
Write-Host ""
Write-Host "[Starting Backend] Running: uvicorn src.api_dual_model:app --reload --host 0.0.0.0 --port 8001" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn src.api_dual_model:app --reload --host 0.0.0.0 --port 8001

Read-Host "Press Enter to exit"
