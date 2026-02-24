# Start Frontend (React on port 3000)
# ===================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multi-Ocular Frontend Server (React)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting React Frontend..." -ForegroundColor Yellow
Write-Host "Frontend will run on: http://localhost:3000" -ForegroundColor Green
Write-Host ""

$scriptPath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Push-Location "$scriptPath\frontend"

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "npm packages not found. Installing dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host ""
Write-Host "[Starting Frontend] Running: npm start" -ForegroundColor Cyan
Write-Host ""

npm start

Read-Host "Press Enter to exit"
