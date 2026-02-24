@echo off
REM Start Frontend (React on port 3000)
REM ===================================

echo.
echo ========================================
echo Multi-Ocular Frontend Server (React)
echo ========================================
echo.
echo Starting React Frontend...
echo Frontend will run on: http://localhost:3000
echo.

cd /d "%~dp0\frontend"

REM Check if node_modules exists
if not exist node_modules (
    echo npm packages not found. Installing dependencies...
    call npm install
)

echo.
echo [Starting Frontend] Running: npm start
echo.

call npm start

pause
