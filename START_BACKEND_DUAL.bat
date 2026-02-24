@echo off
REM Start Backend (Dual-Model API on port 8001)
REM =============================================

echo.
echo ========================================
echo Multi-Ocular Dual-Model Backend Server
echo ========================================
echo.
echo Starting Dual-Model API Server...
echo Backend will run on: http://localhost:8001
echo.

cd /d "%~dp0"

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
)

REM Start backend on port 8001
echo.
echo [Starting Backend] Running: uvicorn src.api_dual_model:app --reload --host 0.0.0.0 --port 8001
echo.

python -m uvicorn src.api_dual_model:app --reload --host 0.0.0.0 --port 8001

pause
