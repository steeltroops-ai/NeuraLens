@echo off
REM =============================================
REM MediLens - Development Server Script
REM Runs Frontend and Backend concurrently
REM =============================================

echo.
echo   MediLens Development Servers
echo   ============================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%..
set ROOT_DIR=%CD%
set BACKEND_DIR=%ROOT_DIR%\backend
set FRONTEND_DIR=%ROOT_DIR%\frontend

REM Check directories
if not exist "%BACKEND_DIR%" (
    echo [ERROR] Backend directory not found!
    exit /b 1
)

if not exist "%FRONTEND_DIR%" (
    echo [ERROR] Frontend directory not found!
    exit /b 1
)

echo [*] Starting Backend on http://localhost:8000
echo [*] Starting Frontend on http://localhost:3000
echo.
echo Press Ctrl+C in each window to stop
echo.

REM Start Backend in new window
start "MediLens Backend" cmd /k "cd /d %BACKEND_DIR% && .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

REM Wait a moment
timeout /t 2 /nobreak > nul

REM Start Frontend in new window
start "MediLens Frontend" cmd /k "cd /d %FRONTEND_DIR% && bun run dev"

echo.
echo [*] Servers started in separate windows
echo [*] Frontend: http://localhost:3000
echo [*] Backend:  http://localhost:8000
echo [*] API Docs: http://localhost:8000/docs
echo.
