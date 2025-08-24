@echo off
REM NeuraLens Backend Environment Setup Script for Windows
REM This script sets up the Python virtual environment and installs dependencies

echo ========================================
echo NeuraLens Backend Environment Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if we're in the backend directory
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please run this script from the backend directory
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
    echo.
) else (
    echo Virtual environment already exists.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Common solutions:
    echo 1. Install Visual C++ Build Tools for Windows
    echo 2. Use conda instead of pip for some packages
    echo 3. Check internet connection
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate the environment, run:
echo   deactivate
echo.
echo To start the development server, run:
echo   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
pause
