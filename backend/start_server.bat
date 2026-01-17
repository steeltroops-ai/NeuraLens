@echo off
echo Starting MediLens Backend Server...
echo.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
