@echo off
echo 🎭 STYLE-SYNTH - Reliable Launcher for Windows
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in virtual environment!
    pause
    exit /b 1
)

REM Start the application
echo 🚀 Starting STYLE-SYNTH...
python start.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo 👋 STYLE-SYNTH stopped.
pause