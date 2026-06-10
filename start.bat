@echo off
chcp 65001 > nul
setlocal EnableExtensions
TITLE Parking Enforcement OCR Server

cd /d "%~dp0"

echo ============================================
echo   Parking Enforcement OCR Server
echo ============================================
echo.

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=requirements.txt"

echo [1/5] Checking required files...
if not exist "ocr.py" (
    echo ERROR: ocr.py was not found.
    pause
    exit /b 1
)
if not exist "%REQUIREMENTS_FILE%" (
    echo ERROR: %REQUIREMENTS_FILE% was not found.
    pause
    exit /b 1
)
echo Required files OK.
echo.

echo [2/5] Preparing Python virtual environment...
call :PrepareEnvironment
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [3/5] Installing Python dependencies...
call :InstallRequirements
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [4/5] Stopping previous processes...
taskkill /f /im cloudflared.exe >nul 2>&1
echo Done.
echo.

echo [5/5] Starting server...
echo.
"%VENV_PY%" ocr.py --server
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Cleaning up...
taskkill /f /im cloudflared.exe >nul 2>&1
echo.
echo ============================================
if not "%EXIT_CODE%"=="0" (
    echo   Server exited with code %EXIT_CODE%.
) else (
    echo   Server stopped.
)
echo ============================================
echo.
pause
exit /b %EXIT_CODE%

:PrepareEnvironment
if not exist "%VENV_PY%" (
    echo Creating .venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Python command failed. Trying Python launcher...
        py -3 -m venv "%VENV_DIR%"
        if errorlevel 1 (
            echo ERROR: Failed to create .venv.
            echo Check that Python 3 is installed and available in PATH.
            exit /b 1
        )
    )
) else (
    echo Using existing .venv.
)
if not exist "%VENV_PY%" (
    echo ERROR: Virtual environment Python was not found.
    echo Delete .venv and run this script again.
    exit /b 1
)
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Installing pip in .venv...
    "%VENV_PY%" -m ensurepip --upgrade >nul 2>&1
    if errorlevel 1 (
        echo ERROR: pip is not available in .venv.
        echo Delete .venv and run this script again.
        exit /b 1
    )
)
echo Virtual environment OK: %CD%\%VENV_DIR%
exit /b 0

:InstallRequirements
"%VENV_PY%" -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)
echo Dependencies OK.
exit /b 0
