@echo off
chcp 65001 > nul
setlocal EnableExtensions
TITLE Parking Enforcement OCR GUI

cd /d "%~dp0"

echo ============================================
echo   Parking Enforcement OCR GUI
echo ============================================
echo.

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=requirements.txt"

echo [1/4] Checking required files...
if not exist "gui.py" (
    echo ERROR: gui.py was not found.
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

echo [2/4] Preparing Python virtual environment...
call :PrepareEnvironment
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [3/4] Installing Python dependencies...
call :InstallRequirements
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [4/4] Starting desktop GUI...
echo Starting desktop GUI...
echo.
"%VENV_PY%" gui.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo ERROR: GUI exited with code %EXIT_CODE%.
) else (
    echo GUI closed.
)
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
