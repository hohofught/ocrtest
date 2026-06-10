@echo off
chcp 65001 > nul
setlocal EnableExtensions
TITLE Parking Enforcement OCR GUI

cd /d "%~dp0"

echo ============================================
echo   Parking Enforcement OCR GUI
echo ============================================
echo.

if not exist "gui.py" (
    echo ERROR: gui.py was not found.
    pause
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON_EXE=%VENV_PY%"
) else (
    echo WARNING: .venv was not found. Falling back to system Python.
    set "PYTHON_EXE=python"
)

echo Starting desktop GUI...
echo.
"%PYTHON_EXE%" gui.py
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
