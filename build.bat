@echo off
chcp 65001 > nul
setlocal EnableExtensions
TITLE Parking Enforcement Builder

echo ============================================
echo   Parking Enforcement - EXE Build
echo   DLL OCR + YOLO26
echo ============================================
echo.

cd /d "%~dp0"
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=requirements.txt"

echo [1/6] Checking required files...
if not exist "ocr.py" (
    echo ERROR: ocr.py was not found.
    pause
    exit /b 1
)
if not exist "dll_extractor.py" (
    echo ERROR: dll_extractor.py was not found.
    pause
    exit /b 1
)
if not exist "%REQUIREMENTS_FILE%" (
    echo ERROR: %REQUIREMENTS_FILE% was not found.
    pause
    exit /b 1
)
set HAS_YOLO_MODEL=0
if exist "best.pt" set HAS_YOLO_MODEL=1
if exist "best_yolo26.pt" set HAS_YOLO_MODEL=1
if exist "yolo26*.pt" set HAS_YOLO_MODEL=1
if "%HAS_YOLO_MODEL%"=="0" (
    echo WARNING: best.pt or yolo26*.pt was not found.
    echo          Runtime may try to download yolo26n.pt.
)
if not exist "templates" (
    echo ERROR: templates directory was not found.
    pause
    exit /b 1
)
echo Required files OK.
echo.

echo [2/6] Preparing build virtual environment...
call :PrepareEnvironment
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [3/6] Checking OCR DLL files...
if not exist "dlls\oneocr.dll" (
    echo OCR DLL files were not found. Running extractor...
    "%VENV_PY%" dll_extractor.py
    if errorlevel 1 (
        echo ERROR: DLL extraction failed.
        echo Install Snipping Tool or Windows Photos, or copy DLL files into dlls.
        pause
        exit /b 1
    )
) else (
    echo OCR DLL files already exist.
)
echo OCR DLL files OK.
echo.

echo [4/6] Installing Python dependencies...
call :InstallRequirements
if errorlevel 1 (
    pause
    exit /b 1
)
echo.

echo [5/6] Cleaning previous build output...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
echo Clean OK.
echo.

echo [6/6] Running PyInstaller...
echo.
set ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1
set YOLO_AUTOINSTALL=false
"%VENV_PY%" -m PyInstaller build.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo ERROR: Build failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build complete
echo ============================================
echo.
echo   Output directory: dist
echo.
echo   Notes:
echo   - The EXE includes bundled DLL and model files.
echo   - Windows Defender may scan the EXE on first launch.
echo.
pause
exit /b 0

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
