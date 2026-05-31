@echo off
chcp 65001 > nul
setlocal EnableExtensions
TITLE 주차 단속 시스템 빌더 (DLL OCR + YOLO26)

echo ============================================
echo   주차 단속 시스템 - EXE 빌드 스크립트
echo   (DLL OCR + YOLO26 지원)
echo ============================================
echo.

:: 현재 디렉토리로 이동
cd /d %~dp0
set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

:: 1. 기본 필수 파일 확인
echo [1/6] 기본 파일 확인 중...
if not exist "ocr.py" (
    echo 오류: ocr.py 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)
if not exist "dll_extractor.py" (
    echo 오류: dll_extractor.py 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)
set HAS_YOLO_MODEL=0
if exist "best.pt" set HAS_YOLO_MODEL=1
if exist "best_yolo26.pt" set HAS_YOLO_MODEL=1
if exist "yolo26*.pt" set HAS_YOLO_MODEL=1
if "%HAS_YOLO_MODEL%"=="0" (
    echo 경고: best.pt 또는 yolo26*.pt 파일이 없습니다.
    echo       실행 시 기본 YOLO26 모델(yolo26n.pt) 다운로드를 시도합니다.
)
if not exist "templates" (
    echo 오류: templates 폴더를 찾을 수 없습니다.
    pause
    exit /b 1
)
echo 기본 파일 확인 완료
echo.

:: 2. 빌드 가상환경 준비
echo [2/6] 빌드 가상환경 준비 중...
if not exist "%VENV_PY%" (
    echo    .venv가 없습니다. 새로 생성합니다...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo 오류: 가상환경 생성 실패
        echo    Python 설치 또는 PATH 설정을 확인하세요.
        pause
        exit /b 1
    )
) else (
    echo .venv 가상환경을 사용합니다.
)
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo 오류: 가상환경의 pip를 실행할 수 없습니다.
    echo    .venv 폴더를 삭제한 뒤 다시 실행해 보세요.
    pause
    exit /b 1
)
echo 가상환경 준비 완료: %CD%\%VENV_DIR%
echo.

:: 3. DLL 자동 추출 (없는 경우)
echo [3/6] OCR DLL 확인 및 추출 중...
if not exist "dlls\oneocr.dll" (
    echo    DLL이 없습니다. 자동 추출을 시작합니다...
    "%VENV_PY%" dll_extractor.py
    if errorlevel 1 (
        echo 오류: DLL 추출 실패
        echo    Snipping Tool 또는 Windows Photos 앱이 설치되어 있는지 확인하세요.
        echo    또는 수동으로 DLL을 dlls 폴더에 복사하세요.
        pause
        exit /b 1
    )
) else (
    echo DLL 파일이 이미 존재합니다.
)
echo DLL 준비 완료
echo.

:: 4. 의존성 패키지 설치
echo [4/6] 의존성 패키지 설치 중...
"%VENV_PY%" -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo 오류: 패키지 설치 실패
    pause
    exit /b 1
)
echo 의존성 설치 완료
echo.

:: 5. 이전 빌드 정리
echo [5/6] 이전 빌드 정리 중...
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build
echo 정리 완료
echo.

:: 6. PyInstaller 빌드 실행
echo [6/6] EXE 빌드 중... (시간이 다소 소요됩니다)
echo.
set ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1
set YOLO_AUTOINSTALL=false
"%VENV_PY%" -m PyInstaller build.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo 빌드 실패! 오류 메시지를 확인하세요.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   빌드 완료!
echo ============================================
echo.
echo   실행 파일 위치: dist\주차단속시스템.exe
echo.
echo   [주의사항]
echo   - EXE 파일에 모든 DLL과 모델이 포함되어 있습니다.
echo   - 첫 실행 시 Windows Defender가 검사할 수 있습니다.
echo.
pause
