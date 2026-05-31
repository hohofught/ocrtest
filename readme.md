# 주차 단속 OCR 시스템

주차 단속 사진에서 차량 번호판을 인식하고, 단속 정보를 Excel 파일로 저장하는 Windows용 OCR 도구입니다. 웹 브라우저와 데스크톱 GUI를 모두 지원하며, 여러 장의 사진을 한 번에 처리할 수 있습니다.

## 주요 기능

- **번호판 자동 인식**: YOLO/YOLO26 모델로 번호판 영역을 찾고 OneOCR DLL로 번호를 읽습니다.
- **일괄 처리**: 여러 이미지를 업로드하거나 폴더 단위로 선택해 순차 분석합니다.
- **결과 확인 및 수정**: 인식 결과를 저장 전에 직접 수정하거나 특정 사진을 제외할 수 있습니다.
- **Excel 저장**: 날짜, 시간대, 단속 위치, 단속 사유, 차량번호를 Excel 파일로 저장합니다.
- **웹 모드 자동 백업**: 웹에서 저장할 때 `backup/` 폴더에 백업 파일을 함께 생성합니다.
- **웹 모드**: PC와 모바일 브라우저에서 사진을 업로드하고 결과를 확인할 수 있습니다.
- **GUI 모드**: Tkinter 기반 로컬 프로그램에서 이미지 선택, 분석, 수정, 저장을 처리할 수 있습니다.
- **외부 접속 지원**: Cloudflare Tunnel을 사용해 임시 외부 접속 주소를 만들 수 있습니다.
- **알림 연동**: Discord Webhook을 설정하면 서버 시작 시 접속 주소를 알림으로 보낼 수 있습니다.

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. OCR DLL 준비

소스 코드로 실행할 때 `dlls/` 폴더에 다음 파일이 필요합니다.

- `oneocr.dll`
- `oneocr.onemodel`
- `onnxruntime.dll`

파일이 없으면 실행 시 Windows의 Snipping Tool 또는 Photos 앱에서 자동 추출을 시도합니다. 수동으로 확인하거나 다시 추출하려면 다음 명령을 사용합니다.

```bash
python dll_extractor.py --info
python dll_extractor.py --force
```

### 3. YOLO 모델 준비

기본으로는 프로그램 폴더의 `best.pt`를 먼저 사용합니다. YOLO26 가중치를 사용하려면 `best_yolo26.pt`, `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` 중 하나를 프로그램 폴더에 두거나, GUI 설정의 `YOLO 모델 설정`에서 `.pt` 파일을 지정합니다.

환경변수로도 모델을 지정할 수 있습니다.

```powershell
$env:PARKING_YOLO_MODEL = "yolo26n.pt"
python ocr.py --server
```

번호판 탐지에는 번호판 데이터로 학습된 모델을 사용하는 것이 가장 좋습니다. 공개 기본 `yolo26n.pt`는 자동 다운로드될 수 있지만, 번호판 전용 모델이 아니면 인식률이 낮을 수 있습니다.

### 4. 실행

```bash
# GUI 모드
python ocr.py

# 웹 서버 모드
python ocr.py --server

# GUI와 웹 서버를 함께 실행
python ocr.py --hybrid
```

웹 서버 모드의 기본 로컬 주소는 `http://127.0.0.1:5000`입니다.

## 사용 흐름

1. 단속 위치, 단속 사유, 오전/오후를 선택합니다.
2. 단속 사진을 한 장 이상 선택합니다.
3. 번호판 자동 인식이 끝날 때까지 기다립니다.
4. 결과 화면에서 차량번호를 확인하고 필요한 경우 수정합니다.
5. 저장하지 않을 항목은 제외한 뒤 Excel로 저장합니다.

## 설정

설정은 프로그램 폴더의 `.settings` 파일에 저장됩니다. GUI의 `설정` 버튼에서 다음 항목을 저장할 수 있으며, 폴더 경로처럼 서버 시작 시 읽는 값은 프로그램 재시작 후 적용됩니다.

- Cloudflare Tunnel 토큰과 고정 도메인
- Discord Webhook URL
- 웹 업로드 폴더
- 웹 백업 폴더
- YOLO 모델 파일
- 마지막으로 선택한 위치, 사유, 시간대

코드 상단의 `SYSTEM_PASSWORD` 값을 설정하면 외부 접속 시 비밀번호 확인을 사용할 수 있습니다. 기본값은 빈 문자열이며, 이 경우 비밀번호 없이 접속됩니다.

## 저장 파일

- 웹 모드 저장 파일: `주차단속내역_YYYY-MM-DD_오전.xlsx` 또는 `주차단속내역_YYYY-MM-DD_오후.xlsx`
- 웹 모드 백업 파일: `backup/YYYY-MM-DD/단속내역_오전_HH시MM분SS초.xlsx`
- GUI 모드 저장 파일: 저장 대화상자에서 사용자가 선택한 `.xlsx` 파일

Excel 컬럼은 `날짜`, `시간대`, `단속위치`, `사유`, `차량번호`입니다.

## 직접 빌드

Windows에서 실행 파일을 만들려면 다음 명령을 실행합니다.

```bat
build.bat
```

빌드 스크립트는 필요한 파일을 확인하고, DLL이 없으면 자동 추출을 시도한 뒤 PyInstaller로 `dist/주차단속시스템.exe`를 생성합니다.

빌드는 프로젝트 안의 `.venv` 가상환경에서 실행됩니다. `.venv`가 없으면 `build.bat`이 자동으로 만들고, 이후 의존성 설치와 PyInstaller 실행은 모두 `.venv\Scripts\python.exe`를 사용합니다.

## 프로젝트 구성

```text
ocrtest/
├─ ocr.py               # Flask 서버, OCR 처리, Excel 저장
├─ gui.py               # Tkinter GUI
├─ dll_extractor.py     # OneOCR DLL 자동 추출
├─ settings_manager.py  # 설정 저장 및 로드
├─ requirements.txt     # Python 의존성
├─ build.spec           # PyInstaller 설정
├─ build.bat            # Windows 빌드 스크립트
├─ start.bat            # 웹 서버 실행 스크립트
├─ best.pt              # 기본 번호판 탐지 모델
├─ yolo26*.pt           # 선택 사항: YOLO26 계열 모델
├─ templates/           # 웹 화면 템플릿
├─ static/              # 정적 파일
├─ dlls/                # OneOCR DLL 파일
├─ uploads/             # 업로드 이미지
└─ backup/              # Excel 백업 파일
```

`dlls/`, `uploads/`, `backup/`, `.settings`, `cloudflared.exe`, 생성된 Excel 파일은 실행 중 생성되거나 로컬 환경에 따라 달라질 수 있습니다.

## 주의 사항

- Windows 환경을 기준으로 동작합니다.
- OneOCR DLL은 Windows Snipping Tool 또는 Photos 앱에서 추출할 수 있어야 합니다.
- 번호판 전용 `best.pt` 또는 YOLO26 학습 모델이 없으면 기본 `yolo26n.pt`를 사용할 수 있지만 번호판 인식률이 낮아질 수 있습니다.
- 외부 접속을 사용하려면 인터넷 연결이 필요하며, 최초 실행 시 `cloudflared.exe`를 다운로드할 수 있습니다.
