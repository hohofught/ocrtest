# 주차 단속 OCR 시스템

주차 단속 사진에서 차량 번호판을 인식하고, 단속 정보를 Excel과 SQLite 기록으로 저장하는 Windows용 OCR 도구입니다. 웹 브라우저와 Tkinter 데스크톱 GUI를 모두 지원하며, 여러 장의 사진을 한 번에 처리할 수 있습니다.

## 주요 기능

- **번호판 자동 인식**: YOLO/YOLO26 모델로 번호판 영역을 찾고 Windows OneOCR DLL로 번호를 읽습니다.
- **웹/GUI 동시 지원**: 브라우저 업로드 방식, 로컬 GUI 방식, GUI와 서버를 함께 쓰는 하이브리드 방식을 제공합니다.
- **일괄 처리**: 웹에서는 여러 이미지를 업로드하고, GUI에서는 파일 또는 폴더 단위로 이미지를 선택합니다.
- **결과 수정**: 저장 전 차량번호를 직접 수정하거나 저장하지 않을 항목을 제외할 수 있습니다.
- **Excel 저장**: `날짜`, `시간대`, `단속위치`, `사유`, `차량번호` 컬럼으로 단속 내역을 저장합니다.
- **SQLite 기록 보관**: 웹/GUI 저장 내역을 `parking_records.sqlite3`에 누적 저장하고 최근 기록을 조회할 수 있습니다.
- **자동 백업**: 웹 저장 시 `backup/` 폴더에 시간별 백업 Excel 파일을 먼저 생성합니다.
- **외부 접속 지원**: Cloudflare Tunnel로 임시 URL 또는 설정된 고정 도메인을 사용할 수 있습니다.
- **Discord 알림**: Webhook URL을 설정하면 서버 시작 시 접속 주소를 전송합니다.

## 요구 사항

- Windows
- Python과 pip
- Windows Snipping Tool 또는 Microsoft Photos 앱
- 번호판 탐지용 YOLO 모델 파일 권장: `best.pt` 또는 `best_yolo26.pt`
- 외부 접속 사용 시 인터넷 연결

## 설치 및 준비

### 1. 의존성 설치

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

빌드만 할 경우 `build.bat`이 `.venv` 생성과 의존성 설치를 자동으로 처리합니다.

### 2. OCR DLL 준비

소스 코드로 실행할 때 `dlls/` 폴더에 다음 파일이 필요합니다.

- `oneocr.dll`
- `oneocr.onemodel`
- `onnxruntime.dll`

파일이 없으면 실행 시 Snipping Tool 또는 Photos 앱에서 자동 추출을 시도합니다. 수동 확인이나 재추출은 다음 명령을 사용합니다.

```powershell
python dll_extractor.py --info
python dll_extractor.py --force
```

### 3. YOLO 모델 준비

모델 선택 우선순위는 다음과 같습니다.

1. 환경변수 `PARKING_YOLO_MODEL`
2. 환경변수 `YOLO_MODEL_PATH`
3. `.settings`의 `yolo_model_path`
4. 프로그램 폴더의 `best.pt`
5. 프로그램 폴더의 `best_yolo26.pt`
6. 프로그램 폴더의 `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`
7. 기본값 `yolo26n.pt`

환경변수 예시:

```powershell
$env:PARKING_YOLO_MODEL = "best.pt"
python ocr.py --server
```

번호판 탐지에는 번호판 데이터로 학습된 모델을 사용하는 것이 가장 좋습니다. 공개 기본 `yolo26n.pt`는 번호판 전용 모델이 아니므로 인식률이 낮을 수 있습니다.

## 실행 방법

```powershell
# GUI 모드, 인수 없이 실행하면 기본값
python ocr.py
python ocr.py --gui

# 웹 서버 모드
python ocr.py --server

# 포트 지정
python ocr.py --server --port 8080

# GUI와 웹 서버를 함께 실행
python ocr.py --hybrid
```

웹 서버 기본 주소는 `http://127.0.0.1:5000`입니다. `start.bat`은 웹 서버를 실행하고 종료 시 `cloudflared.exe` 프로세스를 정리합니다.

## 사용 흐름

1. 단속 위치, 단속 사유, 오전/오후를 선택합니다.
2. 단속 사진을 한 장 이상 선택하거나 업로드합니다.
3. 번호판 자동 인식이 끝날 때까지 기다립니다.
4. 결과 화면에서 차량번호를 확인하고 필요한 경우 수정합니다.
5. 저장하지 않을 항목은 제외한 뒤 Excel로 저장합니다.
6. 저장된 기록은 GUI의 `기록 보기` 또는 웹의 `/report`에서 확인합니다.

## GUI 모드

- `폴더 선택` 또는 `파일 선택`으로 이미지를 불러옵니다.
- `분석 시작`으로 OCR을 실행하고, 결과 목록에서 항목을 선택해 번호판을 수정합니다.
- `Excel 저장`은 저장 대화상자에서 선택한 `.xlsx` 파일로 저장합니다.
- `서버 시작`을 누르면 GUI 안에서 웹 서버와 Cloudflare Tunnel을 시작합니다.
- `주소 복사`로 현재 접속 주소를 클립보드에 복사할 수 있습니다.
- 같은 PC에서 GUI가 중복 실행되지 않도록 락 파일을 사용합니다.

## 웹 모드

- 기본 로컬 주소: `http://127.0.0.1:5000`
- 업로드 화면: `/`
- 진행 상태: `/status/<task_id>`
- 결과 확인: `/result_view/<task_id>`
- 저장 파일 다운로드: `/download/<filename>`
- 기록 리포트: `/report`
- SQLite DB 다운로드: `/download_history_db`

`SYSTEM_PASSWORD` 값을 코드에서 설정하면 외부 접속 시 비밀번호 확인을 사용할 수 있습니다. 기본값은 빈 문자열이며, 이 경우 비밀번호 없이 접속됩니다. 로컬 접속은 자동 로그인됩니다.

## 설정

설정은 프로그램 폴더의 `.settings` 파일에 저장됩니다. GUI의 `설정` 버튼에서 다음 항목을 저장할 수 있습니다.

- Cloudflare Tunnel Token
- Cloudflare 고정 도메인
- Discord Webhook URL
- 로컬 서버 포트
- 웹 업로드 폴더
- 웹 백업 폴더
- Excel 저장 폴더 설정값
- YOLO 모델 파일
- 마지막으로 선택한 위치, 사유, 시간대

서버 포트, 폴더 경로, YOLO 모델처럼 시작 시 읽는 값은 프로그램 또는 서버를 다시 시작해야 적용됩니다.

## 저장 파일

- 웹 모드 메인 Excel: `주차단속내역_YYYY-MM-DD_오전.xlsx` 또는 `주차단속내역_YYYY-MM-DD_오후.xlsx`
- 웹 모드 백업 Excel: `backup/YYYY-MM-DD/단속내역_오전_HH시MM분SS초.xlsx`
- GUI 모드 Excel: 저장 대화상자에서 사용자가 선택한 `.xlsx` 파일
- 누적 기록 DB: `parking_records.sqlite3`

SQLite에는 저장 시각, 날짜, 시간대, 단속 위치, 사유, 차량번호, 원본 파일명, 이미지 경로, 저장 모드, 연결된 Excel 파일 정보가 저장됩니다.

## 직접 빌드

Windows에서 실행 파일을 만들려면 다음 명령을 실행합니다.

```bat
build.bat
```

빌드 스크립트는 다음 작업을 수행합니다.

1. `ocr.py`, `dll_extractor.py`, `templates/`, YOLO 모델 파일을 확인합니다.
2. `.venv`가 없으면 생성합니다.
3. OCR DLL이 없으면 자동 추출을 시도합니다.
4. `requirements.txt` 의존성을 설치합니다.
5. 기존 `dist/`, `build/` 폴더를 정리합니다.
6. PyInstaller로 `dist/주차단속시스템.exe`를 생성합니다.

`build.spec`는 `templates/`, `static/`, `dlls/`, 지원되는 YOLO 모델 파일을 실행 파일에 포함합니다. `cloudflared.exe`는 실행 중 필요하면 다운로드됩니다.

## 프로젝트 구성

```text
ocrtest/
├─ ocr.py               # Flask 서버, OCR 처리, Excel 저장
├─ gui.py               # Tkinter GUI
├─ records_store.py     # SQLite 기록 저장 및 조회
├─ settings_manager.py  # 설정 저장 및 로드
├─ dll_extractor.py     # OneOCR DLL 자동 추출
├─ requirements.txt     # Python 의존성
├─ build.spec           # PyInstaller 설정
├─ build.bat            # Windows 빌드 스크립트
├─ start.bat            # 웹 서버 실행 스크립트
├─ best.pt              # 기본 번호판 탐지 모델
├─ templates/           # 웹 화면 템플릿
├─ static/              # 정적 파일
├─ dlls/                # OneOCR DLL 파일
├─ uploads/             # 웹 업로드 이미지
└─ backup/              # 웹 저장 백업 파일
```

`dlls/`, `uploads/`, `backup/`, `.settings`, `cloudflared.exe`, `parking_records.sqlite3`, 생성된 Excel 파일, `build/`, `dist/`는 로컬 실행 또는 빌드 과정에서 생성됩니다.

## 문제 해결

- **DLL 초기화 실패**: Snipping Tool 또는 Photos 앱을 설치한 뒤 `python dll_extractor.py --force`를 실행합니다.
- **YOLO 모델 로드 실패**: 번호판 학습 모델 파일이 프로그램 폴더에 있는지 확인하거나 GUI 설정에서 `.pt` 파일 경로를 지정합니다.
- **포트 사용 중**: `--port` 옵션이나 GUI 설정의 로컬 포트를 다른 값으로 변경합니다.
- **메인 Excel 저장 실패**: Excel 파일이 열려 있으면 웹 모드에서 메인 파일 업데이트가 실패할 수 있습니다. 이 경우 `backup/` 폴더의 최신 파일을 확인합니다.
- **Cloudflare 연결 실패**: 로컬 주소로는 계속 사용할 수 있습니다. 외부 접속이 필요하면 인터넷 연결과 `cloudflared.exe` 다운로드 가능 여부를 확인합니다.
