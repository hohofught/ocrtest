```markdown
# 📸 주차 단속 시스템 (Parking Enforcement OCR)

이 프로젝트는 주차 단속 현장 사진을 업로드하여 **차량 번호판을 자동으로 인식(OCR)**하고, 단속 내역을 **Excel 파일로 자동 저장**하는 웹 기반 시스템입니다. Flask 웹 서버와 YOLOv8 객체 인식 모델, 그리고 전용 OCR 엔진(DLL)을 사용합니다.

## 📌 주요 기능

1.  **웹 기반 인터페이스**: PC 및 모바일 브라우저에서 사진 업로드 및 관리 가능.
2.  **자동 번호판 인식**:
    * YOLO 모델을 통한 차량/번호판 영역 검출.
    * 이미지 전처리(Grayscale, CLAHE, Threshold) 후 Custom OCR 엔진을 통한 텍스트 추출.
    * 오인식 방지 로직 (번호판 패턴 정규식 필터링).
3.  **데이터 관리 및 리포트**:
    * 단속 위치, 사유, 시간대별 자동 폴더 분류 저장.
    * **Excel 자동 저장**: 날짜별 파일 생성 (`주차단속내역_YYYY-MM-DD.xlsx`).
    * **백업 시스템**: 엑셀 파일이 열려 있어 저장 실패 시, `backup` 폴더에 별도 파일 생성하여 데이터 유실 방지.
4.  **네트워크 접근성**:
    * `Cloudflare Tunnel` 자동 연동으로 외부에서도 접속 가능한 URL 생성.
    * 로컬 접속 시 자동 로그인 / 외부 접속 시 비밀번호 보안 기능.

## 🛠 설치 및 실행 환경 (Prerequisites)

이 시스템은 Windows 환경에서 작동하도록 설계되었습니다. (DLL 의존성 때문)

### 1. 필수 라이브러리 설치
Python 3.8 이상이 필요하며, 아래 명령어로 의존성 패키지를 설치하세요.

```bash
pip install flask waitress opencv-python numpy pandas ultralytics requests

```

### 2. 필수 파일 확인

실행 전 프로젝트 루트 폴더에 다음 파일들이 반드시 존재해야 합니다.

* `ocr.py`: 메인 서버 코드
* `start.bat`: 실행 스크립트
* `best.pt` (또는 `yolov8n.pt`): YOLO 객체 인식 모델
* **OCR 엔진 파일 (필수)**:
* `oneocr.dll`
* `oneocr.onemodel`
* `onnxruntime.dll`



## 📂 폴더 구조

```text
Project_Root/
├── uploads/             # [자동생성] 업로드된 원본 이미지 (날짜/장소/사유별 분류)
├── backup/              # [자동생성] 엑셀 저장 실패 시 백업 파일 저장소
├── templates/           # 웹 페이지 HTML (index.html, progress.html, result.html 등)
├── ocr.py               # Flask 서버 및 OCR 로직 메인 코드
├── start.bat            # 간편 실행 스크립트
├── best.pt              # YOLO 모델 파일
├── oneocr.dll           # OCR 엔진 DLL
├── start.bat
├── onnxruntime.dll      
└── oneocr.onemodel      # OCR 모델 데이터

```

## 🚀 실행 방법 (Usage)

1. **`start.bat` 실행**:
* 폴더 내의 `start.bat` 파일을 더블 클릭합니다.
* 서버가 초기화되고 Cloudflare 터널링이 시작될 때까지 잠시 기다립니다.


2. **웹 접속**:
* **로컬 접속**: 브라우저를 열고 `http://127.0.0.1:5000` 접속 (자동 로그인).
* **외부 접속**: 콘솔 창에 출력된 `https://random-name.trycloudflare.com` 주소로 접속 (비밀번호 입력 필요).


3. **단속 등록**:
* 위치, 단속 사유, 시간대(오전/오후)를 선택합니다.
* 현장 사진을 여러 장 선택하여 업로드합니다.


4. **결과 확인 및 저장**:
* 이미지 분석이 완료되면 번호판 인식 결과가 표시됩니다.
* 오인식된 경우 수동으로 번호판을 수정한 후 **[결과 저장]** 버튼을 누릅니다.



## ⚙️ 설정 (Configuration)

`ocr.py` 파일 상단의 설정을 수정하여 시스템을 커스터마이징 할 수 있습니다.

```python
# ocr.py 내부 설정 영역

# 보안 비밀번호 (외부 접속 시 사용, 비워두면 "" 누구나 접속 가능)
SYSTEM_PASSWORD = "" 

# 단속 위치 및 사유 목록 수정
LOCATIONS = ["1동", "2동", "중앙동", ...]
REASONS = ["주차선 위반", "장애인 구역 위반", ...]

```

## ⚠️ 주의 사항 및 트러블슈팅

* **Excel 저장 오류**: 만약 `주차단속내역.xlsx` 파일을 엑셀로 열어놓은 상태에서 저장을 시도하면, 메인 파일 업데이트에 실패합니다. 이 경우 시스템은 자동으로 `backup/` 폴더에 데이터를 저장하고 경고 메시지를 띄웁니다.
* **DLL 오류**: `oneocr.dll` 파일을 찾을 수 없다는 오류가 뜨면, 해당 파일이 `ocr.py`와 같은 폴더에 있는지 확인하세요.
* **YOLO 모델**: `best.pt` 파일이 없으면 기본적으로 `yolov8n.pt`를 다운로드하여 사용하지만, 인식률이 낮을 수 있습니다.

---

## 사용된 파일들

`best.pt 파일`
`https://github.com/MuhammadMoinFaisal/Computervisionprojects/tree/main/ANPR_YOLOv10/weights`

`oneocr.dll`
`oneocr.onemodel`
`onnxruntime.dll`
`https://github.com/killkimno/MORT_VERSION/releases/download/oneocr/oneocr.zip`

```

```