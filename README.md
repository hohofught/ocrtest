```md
# ScanBot (Windows) — YOLO + Windows OCR 기반 번호판 인식/단속 기록 웹앱

이 프로젝트는 **Flask 웹페이지에서 사진(여러 장)을 업로드**하면, 각 이미지에서 **YOLO로 번호판 후보 영역을 찾고 → Windows OCR(winsdk)로 문자 인식**을 수행한 뒤 결과를 보여주고, 선택한 항목을 **엑셀(.xlsx)**로 저장하는 Windows용 웹앱입니다.

> 실행 엔트리: `ocr.py`  
> 실행 스크립트: `start.bat`  
> 모델 파일: `best.pt` (없으면 `yolov8n.pt` 사용)

---

## 1) 주요 기능

- **웹 로그인(간단 비밀번호 방식)**
  - `/login`에서 비밀번호 입력 후 세션 기반 접근
  - 미로그인 시 주요 페이지 접근 차단

- **다중 이미지 업로드 & 백그라운드 처리**
  - 한 번에 여러 장 업로드(`photos`) 가능
  - 업로드 후 **백그라운드 스레드**가 순차 처리하며 진행률 제공

- **YOLO 기반 번호판 후보 추출**
  - `best.pt`가 있으면 커스텀 모델 사용
  - 없으면 `yolov8n.pt`(기본 모델)로 동작 (정확도 낮을 수 있음)
  - 후보를 만들 때 **감지 박스 + 여백(padding)** 및 **전체 이미지(full scan)**도 후보로 포함

- **이미지 전처리 + Windows OCR(winsdk)**
  - 대비 향상(CLAHE), adaptive threshold, dilation, invert 등 다양한 전처리 조합
  - OCR은 `winsdk.windows.media.ocr` 기반
  - 언어 설정: `ko-KR` (한국어 OCR 엔진)

- **결과 페이지 제공**
  - 파일명 / 인식된 번호 / 업로드 이미지 미리보기 URL 제공
  - 인식 실패 시 빈 값 또는 “인식실패” 형태로 표시될 수 있음

- **엑셀 저장**
  - 결과에서 선택된 항목을 엑셀로 누적 저장
  - 파일명 예: `주차단속내역_YYYY-MM-DD.xlsx`
  - 컬럼: `날짜`, `단속위치`, `사유`, `차량번호`

- **Cloudflare Tunnel 자동 실행(옵션)**
  - `cloudflared.exe`가 없으면 자동 다운로드 후 `trycloudflare.com` 공개 URL를 콘솔에 출력
  - 실패해도 로컬 접속은 가능

---

## 2) 프로젝트 구조


업로드 저장 경로는 코드에서 다음 규칙으로 자동 생성됩니다.

```

uploads/YYYY.MM.DD/<단속위치>/<오전|오후>/<사유>/

````

---

## 3) 동작 흐름(한 눈에 보기)

1. 사용자가 `/login`에서 비밀번호로 로그인
2. `/`(index)에서 **단속위치/사유/오전·오후** 선택 후 사진 업로드
3. 서버가 업로드 파일을 `uploads/...`에 저장하고 `task_id` 생성
4. 백그라운드 스레드가 각 파일에 대해 반복:
   - `detect_best_plate(path)` 수행
   - YOLO로 후보 영역 생성 → 후보별로 `process_and_ocr()`로 전처리 + OCR 시도
   - 가장 먼저 “성공”한 번호판을 해당 이미지 결과로 확정
5. `/status/<task_id>`로 진행률을 폴링(프론트에서 주기적으로 확인)
6. 완료 시 `/result_view/<task_id>`에서 결과 목록 표시
7. 저장 버튼으로 `/save` 호출 → 오늘 날짜 엑셀 파일에 누적 기록

---

## 4) 요구사항(권장)

### OS
- **Windows 10/11 권장**
  - Windows OCR(winsdk) 사용을 전제로 구성되어 있습니다.

### Python
- Python 3.10+ 권장 (코드상 별도 제한은 없으나 최신 환경 권장)

### 주요 파이썬 패키지
- `flask`
- `waitress`
- `ultralytics` (YOLO)
- `opencv-python` (`cv2`)
- `numpy`
- `pandas` (엑셀 저장/읽기)
- `openpyxl` (pandas가 xlsx 다룰 때 필요할 수 있음)
- `requests`
- `winsdk` (**Windows OCR 필수**)

> 참고: `ultralytics`는 내부적으로 PyTorch가 필요할 수 있습니다. 설치 환경에 따라 추가 의존성(특히 CUDA/CPU)이 달라집니다.

---

## 5) 설치 방법

### 1) 가상환경(선택)
```bash
python -m venv .venv
.venv\Scripts\activate
````

### 2) 패키지 설치(예시)

```bash
pip install flask waitress ultralytics opencv-python numpy pandas openpyxl requests winsdk
```

---

## 6) 실행 방법

### A. 배치 파일로 실행(권장)

```bat
start.bat
```

### B. 직접 실행

```bash
python ocr.py
```

실행하면 콘솔에 다음이 출력됩니다.

* 보안 모드 적용 메시지(비밀번호)
* Cloudflare tunnel 성공 시 외부 URL (`https://xxxx.trycloudflare.com`)
* 로컬 접속 주소 (`http://<내부IP>:<PORT>`)

---

## 7) 접속 & 사용 방법

1. 브라우저에서 서버 주소 접속

   * 로컬: `http://localhost:<PORT>` 또는 콘솔에 출력된 로컬 주소
   * 외부: 콘솔에 출력된 `trycloudflare.com` 주소(성공한 경우)

2. 로그인 페이지에서 비밀번호 입력

3. 메인 화면(`/`)에서

   * 단속위치 선택
   * 사유 선택
   * 오전/오후 선택
   * 사진 다중 업로드 후 실행

4. 진행 화면(`/result_view/<task_id>` 또는 progress 페이지)에서 완료까지 대기

5. 결과 화면에서 필요한 항목 저장

---

## 8) 설정 방법(중요)

### 1) 로그인 비밀번호 변경

`ocr.py` 상단의 설정 구역에서 변경합니다.

* `SYSTEM_PASSWORD = "1234"` 값을 원하는 비밀번호로 수정하세요.
* `app.secret_key`도 운영 시에는 반드시 변경 권장

### 2) YOLO 모델 변경

* 기본: `Windows/best.pt`가 존재하면 그 모델을 로드
* 없으면 `yolov8n.pt`로 로드

원하는 커스텀 모델을 사용하려면 `best.pt`를 해당 위치에 두면 됩니다.

### 3) 포트/서버 설정

* 서버 실행은 `waitress.serve(... threads=10 ...)` 형태로 동작합니다.
* 포트는 `ocr.py` 하단의 `PORT` 설정(또는 코드 내부 기본값)에 의해 결정됩니다.

---

## 9) 웹 라우트(API) 정리

* `GET /login` : 로그인 페이지
* `POST /login` : 비밀번호 로그인
* `GET /logout` : 로그아웃
* `GET /` : 메인 업로드 페이지
* `POST /upload` : 사진 업로드 및 task 생성
* `GET /status/<task_id>` : 진행 상태(JSON)

  * 응답 예: `{status, current, total, last_processed}`
* `GET /result_view/<task_id>` : 결과 페이지(처리 중이면 자동 새로고침)
* `POST /save` : 결과를 엑셀로 저장
* `GET /uploads/<path:path>` : 업로드된 이미지 정적 제공
* `GET /help` : 도움말 페이지

---

## 10) 트러블슈팅

### 1) `winsdk` 관련 오류

* 메시지: `❌ 필수: 'winsdk' 라이브러리가 필요합니다. (pip install winsdk)`
* 해결:

  * `pip install winsdk` 설치
  * Windows 환경에서 실행(특히 OCR 기능은 Windows 의존)

### 2) YOLO 기본 모델로만 동작(정확도 낮음)

* 콘솔에 `⚠️ 기본 모델(yolov8n.pt) 로드...`가 뜨면 `best.pt`가 없는 상태입니다.
* 해결:

  * 커스텀 학습 모델 `best.pt`를 `ocr.py`와 같은 폴더에 배치

### 3) Cloudflare Tunnel 실패

* 콘솔에 `❌ Cloudflare 터널 실패 (로컬 접속만 가능)`이 떠도 기능 자체는 정상 동작합니다.
* 해결:

  * 방화벽/네트워크 정책 확인
  * GitHub 릴리즈 다운로드 차단 여부 확인
  * 로컬 URL로 사용

### 4) 엑셀 저장 오류

* 저장 시 `엑셀 저장 오류: ...`가 뜨면,

  * 파일이 다른 프로그램(엑셀 등)에서 열려 잠금 상태일 수 있습니다.
* 해결:

  * 해당 날짜의 `주차단속내역_YYYY-MM-DD.xlsx`를 닫고 다시 저장

---

## 11) 개발 메모(성능/정확도 관련)

* 인식 로직은 “**후보 영역 여러 개** + **전체 이미지 후보**”를 포함하여,
  먼저 성공하는 결과를 채택하는 구조입니다.
* 전처리(Threshold/Dilate/Invert 등)를 여러 변형으로 시도해 OCR 성공률을 높입니다.
* 처리 시간 제한(`timeout`)이 있어, 너무 오래 걸리면 중단될 수 있습니다.

---

## 12) 라이선스 / 고지

사용된 best.pt
https://github.com/MuhammadMoinFaisal/Computervisionprojects/tree/main/ANPR_YOLOv10/weights
