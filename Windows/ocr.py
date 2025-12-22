import os
import sys
import shutil
import re
import asyncio
import threading
import urllib.parse
import math
import cv2
import numpy as np
import uuid
import gc
import time
import socket
import shutil
import requests
import subprocess
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for
from ultralytics import YOLO
from waitress import serve

# --- [1. 설정 및 라이브러리 확인] ---
try:
    import winsdk.windows.media.ocr as windows_ocr
    import winsdk.windows.globalization as globalization
    import winsdk.windows.graphics.imaging as imaging
    import winsdk.windows.storage as storage
except ImportError:
    print("❌ 필수: 'winsdk' 라이브러리가 필요합니다. (pip install winsdk)")
    sys.exit(1)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
TEMP_IMG_PATH = os.path.join(BASE_DIR, 'temp_ocr_processing.jpg')

os.makedirs(UPLOAD_DIR, exist_ok=True)

excel_lock = threading.Lock()
tasks = {}  # 작업 상태를 저장할 전역 딕셔너리

# --- [YOLO 모델 로드] ---
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
if os.path.exists(YOLO_MODEL_PATH):
    print(f"✅ Custom YOLO 모델 로드: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
else:
    print("⚠️ 기본 모델(yolov8n.pt) 로드. 인식률이 낮을 수 있습니다.")
    model = YOLO('yolov8n.pt')

LOCATIONS = [
    "1동", "2동", "3동", "4동", "5동",
    "6동", "7동", "8동", "9동", "10동",
    "11동", "12동", "13동", "14동", "15동", 
    "중앙동", "민원동", "2청사"
]
REASONS = [
    "주차선 외 위반", "경차 구역 위반", "임산부 구역 위반",
    "방문객 전용 구역 위반", "전기차 구역 위반",
    "지하주차장 통로, 통행, 방해주차 위반",
    "장애인 구역 위반, 지정주차 구역(업무용포함)",
    "소방차 전용구역 위반", "주차금지구역위반 (필로티 등)"
]

# --- [OCR 엔진 전역 초기화] ---
try:
    ocr_engine = windows_ocr.OcrEngine.try_create_from_language(globalization.Language("ko-KR"))
except Exception as e:
    print(f"⚠️ OCR 엔진 초기화 실패: {e}")
    ocr_engine = None

# --- [2. 이미지 처리 및 OCR 로직] ---

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def apply_threshold(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

async def run_ocr_on_image(img_path):
    if not ocr_engine: return ""
    try:
        abs_path = os.path.abspath(img_path)
        if not os.path.exists(abs_path): return ""
        file = await storage.StorageFile.get_file_from_path_async(abs_path)
        stream = await file.open_async(storage.FileAccessMode.READ)
        decoder = await imaging.BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await ocr_engine.recognize_async(bitmap)
        return " ".join([line.text for line in result.lines])
    except Exception:
        return ""

def clean_text(text):
    return re.sub(r'[^0-9가-힣]', '', text)

def find_plate_pattern(text):
    match = re.search(r'(\d{2,3}[가-힣]\d{4})', text)
    return match.group(1) if match else None

def process_and_ocr(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    filters = [
        ("Original", gray),
        ("CLAHE", apply_clahe(gray)),
        ("Threshold", apply_threshold(gray))
    ]

    for _, processed_img in filters:
        try:
            processed_img = cv2.resize(processed_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(TEMP_IMG_PATH, processed_img) 
            
            raw_text = asyncio.run(run_ocr_on_image(TEMP_IMG_PATH))
            cleaned = clean_text(raw_text)
            plate = find_plate_pattern(cleaned)
            
            if plate: return [plate]
        except: pass
            
    return []

def detect_best_plate(img_path):
    log_lines = []
    best_plate = ""
    
    original_img = cv2.imread(img_path)
    if original_img is None: return "읽기실패", []

    h, w, _ = original_img.shape
    candidates_boxes = []

    if model:
        try:
            results = model(original_img, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.15)
                    crop = original_img[
                        max(0, y1-pad_y):min(h, y2+pad_y), 
                        max(0, x1-pad_x):min(w, x2+pad_x)
                    ]
                    if crop.size > 0:
                        candidates_boxes.append({'area': area, 'crop': crop})
            del results
        except Exception as e:
            log_lines.append(f"YOLO Error: {e}")

    candidates_boxes.sort(key=lambda x: x['area'], reverse=True)

    plate_found = False
    for item in candidates_boxes:
        found_plates = process_and_ocr(item['crop'])
        if found_plates:
            best_plate = found_plates[0]
            log_lines.append(f" ✅ [최대 크기 선택] {best_plate} (면적: {item['area']})")
            plate_found = True
            break 
    
    if not plate_found:
        log_lines.append(" ⚠️ 박스 감지 실패 또는 인식 불가. 전체 이미지 스캔 시도.")
        found_plates = process_and_ocr(original_img)
        if found_plates:
            best_plate = found_plates[0]
            log_lines.append(f" ✅ [전체 스캔 성공] {best_plate}")
        else:
            log_lines.append(" ❌ 최종 인식 실패")

    del original_img
    return best_plate, log_lines

# --- [3. 백그라운드 작업 스레드 (수정됨)] ---
def background_processing(task_id, file_paths, location, reason, ampm):
    print(f"🚀 [Task {task_id}] 작업 시작 (총 {len(file_paths)}장)")
    
    results_list = []
    total = len(file_paths)
    
    try:
        for idx, path in enumerate(file_paths):
            filename = os.path.basename(path)
            
            # 진행 상황 업데이트
            tasks[task_id]['current'] = idx + 1
            tasks[task_id]['last_processed'] = filename
            
            print(f"   ↳ Processing [{idx+1}/{total}]: {filename} ... ", end='', flush=True)
            
            # 개별 파일 처리 중 에러가 나도 전체가 멈추지 않도록 처리
            try:
                plate, _ = detect_best_plate(path)
            except Exception as e:
                print(f"Error: {e}")
                plate = ""

            print(f"Done. ({plate if plate else '인식실패'})")
            
            web_url = "/uploads/" + urllib.parse.quote(os.path.relpath(path, UPLOAD_DIR).replace('\\', '/'))
            results_list.append({
                'filename': filename,
                'plate': plate,
                'image_url': web_url
            })
            
            gc.collect()

        # 정상 종료 시
        tasks[task_id]['results'] = results_list
        tasks[task_id]['report_text'] = f"{location} {reason} ({ampm}) - 총 {total}건"
        tasks[task_id]['status'] = 'done' # [중요] 상태 변경
        print(f"🏁 [Task {task_id}] 작업 완료.\n")
        
    except Exception as e:
        # 치명적 오류 발생 시
        print(f"🔥 [Task {task_id}] 치명적 오류: {e}")
        tasks[task_id]['status'] = 'error'

# --- [4. Flask 라우트] ---

@app.route('/')
def index():
    return render_template('index.html', locations=LOCATIONS, reasons=REASONS)

@app.route('/upload', methods=['POST'])
def upload():
    loc = request.form['location']
    reason = request.form['reason']
    ampm = request.form.get('ampm', "오전")
    
    save_path = os.path.join(UPLOAD_DIR, datetime.now().strftime('%Y.%m.%d'), loc, ampm, reason)
    os.makedirs(save_path, exist_ok=True)
    
    saved_files = []
    files = request.files.getlist('photos')
    
    for f in files:
        if f.filename:
            safe_name = os.path.basename(f.filename) 
            path = os.path.join(save_path, safe_name)
            f.save(path)
            saved_files.append(path)

    if not saved_files:
        return "파일이 업로드되지 않았습니다.", 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'total': len(saved_files),
        'current': 0,
        'status': 'processing',
        'last_processed': '',
        'results': [],
        'location': loc,
        'reason': reason
    }

    thread = threading.Thread(
        target=background_processing,
        args=(task_id, saved_files, loc, reason, ampm)
    )
    thread.daemon = True
    thread.start()

    return render_template('progress.html', task_id=task_id, total=len(saved_files))

@app.route('/status/<task_id>')
def check_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Unknown task'}), 404
    
    task = tasks[task_id]
    return jsonify({
        'status': task['status'],
        'current': task['current'],
        'total': task['total'],
        'last_processed': task['last_processed']
    })

@app.route('/result_view/<task_id>')
def result_view(task_id):
    # 1. 작업 ID가 메모리에 없는 경우 (서버 재시작 등)
    if task_id not in tasks:
        return f"""
        <h3>❌ 작업을 찾을 수 없습니다.</h3>
        <p>서버가 재시작되었거나, 유효하지 않은 ID입니다.</p>
        <p>현재 메모리에 있는 작업 ID 목록: {list(tasks.keys())}</p>
        <a href="/">메인으로 돌아가기</a>
        """, 404

    task = tasks[task_id]

    # 2. 작업 중 에러가 발생한 경우
    if task['status'] == 'error':
        return f"""
        <h3>🔥 작업 중 오류 발생</h3>
        <p>시스템 로그를 확인해주세요.</p>
        <a href="/">메인으로 돌아가기</a>
        """, 500

    # 3. 아직 진행 중인 경우
    if task['status'] == 'processing':
        return f"""
        <h3>⏳ 아직 분석 중입니다.</h3>
        <p>현재 {task['current']} / {task['total']} 처리 중...</p>
        <script>
            setTimeout(function(){{ location.reload(); }}, 2000);
        </script>
        """, 200

    # 4. 정상 완료 (status == 'done') -> 결과 페이지 표시
    return render_template('result.html', 
                           results=task['results'],
                           report_text=task['report_text'],
                           location=task['location'],
                           reason=task['reason'])

@app.route('/save', methods=['POST'])
def save():
    entries = []
    loc = request.form.get('location', '')
    reason = request.form.get('reason', '')
    
    for k, v in request.form.items():
        if k.startswith('plate_') and v:
            if v.lower().strip() == 's': continue
            entries.append({
                "날짜": datetime.now().strftime('%Y-%m-%d'), 
                "단속위치": loc, 
                "사유": reason, 
                "차량번호": v
            })
    
    fname = f"주차단속내역_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    path = os.path.join(BASE_DIR, fname)
    
    try:
        with excel_lock:
            if os.path.exists(path): 
                df = pd.read_excel(path)
            else: 
                df = pd.DataFrame(columns=["날짜", "단속위치", "사유", "차량번호"])
            
            new_df = pd.DataFrame(entries)
            final_df = pd.concat([df, new_df], ignore_index=True)
            final_df.to_excel(path, index=False)
            
            del df, new_df, final_df
            gc.collect()
            
    except Exception as e:
        return f"엑셀 저장 오류: {e} (파일이 열려있는지 확인하세요)"
    
    return f"""
    <script>
        alert('저장 완료! (총 {len(entries)}건)');
        window.location.href = '/';
    </script>
    """

@app.route('/uploads/<path:path>')
def uploads(path):
    return send_from_directory(UPLOAD_DIR, path)

@app.route('/help')
def help_page():
    return render_template('help.html')


def init_cloudflare_tunnel(port):
    """
    1. cloudflared.exe가 없으면 다운로드
    2. 터널 프로세스 실행
    3. 생성된 외부 접속 URL 파싱하여 반환
    """
    cf_filename = "cloudflared.exe"
    cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

    # 1. 실행 파일 확인 및 다운로드
    if not os.path.exists(cf_filename):
        print(f"⬇️ Cloudflare 실행 파일이 없습니다. 다운로드를 시작합니다... ({cf_filename})")
        try:
            with requests.get(cf_url, stream=True) as r:
                r.raise_for_status()
                with open(cf_filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            print("✅ 다운로드 완료!")
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None

    # 2. 터널 실행 (로그에서 URL을 찾기 위해 subprocess 사용)
    print("Cloudflare Tunnel을 시작합니다...")
    
    # 기존에 실행 중인 cloudflared가 있다면 충돌 방지를 위해 종료 시도 (선택 사항)
    os.system("taskkill /f /im cloudflared.exe >nul 2>&1")

    cmd = [cf_filename, "tunnel", "--url", f"http://localhost:{port}"]
    
    # 프로세스 시작
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8', 
        errors='replace' # 인코딩 에러 방지
    )

    # 3. 로그에서 URL 추출
    tunnel_url = None
    start_time = time.time()
    
    # 10초 동안 로그를 분석하여 URL 찾기
    while time.time() - start_time < 15:
        line = process.stderr.readline()
        if not line:
            break
        
        # URL 패턴 찾기 (trycloudflare.com)
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            tunnel_url = match.group(0)
            break
            
    if tunnel_url:
        return tunnel_url
    else:
        print("⚠️ 터널 URL을 찾지 못했습니다. (잠시 후 다시 시도하거나 로그를 확인하세요)")
        return None

# --- [메인 실행부 수정] ---
from waitress import serve
import socket

if __name__ == '__main__':
    PORT = 5000

    # 1. 내부 IP 찾기
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except:
        host_ip = "127.0.0.1"

    print("=" * 60)
    print(f"🚀 [서버 시작] Waitress WSGI Server Running...")
    
    # 2. Cloudflare 터널 시작 (비동기적으로 실행됨)
    # 터널링은 별도 프로세스로 돌고 있으므로, URL만 따오고 서버를 켭니다.
    public_url = init_cloudflare_tunnel(PORT)

    print("-" * 60)
    if public_url:
        print(f"🌍 [외부 접속 주소] : {public_url}")
        print(f"   (이 주소를 팀원들에게 공유하세요. 전 세계 어디서든 접속 가능)")
    else:
        print("❌ Cloudflare 터널 생성 실패. (방화벽 설정이나 인터넷 연결을 확인하세요)")
    
    print("-" * 60)
    print(f"🏠 [로컬 접속 주소] : http://{host_ip}:{PORT}")
    print(f"👥 최대 동시 접속 : 10명")
    print("=" * 60)

    # 3. 웹 서버 실행
    serve(app, host='0.0.0.0', port=PORT, threads=10, channel_timeout=3000)
