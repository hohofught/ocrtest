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
import requests
import subprocess
from datetime import datetime
from functools import wraps
from collections import Counter
import pandas as pd
from ultralytics import YOLO
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for, session

# --- [1. 설정 및 라이브러리 확인] ---
try:
    import winsdk.windows.media.ocr as windows_ocr
    import winsdk.windows.globalization as globalization
    import winsdk.windows.graphics.imaging as imaging
    import winsdk.windows.storage as storage
    import winsdk.windows.storage.streams as streams
except ImportError:
    print("❌ 필수: 'winsdk' 라이브러리가 필요합니다. (pip install winsdk)")
    sys.exit(1)

app = Flask(__name__)

# ==========================================
# 🔒 [보안 설정 구역] 비밀번호를 여기서 변경하세요
# ==========================================
SYSTEM_PASSWORD = "1234" 
app.secret_key = "super_secret_security_key_change_this"
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

excel_lock = threading.Lock()
tasks = {} 

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

# --- [2. 보안 및 유틸리티 로직] ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['password'] == SYSTEM_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = "❌ 비밀번호가 올바르지 않습니다."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# --- [3. 이미지 처리 및 OCR 코어 로직] ---

def add_padding(img, pad_size=20, color=(255, 255, 255)):
    """이미지 테두리에 흰색 여백 추가 (OCR 인식률 향상)"""
    return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=color)

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def apply_threshold(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

async def run_ocr_on_ndarray(img_np: np.ndarray) -> str:
    """메모리상에서 직접 OCR 수행 (디스크 I/O 없음)"""
    if not ocr_engine:
        return ""
    try:
        if img_np is None or getattr(img_np, "size", 0) == 0:
            return ""
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        if img_np.ndim == 3:
            img_for_ocr = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            img_for_ocr = img_np

        ok, buf = cv2.imencode(".png", img_for_ocr)
        if not ok:
            return ""
        png_bytes = buf.tobytes()

        mem_stream = streams.InMemoryRandomAccessStream()
        writer = streams.DataWriter(mem_stream.get_output_stream_at(0))
        writer.write_bytes(png_bytes)
        await writer.store_async()
        await writer.flush_async()
        writer.detach_stream()
        mem_stream.seek(0)

        decoder = await imaging.BitmapDecoder.create_async(mem_stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await ocr_engine.recognize_async(bitmap)

        return " ".join([line.text for line in result.lines])
    except Exception:
        return ""

def fix_common_errors(text):
    """자주 발생하는 OCR 오인식 문자 교정"""
    text = text.upper()
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('I', '1').replace('l', '1').replace('|', '1')
    text = text.replace('S', '5').replace('s', '5')
    text = text.replace('B', '8')
    text = text.replace('G', '6')
    text = text.replace('Z', '2')
    return text

def clean_text(text):
    text = fix_common_errors(text)
    return re.sub(r'[^0-9가-힣]', '', text)

def find_plate_pattern(text):
    """번호판 정규식 패턴 매칭 (예: 12가3456)"""
    match = re.search(r'(\d{2,3}[가-힣]\d{4})', text)
    return match.group(1) if match else None

def stitch_broken_plate(raw_text):
    """파편화된 텍스트에서 번호판 조합 (짜집기)"""
    text = fix_common_errors(raw_text)
    front_pattern = re.compile(r'\d{2,3}[가-힣]')
    fronts = front_pattern.findall(text)
    back_pattern = re.compile(r'\d{4}')
    backs = back_pattern.findall(text)

    for f in fronts:
        for b in backs:
            combined = f + b
            if find_plate_pattern(combined):
                return combined
    return None

def process_and_ocr(crop_img, start_time, timeout=3.0, is_full_image=False):
    # 기본 그레이스케일
    if crop_img.ndim == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img

    # --- 필터 정의 ---
    filters = []
    filters.append(("Gray+Pad", add_padding(gray))) # 1. 기본
    
    clahe_img = apply_clahe(gray)
    filters.append(("CLAHE", add_padding(clahe_img))) # 2. 대비
    
    thresh_img = apply_threshold(gray)
    filters.append(("Thresh", add_padding(thresh_img))) # 3. 이진화
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1) 
    filters.append(("Dilate", add_padding(dilated))) # 4. 팽창
    
    inverted = cv2.bitwise_not(thresh_img)
    filters.append(("Invert", add_padding(inverted))) # 5. 반전

    # 배율 설정
    if is_full_image:
        scales = [1.0]
    else:
        scales = [2.0, 1.0] # 2배 확대 우선 시도

    candidates = []

    for scale in scales:
        for _, processed_img in filters:
            if time.time() - start_time > timeout:
                break
            
            try:
                # 배율 적용
                if scale != 1.0:
                    target_img = cv2.resize(
                        processed_img, None, 
                        fx=scale, fy=scale, 
                        interpolation=cv2.INTER_CUBIC
                    )
                else:
                    target_img = processed_img

                # OCR 실행
                raw_text = asyncio.run(run_ocr_on_ndarray(target_img))
                cleaned = clean_text(raw_text)
                plate = find_plate_pattern(cleaned)

                if plate:
                    candidates.append(plate)
                    if is_full_image:
                        return [plate]
                    if candidates.count(plate) >= 2:
                        return [plate]
                
                # 전체 스캔 실패 시 짜집기 시도
                elif is_full_image:
                    stitched_plate = stitch_broken_plate(raw_text)
                    if stitched_plate:
                        return [stitched_plate]

            except Exception:
                pass
        
        if time.time() - start_time > timeout:
            break

    if candidates:
        most_common = Counter(candidates).most_common(1)
        return [most_common[0][0]]
    
    return []

def detect_best_plate(img_path):
    start_time = time.time()
    timeout = 3.0

    log_lines = []
    best_plate = ""

    original_img = cv2.imread(img_path)
    if original_img is None:
        return "읽기실패", []

    h, w, _ = original_img.shape
    candidates_boxes = []

    # 1. YOLO 객체 탐지
    if model:
        try:
            results = model(original_img, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    
                    # 여백 조정 (타이트하게)
                    pad_x = int(box_w * 0.10) 
                    pad_y = int(box_h * 0.12)

                    crop = original_img[
                        max(0, y1 - pad_y):min(h, y2 + pad_y),
                        max(0, x1 - pad_x):min(w, x2 + pad_x)
                    ]
                    
                    if crop.size > 0:
                        candidates_boxes.append({
                            'y2': y2,
                            'area': box_w * box_h,
                            'crop': crop,
                            'is_full': False 
                        })
            del results
        except Exception as e:
            log_lines.append(f"YOLO Error: {e}")

    # 2. '전체 이미지'도 후보군에 등록
    candidates_boxes.append({
        'y2': h,           
        'area': h * w,     
        'crop': original_img,
        'is_full': True 
    })

    # 3. 정렬 우선순위
    candidates_boxes.sort(key=lambda x: (x['is_full'], -x['y2']))

    plate_found = False
    
    for item in candidates_boxes:
        if time.time() - start_time > timeout:
            log_lines.append(" ⚠️ [Timeout] 시간 초과로 강제 종료")
            break

        is_full = item['is_full']
        label = "전체 스캔" if is_full else f"박스(y2:{item['y2']})"

        found_plates = process_and_ocr(item['crop'], start_time, timeout, is_full_image=is_full)
        
        if found_plates:
            best_plate = found_plates[0]
            log_lines.append(f" ✅ [인식 성공] {best_plate} - {label}")
            plate_found = True
            break 

    if not plate_found:
        log_lines.append(" ❌ 최종 인식 실패")

    del original_img
    return best_plate, log_lines

# --- [4. 백그라운드 작업 스레드] ---
def background_processing(task_id, file_paths, location, reason, ampm):
    print(f"🚀 [Task {task_id}] 작업 시작 (총 {len(file_paths)}장)")

    results_list = []
    total = len(file_paths)

    try:
        for idx, path in enumerate(file_paths):
            filename = os.path.basename(path)

            tasks[task_id]['current'] = idx + 1
            tasks[task_id]['last_processed'] = filename

            print(f"   ↳ Processing [{idx+1}/{total}]: {filename} ... ", end='', flush=True)

            try:
                plate, _ = detect_best_plate(path)
            except Exception as e:
                print(f"Error: {e}")
                plate = ""

            print(f"Done. ({plate if plate else '인식실패'})")

            web_url = "/uploads/" + urllib.parse.quote(
                os.path.relpath(path, UPLOAD_DIR).replace('\\', '/')
            )
            results_list.append({
                'filename': filename,
                'plate': plate,
                'image_url': web_url
            })

            gc.collect()

        tasks[task_id]['results'] = results_list
        tasks[task_id]['report_text'] = f"{location} {reason} ({ampm}) - 총 {total}건"
        tasks[task_id]['status'] = 'done'
        print(f"🏁 [Task {task_id}] 작업 완료.\n")

    except Exception as e:
        print(f"🔥 [Task {task_id}] 치명적 오류: {e}")
        tasks[task_id]['status'] = 'error'

# --- [5. Flask 라우트] ---

@app.route('/')
@login_required
def index():
    return render_template('index.html', locations=LOCATIONS, reasons=REASONS)

@app.route('/changelog')
@login_required
def changelog():
    return render_template('changelog.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    # [수정] 입력값 없을 시 기본값 자동 채움
    loc = request.form.get('location')
    if not loc or loc.strip() == "":
        loc = "위치 미지정"
        
    reason = request.form.get('reason')
    if not reason or reason.strip() == "":
        reason = "사유 미지정"
    
    ampm = request.form.get('ampm', "오전")

    save_path = os.path.join(
        UPLOAD_DIR,
        datetime.now().strftime('%Y.%m.%d'),
        loc, ampm, reason
    )
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
@login_required
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
@login_required
def result_view(task_id):
    if task_id not in tasks:
        return f"<h3>❌ 작업을 찾을 수 없습니다.</h3><a href='/'>메인으로</a>", 404

    task = tasks[task_id]
    if task['status'] == 'error':
        return f"<h3>🔥 오류 발생</h3><a href='/'>메인으로</a>", 500

    if task['status'] == 'processing':
        return f"""
        <h3>⏳ 분석 중... ({task['current']} / {task['total']})</h3>
        <script>setTimeout(function(){{ location.reload(); }}, 2000);</script>
        """, 200

    return render_template(
        'result.html',
        results=task['results'],
        report_text=task['report_text'],
        location=task['location'],
        reason=task['reason']
    )

@app.route('/save', methods=['POST'])
@login_required
def save():
    entries = []
    loc = request.form.get('location', '')
    reason = request.form.get('reason', '')
    report_text = request.form.get('report_text', '단속 결과')

    # [수정] 파일명 시간대(오전/오후) 구분 로직
    if '(오후)' in report_text:
        time_suffix = "오후"
    elif '(오전)' in report_text:
        time_suffix = "오전"
    else:
        time_suffix = "오전" if datetime.now().hour < 12 else "오후"

    for k, v in request.form.items():
        if k.startswith('plate_') and v:
            if v.lower().strip() == 's':
                continue
            entries.append({
                "날짜": datetime.now().strftime('%Y-%m-%d'),
                "시간대": time_suffix,
                "단속위치": loc,
                "사유": reason,
                "차량번호": v
            })

    fname = f"주차단속내역_{datetime.now().strftime('%Y-%m-%d')}_{time_suffix}.xlsx"
    path = os.path.join(BASE_DIR, fname)

    try:
        with excel_lock:
            if os.path.exists(path):
                df = pd.read_excel(path)
            else:
                df = pd.DataFrame(columns=["날짜", "시간대", "단속위치", "사유", "차량번호"])

            new_df = pd.DataFrame(entries)
            if not new_df.empty:
                final_df = pd.concat([df, new_df], ignore_index=True)
                final_df.to_excel(path, index=False)

            del df, new_df
            if 'final_df' in locals(): del final_df
            gc.collect()

    except Exception as e:
        return f"엑셀 저장 오류: {e}"

    return render_template(
        'success.html',
        count=len(entries),
        excel_file=fname,
        report_text=report_text
    )

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        return send_from_directory(BASE_DIR, filename, as_attachment=True)
    except Exception as e:
        return f"파일을 찾을 수 없습니다: {e}", 404

@app.route('/report')
@login_required
def report_page():
    # 엑셀 파일 목록 조회 (최신순)
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.xlsx') and '주차단속내역' in f]
    files.sort(reverse=True)
    
    file_list_html = "".join([f'<li><a href="/download/{f}">{f}</a></li>' for f in files])
    
    return f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>단속 리포트</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; max-width: 600px; margin: auto; }}
            h2 {{ color: #2c3e50; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-bottom: 1px solid #ddd; }}
            a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
            .btn {{ display:inline-block; margin-top:20px; padding:10px 20px; background:#6c757d; color:white; text-decoration:none; border-radius:5px; }}
        </style>
    </head>
    <body>
        <h2>📊 주차 단속 엑셀 파일 목록</h2>
        <ul>
            {file_list_html if files else "<li>저장된 내역이 없습니다.</li>"}
        </ul>
        <a href="/" class="btn">🏠 홈으로 돌아가기</a>
    </body>
    </html>
    """

@app.route('/uploads/<path:path>')
@login_required
def uploads(path):
    return send_from_directory(UPLOAD_DIR, path)

@app.route('/help')
@login_required
def help_page():
    try:
        return render_template('help.html')
    except:
        return "<h3>도움말 페이지 준비 중입니다.</h3><a href='/'>돌아가기</a>"

# --- [Cloudflare Tunnel 자동 설정] ---
def init_cloudflare_tunnel(port):
    cf_filename = "cloudflared.exe"
    cf_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

    if not os.path.exists(cf_filename):
        print(f"⬇️ Cloudflare 다운로드 중...")
        try:
            with requests.get(cf_url, stream=True) as r:
                r.raise_for_status()
                with open(cf_filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
        except Exception:
            return None

    print("Cloudflare Tunnel 시작...")
    os.system("taskkill /f /im cloudflared.exe >nul 2>&1")

    cmd = [cf_filename, "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    tunnel_url = None
    start_time = time.time()
    while time.time() - start_time < 15:
        line = process.stderr.readline()
        if not line:
            break
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            tunnel_url = match.group(0)
            break

    return tunnel_url

# --- [메인 실행부] ---
from waitress import serve

if __name__ == '__main__':
    PORT = 5000

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()
    except:
        host_ip = "127.0.0.1"

    print("=" * 60)
    print(f"🚀 [서버 시작] 보안 모드 적용됨 (비밀번호: {SYSTEM_PASSWORD})")

    public_url = init_cloudflare_tunnel(PORT)

    print("-" * 60)
    if public_url:
        print(f"🌍 [외부 접속 주소] : {public_url}")
    else:
        print("❌ Cloudflare 터널 실패 (로컬 접속만 가능)")

    print("-" * 60)
    print(f"🏠 [로컬 접속 주소] : http://{host_ip}:{PORT}")
    print("=" * 60)

    serve(app, host='0.0.0.0', port=PORT, threads=10, channel_timeout=3000)