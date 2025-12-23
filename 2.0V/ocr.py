import os
import sys
import shutil
import re
import threading
import urllib.parse
import cv2
import numpy as np
import uuid
import gc
import time
import socket
import requests
import subprocess
import ctypes
import pandas as pd
from datetime import datetime
from functools import wraps
from collections import Counter
from ctypes import Structure, byref, POINTER, c_int64, c_int32, c_float, c_ubyte, c_char, c_char_p
from contextlib import contextmanager

from ultralytics import YOLO
from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for, session
from waitress import serve

# ==========================================
# 1. 시스템 설정 및 라이브러리 로드
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
BACKUP_DIR = os.path.join(BASE_DIR, 'backup')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# OCR 설정 (DLL 및 모델 파일명)
MODEL_NAME = 'oneocr.onemodel'
DLL_NAME = 'oneocr.dll'
MODEL_KEY = b"kj)TGtrK>f]b[Piow.gU+nC@s\"\"\"\"\"\"4"

# ==========================================
# [보안 설정]
# 비밀번호를 비워두면("") 외부 접속도 비밀번호 없이 통과됩니다.
# ==========================================
SYSTEM_PASSWORD = ""  
SECRET_KEY = "super_secret_security_key_change_this"

app = Flask(__name__)
app.secret_key = SECRET_KEY

# 전역 변수
excel_lock = threading.Lock()
tasks = {}

# ==========================================
# 2. Ctypes 구조체 및 DLL 정의 (OCR)
# ==========================================
c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_ubyte_p = POINTER(c_ubyte)

class ImageStructure(Structure):
    _fields_ = [
        ('type', c_int32),
        ('width', c_int32),
        ('height', c_int32),
        ('_reserved', c_int32),
        ('step_size', c_int64),
        ('data_ptr', c_ubyte_p)
    ]

class BoundingBox(Structure):
    _fields_ = [
        ('x1', c_float), ('y1', c_float), ('x2', c_float), ('y2', c_float),
        ('x3', c_float), ('y3', c_float), ('x4', c_float), ('y4', c_float)
    ]

DLL_FUNCTIONS = [
    ('CreateOcrInitOptions', [c_int64_p], c_int64),
    ('OcrInitOptionsSetUseModelDelayLoad', [c_int64, c_char], c_int64),
    ('CreateOcrPipeline', [c_char_p, c_char_p, c_int64, c_int64_p], c_int64),
    ('CreateOcrProcessOptions', [c_int64_p], c_int64),
    ('OcrProcessOptionsSetMaxRecognitionLineCount', [c_int64, c_int64], c_int64),
    ('RunOcrPipeline', [c_int64, POINTER(ImageStructure), c_int64, c_int64_p], c_int64),
    ('GetOcrLineCount', [c_int64, c_int64_p], c_int64),
    ('GetOcrLine', [c_int64, c_int64, c_int64_p], c_int64),
    ('GetOcrLineContent', [c_int64, POINTER(c_char_p)], c_int64),
    ('ReleaseOcrResult', [c_int64], None),
    ('ReleaseOcrInitOptions', [c_int64], None),
    ('ReleaseOcrPipeline', [c_int64], None),
    ('ReleaseOcrProcessOptions', [c_int64], None)
]

# DLL 로드
ocr_dll = None
try:
    dll_path = os.path.join(BASE_DIR, DLL_NAME)
    if not os.path.exists(dll_path):
        print(f"❌ 오류: {DLL_NAME} 파일을 찾을 수 없습니다.")
        sys.exit(1)
        
    ocr_dll = ctypes.WinDLL(dll_path)
    for name, argtypes, restype in DLL_FUNCTIONS:
        if hasattr(ocr_dll, name):
            func = getattr(ocr_dll, name)
            func.argtypes = argtypes
            func.restype = restype
        else:
            print(f"⚠️ 경고: DLL 함수 '{name}'를 찾을 수 없습니다.")
    print(f"✅ Custom OCR DLL 로드 성공")
except Exception as e:
    print(f"❌ DLL 초기화 실패: {e}")
    sys.exit(1)

# ==========================================
# 3. OcrEngine 클래스
# ==========================================
@contextmanager
def suppress_output():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)

class OcrEngine:
    def __init__(self):
        self.init_opts = c_int64()
        self._check(ocr_dll.CreateOcrInitOptions(byref(self.init_opts)), "InitOptions 생성 실패")
        self._check(ocr_dll.OcrInitOptionsSetUseModelDelayLoad(self.init_opts, 0), "DelayLoad 설정 실패")
        
        model_path = os.path.join(BASE_DIR, MODEL_NAME).encode()
        self.pipeline = c_int64()
        
        with suppress_output():
            self._check(ocr_dll.CreateOcrPipeline(
                model_path, 
                ctypes.create_string_buffer(MODEL_KEY), 
                self.init_opts, 
                byref(self.pipeline)
            ), "파이프라인 생성 실패 (모델 파일 확인 필요)")

        self.proc_opts = c_int64()
        self._check(ocr_dll.CreateOcrProcessOptions(byref(self.proc_opts)), "ProcessOptions 생성 실패")
        ocr_dll.OcrProcessOptionsSetMaxRecognitionLineCount(self.proc_opts, 1000)

    def _check(self, code, msg):
        if code != 0:
            raise RuntimeError(f"{msg} (Code: {code})")

    def recognize_numpy(self, img_np):
        if img_np is None or img_np.size == 0:
            return ""

        if len(img_np.shape) == 2:
            img_bgra = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGRA)
        elif len(img_np.shape) == 3:
            img_bgra = cv2.cvtColor(img_np, cv2.COLOR_BGR2BGRA)
        else:
            return ""

        h, w = img_bgra.shape[:2]
        step = w * 4
        
        img_struct = ImageStructure(
            type=3, width=w, height=h, _reserved=0, step_size=step,
            data_ptr=img_bgra.ctypes.data_as(c_ubyte_p)
        )

        res_handle = c_int64()
        if ocr_dll.RunOcrPipeline(self.pipeline, byref(img_struct), self.proc_opts, byref(res_handle)) != 0:
            return ""

        line_count = c_int64()
        ocr_dll.GetOcrLineCount(res_handle, byref(line_count))
        
        full_text = []
        for i in range(line_count.value):
            l_handle = c_int64()
            ocr_dll.GetOcrLine(res_handle, i, byref(l_handle))
            content = c_char_p()
            ocr_dll.GetOcrLineContent(l_handle, byref(content))
            if content.value:
                try:
                    text = content.value.decode('utf-8', errors='ignore')
                    full_text.append(text)
                except:
                    pass
        
        ocr_dll.ReleaseOcrResult(res_handle)
        return " ".join(full_text)

try:
    global_ocr = OcrEngine()
    print(f"✅ OCR 엔진 초기화 완료")
except Exception as e:
    print(f"❌ OCR 엔진 초기화 실패: {e}")
    sys.exit(1)

# YOLO 모델 로드
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
if os.path.exists(YOLO_MODEL_PATH):
    print(f"✅ YOLO 모델 로드: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
else:
    print("⚠️ 기본 모델(yolov8n.pt) 로드. 인식률이 낮을 수 있습니다.")
    model = YOLO('yolov8n.pt')

LOCATIONS = [
    "1동", "2동", "3동", "4동", "5동", "6동", "7동", "8동", "9동", "10동",
    "11동", "12동", "13동", "14동", "15동", "중앙동", "민원동", "2청사"
]
REASONS = [
    "주차선 외 위반", "경차 구역 위반", "임산부 구역 위반",
    "방문객 전용 구역 위반", "전기차 구역 위반", "지하주차장 통로 위반",
    "장애인 구역 위반", "소방차 전용구역 위반", "주차금지구역위반"
]

# ==========================================
# 4. 이미지 처리 및 유틸리티
# ==========================================

# [스마트 로그인] 로컬/Cloudflare 접속 구분
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. 시스템 비밀번호가 없는 경우 무조건 통과
        if not SYSTEM_PASSWORD:
            session['logged_in'] = True
            return f(*args, **kwargs)

        # 2. 이미 로그인된 세션인 경우 통과
        if session.get('logged_in'):
            return f(*args, **kwargs)
        
        # 3. 로컬 접속(127.0.0.1) 자동 통과 처리
        is_localhost = request.remote_addr == '127.0.0.1'
        is_cloudflare = request.headers.get('CF-Ray') is not None
        
        if is_localhost and not is_cloudflare:
            session['logged_in'] = True # 로컬은 자동 로그인
            return f(*args, **kwargs)

        # 4. 외부 접속(Cloudflare)이면서 로그인이 안 된 경우 로그인 페이지로
        return redirect(url_for('login'))
        
    return decorated_function

def add_padding(img, pad_size=20, color=(255, 255, 255)):
    return cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=color)

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def apply_threshold(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def fix_common_errors(text):
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
    match = re.search(r'(\d{2,3}[가-힣]\d{4})', text)
    return match.group(1) if match else None

def mask_side_regions(img, ratio=0.1):
    h, w = img.shape[:2]
    w_cut = int(w * ratio)
    masked = img.copy()
    cv2.rectangle(masked, (0, 0), (w_cut, h), (255, 255, 255), -1)
    cv2.rectangle(masked, (w - w_cut, 0), (w, h), (255, 255, 255), -1)
    return masked

def smart_plate_filter(text):
    text = clean_text(text)
    if re.fullmatch(r'\d{2,3}[가-힣]\d{4}', text):
        return text
    front_bolt_match = re.search(r'(\d{2,3}[가-힣])0(\d{4})', text)
    if front_bolt_match:
        return front_bolt_match.group(1) + front_bolt_match.group(2)
    rear_bolt_match = re.search(r'(\d{2,3}[가-힣]\d{4})0', text)
    if rear_bolt_match:
        return rear_bolt_match.group(1)
    return find_plate_pattern(text)

def stitch_broken_plate(raw_text):
    text = fix_common_errors(raw_text)
    fronts = re.findall(r'\d{2,3}[가-힣]', text)
    backs = re.findall(r'\d{4}', text)
    for f in fronts:
        for b in backs:
            combined = f + b
            if find_plate_pattern(combined):
                return combined
    return None

def process_and_ocr(crop_img, start_time, timeout=3.0, is_full_image=False):
    if crop_img.ndim == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img

    if not is_full_image:
        gray = mask_side_regions(gray, ratio=0.1)

    filters = []
    filters.append(("Gray+Pad", add_padding(gray)))
    filters.append(("CLAHE", add_padding(apply_clahe(gray))))
    filters.append(("Thresh", add_padding(apply_threshold(gray))))
    
    kernel = np.ones((3, 3), np.uint8)
    filters.append(("Dilate", add_padding(cv2.dilate(apply_threshold(gray), kernel, iterations=1))))
    filters.append(("Invert", add_padding(cv2.bitwise_not(apply_threshold(gray)))))

    if is_full_image:
        scales = [1.0]
    else:
        scales = [2.0, 1.0]

    candidates = []

    for scale in scales:
        for _, processed_img in filters:
            if time.time() - start_time > timeout:
                break
            
            try:
                if scale != 1.0:
                    target_img = cv2.resize(processed_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                else:
                    target_img = processed_img

                raw_text = global_ocr.recognize_numpy(target_img)
                plate = smart_plate_filter(raw_text)

                if plate:
                    candidates.append(plate)
                    if is_full_image:
                        return [plate]
                    if candidates.count(plate) >= 2:
                        return [plate]
                
                elif is_full_image:
                    stitched = stitch_broken_plate(raw_text)
                    if stitched:
                        return [stitched]

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

    if model:
        try:
            results = model(original_img, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    pad_x = int(box_w * 0.10) 
                    pad_y = int(box_h * 0.12)
                    crop = original_img[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
                    if crop.size > 0:
                        candidates_boxes.append({'y2': y2, 'crop': crop, 'is_full': False})
            del results
        except Exception as e:
            log_lines.append(f"YOLO Error: {e}")

    candidates_boxes.append({'y2': h, 'crop': original_img, 'is_full': True})
    candidates_boxes.sort(key=lambda x: (x['is_full'], -x['y2']))

    plate_found = False
    for item in candidates_boxes:
        if time.time() - start_time > timeout:
            log_lines.append(" ⚠️ [Timeout] 시간 초과")
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

def background_processing(task_id, file_paths, location, reason, ampm):
    print(f"🚀 [Task {task_id}] 작업 시작 (총 {len(file_paths)}장)")
    results_list = []
    total = len(file_paths)

    try:
        for idx, path in enumerate(file_paths):
            filename = os.path.basename(path)
            tasks[task_id]['current'] = idx + 1
            tasks[task_id]['last_processed'] = filename
            print(f"  ↳ Processing [{idx+1}/{total}]: {filename} ... ", end='', flush=True)

            try:
                plate, _ = detect_best_plate(path)
            except Exception as e:
                print(f"Error: {e}")
                plate = ""
            
            print(f"Done ({plate if plate else '인식실패'})")
            
            web_url = "/uploads/" + urllib.parse.quote(os.path.relpath(path, UPLOAD_DIR).replace('\\', '/'))
            results_list.append({'filename': filename, 'plate': plate, 'image_url': web_url})
            gc.collect()

        tasks[task_id]['results'] = results_list
        tasks[task_id]['report_text'] = f"{location} {reason} ({ampm}) - 총 {total}건"
        tasks[task_id]['status'] = 'done'
        print(f"🏁 [Task {task_id}] 작업 완료.\n")

    except Exception as e:
        print(f"🔥 [Task {task_id}] 오류: {e}")
        tasks[task_id]['status'] = 'error'

# ==========================================
# 5. Flask 라우트 정의
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not SYSTEM_PASSWORD:
        session['logged_in'] = True
        return redirect(url_for('index'))

    if request.method == 'POST':
        if request.form['password'] == SYSTEM_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="❌ 비밀번호가 올바르지 않습니다.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

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
    loc = request.form.get('location', "위치 미지정")
    reason = request.form.get('reason', "사유 미지정")
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
        'total': len(saved_files), 'current': 0, 'status': 'processing',
        'last_processed': '', 'results': [], 'location': loc, 'reason': reason
    }

    thread = threading.Thread(target=background_processing, args=(task_id, saved_files, loc, reason, ampm))
    thread.daemon = True
    thread.start()

    return render_template('progress.html', task_id=task_id, total=len(saved_files))

@app.route('/status/<task_id>')
@login_required
def check_status(task_id):
    if task_id not in tasks: return jsonify({'error': 'Unknown task'}), 404
    return jsonify({
        'status': tasks[task_id]['status'],
        'current': tasks[task_id]['current'],
        'total': tasks[task_id]['total'],
        'last_processed': tasks[task_id]['last_processed']
    })

@app.route('/result_view/<task_id>')
@login_required
def result_view(task_id):
    if task_id not in tasks: return f"<h3>❌ 작업을 찾을 수 없습니다.</h3><a href='/'>메인으로</a>", 404
    task = tasks[task_id]
    if task['status'] == 'error': return f"<h3>🔥 오류 발생</h3><a href='/'>메인으로</a>", 500
    if task['status'] == 'processing':
        return f"<h3>⏳ 분석 중... ({task['current']} / {task['total']})</h3><script>setTimeout(function(){{ location.reload(); }}, 2000);</script>", 200
    return render_template('result.html', results=task['results'], report_text=task['report_text'], location=task['location'], reason=task['reason'])

@app.route('/save', methods=['POST'])
@login_required
def save():
    entries = []
    loc = request.form.get('location', '')
    reason = request.form.get('reason', '')
    report_text = request.form.get('report_text', '')
    
    if '(오후)' in report_text:
        time_suffix = "오후"
    elif '(오전)' in report_text:
        time_suffix = "오전"
    else:
        time_suffix = "오전" if datetime.now().hour < 12 else "오후"

    for k, v in request.form.items():
        if k.startswith('plate_') and v and v.lower() != 's':
            entries.append({
                "날짜": datetime.now().strftime('%Y-%m-%d'),
                "시간대": time_suffix,
                "단속위치": loc,
                "사유": reason,
                "차량번호": v
            })

    if not entries:
        return "저장할 데이터가 없습니다.", 400

    today_str = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.now().strftime('%H시%M분%S초')
    
    root_filename = f"주차단속내역_{today_str}_{time_suffix}.xlsx"
    root_path = os.path.join(BASE_DIR, root_filename)

    backup_folder = os.path.join(BACKUP_DIR, today_str)
    os.makedirs(backup_folder, exist_ok=True)
    backup_filename = f"단속내역_{time_suffix}_{timestamp}.xlsx"
    backup_path = os.path.join(backup_folder, backup_filename)

    messages = []

    with excel_lock:
        try:
            # 1. 메인 엑셀 파일 로드 (없으면 생성)
            if os.path.exists(root_path):
                try:
                    df = pd.read_excel(root_path)
                except Exception:
                    df = pd.DataFrame(columns=["날짜", "시간대", "단속위치", "사유", "차량번호"])
            else:
                df = pd.DataFrame(columns=["날짜", "시간대", "단속위치", "사유", "차량번호"])
            
            new_df = pd.DataFrame(entries)
            final_df = pd.concat([df, new_df], ignore_index=True)

            # 2. [중요] 백업 파일 우선 저장
            final_df.to_excel(backup_path, index=False)
            messages.append(f"✅ <b>데이터 안전 저장됨 (Backup):</b> {today_str}/{backup_filename}")

            # 3. 메인 파일 덮어쓰기 시도
            try:
                shutil.copy2(backup_path, root_path)
                messages.append(f"✅ <b>메인 파일 업데이트됨:</b> {root_filename}")
                main_status = "성공"
            except PermissionError:
                messages.append(
                    f"<br>⚠️ <b>[주의] 메인 엑셀 파일이 열려있어 업데이트하지 못했습니다.</b><br>"
                    f"하지만 데이터는 <b>backup 폴더</b>에 안전하게 저장되었습니다.<br>"
                    f"최신 내용을 보려면 엑셀을 닫고 다시 저장하거나 backup 폴더를 확인하세요."
                )
                main_status = "실패"

        except Exception as e:
            return f"<h3>❌ 치명적 저장 오류</h3><p>{str(e)}</p>", 500
    
    # [수정된 부분] 백업 파일의 경로를 상대 경로로 올바르게 전달
    backup_relative_path = os.path.join('backup', today_str, backup_filename)

    return render_template(
        'success.html', 
        count=len(entries), 
        excel_file=root_filename if main_status == "성공" else backup_relative_path,
        report_text=report_text,
        extra_message="<br>".join(messages)
    )

@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        return send_from_directory(BASE_DIR, filename, as_attachment=True)
    except:
        return "파일을 찾을 수 없습니다.", 404

@app.route('/uploads/<path:path>')
@login_required
def uploads(path):
    return send_from_directory(UPLOAD_DIR, path)

@app.route('/help')
@login_required
def help_page():
    return render_template('help.html') if os.path.exists(os.path.join(BASE_DIR, 'templates', 'help.html')) else "<h3>도움말 준비 중</h3><a href='/'>홈으로</a>"

@app.route('/report')
@login_required
def report_page():
    files = [f for f in os.listdir(BASE_DIR) if f.endswith('.xlsx') and '주차단속내역' in f]
    files.sort(reverse=True)
    file_list = "".join([f'<li><a href="/download/{f}">{f}</a></li>' for f in files])
    
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
            {file_list if files else "<li>저장된 내역이 없습니다.</li>"}
        </ul>
        <hr>
        <p>※ 파일이 열려있어 저장이 안 된 경우, <b>backup</b> 폴더를 확인하세요.</p>
        <a href="/" class="btn">🏠 홈으로 돌아가기</a>
    </body>
    </html>
    """

# ==========================================
# 6. 서버 실행 및 터널링
# ==========================================
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

    os.system("taskkill /f /im cloudflared.exe >nul 2>&1")
    cmd = [cf_filename, "tunnel", "--url", f"http://localhost:{port}"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        text=True, encoding='utf-8', errors='replace'
    )

    tunnel_url = None
    start_time = time.time()
    while time.time() - start_time < 15:
        line = process.stderr.readline()
        if not line: break
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            tunnel_url = match.group(0)
            break
    return tunnel_url

if __name__ == '__main__':
    PORT = 5000
    HOST_IP = '127.0.0.1' 
    
    print("=" * 60)
    print(f"🚀 [서버 시작] 보안 모드 (v2.0 Updated)")
    if SYSTEM_PASSWORD:
        print(f"🔑 외부 접속 비밀번호: {SYSTEM_PASSWORD}")
    else:
        print(f"🔓 비밀번호 미설정 (누구나 접속 가능)")
        
    print(f"📂 백업 폴더: {BACKUP_DIR}")

    public_url = init_cloudflare_tunnel(PORT)
    print("-" * 60)
    if public_url:
        print(f"🌍 [외부 접속 주소] : {public_url}")
    else:
        print("❌ Cloudflare 터널 실패 (로컬 접속만 가능)")

    print("-" * 60)
    print(f"🏠 [로컬 접속 주소] : http://{HOST_IP}:{PORT}")
    print("   (로컬 접속 시 비밀번호 없이 자동 로그인됩니다)")
    print("=" * 60)

    serve(app, host=HOST_IP, port=PORT, threads=10, channel_timeout=3000)