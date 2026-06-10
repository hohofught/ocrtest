# -*- coding: utf-8 -*-
"""
주차 단속 시스템 - 로컬 GUI 모드
Tkinter 기반 데스크톱 애플리케이션
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from PIL import Image, ImageTk
import pandas as pd
import atexit

# 설정 관리자 import
from settings_manager import get_settings, init_settings
from records_store import add_records, fetch_daily_counts, fetch_recent_records, fetch_record_count

# OCR 엔진 및 처리 함수 import (ocr.py에서)
try:
    from ocr import (
        detect_best_plate,
        LOCATIONS, REASONS, BASE_DIR, BACKUP_DIR, DB_PATH
    )
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"OCR 모듈 로드 실패: {e}")
    OCR_AVAILABLE = False


UI_LANGUAGES = {
    "ko": "한국어",
    "en": "English",
}

UI_LANGUAGE_CODES_BY_LABEL = {label: code for code, label in UI_LANGUAGES.items()}

UI_TEXT = {
    "ko": {
        "app_title": "주차 단속 시스템 (로컬 모드)",
        "button_select_folder": "폴더 선택",
        "button_select_files": "파일 선택",
        "button_start_analysis": "분석 시작",
        "button_stop": "중지",
        "button_report": "기록 보기",
        "button_settings": "설정",
        "button_server_start": "서버 시작",
        "button_server_stop": "서버 중지",
        "button_copy_url": "주소 복사",
        "label_location": "위치:",
        "label_reason": "사유:",
        "label_time_period": "시간대:",
        "ampm_morning": "오전",
        "ampm_afternoon": "오후",
        "image_select_prompt": "이미지를 선택하세요",
        "button_prev": "이전",
        "button_next": "다음",
        "label_language": "언어:",
        "results_title": "인식 결과",
        "label_plate": "번호판:",
        "button_edit": "수정",
        "button_save_excel": "Excel 저장",
        "button_reset": "초기화",
        "button_save": "저장",
        "button_cancel": "취소",
        "button_close": "닫기",
        "button_browse": "찾아보기",
        "status_ready": "준비됨",
        "status_ocr_load_failed": "OCR 모듈 로드 실패",
        "status_images_loaded": "{count}개 이미지 로드됨",
        "status_plate_updated": "번호판 수정됨: {plate}",
        "status_analysis_complete": "분석 완료!",
        "status_processing": "분석 중... {index}/{total}",
        "status_processing_stopped": "분석 중지됨",
        "status_excel_saved": "Excel 저장 완료: {count}건",
        "status_reset": "초기화됨",
        "status_url_copied": "주소 복사됨: {url}",
        "status_server_not_running": "서버가 실행되지 않았습니다",
        "status_open_report_page": "기록 페이지 열기",
        "status_db_path_copied": "SQLite DB 경로 복사됨",
        "status_history_displayed": "과거 기록 표시",
        "status_server_stopped": "서버 중지됨 (다시 시작 버튼 클릭)",
        "status_server_starting": "웹 서버 시작 중...",
        "status_language_changed": "언어가 한국어로 변경되었습니다.",
        "result_done": "완료",
        "result_waiting": "대기",
        "folder_dialog_title": "이미지 폴더 선택",
        "file_dialog_title": "이미지 파일 선택",
        "image_file_type": "이미지 파일",
        "all_files_type": "모든 파일",
        "image_load_failed": "이미지 로드 실패: {error}",
        "error_title": "오류",
        "warning_title": "경고",
        "success_title": "성공",
        "copied_title": "복사 완료",
        "confirm_title": "확인",
        "server_error_title": "서버 오류",
        "ocr_module_unavailable": "OCR 모듈이 로드되지 않았습니다.",
        "select_image_first": "먼저 이미지를 선택해주세요.",
        "save_no_data": "저장할 데이터가 없습니다.",
        "save_no_plate": "인식된 번호판이 없습니다.",
        "excel_file_type": "Excel 파일",
        "excel_filename_prefix": "주차단속내역",
        "excel_col_date": "날짜",
        "excel_col_time_period": "시간대",
        "excel_col_location": "단속위치",
        "excel_col_reason": "사유",
        "excel_col_plate": "차량번호",
        "sqlite_save_complete": "SQLite 기록 저장 완료",
        "sqlite_save_failed": "SQLite 기록 저장 실패: {error}",
        "save_complete_message": "저장 완료: {filepath}\n총 {count}건{history_message}",
        "save_failed": "저장 실패: {error}",
        "copy_url_message": "서버 주소가 복사되었습니다:\n{url}",
        "history_title": "과거 기록",
        "history_total": "누적 기록: {count}건",
        "history_load_failed": "로드 실패: {error}",
        "button_copy_db_path": "DB 경로 복사",
        "button_open_web_report": "웹 기록 열기",
        "history_daily_title": "최근 일자별 기록",
        "history_count": "{count}건",
        "history_empty": "저장된 기록이 없습니다.",
        "heading_created_at": "저장시각",
        "heading_date": "날짜",
        "heading_time_period": "시간대",
        "heading_location": "위치",
        "heading_reason": "사유",
        "heading_plate": "차량번호",
        "heading_mode": "모드",
        "heading_source": "원본",
        "settings_title": "설정",
        "settings_section_tunnel": "링크 고정 설정 (Cloudflare Tunnel)",
        "settings_tunnel_domain": "고정 도메인:",
        "settings_tunnel_help": "토큰이 설정되면 고정 도메인으로 접속합니다.\n비워두면 임시 URL(trycloudflare.com)을 사용합니다.",
        "settings_section_discord": "Discord 알림",
        "settings_discord_help": "서버 시작 시 Discord로 알림을 보냅니다.",
        "settings_section_server": "서버 설정",
        "settings_local_port": "로컬 포트:",
        "settings_port_help": "예: 5000, 8080, 18080\n변경 후 서버를 다시 시작해야 적용됩니다.",
        "settings_section_folders": "폴더 경로 설정",
        "settings_input_folder": "입력 폴더:",
        "settings_backup_folder": "백업 폴더:",
        "settings_excel_save": "Excel 저장:",
        "settings_folder_help": "비워두면 기본 경로(프로그램 폴더)를 사용합니다.",
        "settings_section_yolo": "YOLO 모델 설정",
        "settings_model_file": "모델 파일:",
        "settings_model_help": "예: best.pt, best_yolo26.pt, yolo26n.pt 또는 .pt 파일 경로\n변경 후 프로그램을 재시작해야 적용됩니다.",
        "model_dialog_title": "YOLO 모델 파일 선택",
        "model_file_type": "YOLO 모델",
        "port_invalid": "로컬 포트는 1부터 65535 사이의 숫자여야 합니다.",
        "settings_saved_message": "설정이 저장되었습니다.\n일부 설정은 서버 재시작 후 적용됩니다.",
        "settings_save_failed": "설정 저장에 실패했습니다.",
        "settings_reset_confirm": "모든 설정을 초기화하시겠습니까?",
        "port_in_use": "포트 {port}가 이미 사용 중입니다.\n다른 서버가 실행 중인지 확인하세요.",
        "server_status_local_only": "로컬만",
        "lock_warning": "프로그램이 이미 실행 중입니다.\n기존 창을 확인해주세요.",
        "lock_activated": "기존 프로그램 창을 활성화했습니다.",
        "pillow_missing": "Pillow 라이브러리가 필요합니다: pip install Pillow",
    },
    "en": {
        "app_title": "Parking Enforcement System (Local Mode)",
        "button_select_folder": "Select Folder",
        "button_select_files": "Select Files",
        "button_start_analysis": "Start Analysis",
        "button_stop": "Stop",
        "button_report": "View Records",
        "button_settings": "Settings",
        "button_server_start": "Start Server",
        "button_server_stop": "Stop Server",
        "button_copy_url": "Copy URL",
        "label_location": "Location:",
        "label_reason": "Reason:",
        "label_time_period": "Time:",
        "ampm_morning": "AM",
        "ampm_afternoon": "PM",
        "image_select_prompt": "Select an image",
        "button_prev": "Previous",
        "button_next": "Next",
        "label_language": "Language:",
        "results_title": "Recognition Results",
        "label_plate": "Plate:",
        "button_edit": "Edit",
        "button_save_excel": "Save Excel",
        "button_reset": "Reset",
        "button_save": "Save",
        "button_cancel": "Cancel",
        "button_close": "Close",
        "button_browse": "Browse",
        "status_ready": "Ready",
        "status_ocr_load_failed": "Failed to load OCR module",
        "status_images_loaded": "{count} image(s) loaded",
        "status_plate_updated": "Plate updated: {plate}",
        "status_analysis_complete": "Analysis complete.",
        "status_processing": "Analyzing... {index}/{total}",
        "status_processing_stopped": "Analysis stopped",
        "status_excel_saved": "Excel saved: {count} record(s)",
        "status_reset": "Reset complete",
        "status_url_copied": "URL copied: {url}",
        "status_server_not_running": "Server is not running",
        "status_open_report_page": "Opening records page",
        "status_db_path_copied": "SQLite DB path copied",
        "status_history_displayed": "Showing records",
        "status_server_stopped": "Server stopped (click start to run again)",
        "status_server_starting": "Starting web server...",
        "status_language_changed": "Language changed to English.",
        "result_done": "Done",
        "result_waiting": "Waiting",
        "folder_dialog_title": "Select Image Folder",
        "file_dialog_title": "Select Image Files",
        "image_file_type": "Image Files",
        "all_files_type": "All Files",
        "image_load_failed": "Failed to load image: {error}",
        "error_title": "Error",
        "warning_title": "Warning",
        "success_title": "Success",
        "copied_title": "Copied",
        "confirm_title": "Confirm",
        "server_error_title": "Server Error",
        "ocr_module_unavailable": "OCR module is not loaded.",
        "select_image_first": "Please select images first.",
        "save_no_data": "There is no data to save.",
        "save_no_plate": "No recognized plates found.",
        "excel_file_type": "Excel Files",
        "excel_filename_prefix": "Parking_Enforcement",
        "excel_col_date": "Date",
        "excel_col_time_period": "Time Period",
        "excel_col_location": "Location",
        "excel_col_reason": "Reason",
        "excel_col_plate": "Plate Number",
        "sqlite_save_complete": "SQLite record saved",
        "sqlite_save_failed": "Failed to save SQLite record: {error}",
        "save_complete_message": "Saved: {filepath}\nTotal {count} record(s){history_message}",
        "save_failed": "Save failed: {error}",
        "copy_url_message": "Server URL copied:\n{url}",
        "history_title": "Past Records",
        "history_total": "Total records: {count}",
        "history_load_failed": "Load failed: {error}",
        "button_copy_db_path": "Copy DB Path",
        "button_open_web_report": "Open Web Records",
        "history_daily_title": "Recent Daily Records",
        "history_count": "{count} record(s)",
        "history_empty": "No saved records.",
        "heading_created_at": "Saved At",
        "heading_date": "Date",
        "heading_time_period": "Time",
        "heading_location": "Location",
        "heading_reason": "Reason",
        "heading_plate": "Plate Number",
        "heading_mode": "Mode",
        "heading_source": "Source",
        "settings_title": "Settings",
        "settings_section_tunnel": "Fixed Link Settings (Cloudflare Tunnel)",
        "settings_tunnel_domain": "Fixed Domain:",
        "settings_tunnel_help": "When a token is set, the fixed domain is used.\nLeave blank to use a temporary URL (trycloudflare.com).",
        "settings_section_discord": "Discord Notifications",
        "settings_discord_help": "Sends a Discord notification when the server starts.",
        "settings_section_server": "Server Settings",
        "settings_local_port": "Local Port:",
        "settings_port_help": "Examples: 5000, 8080, 18080\nRestart the server after changing this.",
        "settings_section_folders": "Folder Paths",
        "settings_input_folder": "Input Folder:",
        "settings_backup_folder": "Backup Folder:",
        "settings_excel_save": "Excel Save:",
        "settings_folder_help": "Leave blank to use the default path (program folder).",
        "settings_section_yolo": "YOLO Model Settings",
        "settings_model_file": "Model File:",
        "settings_model_help": "Examples: best.pt, best_yolo26.pt, yolo26n.pt, or a .pt file path\nRestart the program after changing this.",
        "model_dialog_title": "Select YOLO Model File",
        "model_file_type": "YOLO Model",
        "port_invalid": "Local port must be a number from 1 to 65535.",
        "settings_saved_message": "Settings saved.\nSome settings apply after restarting the server.",
        "settings_save_failed": "Failed to save settings.",
        "settings_reset_confirm": "Reset all settings?",
        "port_in_use": "Port {port} is already in use.\nCheck whether another server is running.",
        "server_status_local_only": "Local only",
        "lock_warning": "The program is already running.\nCheck the existing window.",
        "lock_activated": "Activated the existing program window.",
        "pillow_missing": "Pillow is required: pip install Pillow",
    },
}

APP_WINDOW_TITLES = tuple(texts["app_title"] for texts in UI_TEXT.values())


def normalize_ui_language(value):
    """설정 파일에 저장된 언어 값을 ko/en 코드로 정규화"""
    if value is None:
        return "ko"

    normalized = str(value).strip().lower()
    if normalized in ("en", "english", "eng", "영어"):
        return "en"
    if normalized in ("ko", "kr", "kor", "korean", "한국어"):
        return "ko"
    return "ko"


def translate_ui(language, key, **kwargs):
    """UI 문자열을 현재 언어로 반환하고 누락 시 한국어/키로 폴백"""
    language = normalize_ui_language(language)
    template = UI_TEXT.get(language, UI_TEXT["ko"]).get(key, UI_TEXT["ko"].get(key, key))
    return template.format(**kwargs)


class ParkingEnforcementGUI:
    """주차 단속 GUI 애플리케이션"""
    
    def __init__(self, root, start_server=False):
        self.root = root
        
        # 설정 관리자 초기화 및 로드
        self.settings = get_settings()
        self.ui_language = normalize_ui_language(self.settings.get("ui_language", "ko"))
        self.current_status_key = None
        self.current_status_kwargs = {}

        self.root.title(self.t("app_title"))
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 다크 테마 설정
        self.setup_theme()
        
        # 데이터 저장소
        self.image_files = []
        self.results = []
        self.current_index = 0
        self.processing = False
        
        # 서버 관련
        self.server_thread = None
        self.http_server = None
        self.server_lock = threading.Lock()
        self.server_stop_event = None
        self.server_running = False
        self.server_url = None
        
        # UI 구성
        self.create_widgets()
        
        # 저장된 설정 적용
        self.apply_loaded_settings()
        
        # 창 닫기 핸들러 등록 (서버 정리)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 하이브리드 모드: 서버 자동 시작
        if start_server:
            self.root.after(500, self.toggle_server)
        
        # 상태 표시
        self.set_status("status_ready" if OCR_AVAILABLE else "status_ocr_load_failed")

    def t(self, key, **kwargs):
        """현재 UI 언어에 맞는 문자열 반환"""
        return translate_ui(self.ui_language, key, **kwargs)

    def get_ampm_key(self, value):
        """시간대 표시 문자열을 내부 키로 변환"""
        normalized = str(value or "").strip().lower()
        if normalized in ("오전", "am", "morning"):
            return "morning"
        if normalized in ("오후", "pm", "afternoon"):
            return "afternoon"
        return None

    def get_ampm_options(self):
        """현재 언어의 시간대 콤보박스 옵션"""
        return [self.t("ampm_morning"), self.t("ampm_afternoon")]

    def get_current_ampm_label(self):
        """현재 언어로 표시되는 시간대 값"""
        ampm_key = self.get_ampm_key(self.ampm_var.get())
        if ampm_key == "morning":
            return self.t("ampm_morning")
        if ampm_key == "afternoon":
            return self.t("ampm_afternoon")
        return self.ampm_var.get()

    def on_language_changed(self, event=None):
        """사이드 UI 언어 선택 변경"""
        selected = self.language_var.get()
        language = UI_LANGUAGE_CODES_BY_LABEL.get(selected, normalize_ui_language(selected))
        if language == self.ui_language:
            return

        ampm_key = self.get_ampm_key(self.ampm_var.get())
        self.ui_language = language
        self.settings.set("ui_language", self.ui_language)
        self.settings.save()

        self.refresh_ui_texts(ampm_key=ampm_key)
        self.set_status("status_language_changed")

    def refresh_ui_texts(self, ampm_key=None):
        """현재 생성된 메인 UI 텍스트를 현재 언어로 갱신"""
        self.root.title(self.t("app_title"))
        self.select_folder_btn.configure(text=self.t("button_select_folder"))
        self.select_files_btn.configure(text=self.t("button_select_files"))
        self.start_processing_btn.configure(text=self.t("button_start_analysis"))
        self.stop_processing_btn.configure(text=self.t("button_stop"))
        self.report_btn.configure(text=self.t("button_report"))
        self.settings_btn.configure(text=self.t("button_settings"))
        self.server_btn.configure(text=self.t("button_server_stop" if self.server_running else "button_server_start"))
        self.copy_url_btn.configure(text=self.t("button_copy_url"))

        self.location_label.configure(text=self.t("label_location"))
        self.reason_label.configure(text=self.t("label_reason"))
        self.ampm_label.configure(text=self.t("label_time_period"))
        self.prev_btn.configure(text=self.t("button_prev"))
        self.next_btn.configure(text=self.t("button_next"))
        self.language_label.configure(text=self.t("label_language"))
        self.results_title_label.configure(text=self.t("results_title"))
        self.plate_label.configure(text=self.t("label_plate"))
        self.update_plate_btn.configure(text=self.t("button_edit"))
        self.save_excel_btn.configure(text=self.t("button_save_excel"))
        self.reset_btn.configure(text=self.t("button_reset"))

        self.language_var.set(UI_LANGUAGES[self.ui_language])
        self.ampm_combo.configure(values=self.get_ampm_options())
        if ampm_key == "morning":
            self.ampm_var.set(self.t("ampm_morning"))
        elif ampm_key == "afternoon":
            self.ampm_var.set(self.t("ampm_afternoon"))

        if self.server_running and self.server_url and self.server_url.startswith("http://127.0.0.1"):
            self.server_status_label.configure(text=self.t("server_status_local_only"))

        if not self.image_files:
            self.image_label.configure(text=self.t("image_select_prompt"))
        self.update_result_list()
        if self.current_status_key:
            self.status_label.configure(text=self.t(self.current_status_key, **self.current_status_kwargs))
    
    def apply_loaded_settings(self):
        """저장된 설정을 UI에 적용"""
        # 마지막 선택 값 복원
        last_location = self.settings.get("last_location", "")
        last_reason = self.settings.get("last_reason", "")
        last_ampm = self.settings.get("last_ampm", "")
        
        if last_location and OCR_AVAILABLE:
            if last_location in LOCATIONS:
                self.location_var.set(last_location)
        
        if last_reason and OCR_AVAILABLE:
            if last_reason in REASONS:
                self.reason_var.set(last_reason)
        
        if last_ampm:
            ampm_key = self.get_ampm_key(last_ampm)
            if ampm_key == "morning":
                self.ampm_var.set(self.t("ampm_morning"))
            elif ampm_key == "afternoon":
                self.ampm_var.set(self.t("ampm_afternoon"))
            else:
                self.ampm_var.set(last_ampm)
    
    def save_current_settings(self):
        """현재 UI 상태를 설정에 저장"""
        self.settings.set("ui_language", self.ui_language)
        self.settings.set("last_location", self.location_var.get())
        self.settings.set("last_reason", self.reason_var.get())
        self.settings.set("last_ampm", self.get_ampm_key(self.ampm_var.get()) or self.ampm_var.get())
        self.settings.save()
    
    def on_closing(self):
        """프로그램 종료 시 정리"""
        import subprocess
        
        # 현재 설정 저장
        self.save_current_settings()
        
        # 서버 중지
        if self.server_running:
            self.stop_background_server()
        
        # Cloudflare 프로세스 종료
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", "cloudflared.exe"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except:
            pass
        
        self.root.destroy()
    
    def setup_theme(self):
        """다크 테마 설정"""
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#0078d4"
        self.entry_bg = "#2d2d2d"
        self.button_bg = "#3c3c3c"
        self.success_color = "#107c10"
        
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background=self.bg_color, foreground=self.fg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=("맑은 고딕", 10))
        style.configure("TButton", background=self.button_bg, foreground=self.fg_color, font=("맑은 고딕", 10), padding=8)
        style.map("TButton", background=[("active", self.accent_color)])
        style.configure("Accent.TButton", background=self.accent_color, foreground=self.fg_color)
        style.configure("Success.TButton", background=self.success_color, foreground=self.fg_color)
        style.configure("TEntry", fieldbackground=self.entry_bg, foreground=self.fg_color)
        style.configure("TCombobox", fieldbackground=self.entry_bg, foreground=self.fg_color)
        style.configure("Horizontal.TProgressbar", background=self.accent_color, troughcolor=self.entry_bg)
    
    def create_widgets(self):
        """UI 위젯 생성"""
        # 상단 툴바
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=10)
        
        self.select_folder_btn = ttk.Button(toolbar, text=self.t("button_select_folder"), command=self.select_folder)
        self.select_folder_btn.pack(side=tk.LEFT, padx=5)
        self.select_files_btn = ttk.Button(toolbar, text=self.t("button_select_files"), command=self.select_files)
        self.select_files_btn.pack(side=tk.LEFT, padx=5)
        self.start_processing_btn = ttk.Button(toolbar, text=self.t("button_start_analysis"), command=self.start_processing, style="Accent.TButton")
        self.start_processing_btn.pack(side=tk.LEFT, padx=5)
        self.stop_processing_btn = ttk.Button(toolbar, text=self.t("button_stop"), command=self.stop_processing)
        self.stop_processing_btn.pack(side=tk.LEFT, padx=5)
        self.report_btn = ttk.Button(toolbar, text=self.t("button_report"), command=self.open_report_page)
        self.report_btn.pack(side=tk.LEFT, padx=5)
        self.settings_btn = ttk.Button(toolbar, text=self.t("button_settings"), command=self.open_settings_dialog)
        self.settings_btn.pack(side=tk.LEFT, padx=5)
        
        # 서버 토글 버튼
        self.server_btn = ttk.Button(toolbar, text=self.t("button_server_start"), command=self.toggle_server)
        self.server_btn.pack(side=tk.RIGHT, padx=5)
        
        # 주소 복사 버튼
        self.copy_url_btn = ttk.Button(toolbar, text=self.t("button_copy_url"), command=self.copy_server_url, state=tk.DISABLED)
        self.copy_url_btn.pack(side=tk.RIGHT, padx=5)
        
        self.server_status_label = ttk.Label(toolbar, text="")
        self.server_status_label.pack(side=tk.RIGHT, padx=5)
        
        # 설정 영역
        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.location_label = ttk.Label(settings_frame, text=self.t("label_location"))
        self.location_label.pack(side=tk.LEFT, padx=5)
        self.location_var = tk.StringVar(value=LOCATIONS[0] if OCR_AVAILABLE else "")
        location_combo = ttk.Combobox(settings_frame, textvariable=self.location_var, 
                                       values=LOCATIONS if OCR_AVAILABLE else [], width=15)
        location_combo.pack(side=tk.LEFT, padx=5)
        
        self.reason_label = ttk.Label(settings_frame, text=self.t("label_reason"))
        self.reason_label.pack(side=tk.LEFT, padx=5)
        self.reason_var = tk.StringVar(value=REASONS[0] if OCR_AVAILABLE else "")
        reason_combo = ttk.Combobox(settings_frame, textvariable=self.reason_var,
                                     values=REASONS if OCR_AVAILABLE else [], width=25)
        reason_combo.pack(side=tk.LEFT, padx=5)
        
        self.ampm_label = ttk.Label(settings_frame, text=self.t("label_time_period"))
        self.ampm_label.pack(side=tk.LEFT, padx=5)
        default_ampm_key = "morning" if datetime.now().hour < 12 else "afternoon"
        self.ampm_var = tk.StringVar(value=self.t(f"ampm_{default_ampm_key}"))
        self.ampm_combo = ttk.Combobox(settings_frame, textvariable=self.ampm_var, 
                                       values=self.get_ampm_options(), width=8, state="readonly")
        self.ampm_combo.pack(side=tk.LEFT, padx=5)
        
        # 메인 컨텐츠 영역
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 왼쪽: 이미지 미리보기
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(left_frame, bg=self.entry_bg, text=self.t("image_select_prompt"),
                                     fg=self.fg_color, font=("맑은 고딕", 12))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 이미지 네비게이션
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        self.prev_btn = ttk.Button(nav_frame, text=self.t("button_prev"), command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.nav_label = ttk.Label(nav_frame, text="0 / 0")
        self.nav_label.pack(side=tk.LEFT, expand=True)
        self.next_btn = ttk.Button(nav_frame, text=self.t("button_next"), command=self.next_image)
        self.next_btn.pack(side=tk.RIGHT, padx=5)
        
        # 오른쪽: 결과 목록
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_frame.pack_propagate(False)
        
        language_frame = ttk.Frame(right_frame)
        language_frame.pack(fill=tk.X, pady=(0, 8))

        self.language_label = ttk.Label(language_frame, text=self.t("label_language"))
        self.language_label.pack(side=tk.LEFT, padx=5)
        self.language_var = tk.StringVar(value=UI_LANGUAGES[self.ui_language])
        self.language_combo = ttk.Combobox(
            language_frame,
            textvariable=self.language_var,
            values=list(UI_LANGUAGES.values()),
            state="readonly",
            width=12
        )
        self.language_combo.pack(side=tk.LEFT, padx=5)
        self.language_combo.bind("<<ComboboxSelected>>", self.on_language_changed)

        self.results_title_label = ttk.Label(right_frame, text=self.t("results_title"), font=("맑은 고딕", 12, "bold"))
        self.results_title_label.pack(pady=5)
        
        # 결과 리스트박스
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_listbox = tk.Listbox(list_frame, bg=self.entry_bg, fg=self.fg_color,
                                          font=("Consolas", 10), selectmode=tk.SINGLE,
                                          yscrollcommand=scrollbar.set)
        self.result_listbox.pack(fill=tk.BOTH, expand=True)
        self.result_listbox.bind('<<ListboxSelect>>', self.on_result_select)
        scrollbar.config(command=self.result_listbox.yview)
        
        # 수정 영역
        edit_frame = ttk.Frame(right_frame)
        edit_frame.pack(fill=tk.X, pady=10)
        
        self.plate_label = ttk.Label(edit_frame, text=self.t("label_plate"))
        self.plate_label.pack(side=tk.LEFT, padx=5)
        self.plate_entry = ttk.Entry(edit_frame, font=("맑은 고딕", 12), width=15)
        self.plate_entry.pack(side=tk.LEFT, padx=5)
        self.update_plate_btn = ttk.Button(edit_frame, text=self.t("button_edit"), command=self.update_plate)
        self.update_plate_btn.pack(side=tk.LEFT, padx=5)
        
        # 하단 상태 및 진행률
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, 
                                             maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        status_frame = ttk.Frame(bottom_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text=self.t("status_ready"))
        self.status_label.pack(side=tk.LEFT)
        
        self.save_excel_btn = ttk.Button(status_frame, text=self.t("button_save_excel"), command=self.save_to_excel,
                                         style="Accent.TButton")
        self.save_excel_btn.pack(side=tk.RIGHT, padx=5)
        self.reset_btn = ttk.Button(status_frame, text=self.t("button_reset"), command=self.reset_all)
        self.reset_btn.pack(side=tk.RIGHT, padx=5)
    
    def select_folder(self):
        """폴더 선택"""
        folder = filedialog.askdirectory(title=self.t("folder_dialog_title"))
        if folder:
            self.image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                self.image_files.extend(glob.glob(os.path.join(folder, ext)))
                self.image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
            
            self.image_files.sort()
            self.results = [{"filename": os.path.basename(f), "path": f, "plate": ""} 
                           for f in self.image_files]
            self.current_index = 0
            self.update_result_list()
            self.show_current_image()
            self.set_status("status_images_loaded", count=len(self.image_files))
    
    def select_files(self):
        """파일 선택"""
        files = filedialog.askopenfilenames(
            title=self.t("file_dialog_title"),
            filetypes=[(self.t("image_file_type"), "*.jpg *.jpeg *.png *.bmp"), (self.t("all_files_type"), "*.*")]
        )
        if files:
            self.image_files = list(files)
            self.results = [{"filename": os.path.basename(f), "path": f, "plate": ""} 
                           for f in self.image_files]
            self.current_index = 0
            self.update_result_list()
            self.show_current_image()
            self.set_status("status_images_loaded", count=len(self.image_files))
    
    def show_current_image(self):
        """현재 이미지 표시"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        try:
            img_path = self.image_files[self.current_index]
            img = Image.open(img_path)
            
            # 이미지 크기 조정
            max_size = (450, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            self.nav_label.configure(text=f"{self.current_index + 1} / {len(self.image_files)}")
            
            # 현재 결과의 번호판 표시
            if self.results:
                self.plate_entry.delete(0, tk.END)
                self.plate_entry.insert(0, self.results[self.current_index].get("plate", ""))
        except Exception as e:
            self.image_label.configure(image="", text=self.t("image_load_failed", error=e))
    
    def prev_image(self):
        """이전 이미지"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.result_listbox.selection_clear(0, tk.END)
            self.result_listbox.selection_set(self.current_index)
            self.result_listbox.see(self.current_index)
    
    def next_image(self):
        """다음 이미지"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_current_image()
            self.result_listbox.selection_clear(0, tk.END)
            self.result_listbox.selection_set(self.current_index)
            self.result_listbox.see(self.current_index)
    
    def update_result_list(self):
        """결과 목록 업데이트"""
        self.result_listbox.delete(0, tk.END)
        for i, result in enumerate(self.results):
            plate = result.get("plate", "")
            status = self.t("result_done") if plate else self.t("result_waiting")
            self.result_listbox.insert(tk.END, f"{status} {result['filename']}: {plate}")
    
    def on_result_select(self, event):
        """결과 항목 선택"""
        selection = self.result_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self.show_current_image()
    
    def update_plate(self):
        """번호판 수정"""
        if self.results and self.current_index < len(self.results):
            new_plate = self.plate_entry.get().strip()
            self.results[self.current_index]["plate"] = new_plate
            self.update_result_list()
            self.result_listbox.selection_set(self.current_index)
            self.set_status("status_plate_updated", plate=new_plate)
    
    def start_processing(self):
        """분석 시작"""
        if not OCR_AVAILABLE:
            messagebox.showerror(self.t("error_title"), self.t("ocr_module_unavailable"))
            return
        
        if not self.image_files:
            messagebox.showwarning(self.t("warning_title"), self.t("select_image_first"))
            return
        
        if self.processing:
            return
        
        self.processing = True
        threading.Thread(target=self._process_images, daemon=True).start()
    
    def _process_images(self):
        """이미지 처리 (백그라운드 스레드)"""
        total = len(self.image_files)
        
        for i, img_path in enumerate(self.image_files):
            if not self.processing:
                break
            
            try:
                plate, _ = detect_best_plate(img_path)
                self.results[i]["plate"] = plate if plate else ""
            except Exception as e:
                self.results[i]["plate"] = ""
            
            # UI 업데이트 (메인 스레드에서)
            progress = ((i + 1) / total) * 100
            self.root.after(0, lambda p=progress, idx=i: self._update_progress(p, idx))
        
        self.processing = False
        self.root.after(0, lambda: self.set_status("status_analysis_complete"))
    
    def _update_progress(self, progress, index):
        """진행률 업데이트"""
        self.progress_var.set(progress)
        self.update_result_list()
        self.result_listbox.see(index)
        self.set_status("status_processing", index=index + 1, total=len(self.image_files))
    
    def stop_processing(self):
        """분석 중지"""
        self.processing = False
        self.set_status("status_processing_stopped")
    
    def save_to_excel(self):
        """Excel 저장"""
        if not self.results:
            messagebox.showwarning(self.t("warning_title"), self.t("save_no_data"))
            return
        
        # 유효한 번호판만 필터링
        valid_results = [r for r in self.results if r.get("plate")]
        
        if not valid_results:
            messagebox.showwarning(self.t("warning_title"), self.t("save_no_plate"))
            return
        
        # DataFrame 생성
        entries = []
        db_records = []
        today_value = datetime.now().strftime('%Y-%m-%d')
        ampm_label = self.get_current_ampm_label()
        for r in valid_results:
            entries.append({
                self.t("excel_col_date"): today_value,
                self.t("excel_col_time_period"): ampm_label,
                self.t("excel_col_location"): self.location_var.get(),
                self.t("excel_col_reason"): self.reason_var.get(),
                self.t("excel_col_plate"): r["plate"]
            })
            db_records.append({
                "date": today_value,
                "time_period": ampm_label,
                "location": self.location_var.get(),
                "reason": self.reason_var.get(),
                "plate_number": r["plate"],
                "source_filename": r.get("filename", ""),
                "image_path": r.get("path", ""),
                "mode": "gui",
            })
        
        df = pd.DataFrame(entries)
        
        # 파일 저장 대화상자
        filename = f"{self.t('excel_filename_prefix')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.xlsx"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[(self.t("excel_file_type"), "*.xlsx")],
            initialfile=filename
        )
        
        if filepath:
            try:
                df.to_excel(filepath, index=False)
                for record in db_records:
                    record["excel_file"] = filepath
                try:
                    add_records(DB_PATH, db_records)
                    history_message = "\n" + self.t("sqlite_save_complete")
                except Exception as history_error:
                    history_message = "\n" + self.t("sqlite_save_failed", error=history_error)

                messagebox.showinfo(
                    self.t("success_title"),
                    self.t("save_complete_message", filepath=filepath, count=len(entries), history_message=history_message)
                )
                self.set_status("status_excel_saved", count=len(entries))
            except Exception as e:
                messagebox.showerror(self.t("error_title"), self.t("save_failed", error=e))
    
    def reset_all(self):
        """초기화"""
        self.image_files = []
        self.results = []
        self.current_index = 0
        self.progress_var.set(0)
        self.result_listbox.delete(0, tk.END)
        self.plate_entry.delete(0, tk.END)
        self.image_label.configure(image="", text=self.t("image_select_prompt"))
        self.nav_label.configure(text="0 / 0")
        self.set_status("status_reset")
    
    def update_status(self, text):
        """상태 업데이트"""
        self.current_status_key = None
        self.current_status_kwargs = {}
        self.status_label.configure(text=text)

    def set_status(self, key, **kwargs):
        """번역 가능한 상태 업데이트"""
        self.current_status_key = key
        self.current_status_kwargs = kwargs
        self.status_label.configure(text=self.t(key, **kwargs))
    
    def copy_server_url(self):
        """서버 URL을 클립보드에 복사"""
        if self.server_url:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.server_url)
            self.root.update()  # 클립보드 업데이트 강제
            self.set_status("status_url_copied", url=self.server_url)
            messagebox.showinfo(self.t("copied_title"), self.t("copy_url_message", url=self.server_url))
        else:
            self.set_status("status_server_not_running")

    def get_configured_server_port(self):
        try:
            port = int(self.settings.get("server_port", "5000"))
            if 1 <= port <= 65535:
                return port
        except (TypeError, ValueError):
            pass
        return 5000

    def open_report_page(self):
        """기록 보기"""
        import webbrowser

        if self.server_running and self.server_url:
            webbrowser.open(self.server_url.rstrip("/") + "/report")
            self.set_status("status_open_report_page")
            return

        self.open_history_window()

    def open_history_window(self):
        """서버 없이 SQLite 기록을 직접 표시"""
        dialog = tk.Toplevel(self.root)
        dialog.title(self.t("history_title"))
        dialog.geometry("900x560")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)

        try:
            total_count = fetch_record_count(DB_PATH)
            daily_counts = fetch_daily_counts(DB_PATH, limit=14)
            recent_records = fetch_recent_records(DB_PATH, limit=200)
            load_error = ""
        except Exception as e:
            total_count = 0
            daily_counts = []
            recent_records = []
            load_error = str(e)

        top_frame = ttk.Frame(dialog)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text=self.t("history_total", count=total_count)).pack(side=tk.LEFT, padx=5)
        if load_error:
            ttk.Label(top_frame, text=self.t("history_load_failed", error=load_error)).pack(side=tk.LEFT, padx=5)

        def copy_db_path():
            dialog.clipboard_clear()
            dialog.clipboard_append(DB_PATH)
            dialog.update()
            self.set_status("status_db_path_copied")

        ttk.Button(top_frame, text=self.t("button_copy_db_path"), command=copy_db_path).pack(side=tk.RIGHT, padx=5)
        if self.server_running and self.server_url:
            ttk.Button(
                top_frame,
                text=self.t("button_open_web_report"),
                command=lambda: webbrowser.open(self.server_url.rstrip("/") + "/report")
            ).pack(side=tk.RIGHT, padx=5)

        daily_frame = ttk.LabelFrame(dialog, text=self.t("history_daily_title"), padding=8)
        daily_frame.pack(fill=tk.X, padx=10, pady=5)
        daily_text = ", ".join([f"{row['date']}: {self.t('history_count', count=row['count'])}" for row in daily_counts])
        ttk.Label(daily_frame, text=daily_text if daily_text else self.t("history_empty")).pack(anchor="w")

        table_frame = ttk.Frame(dialog)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("created_at", "date", "time_period", "location", "reason", "plate", "mode", "source")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "created_at": self.t("heading_created_at"),
            "date": self.t("heading_date"),
            "time_period": self.t("heading_time_period"),
            "location": self.t("heading_location"),
            "reason": self.t("heading_reason"),
            "plate": self.t("heading_plate"),
            "mode": self.t("heading_mode"),
            "source": self.t("heading_source"),
        }
        widths = {
            "created_at": 145,
            "date": 95,
            "time_period": 70,
            "location": 90,
            "reason": 160,
            "plate": 110,
            "mode": 60,
            "source": 160,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for record in recent_records:
            tree.insert("", tk.END, values=(
                record.get("created_at", ""),
                record.get("date", ""),
                record.get("time_period", ""),
                record.get("location", ""),
                record.get("reason", ""),
                record.get("plate_number", ""),
                record.get("mode", ""),
                record.get("source_filename", ""),
            ))

        ttk.Button(dialog, text=self.t("button_close"), command=dialog.destroy).pack(pady=10)
        self.set_status("status_history_displayed")
    
    def open_settings_dialog(self):
        """설정 다이얼로그 열기"""
        dialog = tk.Toplevel(self.root)
        dialog.title(self.t("settings_title"))
        dialog.geometry("550x500")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(dialog, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 설정 입력 필드들 저장
        entries = {}
        
        # === 링크 고정 설정 섹션 ===
        section1 = ttk.LabelFrame(scrollable_frame, text=self.t("settings_section_tunnel"), padding=10)
        section1.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(section1, text="Tunnel Token:").grid(row=0, column=0, sticky="w", pady=5)
        entries["cloudflare_tunnel_token"] = ttk.Entry(section1, width=50)
        entries["cloudflare_tunnel_token"].grid(row=0, column=1, padx=5, pady=5)
        entries["cloudflare_tunnel_token"].insert(0, self.settings.get("cloudflare_tunnel_token", ""))
        
        ttk.Label(section1, text=self.t("settings_tunnel_domain")).grid(row=1, column=0, sticky="w", pady=5)
        entries["cloudflare_tunnel_domain"] = ttk.Entry(section1, width=50)
        entries["cloudflare_tunnel_domain"].grid(row=1, column=1, padx=5, pady=5)
        entries["cloudflare_tunnel_domain"].insert(0, self.settings.get("cloudflare_tunnel_domain", ""))
        
        ttk.Label(section1, text=self.t("settings_tunnel_help"),
                 foreground="#888888").grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        # === Discord 알림 설정 ===
        section2 = ttk.LabelFrame(scrollable_frame, text=self.t("settings_section_discord"), padding=10)
        section2.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(section2, text="Webhook URL:").grid(row=0, column=0, sticky="w", pady=5)
        entries["discord_webhook_url"] = ttk.Entry(section2, width=50)
        entries["discord_webhook_url"].grid(row=0, column=1, padx=5, pady=5)
        entries["discord_webhook_url"].insert(0, self.settings.get("discord_webhook_url", ""))
        
        ttk.Label(section2, text=self.t("settings_discord_help"),
                 foreground="#888888").grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        # === 서버 설정 ===
        section_server = ttk.LabelFrame(scrollable_frame, text=self.t("settings_section_server"), padding=10)
        section_server.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(section_server, text=self.t("settings_local_port")).grid(row=0, column=0, sticky="w", pady=5)
        entries["server_port"] = ttk.Entry(section_server, width=15)
        entries["server_port"].grid(row=0, column=1, sticky="w", padx=5, pady=5)
        entries["server_port"].insert(0, self.settings.get("server_port", "5000"))

        ttk.Label(section_server, text=self.t("settings_port_help"),
                 foreground="#888888").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        # === 폴더 경로 설정 ===
        section3 = ttk.LabelFrame(scrollable_frame, text=self.t("settings_section_folders"), padding=10)
        section3.pack(fill=tk.X, padx=10, pady=10)
        
        def browse_folder(key, entry_widget):
            folder = filedialog.askdirectory(title=self.t("folder_dialog_title"))
            if folder:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, folder)

        def browse_model_file(entry_widget):
            filepath = filedialog.askopenfilename(
                title=self.t("model_dialog_title"),
                filetypes=[(self.t("model_file_type"), "*.pt"), (self.t("all_files_type"), "*.*")]
            )
            if filepath:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, filepath)
        
        # 입력 폴더
        ttk.Label(section3, text=self.t("settings_input_folder")).grid(row=0, column=0, sticky="w", pady=5)
        entries["input_folder"] = ttk.Entry(section3, width=40)
        entries["input_folder"].grid(row=0, column=1, padx=5, pady=5)
        entries["input_folder"].insert(0, self.settings.get("input_folder", ""))
        ttk.Button(section3, text=self.t("button_browse"), 
                  command=lambda: browse_folder("input_folder", entries["input_folder"])).grid(row=0, column=2, padx=5)
        
        # 백업 폴더 (출력)
        ttk.Label(section3, text=self.t("settings_backup_folder")).grid(row=1, column=0, sticky="w", pady=5)
        entries["output_folder"] = ttk.Entry(section3, width=40)
        entries["output_folder"].grid(row=1, column=1, padx=5, pady=5)
        entries["output_folder"].insert(0, self.settings.get("output_folder", ""))
        ttk.Button(section3, text=self.t("button_browse"),
                  command=lambda: browse_folder("output_folder", entries["output_folder"])).grid(row=1, column=2, padx=5)
        
        # Excel 저장 폴더
        ttk.Label(section3, text=self.t("settings_excel_save")).grid(row=2, column=0, sticky="w", pady=5)
        entries["excel_save_folder"] = ttk.Entry(section3, width=40)
        entries["excel_save_folder"].grid(row=2, column=1, padx=5, pady=5)
        entries["excel_save_folder"].insert(0, self.settings.get("excel_save_folder", ""))
        ttk.Button(section3, text=self.t("button_browse"),
                  command=lambda: browse_folder("excel_save_folder", entries["excel_save_folder"])).grid(row=2, column=2, padx=5)
        
        ttk.Label(section3, text=self.t("settings_folder_help"),
                 foreground="#888888").grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

        # === YOLO 모델 설정 ===
        section4 = ttk.LabelFrame(scrollable_frame, text=self.t("settings_section_yolo"), padding=10)
        section4.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(section4, text=self.t("settings_model_file")).grid(row=0, column=0, sticky="w", pady=5)
        entries["yolo_model_path"] = ttk.Entry(section4, width=40)
        entries["yolo_model_path"].grid(row=0, column=1, padx=5, pady=5)
        entries["yolo_model_path"].insert(0, self.settings.get("yolo_model_path", ""))
        ttk.Button(section4, text=self.t("button_browse"),
                  command=lambda: browse_model_file(entries["yolo_model_path"])).grid(row=0, column=2, padx=5)

        ttk.Label(section4, text=self.t("settings_model_help"),
                 foreground="#888888").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        # === 버튼 영역 ===
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        def save_settings():
            """설정 저장"""
            port_value = entries["server_port"].get().strip()
            try:
                port_int = int(port_value)
                if not 1 <= port_int <= 65535:
                    raise ValueError
            except ValueError:
                messagebox.showerror(self.t("error_title"), self.t("port_invalid"))
                return

            for key, entry in entries.items():
                self.settings.set(key, entry.get().strip())
            
            if self.settings.save():
                messagebox.showinfo(self.t("success_title"), self.t("settings_saved_message"))
                dialog.destroy()
            else:
                messagebox.showerror(self.t("error_title"), self.t("settings_save_failed"))
        
        def reset_settings():
            """설정 초기화"""
            if messagebox.askyesno(self.t("confirm_title"), self.t("settings_reset_confirm")):
                self.settings.reset()
                dialog.destroy()
                self.open_settings_dialog()  # 다이얼로그 다시 열기
        
        ttk.Button(button_frame, text=self.t("button_save"), command=save_settings,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=self.t("button_reset"), command=reset_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text=self.t("button_cancel"), command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 스크롤 레이아웃
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def toggle_server(self):
        """웹 서버 시작/중지 토글"""
        if self.server_running:
            # 서버 중지
            self.stop_background_server()
        else:
            # 서버 시작
            self.start_background_server()
    
    def stop_background_server(self):
        """백그라운드 웹 서버 중지"""
        import subprocess

        if self.server_stop_event:
            self.server_stop_event.set()
        
        # Cloudflare 프로세스 종료
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", "cloudflared.exe"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except:
            pass

        with self.server_lock:
            server = self.http_server
            self.http_server = None

        if server:
            self._close_waitress_server(server)

        if self.server_thread and self.server_thread.is_alive() and self.server_thread is not threading.current_thread():
            self.server_thread.join(timeout=2)
        
        self.server_running = False
        self.server_thread = None
        self.server_btn.configure(text=self.t("button_server_start"))
        self.copy_url_btn.configure(state=tk.DISABLED)
        self.server_status_label.configure(text="")
        self.server_url = None
        self.set_status("status_server_stopped")

    def _close_waitress_server(self, server):
        """Waitress 서버와 작업 스레드를 실제로 종료"""
        try:
            for channel in list(getattr(server, "active_channels", {}).values()):
                try:
                    channel.close()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            server.close()
        except Exception:
            pass

        dispatcher = getattr(server, "task_dispatcher", None)
        if dispatcher:
            try:
                dispatcher.shutdown(timeout=2)
            except Exception:
                pass

    def start_background_server(self):
        """백그라운드 웹 서버 시작"""
        import socket
        import subprocess

        if self.server_running:
            return
        
        # 기존 cloudflared 프로세스 정리
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", "cloudflared.exe"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        except:
            pass
        
        # 포트 사용 확인
        port = self.get_configured_server_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            messagebox.showerror(self.t("error_title"), self.t("port_in_use", port=port))
            return
        finally:
            sock.close()

        stop_event = threading.Event()
        self.server_stop_event = stop_event
        
        def run_server():
            http_server = None
            try:
                from ocr import app, init_cloudflare_tunnel, send_discord_webhook
                from waitress import create_server

                http_server = create_server(
                    app, host='0.0.0.0', port=port,
                    threads=10, channel_timeout=3000
                )

                with self.server_lock:
                    self.http_server = http_server
                
                # Cloudflare Tunnel 시도
                public_url = init_cloudflare_tunnel(port)

                if stop_event.is_set():
                    return
                
                if public_url:
                    self.server_url = public_url
                    self.root.after(0, lambda: self.server_status_label.configure(
                        text=f"{public_url[:30]}..."))
                    # Discord 알림
                    try:
                        send_discord_webhook(public_url, port)
                    except:
                        pass
                else:
                    self.server_url = f"http://127.0.0.1:{port}"
                    self.root.after(0, lambda: self.server_status_label.configure(
                        text=self.t("server_status_local_only")))

                if stop_event.is_set():
                    return

                http_server.run()
            except Exception as e:
                if not stop_event.is_set():
                    self.root.after(0, lambda: messagebox.showerror(self.t("server_error_title"), str(e)))
                    self.root.after(0, lambda: self.stop_background_server())
            finally:
                if http_server:
                    self._close_waitress_server(http_server)
                with self.server_lock:
                    if self.http_server is http_server:
                        self.http_server = None
                if self.server_stop_event is stop_event:
                    self.server_stop_event = None
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.server_running = True
        self.server_btn.configure(text=self.t("button_server_stop"))
        self.copy_url_btn.configure(state=tk.NORMAL)
        self.server_url = f"http://127.0.0.1:{port}"
        self.set_status("status_server_starting")

# 전역 락 파일 핸들
_lock_file_handle = None
_lock_file_path = None

def acquire_lock():
    """락 파일을 획득하여 중복 실행 방지"""
    global _lock_file_handle, _lock_file_path
    
    import tempfile
    import msvcrt
    
    # 락 파일 경로 (사용자 temp 디렉토리)
    _lock_file_path = os.path.join(tempfile.gettempdir(), "parking_enforcement_gui.lock")
    
    try:
        # 락 파일 열기 또는 생성
        _lock_file_handle = open(_lock_file_path, 'w')
        # 독점적 락 시도 (비차단)
        msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        # 현재 PID 기록
        _lock_file_handle.write(str(os.getpid()))
        _lock_file_handle.flush()
        return True
    except (IOError, OSError):
        # 이미 다른 인스턴스가 실행 중
        if _lock_file_handle:
            try:
                _lock_file_handle.close()
            except:
                pass
            _lock_file_handle = None
        return False
    except Exception as e:
        print(f"락 획득 중 오류: {e}")
        return False

def release_lock():
    """락 파일 해제"""
    global _lock_file_handle, _lock_file_path
    
    import msvcrt
    
    if _lock_file_handle:
        try:
            # 파일 디스크립터 유효성 확인 후 언락
            fileno = _lock_file_handle.fileno()
            if fileno >= 0:
                msvcrt.locking(fileno, msvcrt.LK_UNLCK, 1)
            _lock_file_handle.close()
        except:
            pass
        _lock_file_handle = None
    
    if _lock_file_path and os.path.exists(_lock_file_path):
        try:
            os.remove(_lock_file_path)
        except:
            pass

def bring_existing_window_to_front():
    """기존에 실행 중인 프로그램 창을 앞으로 가져오기 시도"""
    try:
        import ctypes
        from ctypes import wintypes
        
        # Windows API 함수 정의
        user32 = ctypes.windll.user32
        
        # 창 제목으로 찾기
        for title in APP_WINDOW_TITLES:
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                # 창 복원 (최소화된 경우)
                SW_RESTORE = 9
                user32.ShowWindow(hwnd, SW_RESTORE)
                # 창을 맨 앞으로
                user32.SetForegroundWindow(hwnd)
                return True
    except Exception:
        pass
    return False

def main(start_server=False):
    """GUI 애플리케이션 실행"""
    ui_language = normalize_ui_language(get_settings().get("ui_language", "ko"))

    # PIL 설치 확인
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print(translate_ui(ui_language, "pillow_missing"))
        sys.exit(1)
    
    # 중복 실행 방지
    if not acquire_lock():
        # 기존 창을 앞으로 가져오기 시도
        if bring_existing_window_to_front():
            print(translate_ui(ui_language, "lock_activated"))
        else:
            # 창을 찾지 못한 경우 메시지 표시
            # Tk 인스턴스 생성하여 메시지박스 표시
            temp_root = tk.Tk()
            temp_root.withdraw()  # 임시 창 숨기기
            messagebox.showwarning(translate_ui(ui_language, "warning_title"), translate_ui(ui_language, "lock_warning"))
            temp_root.destroy()
        sys.exit(0)
    
    # 종료 시 락 해제 등록
    atexit.register(release_lock)
    
    try:
        root = tk.Tk()
        app = ParkingEnforcementGUI(root, start_server=start_server)
        root.mainloop()
    except Exception as e:
        print(f"프로그램 오류: {e}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
