# -*- coding: utf-8 -*-
"""
주차 단속 시스템 - 로컬 GUI 모드
Tkinter 기반 데스크톱 애플리케이션
"""
# OCR 작업 흐름을 위한 Tkinter 데스크톱 화면이다.
#
# 이 GUI는 ocr.py와 같은 OCR, SQLite 이력, 선택적 웹 서버 코드를 재사용한다.
# 대부분의 메서드는 이미지 선택, 미리보기 표시, 작업 스레드 OCR 실행,
# 결과 저장, 웹 서버 시작/중지 같은 화면 흐름을 조율한다.

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
    # ocr.py를 import하면 네이티브 OCR DLL과 YOLO 모델이 한 번 초기화된다.
    from ocr import (
        detect_best_plate,
        LOCATIONS, REASONS, BASE_DIR, BACKUP_DIR, DB_PATH
    )
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"OCR 모듈 로드 실패: {e}")
    OCR_AVAILABLE = False


class ParkingEnforcementGUI:
    """주차 단속 GUI 애플리케이션"""
    
    def __init__(self, root, start_server=False):
        # 루트 창의 소유권은 main()에 두고, 이 클래스는 구성만 담당한다.
        self.root = root
        self.root.title("주차 단속 시스템 (로컬 모드)")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)
        
        # 설정 관리자 초기화 및 로드
        self.settings = get_settings()
        
        # Tkinter 위젯은 메인 스레드에서 만들고 갱신해야 한다.
        # 다크 테마 설정
        self.setup_theme()
        
        # 데이터 저장소
        # image_files와 results는 탐색 중 같은 인덱스를 기준으로 맞춰진다.
        self.image_files = []
        self.results = []
        self.current_index = 0
        self.processing = False
        
        # 서버 관련
        # waitress 서버 객체가 UI 스레드와 서버 스레드 사이를 오갈 때는
        # server_lock으로 보호한다.
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
            # 초기 UI가 보인 뒤 서버를 시작하도록 약간 지연한다.
            self.root.after(500, self.toggle_server)
        
        # 상태 표시
        self.update_status("준비됨" if OCR_AVAILABLE else "OCR 모듈 로드 실패")
    
    def apply_loaded_settings(self):
        """저장된 설정을 UI에 적용"""
        # 현재 콤보박스 옵션에 남아 있는 값만 복원한다.
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
            self.ampm_var.set(last_ampm)
    
    def save_current_settings(self):
        """현재 UI 상태를 설정에 저장"""
        # 반복 사용 시 마지막 위치/사유/시간대에서 시작하도록 작은 UI 상태를 저장한다.
        self.settings.set("last_location", self.location_var.get())
        self.settings.set("last_reason", self.reason_var.get())
        self.settings.set("last_ampm", self.ampm_var.get())
        self.settings.save()
    
    def on_closing(self):
        """프로그램 종료 시 정리"""
        import subprocess
        
        # 보조 프로세스를 종료하기 전에 설정을 저장해 사용자의 마지막 상태를 보존한다.
        # 현재 설정 저장
        self.save_current_settings()
        
        # 서버 중지
        if self.server_running:
            self.stop_background_server()
        
        # Cloudflare 프로세스 종료
        try:
            # cloudflared는 백그라운드 서버 스레드가 시작했을 수 있다.
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
        # 테마 변경 시 모든 위젯 선언을 고치지 않도록 색상은 한곳에 모아둔다.
        self.bg_color = "#202020"
        self.sidebar_bg = "#1c1c1c"
        self.panel_bg = "#272727"
        self.panel_alt_bg = "#2b2b2b"
        self.preview_bg = "#181818"
        self.border_color = "#3a3a3a"
        self.fg_color = "#f5f5f5"
        self.muted_fg = "#b9b9b9"
        self.accent_color = "#67b7ff"
        self.accent_hover = "#89c8ff"
        self.entry_bg = "#303030"
        self.button_bg = "#323232"
        self.button_hover = "#3f3f3f"
        self.success_color = "#44c060"
        
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background=self.bg_color, foreground=self.fg_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("Panel.TFrame", background=self.panel_bg, relief=tk.FLAT)
        style.configure("Sidebar.TFrame", background=self.sidebar_bg)
        style.configure("Card.TLabelframe", background=self.panel_bg, bordercolor=self.border_color, relief=tk.SOLID)
        style.configure("Card.TLabelframe.Label", background=self.panel_bg, foreground=self.fg_color, font=("맑은 고딕", 11, "bold"))
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=("맑은 고딕", 10))
        style.configure("Muted.TLabel", background=self.bg_color, foreground=self.muted_fg, font=("맑은 고딕", 9))
        style.configure("Panel.TLabel", background=self.panel_bg, foreground=self.fg_color, font=("맑은 고딕", 10))
        style.configure("PanelMuted.TLabel", background=self.panel_bg, foreground=self.muted_fg, font=("맑은 고딕", 9))
        style.configure("Title.TLabel", background=self.bg_color, foreground=self.fg_color, font=("맑은 고딕", 24, "bold"))
        style.configure("Subtitle.TLabel", background=self.bg_color, foreground=self.muted_fg, font=("맑은 고딕", 10))
        style.configure("SidebarTitle.TLabel", background=self.sidebar_bg, foreground=self.fg_color, font=("맑은 고딕", 15, "bold"))
        style.configure("SidebarMuted.TLabel", background=self.sidebar_bg, foreground=self.muted_fg, font=("맑은 고딕", 9))
        style.configure("TButton", background=self.button_bg, foreground=self.fg_color, font=("맑은 고딕", 10), padding=(12, 8), borderwidth=0)
        style.map("TButton", background=[("active", self.button_hover), ("pressed", self.panel_alt_bg)])
        style.configure("Accent.TButton", background=self.accent_color, foreground="#111111", font=("맑은 고딕", 10, "bold"), padding=(12, 8), borderwidth=0)
        style.map("Accent.TButton", background=[("active", self.accent_hover), ("pressed", self.accent_color)])
        style.configure("Success.TButton", background=self.success_color, foreground=self.fg_color)
        style.configure("Sidebar.TButton", background=self.sidebar_bg, foreground=self.fg_color, font=("맑은 고딕", 10), padding=(14, 10), borderwidth=0, anchor="w")
        style.map("Sidebar.TButton", background=[("active", self.button_hover), ("pressed", self.panel_alt_bg)])
        style.configure("TEntry", fieldbackground=self.entry_bg, foreground=self.fg_color, insertcolor=self.fg_color, bordercolor=self.border_color, lightcolor=self.border_color, darkcolor=self.border_color)
        style.configure("TCombobox", fieldbackground=self.entry_bg, background=self.entry_bg, foreground=self.fg_color, arrowcolor=self.fg_color, bordercolor=self.border_color)
        style.configure("Horizontal.TProgressbar", background=self.accent_color, troughcolor=self.entry_bg, bordercolor=self.entry_bg, lightcolor=self.accent_color, darkcolor=self.accent_color)
    
    def create_widgets(self):
        """UI 위젯 생성"""
        # Starward/Fluent 계열처럼 왼쪽 네비게이션과 넓은 작업 영역으로 나눈다.
        shell = ttk.Frame(self.root)
        shell.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="OCR 단속", style="SidebarTitle.TLabel").pack(anchor="w", padx=20, pady=(22, 2))
        ttk.Label(sidebar, text="로컬 작업 공간", style="SidebarMuted.TLabel").pack(anchor="w", padx=20, pady=(0, 20))

        ttk.Button(sidebar, text="폴더 선택", command=self.select_folder, style="Sidebar.TButton").pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Button(sidebar, text="파일 선택", command=self.select_files, style="Sidebar.TButton").pack(fill=tk.X, padx=14, pady=6)
        ttk.Button(sidebar, text="분석 시작", command=self.start_processing, style="Accent.TButton").pack(fill=tk.X, padx=14, pady=(18, 6))
        ttk.Button(sidebar, text="분석 중지", command=self.stop_processing, style="Sidebar.TButton").pack(fill=tk.X, padx=14, pady=6)
        ttk.Button(sidebar, text="기록 보기", command=self.open_report_page, style="Sidebar.TButton").pack(fill=tk.X, padx=14, pady=(18, 6))
        ttk.Button(sidebar, text="설정", command=self.open_settings_dialog, style="Sidebar.TButton").pack(fill=tk.X, padx=14, pady=6)

        server_panel = ttk.Frame(sidebar, style="Sidebar.TFrame")
        server_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=18)
        self.server_status_label = ttk.Label(server_panel, text="", style="SidebarMuted.TLabel")
        self.server_status_label.pack(anchor="w", pady=(0, 8))
        self.copy_url_btn = ttk.Button(server_panel, text="주소 복사", command=self.copy_server_url,
                                       state=tk.DISABLED, style="Sidebar.TButton")
        self.copy_url_btn.pack(fill=tk.X, pady=(0, 6))
        self.server_btn = ttk.Button(server_panel, text="서버 시작", command=self.toggle_server,
                                     style="Sidebar.TButton")
        self.server_btn.pack(fill=tk.X)

        content = ttk.Frame(shell)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=24, pady=20)

        header = ttk.Frame(content)
        header.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(header, text="주차 단속 OCR", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="로컬 이미지 분석과 기록 저장", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        
        # 단속 정보 입력 영역
        settings_frame = ttk.LabelFrame(content, text="단속 정보", style="Card.TLabelframe", padding=(14, 10))
        settings_frame.pack(fill=tk.X, pady=(0, 14))
        
        ttk.Label(settings_frame, text="위치", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.location_var = tk.StringVar(value=LOCATIONS[0] if OCR_AVAILABLE else "")
        location_combo = ttk.Combobox(settings_frame, textvariable=self.location_var, 
                                       values=LOCATIONS if OCR_AVAILABLE else [], width=18)
        location_combo.grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=4)
        
        ttk.Label(settings_frame, text="사유", style="Panel.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=4)
        self.reason_var = tk.StringVar(value=REASONS[0] if OCR_AVAILABLE else "")
        reason_combo = ttk.Combobox(settings_frame, textvariable=self.reason_var,
                                     values=REASONS if OCR_AVAILABLE else [], width=30)
        reason_combo.grid(row=0, column=3, sticky="ew", padx=(0, 18), pady=4)
        
        ttk.Label(settings_frame, text="시간대", style="Panel.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 8), pady=4)
        self.ampm_var = tk.StringVar(value="오전" if datetime.now().hour < 12 else "오후")
        ttk.Combobox(settings_frame, textvariable=self.ampm_var, 
                     values=["오전", "오후"], width=8).grid(row=0, column=5, sticky="ew", pady=4)
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=2)
        
        # 메인 콘텐츠 영역
        main_frame = ttk.Frame(content)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 왼쪽: 이미지 미리보기
        left_frame = ttk.LabelFrame(main_frame, text="이미지 미리보기", style="Card.TLabelframe", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))
        
        self.image_label = tk.Label(left_frame, bg=self.preview_bg, text="이미지를 선택하세요",
                                     fg=self.muted_fg, font=("맑은 고딕", 13))
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 10))
        
        # 이미지 네비게이션
        nav_frame = ttk.Frame(left_frame, style="Panel.TFrame")
        nav_frame.pack(fill=tk.X)
        ttk.Button(nav_frame, text="이전", command=self.prev_image).pack(side=tk.LEFT)
        self.nav_label = ttk.Label(nav_frame, text="0 / 0", style="PanelMuted.TLabel")
        self.nav_label.pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="다음", command=self.next_image).pack(side=tk.RIGHT)
        
        # 오른쪽: 결과 목록
        right_frame = ttk.LabelFrame(main_frame, text="인식 결과", style="Card.TLabelframe", padding=10, width=360)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_frame.pack_propagate(False)
        
        # 결과 리스트박스
        list_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_listbox = tk.Listbox(list_frame, bg=self.preview_bg, fg=self.fg_color,
                                          selectbackground=self.accent_color, selectforeground="#111111",
                                          highlightthickness=1, highlightbackground=self.border_color,
                                          relief=tk.FLAT, borderwidth=0, activestyle="none",
                                          font=("Consolas", 10), selectmode=tk.SINGLE,
                                          yscrollcommand=scrollbar.set)
        self.result_listbox.pack(fill=tk.BOTH, expand=True)
        self.result_listbox.bind('<<ListboxSelect>>', self.on_result_select)
        scrollbar.config(command=self.result_listbox.yview)
        
        # 수정 영역
        edit_frame = ttk.Frame(right_frame, style="Panel.TFrame")
        edit_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(edit_frame, text="번호판", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.plate_entry = ttk.Entry(edit_frame, font=("맑은 고딕", 12), width=15)
        self.plate_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(edit_frame, text="수정", command=self.update_plate).pack(side=tk.LEFT)
        
        # 하단 상태 및 진행률
        bottom_frame = ttk.LabelFrame(content, text="작업 상태", style="Card.TLabelframe", padding=(12, 8))
        bottom_frame.pack(fill=tk.X, pady=(14, 0))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, 
                                             maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))
        
        status_frame = ttk.Frame(bottom_frame, style="Panel.TFrame")
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="준비됨", style="PanelMuted.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        ttk.Button(status_frame, text="Excel 저장", command=self.save_to_excel,
                   style="Accent.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(status_frame, text="초기화", command=self.reset_all).pack(side=tk.RIGHT, padx=5)
    
    def select_folder(self):
        """폴더 선택"""
        folder = filedialog.askdirectory(title="이미지 폴더 선택")
        if folder:
            # 새 폴더 기준으로 상태를 다시 만들어 이전 결과가 남지 않게 한다.
            self.image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                # Windows에서는 대소문자 확장자가 섞일 수 있어 둘 다 포함한다.
                self.image_files.extend(glob.glob(os.path.join(folder, ext)))
                self.image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
            
            self.image_files.sort()
            self.results = [{"filename": os.path.basename(f), "path": f, "plate": ""} 
                           for f in self.image_files]
            self.current_index = 0
            self.update_result_list()
            self.show_current_image()
            self.update_status(f"{len(self.image_files)}개 이미지 로드됨")
    
    def select_files(self):
        """파일 선택"""
        files = filedialog.askopenfilenames(
            title="이미지 파일 선택",
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")]
        )
        if files:
            # 파일 선택 창에서 사용자가 고른 순서를 유지한다.
            self.image_files = list(files)
            self.results = [{"filename": os.path.basename(f), "path": f, "plate": ""} 
                           for f in self.image_files]
            self.current_index = 0
            self.update_result_list()
            self.show_current_image()
            self.update_status(f"{len(self.image_files)}개 이미지 로드됨")
    
    def show_current_image(self):
        """현재 이미지 표시"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        try:
            img_path = self.image_files[self.current_index]
            img = Image.open(img_path)
            
            # 미리보기만 줄이고, OCR 파이프라인은 원본 이미지를 읽는다.
            # 이미지 크기 조정
            max_size = (640, 460)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo, text="")
            # Tkinter는 Python 참조가 없으면 이미지를 해제하므로 참조를 보관한다.
            self.image_label.image = photo
            
            self.nav_label.configure(text=f"{self.current_index + 1} / {len(self.image_files)}")
            
            # 현재 결과의 번호판 표시
            if self.results:
                self.plate_entry.delete(0, tk.END)
                self.plate_entry.insert(0, self.results[self.current_index].get("plate", ""))
        except Exception as e:
            self.image_label.configure(image="", text=f"이미지 로드 실패: {e}")
    
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
        # UI 상태가 예측 가능하도록 self.results에서 목록을 다시 그린다.
        self.result_listbox.delete(0, tk.END)
        for i, result in enumerate(self.results):
            plate = result.get("plate", "")
            status = "완료" if plate else "대기"
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
            # 수동 수정값은 Excel 저장에 쓰는 같은 구조에 반영한다.
            new_plate = self.plate_entry.get().strip()
            self.results[self.current_index]["plate"] = new_plate
            self.update_result_list()
            self.result_listbox.selection_set(self.current_index)
            self.update_status(f"번호판 수정됨: {new_plate}")
    
    def start_processing(self):
        """분석 시작"""
        if not OCR_AVAILABLE:
            messagebox.showerror("오류", "OCR 모듈이 로드되지 않았습니다.")
            return
        
        if not self.image_files:
            messagebox.showwarning("경고", "먼저 이미지를 선택해주세요.")
            return
        
        if self.processing:
            return
        
        self.processing = True
        # OCR은 몇 초 동안 막힐 수 있으므로 Tkinter 이벤트 루프 밖에서 실행한다.
        threading.Thread(target=self._process_images, daemon=True).start()
    
    def _process_images(self):
        """이미지 처리 (백그라운드 스레드)"""
        total = len(self.image_files)
        
        for i, img_path in enumerate(self.image_files):
            if not self.processing:
                break
            
            try:
                # detect_best_plate는 ocr.py의 추론 워커 안에서 순차 실행된다.
                plate, _ = detect_best_plate(img_path)
                self.results[i]["plate"] = plate if plate else ""
            except Exception as e:
                self.results[i]["plate"] = ""
            
            # UI 업데이트 (메인 스레드에서)
            progress = ((i + 1) / total) * 100
            # 작업 스레드에서는 root.after를 통해서만 Tkinter 위젯을 갱신한다.
            self.root.after(0, lambda p=progress, idx=i: self._update_progress(p, idx))
        
        self.processing = False
        self.root.after(0, lambda: self.update_status("분석 완료!"))
    
    def _update_progress(self, progress, index):
        """진행률 업데이트"""
        self.progress_var.set(progress)
        self.update_result_list()
        self.result_listbox.see(index)
        self.update_status(f"분석 중... {index + 1}/{len(self.image_files)}")
    
    def stop_processing(self):
        """분석 중지"""
        self.processing = False
        self.update_status("분석 중지됨")
    
    def save_to_excel(self):
        """Excel 저장"""
        if not self.results:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다.")
            return
        
        # 유효한 번호판만 필터링
        # 빈 OCR 결과는 저장하지 않는다. 포함해야 하는 값은 사용자가 저장 전에 수정할 수 있다.
        valid_results = [r for r in self.results if r.get("plate")]
        
        if not valid_results:
            messagebox.showwarning("경고", "인식된 번호판이 없습니다.")
            return
        
        # DataFrame 생성
        entries = []
        db_records = []
        today_value = datetime.now().strftime('%Y-%m-%d')
        for r in valid_results:
            # Excel 행과 SQLite 행을 같은 원본 결과에서 함께 만든다.
            entries.append({
                "날짜": today_value,
                "시간대": self.ampm_var.get(),
                "단속위치": self.location_var.get(),
                "사유": self.reason_var.get(),
                "차량번호": r["plate"]
            })
            db_records.append({
                "date": today_value,
                "time_period": self.ampm_var.get(),
                "location": self.location_var.get(),
                "reason": self.reason_var.get(),
                "plate_number": r["plate"],
                "source_filename": r.get("filename", ""),
                "image_path": r.get("path", ""),
                "mode": "gui",
            })
        
        df = pd.DataFrame(entries)
        
        # 파일 저장 대화상자
        filename = f"주차단속내역_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.xlsx"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 파일", "*.xlsx")],
            initialfile=filename
        )
        
        if filepath:
            try:
                df.to_excel(filepath, index=False)
                for record in db_records:
                    # GUI에서 선택한 정확한 Excel 경로를 이력 행에 저장한다.
                    record["excel_file"] = filepath
                try:
                    # SQLite에 일시 오류가 있어도 Excel 저장은 성공할 수 있게 이력 저장은 부가 처리한다.
                    add_records(DB_PATH, db_records)
                    history_message = "\nSQLite 기록 저장 완료"
                except Exception as history_error:
                    history_message = f"\nSQLite 기록 저장 실패: {history_error}"

                messagebox.showinfo("성공", f"저장 완료: {filepath}\n총 {len(entries)}건{history_message}")
                self.update_status(f"Excel 저장 완료: {len(entries)}건")
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {e}")
    
    def reset_all(self):
        """초기화"""
        # 새 세션 상태로 돌아가도록 모델 데이터와 표시 위젯을 모두 비운다.
        self.image_files = []
        self.results = []
        self.current_index = 0
        self.progress_var.set(0)
        self.result_listbox.delete(0, tk.END)
        self.plate_entry.delete(0, tk.END)
        self.image_label.configure(image="", text="이미지를 선택하세요")
        self.nav_label.configure(text="0 / 0")
        self.update_status("초기화됨")
    
    def update_status(self, text):
        """상태 업데이트"""
        self.status_label.configure(text=text)
    
    def copy_server_url(self):
        """서버 URL을 클립보드에 복사"""
        if self.server_url:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.server_url)
            self.root.update()  # 클립보드 업데이트 강제
            self.update_status(f"주소 복사됨: {self.server_url}")
            messagebox.showinfo("복사 완료", f"서버 주소가 복사되었습니다:\n{self.server_url}")
        else:
            self.update_status("서버가 실행되지 않았습니다")

    def get_configured_server_port(self):
        """유효한 설정 포트 또는 Flask 기본 포트를 반환한다."""
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

        # 웹 서버가 실행 중이면 더 풍부한 웹 보고서 페이지를 우선 연다.
        if self.server_running and self.server_url:
            webbrowser.open(self.server_url.rstrip("/") + "/report")
            self.update_status("기록 페이지 열기")
            return

        self.open_history_window()

    def open_history_window(self):
        """서버 없이 SQLite 기록을 직접 표시"""
        dialog = tk.Toplevel(self.root)
        dialog.title("과거 기록")
        dialog.geometry("900x560")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)

        try:
            # 테이블 생성 전에 요약 데이터를 먼저 조회해 오류를 상단에 한 번만 표시한다.
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

        ttk.Label(top_frame, text=f"누적 기록: {total_count}건").pack(side=tk.LEFT, padx=5)
        if load_error:
            ttk.Label(top_frame, text=f"로드 실패: {load_error}").pack(side=tk.LEFT, padx=5)

        def copy_db_path():
            # 사용자가 백업이나 수동 확인을 위해 DB 위치를 찾을 수 있게 한다.
            dialog.clipboard_clear()
            dialog.clipboard_append(DB_PATH)
            dialog.update()
            self.update_status("SQLite DB 경로 복사됨")

        ttk.Button(top_frame, text="DB 경로 복사", command=copy_db_path).pack(side=tk.RIGHT, padx=5)
        if self.server_running and self.server_url:
            ttk.Button(
                top_frame,
                text="웹 기록 열기",
                command=lambda: webbrowser.open(self.server_url.rstrip("/") + "/report")
            ).pack(side=tk.RIGHT, padx=5)

        daily_frame = ttk.LabelFrame(dialog, text="최근 일자별 기록", padding=8)
        daily_frame.pack(fill=tk.X, padx=10, pady=5)
        daily_text = ", ".join([f"{row['date']}: {row['count']}건" for row in daily_counts])
        ttk.Label(daily_frame, text=daily_text if daily_text else "저장된 기록이 없습니다.").pack(anchor="w")

        table_frame = ttk.Frame(dialog)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("created_at", "date", "time_period", "location", "reason", "plate", "mode", "source")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "created_at": "저장시각",
            "date": "날짜",
            "time_period": "시간대",
            "location": "위치",
            "reason": "사유",
            "plate": "차량번호",
            "mode": "모드",
            "source": "원본",
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
            # 작은 화면에서도 이력 창이 읽기 쉽도록 열 너비를 고정한다.
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for record in recent_records:
            # 선택 컬럼이 없는 오래된 DB 행도 표시되도록 .get()을 사용한다.
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

        ttk.Button(dialog, text="닫기", command=dialog.destroy).pack(pady=10)
        self.update_status("과거 기록 표시")
    
    def open_settings_dialog(self):
        """설정 다이얼로그 열기"""
        dialog = tk.Toplevel(self.root)
        dialog.title("설정")
        dialog.geometry("550x500")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 스크롤 가능한 설정 패널을 써서 작은 화면에서도 옵션이 숨지 않게 한다.
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
        section1 = ttk.LabelFrame(scrollable_frame, text="링크 고정 설정 (Cloudflare Tunnel)", padding=10)
        section1.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(section1, text="Tunnel Token:").grid(row=0, column=0, sticky="w", pady=5)
        entries["cloudflare_tunnel_token"] = ttk.Entry(section1, width=50)
        entries["cloudflare_tunnel_token"].grid(row=0, column=1, padx=5, pady=5)
        entries["cloudflare_tunnel_token"].insert(0, self.settings.get("cloudflare_tunnel_token", ""))
        
        ttk.Label(section1, text="고정 도메인:").grid(row=1, column=0, sticky="w", pady=5)
        entries["cloudflare_tunnel_domain"] = ttk.Entry(section1, width=50)
        entries["cloudflare_tunnel_domain"].grid(row=1, column=1, padx=5, pady=5)
        entries["cloudflare_tunnel_domain"].insert(0, self.settings.get("cloudflare_tunnel_domain", ""))
        
        ttk.Label(section1, text="토큰이 설정되면 고정 도메인으로 접속합니다.\n비워두면 임시 URL(trycloudflare.com)을 사용합니다.",
                 foreground="#888888").grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        # === Discord 알림 설정 ===
        section2 = ttk.LabelFrame(scrollable_frame, text="Discord 알림", padding=10)
        section2.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(section2, text="Webhook URL:").grid(row=0, column=0, sticky="w", pady=5)
        entries["discord_webhook_url"] = ttk.Entry(section2, width=50)
        entries["discord_webhook_url"].grid(row=0, column=1, padx=5, pady=5)
        entries["discord_webhook_url"].insert(0, self.settings.get("discord_webhook_url", ""))
        
        ttk.Label(section2, text="서버 시작 시 Discord로 알림을 보냅니다.",
                 foreground="#888888").grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        # === 서버 설정 ===
        section_server = ttk.LabelFrame(scrollable_frame, text="서버 설정", padding=10)
        section_server.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(section_server, text="로컬 포트:").grid(row=0, column=0, sticky="w", pady=5)
        entries["server_port"] = ttk.Entry(section_server, width=15)
        entries["server_port"].grid(row=0, column=1, sticky="w", padx=5, pady=5)
        entries["server_port"].insert(0, self.settings.get("server_port", "5000"))

        ttk.Label(section_server, text="예: 5000, 8080, 18080\n변경 후 서버를 다시 시작해야 적용됩니다.",
                 foreground="#888888").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        # === 폴더 경로 설정 ===
        section3 = ttk.LabelFrame(scrollable_frame, text="폴더 경로 설정", padding=10)
        section3.pack(fill=tk.X, padx=10, pady=10)
        
        def browse_folder(key, entry_widget):
            # 모든 폴더 경로 입력에서 쓰는 공용 선택 함수다.
            folder = filedialog.askdirectory(title="폴더 선택")
            if folder:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, folder)

        def browse_model_file(entry_widget):
            # YOLO 모델 선택은 프로젝트 내부 파일과 절대 경로 .pt 파일을 모두 허용한다.
            filepath = filedialog.askopenfilename(
                title="YOLO 모델 파일 선택",
                filetypes=[("YOLO 모델", "*.pt"), ("모든 파일", "*.*")]
            )
            if filepath:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, filepath)
        
        # 입력 폴더
        ttk.Label(section3, text="입력 폴더:").grid(row=0, column=0, sticky="w", pady=5)
        entries["input_folder"] = ttk.Entry(section3, width=40)
        entries["input_folder"].grid(row=0, column=1, padx=5, pady=5)
        entries["input_folder"].insert(0, self.settings.get("input_folder", ""))
        ttk.Button(section3, text="찾아보기", 
                  command=lambda: browse_folder("input_folder", entries["input_folder"])).grid(row=0, column=2, padx=5)
        
        # 백업 폴더 (출력)
        ttk.Label(section3, text="백업 폴더:").grid(row=1, column=0, sticky="w", pady=5)
        entries["output_folder"] = ttk.Entry(section3, width=40)
        entries["output_folder"].grid(row=1, column=1, padx=5, pady=5)
        entries["output_folder"].insert(0, self.settings.get("output_folder", ""))
        ttk.Button(section3, text="찾아보기",
                  command=lambda: browse_folder("output_folder", entries["output_folder"])).grid(row=1, column=2, padx=5)
        
        # Excel 저장 폴더
        ttk.Label(section3, text="Excel 저장:").grid(row=2, column=0, sticky="w", pady=5)
        entries["excel_save_folder"] = ttk.Entry(section3, width=40)
        entries["excel_save_folder"].grid(row=2, column=1, padx=5, pady=5)
        entries["excel_save_folder"].insert(0, self.settings.get("excel_save_folder", ""))
        ttk.Button(section3, text="찾아보기",
                  command=lambda: browse_folder("excel_save_folder", entries["excel_save_folder"])).grid(row=2, column=2, padx=5)
        
        ttk.Label(section3, text="비워두면 기본 경로(프로그램 폴더)를 사용합니다.",
                 foreground="#888888").grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

        # === YOLO 모델 설정 ===
        section4 = ttk.LabelFrame(scrollable_frame, text="YOLO 모델 설정", padding=10)
        section4.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(section4, text="모델 파일:").grid(row=0, column=0, sticky="w", pady=5)
        entries["yolo_model_path"] = ttk.Entry(section4, width=40)
        entries["yolo_model_path"].grid(row=0, column=1, padx=5, pady=5)
        entries["yolo_model_path"].insert(0, self.settings.get("yolo_model_path", ""))
        ttk.Button(section4, text="찾아보기",
                  command=lambda: browse_model_file(entries["yolo_model_path"])).grid(row=0, column=2, padx=5)

        ttk.Label(section4, text="예: best.pt, best_yolo26.pt, yolo26n.pt 또는 .pt 파일 경로\n변경 후 프로그램을 재시작해야 적용됩니다.",
                 foreground="#888888").grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        # === 버튼 영역 ===
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        def save_settings():
            """설정 저장"""
            port_value = entries["server_port"].get().strip()
            try:
                # 잘못된 포트가 다음 실행을 망치지 않도록 저장 전에 검증한다.
                port_int = int(port_value)
                if not 1 <= port_int <= 65535:
                    raise ValueError
            except ValueError:
                messagebox.showerror("오류", "로컬 포트는 1부터 65535 사이의 숫자여야 합니다.")
                return

            for key, entry in entries.items():
                # 설정 폼의 모든 값은 문자열로 저장한다.
                self.settings.set(key, entry.get().strip())
            
            if self.settings.save():
                messagebox.showinfo("저장 완료", "설정이 저장되었습니다.\n일부 설정은 서버 재시작 후 적용됩니다.")
                dialog.destroy()
            else:
                messagebox.showerror("오류", "설정 저장에 실패했습니다.")
        
        def reset_settings():
            """설정 초기화"""
            if messagebox.askyesno("확인", "모든 설정을 초기화하시겠습니까?"):
                # 초기화된 기본값이 바로 보이도록 설정창을 다시 연다.
                self.settings.reset()
                dialog.destroy()
                self.open_settings_dialog()  # 다이얼로그 다시 열기
        
        ttk.Button(button_frame, text="저장", command=save_settings,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="초기화", command=reset_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="취소", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # 스크롤 레이아웃
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def toggle_server(self):
        """웹 서버 시작/중지 토글"""
        # 서버 제어는 툴바 버튼 하나가 담당하도록 유지한다.
        if self.server_running:
            # 서버 중지
            self.stop_background_server()
        else:
            # 서버 시작
            self.start_background_server()
    
    def stop_background_server(self):
        """백그라운드 웹 서버 중지"""
        import subprocess

        # 소켓/프로세스를 닫기 전에 서버 스레드에 중지 신호를 보낸다.
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
            # 경쟁 중인 스레드가 같은 서버 객체를 두 번 닫지 않도록 먼저 참조를 분리한다.
            server = self.http_server
            self.http_server = None

        if server:
            self._close_waitress_server(server)

        if self.server_thread and self.server_thread.is_alive() and self.server_thread is not threading.current_thread():
            self.server_thread.join(timeout=2)
        
        self.server_running = False
        self.server_thread = None
        self.server_btn.configure(text="서버 시작")
        self.copy_url_btn.configure(state=tk.DISABLED)
        self.server_status_label.configure(text="")
        self.server_url = None
        self.update_status("서버 중지됨 (다시 시작 버튼 클릭)")

    def _close_waitress_server(self, server):
        """Waitress 서버와 작업 스레드를 실제로 종료"""
        try:
            # waitress.run()이 빠져나올 수 있도록 활성 채널을 먼저 닫는다.
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
                # waitress가 사용한 작업 스레드를 종료한다.
                dispatcher.shutdown(timeout=2)
            except Exception:
                pass

    def start_background_server(self):
        """백그라운드 웹 서버 시작"""
        import socket
        import subprocess

        if self.server_running:
            return
        
        # 새 공개 URL을 만들기 전에 남아 있는 터널 프로세스를 정리한다.
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
            # 명확한 오류를 보여주기 위해 waitress 시작 전에 포트 바인딩을 시험한다.
            sock.bind(('127.0.0.1', port))
        except OSError:
            messagebox.showerror("오류", f"포트 {port}가 이미 사용 중입니다.\n다른 서버가 실행 중인지 확인하세요.")
            return
        finally:
            sock.close()

        stop_event = threading.Event()
        self.server_stop_event = stop_event
        
        def run_server():
            # 서버 스레드가 waitress.run()을 소유하고, UI 갱신은 root.after로 전달한다.
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
                    # 공개 터널 URL이 준비되면 로컬 URL 대신 사용한다.
                    self.server_url = public_url
                    self.root.after(0, lambda: self.server_status_label.configure(
                        text=f"{public_url[:30]}..."))
                    # Discord 알림
                    try:
                        send_discord_webhook(public_url, port)
                    except:
                        pass
                else:
                    # 터널링에 실패하면 로컬 전용 접속으로 대체한다.
                    self.server_url = f"http://127.0.0.1:{port}"
                    self.root.after(0, lambda: self.server_status_label.configure(
                        text="로컬만"))

                if stop_event.is_set():
                    return

                http_server.run()
            except Exception as e:
                if not stop_event.is_set():
                    self.root.after(0, lambda: messagebox.showerror("서버 오류", str(e)))
                    self.root.after(0, lambda: self.stop_background_server())
            finally:
                if http_server:
                    self._close_waitress_server(http_server)
                with self.server_lock:
                    # 이 스레드가 아직 해당 서버를 소유한 경우에만 현재 서버 참조를 비운다.
                    if self.http_server is http_server:
                        self.http_server = None
                if self.server_stop_event is stop_event:
                    self.server_stop_event = None
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.server_running = True
        self.server_btn.configure(text="서버 중지")
        self.copy_url_btn.configure(state=tk.NORMAL)
        self.server_url = f"http://127.0.0.1:{port}"
        self.update_status("웹 서버 시작 중...")

# 전역 락 파일 핸들
_lock_file_handle = None
_lock_file_path = None

def acquire_lock():
    """락 파일을 획득하여 중복 실행 방지"""
    global _lock_file_handle, _lock_file_path
    
    import tempfile
    import msvcrt
    
    # 임시 파일 락은 같은 PC에서 EXE를 여러 번 실행하는 경우도 감지할 수 있다.
    # 락 파일 경로 (사용자 temp 디렉토리)
    _lock_file_path = os.path.join(tempfile.gettempdir(), "parking_enforcement_gui.lock")
    
    try:
        # 락 파일 열기 또는 생성
        _lock_file_handle = open(_lock_file_path, 'w')
        # 비차단 락을 사용해 두 번째 인스턴스가 빠르게 실패하게 한다.
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
            # 다른 실행이 바로 락을 얻을 수 있도록 닫기 전에 먼저 언락한다.
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
        
        # 창 제목은 __init__에서 설정한 메인 창 제목과 일치해야 한다.
        # 창 제목으로 찾기
        hwnd = user32.FindWindowW(None, "주차 단속 시스템 (로컬 모드)")
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
    # OCR 사용 가능 여부와 관계없이 미리보기 렌더링에는 Pillow가 필요하다.
    # PIL 설치 확인
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow 라이브러리가 필요합니다: pip install Pillow")
        sys.exit(1)
    
    # 중복 실행 방지
    if not acquire_lock():
        # 두 번째 실행은 새 OCR/서버 프로세스를 만들지 말고 기존 창을 앞으로 가져와야 한다.
        # 기존 창을 앞으로 가져오기 시도
        if bring_existing_window_to_front():
            print("기존 프로그램 창을 활성화했습니다.")
        else:
            # 창을 찾지 못한 경우 메시지 표시
            # Tk 인스턴스 생성하여 메시지박스 표시
            temp_root = tk.Tk()
            temp_root.withdraw()  # 임시 창 숨기기
            messagebox.showwarning("경고", "프로그램이 이미 실행 중입니다.\n기존 창을 확인해주세요.")
            temp_root.destroy()
        sys.exit(0)
    
    # 종료 시 락 해제 등록
    # atexit에 등록하면 일반 창 닫기와 sys.exit 경로 모두에서 락을 해제할 수 있다.
    atexit.register(release_lock)
    
    try:
        root = tk.Tk()
        # start_server=True는 ocr.py의 하이브리드 모드에서 사용한다.
        app = ParkingEnforcementGUI(root, start_server=start_server)
        root.mainloop()
    except Exception as e:
        print(f"프로그램 오류: {e}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
