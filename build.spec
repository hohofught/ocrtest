# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 빌드 스펙 파일 - DLL OCR 버전 (2.0V)
# 
# 빌드 전 주의사항:
# - dlls/ 폴더에 DLL이 있어야 합니다 (build.bat이 자동으로 추출함)
# - 또는 python dll_extractor.py 를 먼저 실행하세요

import os
import sys
from PyInstaller.utils.hooks.tcl_tk import tcltk_info

block_cipher = None
base_dir = os.path.dirname(os.path.abspath(SPEC))
python_base = os.path.dirname(sys.executable)
python_dll_dir = os.path.join(python_base, 'DLLs')
python_tcl_dir = os.path.join(python_base, 'tcl')
tcl_data_dir = os.path.join(python_tcl_dir, 'tcl8.6')
tk_data_dir = os.path.join(python_tcl_dir, 'tk8.6')
tcl_module_dir = os.path.join(python_tcl_dir, 'tcl8')

os.environ.setdefault('ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS', '1')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('TCL_LIBRARY', tcl_data_dir)
os.environ.setdefault('TK_LIBRARY', tk_data_dir)

manual_tcl_tk_binaries = [
    (path, '.')
    for path in [
        os.path.join(python_dll_dir, '_tkinter.pyd'),
        os.path.join(python_dll_dir, 'tcl86t.dll'),
        os.path.join(python_dll_dir, 'tk86t.dll'),
    ]
    if os.path.exists(path)
]

manual_tcl_tk_datas = []
for src, dest in [
    (tcl_data_dir, '_tcl_data'),
    (tk_data_dir, '_tk_data'),
    (tcl_module_dir, 'tcl8'),
]:
    if os.path.isdir(src):
        manual_tcl_tk_datas.append((src, dest))

# The sandboxed build process can fail PyInstaller's Tcl probe even when the
# local Python installation has the files. Seed the cached metadata so tkinter
# is not incorrectly excluded from the GUI build.
if os.path.exists(os.path.join(python_dll_dir, '_tkinter.pyd')) and os.path.isdir(tcl_data_dir) and os.path.isdir(tk_data_dir):
    tcltk_info.available = True
    tcltk_info.tkinter_extension_file = os.path.join(python_dll_dir, '_tkinter.pyd')
    tcltk_info.tcl_version = (8, 6)
    tcltk_info.tk_version = (8, 6)
    tcltk_info.tcl_threaded = True
    tcltk_info.tcl_data_dir = tcl_data_dir
    tcltk_info.tk_data_dir = tk_data_dir
    tcltk_info.tcl_module_dir = tcl_module_dir
    tcltk_info.is_macos_system_framework = False
    tcltk_info.tcl_shared_library = os.path.join(python_dll_dir, 'tcl86t.dll')
    tcltk_info.tk_shared_library = os.path.join(python_dll_dir, 'tk86t.dll')
    tcltk_info.data_files = []
    tcltk_info.data_files += tcltk_info._collect_files_from_directory(
        tcl_data_dir,
        prefix=tcltk_info.TCL_ROOTNAME,
        excludes=['demos', '*.lib', 'tclConfig.sh'],
    )
    tcltk_info.data_files += tcltk_info._collect_files_from_directory(
        tk_data_dir,
        prefix=tcltk_info.TK_ROOTNAME,
        excludes=['demos', '*.lib', 'tkConfig.sh'],
    )
    if os.path.isdir(tcl_module_dir):
        tcltk_info.data_files += tcltk_info._collect_files_from_directory(
            tcl_module_dir,
            prefix=os.path.basename(tcl_module_dir),
        )

# These packages may be installed in a developer's global Python for other
# projects, but this app does not use them. Excluding them prevents PyInstaller
# from bundling incompatible native .pyd/.dll files such as torchaudio.
unused_native_modules = [
    'torchaudio',
    'torch_audiomentations',
    'torch_pitch_shift',
    'torchcodec',
    'pyannote',
    'pyannote.audio',
    'whisperx',
    'faster_whisper',
    'ctranslate2',
    'paddleocr',
    'paddle',
    'paddlepaddle',
    'paddlex',
    'onnxruntime',
    'tensorflow',
    'lightning',
    'pytorch_lightning',
    'torchmetrics',
    'lap',
    'onnx',
    'onnxruntime-gpu',
    'onnxruntime-extensions',
    'onnxslim',
    'onnxscript',
    'onnx_ir',
    'onnx2tf',
    'openvino',
    'tensorrt',
    'coremltools',
    'tensorflowjs',
    'tflite_runtime',
    'ncnn',
    'MNN',
    'rknn',
    'rknnlite',
    'x2paddle',
    'executorch',
    'edgemdt_cl',
    'edgemdt_tpc',
    'model_compression_toolkit',
    'mct_quantizers',
    'dx_engine',
    'dx_com',
    'tritonclient',
    'streamlit',
    'roboflow',
    'hub_sdk',
    'sentry_sdk',
    'transformers',
    'tokenizers',
    'safetensors',
    'huggingface_hub',
    'av',
    'nltk',
    'sklearn',
    'scikit_learn',
    'shapely',
    'sqlalchemy',
    'grpc',
    'opentelemetry',
    'mako',
    'optuna',
    'chardet',
]

yolo_model_datas = []
for filename in os.listdir(base_dir):
    filename_lower = filename.lower()
    if filename_lower == 'best.pt' or filename_lower == 'best_yolo26.pt' or (
        filename_lower.startswith('yolo26') and filename_lower.endswith('.pt')
    ):
        yolo_model_datas.append((os.path.join(base_dir, filename), '.'))

a = Analysis(
    ['ocr.py'],
    pathex=[base_dir],
    binaries=[
        ('dlls/oneocr.dll', 'dlls'),
        ('dlls/onnxruntime.dll', 'dlls'),
    ] + manual_tcl_tk_binaries,
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('dlls/oneocr.onemodel', 'dlls'),
    ] + yolo_model_datas + manual_tcl_tk_datas,
    hiddenimports=[
        'gui',
        'dll_extractor',
        '_tkinter',
        'flask',
        'waitress',
        'pandas',
        'openpyxl',
        'cv2',
        'numpy',
        'ultralytics',
        'ctypes',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=unused_native_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='주차단속시스템',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 모드 기본 (콘솔 창 숨김)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
