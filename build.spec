# -*- mode: python ; coding: utf-8 -*-

import os

base_dir = os.path.dirname(os.path.abspath(SPEC))

os.environ.setdefault('ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS', '1')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


def collect_yolo_models():
    models = []
    for filename in os.listdir(base_dir):
        lower = filename.lower()
        is_supported_model = (
            lower in {'best.pt', 'best_yolo26.pt'}
            or (lower.startswith('yolo26') and lower.endswith('.pt'))
        )
        if is_supported_model:
            models.append((os.path.join(base_dir, filename), '.'))
    return models


binaries = [
    ('dlls/oneocr.dll', 'dlls'),
    ('dlls/onnxruntime.dll', 'dlls'),
]

datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('dlls/oneocr.onemodel', 'dlls'),
] + collect_yolo_models()

hiddenimports = [
    'gui',
    'dll_extractor',
    'records_store',
    'PIL.ImageTk',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
]

excludes = [
    'torchaudio',
    'torchcodec',
    'paddle',
    'paddleocr',
    'paddlepaddle',
    'tensorflow',
    'onnxruntime',
    'streamlit',
]

a = Analysis(
    ['ocr.py'],
    pathex=[base_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='주차단속시스템',
    console=False,
    disable_windowed_traceback=False,
)
