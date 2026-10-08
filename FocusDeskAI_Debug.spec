# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# ---------------------------------------------------------
# Collect dependencies
# ---------------------------------------------------------

mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
torch_datas, torch_binaries, torch_hidden = collect_all("torch")
tv_datas, tv_binaries, tv_hidden = collect_all("torchvision")
ultra_datas, ultra_binaries, ultra_hidden = collect_all("ultralytics")

# Ship the models with the app instead of downloading them on first run (the
# install folder may be read-only, and a windowed build has no console for
# download progress output).
model_datas = [
    ("models/yolo11n.pt", "models"),
    ("models/face_landmarker.task", "models"),
    ("models/pose_landmarker_lite.task", "models"),
]

datas = (
    mp_datas
    + torch_datas
    + tv_datas
    + ultra_datas
    + model_datas
)

binaries = (
    mp_binaries
    + torch_binaries
    + tv_binaries
    + ultra_binaries
)

hiddenimports = (
    mp_hidden
    + torch_hidden
    + tv_hidden
    + ultra_hidden
    + [
        "mediapipe.tasks.c",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "mediapipe.tasks.python.core",
        "torchvision",
        "torchvision.ops",
        "torchvision._C",
    ]
)

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------
# Python archive
# ---------------------------------------------------------

pyz = PYZ(a.pure)

# ---------------------------------------------------------
# Executable
# ---------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FocusDeskAI_Debug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ---------------------------------------------------------
# Collect
# ---------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FocusDeskAI_Debug",
)