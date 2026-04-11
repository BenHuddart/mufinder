# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path.cwd()
src_root = project_root / "src"

datas = collect_data_files("mufinder.assets")
hiddenimports = [
    "mufinder.app.main_window",
    "mufinder.app.splash_screen",
    "mufinder.app.theme",
    "mufinder.branding",
    "mufinder.castep_setup",
]

a = Analysis(
    [str(src_root / "mufinder" / "main.py")],
    pathex=[str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MuFinder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MuFinder",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MuFinder.app",
        icon=None,
        bundle_identifier="io.github.benhuddart.mufinder",
    )
