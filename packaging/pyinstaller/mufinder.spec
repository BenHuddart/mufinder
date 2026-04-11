# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path.cwd()
src_root = project_root / "src"
build_root = project_root / "packaging" / "build"
logo_source = src_root / "mufinder" / "assets" / "mufinder_logo.png"


def _build_icon(extension: str) -> str:
    build_root.mkdir(parents=True, exist_ok=True)
    output = build_root / f"mufinder{extension}"
    with Image.open(logo_source) as image:
        icon = image.convert("RGBA")
        if extension == ".ico":
            icon.save(
                output,
                format="ICO",
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
            )
        elif extension == ".icns":
            icon.save(output, format="ICNS")
        else:
            raise ValueError(f"Unsupported icon extension: {extension}")
    return str(output)


exe_icon = _build_icon(".ico") if sys.platform.startswith("win") else None
bundle_icon = _build_icon(".icns") if sys.platform == "darwin" else None

datas = collect_data_files("mufinder.assets")
hiddenimports = [
    "mufinder.app.main_window",
    "mufinder.app.splash_screen",
    "mufinder.app.theme",
    "mufinder.branding",
    "mufinder.castep_setup",
] + collect_submodules("ase.io")

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
    icon=exe_icon,
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
        icon=bundle_icon,
        bundle_identifier="io.github.benhuddart.mufinder",
    )
