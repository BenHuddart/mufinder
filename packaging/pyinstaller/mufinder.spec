# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


project_root = Path.cwd()
src_root = project_root / "src"
build_root = project_root / "packaging" / "build"
logo_source = src_root / "mufinder" / "assets" / "mufinder_logo.png"

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


def _require_installed_modules(*module_names: str) -> None:
    missing = [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ModuleNotFoundError(
            f"PyInstaller build requires installed runtime modules: {missing_text}"
        )


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

_require_installed_modules(
    "PySide6",
    "ase",
    "paramiko",
    "pyvista",
    "pyvistaqt",
    "vtkmodules",
    "muesr",
    "mulfc",
    "soprano",
)

datas = collect_data_files("PySide6")
datas += collect_data_files("mufinder.assets")
datas += collect_data_files("mufinder.data")
# Expose the source tree's mufinder.egg-info to importlib.metadata so PyInstaller
# can copy runtime package metadata even when CI builds from an editable checkout.
datas += copy_metadata("mufinder")
# ASE's CIF reader resolves symmetry data relative to ase.spacegroup.__file__.
# Bundle ASE package data so frozen macOS apps can load spacegroup.dat.
datas += collect_data_files("ase")
# muesr ships core lookup data alongside its Python modules.
datas += collect_data_files("muesr")
datas += collect_data_files("pyvista")
datas += collect_data_files("vtkmodules")
# The UEP clustering dialog imports Soprano's phylogenetic clustering stack lazily
# when the batch-search clustering window is opened.
# Magnetic-field calculations import muesr/muLFC only when a calculation starts.
hiddenimports = [
    "lfclib",
    "mufinder.app.main_window",
    "mufinder.app.splash_screen",
    "mufinder.app.theme",
    "mufinder.branding",
    "mufinder.castep_setup",
] + (
    collect_submodules("PySide6")
    + collect_submodules("ase.io")
    + collect_submodules("mufinder.app")
    + collect_submodules("muesr")
    + collect_submodules("mulfc")
    + collect_submodules("paramiko")
    + collect_submodules("pyvista")
    + collect_submodules("pyvistaqt")
    + collect_submodules("soprano")
    + collect_submodules("vtkmodules")
)

# PyVista/VTK is imported lazily in the viewer stack, so static analysis does not
# reliably discover VTK's native libraries during preview builds on Windows.
binaries = collect_dynamic_libs("PySide6") + collect_dynamic_libs("vtkmodules")

a = Analysis(
    [str(src_root / "mufinder" / "main.py")],
    pathex=[str(src_root)],
    binaries=binaries,
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
