from __future__ import annotations

from pathlib import Path


PROJECT_HOME_ASSETS = [
    "project_home_convergence_cutoff.png",
    "project_home_convergence_kpoint.png",
    "project_home_energy_barriers.png",
    "project_home_host_structure.png",
    "project_home_magnetic_fields.png",
    "project_home_map.png",
    "project_home_map_color.png",
    "project_home_muon_candidates.png",
    "project_home_pristine_relaxation.png",
    "project_home_provenance_log.png",
    "project_home_quantum_corrections.png",
    "project_home_supercell_editor.png",
    "project_home_uep.png",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pyinstaller_spec_text() -> str:
    return (_repo_root() / "packaging" / "pyinstaller" / "mufinder.spec").read_text(
        encoding="utf-8"
    )


def test_pyproject_packages_runtime_assets_and_science_data() -> None:
    repo_root = _repo_root()
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"mufinder.assets" = ["*.png", "*.ico", "*.yaml"]' in pyproject_text
    assert '"mufinder.data" = ["*.ini"]' in pyproject_text

    assets_root = repo_root / "src" / "mufinder" / "assets"
    for asset_name in PROJECT_HOME_ASSETS:
        assert (assets_root / asset_name).is_file()

    assert (repo_root / "src" / "mufinder" / "data" / "isotopes.ini").is_file()


def test_pyinstaller_spec_bundles_mufinder_runtime_assets_and_data() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'if str(src_root) not in sys.path:' in spec_text
    assert 'sys.path.insert(0, str(src_root))' in spec_text
    assert 'def _require_installed_modules(*module_names: str) -> None:' in spec_text
    assert 'PyInstaller build requires installed runtime modules:' in spec_text
    assert 'collect_data_files("mufinder.assets")' in spec_text
    assert 'collect_data_files("mufinder.data")' in spec_text
    assert 'copy_metadata("mufinder")' in spec_text


def test_pyinstaller_spec_bundles_ase_spacegroup_data() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'collect_data_files("ase")' in spec_text


def test_pyinstaller_spec_bundles_pyvista_and_vtk_runtime_assets() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'collect_data_files("PySide6")' in spec_text
    assert 'collect_submodules("PySide6")' in spec_text
    assert 'collect_dynamic_libs("PySide6")' in spec_text
    assert 'collect_data_files("pyvista")' in spec_text
    assert 'collect_data_files("vtkmodules")' in spec_text
    assert 'collect_dynamic_libs("vtkmodules")' in spec_text
    assert 'collect_submodules("pyvista")' in spec_text
    assert 'collect_submodules("pyvistaqt")' in spec_text
    assert 'collect_submodules("vtkmodules")' in spec_text


def test_pyinstaller_spec_bundles_soprano_clustering_modules() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'collect_submodules("soprano")' in spec_text


def test_pyinstaller_spec_bundles_magnetic_field_backend_modules() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'collect_data_files("muesr")' in spec_text
    assert 'collect_submodules("muesr")' in spec_text
    assert 'collect_submodules("mulfc")' in spec_text
    assert '"lfclib"' in spec_text


def test_pyinstaller_spec_builds_windows_icon() -> None:
    spec_text = _pyinstaller_spec_text()

    assert 'build_root = project_root / "packaging" / "build"' in spec_text
    assert 'exe_icon = _build_icon(".ico") if sys.platform.startswith("win") else None' in spec_text
    assert "icon=exe_icon" in spec_text


def test_windows_installer_uses_application_icon() -> None:
    nsi_path = _repo_root() / "packaging" / "windows" / "mufinder.nsi"
    nsi_text = nsi_path.read_text(encoding="utf-8")

    assert "!ifdef ICON_FILE" in nsi_text
    assert '!define APP_ICON "${ICON_FILE}"' in nsi_text
    assert '!define APP_ICON "packaging\\build\\mufinder.ico"' in nsi_text
    assert '!define INSTALLED_ICON_NAME "mufinder.ico"' in nsi_text
    assert 'Icon "${APP_ICON}"' in nsi_text
    assert '!define MUI_ICON "${APP_ICON}"' in nsi_text
    assert 'File /oname=${INSTALLED_ICON_NAME} "${APP_ICON}"' in nsi_text
    assert '"$INSTDIR\\${EXECUTABLE_NAME}" "" "$INSTDIR\\${INSTALLED_ICON_NAME}" 0' in nsi_text
    assert 'WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayIcon" "$INSTDIR\\${INSTALLED_ICON_NAME}"' in nsi_text
