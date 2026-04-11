from __future__ import annotations

from pathlib import Path


def test_pyinstaller_spec_bundles_ase_spacegroup_data() -> None:
    spec_path = (
        Path(__file__).resolve().parents[2] / "packaging" / "pyinstaller" / "mufinder.spec"
    )
    spec_text = spec_path.read_text(encoding="utf-8")

    assert 'collect_data_files("ase")' in spec_text


def test_pyinstaller_spec_builds_windows_icon() -> None:
    spec_path = (
        Path(__file__).resolve().parents[2] / "packaging" / "pyinstaller" / "mufinder.spec"
    )
    spec_text = spec_path.read_text(encoding="utf-8")

    assert 'build_root = project_root / "packaging" / "build"' in spec_text
    assert 'exe_icon = _build_icon(".ico") if sys.platform.startswith("win") else None' in spec_text
    assert "icon=exe_icon" in spec_text


def test_windows_installer_uses_application_icon() -> None:
    nsi_path = Path(__file__).resolve().parents[2] / "packaging" / "windows" / "mufinder.nsi"
    nsi_text = nsi_path.read_text(encoding="utf-8")

    assert '!define APP_ICON "packaging\\build\\mufinder.ico"' in nsi_text
    assert 'Icon "${APP_ICON}"' in nsi_text
    assert '!define MUI_ICON "${APP_ICON}"' in nsi_text
    assert '"$INSTDIR\\${EXECUTABLE_NAME}" "" "$INSTDIR\\${EXECUTABLE_NAME}" 0' in nsi_text
