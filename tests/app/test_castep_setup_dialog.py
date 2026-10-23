"""Tests for CASTEP setup dialog behavior."""

from __future__ import annotations

# Example paths for the keyword-generation command tests. Nothing here is run,
# so the paths do not need to exist: the tests only check how an MPI-built
# CASTEP binary is wrapped in its launcher.
MPI_CASTEP_EXECUTABLE = "/opt/castep/bin/castep.mpi"
MPIRUN_LAUNCHER = "/opt/openmpi/bin/mpirun"

from dataclasses import replace
import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel

import mufinder.app.castep_setup_dialog as dialog_module
import mufinder.castep_io.schema as schema_module
import mufinder.castep_setup as castep_setup_module
from mufinder.castep_io.keyword_catalog_file import KeywordCatalogValidationError
from mufinder.castep_io.schema import SETUP_KEYWORDS_JSON_ENV_VAR, keyword_catalog_status
from mufinder.castep_setup import (
    CASTEP_EXECUTABLE_KEY,
    CASTEP_MPIRUN_KEY,
    CASTEP_KEYWORDS_JSON_KEY,
    CastepSetup,
    apply_castep_environment,
    apply_saved_castep_environment,
    build_castep_keyword_command,
    discover_mpirun,
    generate_castep_keywords_json,
    install_keyword_catalog,
    load_castep_setup,
    managed_keywords_json_path,
    save_castep_setup,
)
from tests.support.keyword_catalogs import write_complete_catalog
from mufinder.app.castep_setup_dialog import CastepSetupDialog
from mufinder.services.ssh_client import SSHCredentials, SSHUnknownHostKeyError
from tests.support.qt import pump_until


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def value(self, key: str, default: str = "", value_type: type[str] | None = None) -> str:
        value = self._values.get(key, default)
        return str(value) if value_type is str else value

    def setValue(self, key: str, value: str) -> None:  # noqa: N802 - Qt API naming.
        self._values[key] = value


def test_castep_setup_roundtrip_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _FakeSettings()
    setup = CastepSetup(
        executable_path="/opt/castep/bin/castep.serial",
        keywords_json_path="/tmp/castep_keywords.json",
        mpirun_path="/opt/mpi/bin/mpirun",
    )

    save_castep_setup(settings, setup)
    loaded = load_castep_setup(settings)
    assert loaded == setup

    monkeypatch.delenv("CASTEP_COMMAND", raising=False)
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", "/tmp/user-override.json")
    apply_castep_environment(setup)

    assert "CASTEP_COMMAND" not in os.environ
    assert os.environ[SETUP_KEYWORDS_JSON_ENV_VAR] == setup.keywords_json_path
    # The user-facing variable is an override and is never touched by setup.
    assert os.environ["MUFINDER_CASTEP_KEYWORDS_JSON"] == "/tmp/user-override.json"


def test_apply_saved_castep_environment_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _FakeSettings()
    settings.setValue(CASTEP_EXECUTABLE_KEY, "/usr/local/bin/castep")
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, "/tmp/generated/castep_keywords.json")
    settings.setValue(CASTEP_MPIRUN_KEY, "/opt/mpi/bin/mpirun")

    monkeypatch.delenv("CASTEP_COMMAND", raising=False)

    setup = apply_saved_castep_environment(settings)

    assert setup.executable_path == "/usr/local/bin/castep"
    assert setup.mpirun_path == "/opt/mpi/bin/mpirun"
    assert "CASTEP_COMMAND" not in os.environ
    assert os.environ[SETUP_KEYWORDS_JSON_ENV_VAR] == "/tmp/generated/castep_keywords.json"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="discover_mpirun fallback candidates are POSIX-only paths",
)
def test_discover_mpirun_prefers_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dialog_module, "discover_mpirun", discover_mpirun)
    monkeypatch.setattr("mufinder.castep_setup.shutil.which", lambda _name: "/usr/local/bin/mpirun")

    discovered = discover_mpirun()

    assert discovered is not None
    assert discovered.endswith("/usr/local/bin/mpirun")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="discover_mpirun fallback candidates are POSIX-only paths",
)
def test_discover_mpirun_falls_back_to_known_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mufinder.castep_setup.shutil.which", lambda _name: None)

    class _FakePath:
        def __init__(self, candidate: str) -> None:
            self._candidate = candidate

        def exists(self) -> bool:
            return self._candidate == "/opt/homebrew/bin/mpirun"

        def is_file(self) -> bool:
            return self.exists()

        def resolve(self) -> Path:
            return Path(self._candidate)

    monkeypatch.setattr("mufinder.castep_setup.Path", _FakePath)
    monkeypatch.setattr("mufinder.castep_setup.os.access", lambda _path, _mode: True)

    discovered = discover_mpirun()

    assert discovered == "/opt/homebrew/bin/mpirun"


def test_discover_mpirun_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mufinder.castep_setup.shutil.which", lambda _name: None)

    class _MissingPath:
        def __init__(self, _candidate: str) -> None:
            pass

        def exists(self) -> bool:
            return False

        def is_file(self) -> bool:
            return False

    monkeypatch.setattr("mufinder.castep_setup.Path", _MissingPath)

    discovered = discover_mpirun()

    assert discovered is None


def test_generate_castep_keywords_json_uses_ase_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str, bool]] = []

    def fake_create(
        command: str, filename: str, force_write: bool, path: str, fetch_only=None
    ) -> None:
        calls.append((command, filename, path, force_write))
        (Path(path) / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("ase.calculators.castep.create_castep_keywords", fake_create)

    output = generate_castep_keywords_json(
        executable_path="/opt/castep/bin/castep",
        output_path=tmp_path / "castep_keywords.json",
    )

    assert calls == [("/opt/castep/bin/castep", "castep_keywords.json", str(tmp_path), True)]
    assert output == (tmp_path / "castep_keywords.json").resolve()


def test_build_castep_keyword_command_passes_serial_executable_through() -> None:
    assert build_castep_keyword_command("/opt/castep/bin/castep") == "/opt/castep/bin/castep"


def test_build_castep_keyword_command_wraps_executable_in_mpi_launcher() -> None:
    # MPI-built binaries hang when run directly, so they must be launched via mpirun.
    assert (
        build_castep_keyword_command(MPI_CASTEP_EXECUTABLE, MPIRUN_LAUNCHER)
        == f"{MPIRUN_LAUNCHER} -np 1 {MPI_CASTEP_EXECUTABLE}"
    )


def test_generate_castep_keywords_json_wraps_command_with_mpi_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[str] = []

    def fake_create(
        command: str, filename: str, force_write: bool, path: str, fetch_only=None
    ) -> None:
        commands.append(command)
        (Path(path) / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("ase.calculators.castep.create_castep_keywords", fake_create)

    generate_castep_keywords_json(
        executable_path=MPI_CASTEP_EXECUTABLE,
        output_path=tmp_path / "castep_keywords.json",
        mpirun_path=MPIRUN_LAUNCHER,
    )

    assert commands == [f"{MPIRUN_LAUNCHER} -np 1 {MPI_CASTEP_EXECUTABLE}"]


def test_generate_castep_keywords_json_raises_when_generation_produces_no_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "ase.calculators.castep.create_castep_keywords",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="did not produce"):
        generate_castep_keywords_json(
            executable_path="/opt/castep/bin/castep",
            output_path=tmp_path / "missing.json",
        )


def test_apply_castep_environment_clears_keywords_env_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SETUP_KEYWORDS_JSON_ENV_VAR, "/tmp/old.json")

    apply_castep_environment(CastepSetup(keywords_json_path="   "))

    assert SETUP_KEYWORDS_JSON_ENV_VAR not in os.environ


@pytest.fixture
def no_catalog_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a machine with no catalog configured or discoverable."""

    for name in (
        "MUFINDER_CASTEP_KEYWORDS_JSON",
        "CASTEP_KEYWORDS_JSON",
        SETUP_KEYWORDS_JSON_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(schema_module, "_ENV_ROOT_VARS", ())
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: tuple(
            schema_module.CatalogSourceInfo(path=Path(os.environ[name]), origin=origin)
            for name, origin in (
                ("MUFINDER_CASTEP_KEYWORDS_JSON", "env_json:MUFINDER_CASTEP_KEYWORDS_JSON"),
                (SETUP_KEYWORDS_JSON_ENV_VAR, "setup"),
            )
            if os.environ.get(name)
        ),
    )
    schema_module.refresh_keyword_catalog(allow_fallback=True)


def test_startup_loads_the_installed_catalog(no_catalog_env, tmp_path: Path) -> None:
    """Regression: a reopened MuFinder reported no catalog although one was installed.

    The schema builds itself when first imported, which happens before the saved
    setup is applied at startup, so applying the setup must rebuild it.
    """

    settings = _FakeSettings()
    install_keyword_catalog(write_complete_catalog(tmp_path / "source.json", "26.11"), settings)

    # A fresh process: the schema has been built with nothing configured.
    os.environ.pop(SETUP_KEYWORDS_JSON_ENV_VAR, None)
    schema_module.refresh_keyword_catalog(allow_fallback=True)
    assert keyword_catalog_status().is_starter is True

    apply_saved_castep_environment(settings)

    status = keyword_catalog_status()
    assert (status.origin, status.is_starter, status.castep_version) == ("setup", False, "26.11")
    assert status.path == managed_keywords_json_path()


def test_install_keyword_catalog_copies_validates_and_activates(
    no_catalog_env,
    tmp_path: Path,
) -> None:
    settings = _FakeSettings()
    settings.setValue(CASTEP_EXECUTABLE_KEY, "/opt/castep/bin/castep")
    source = write_complete_catalog(tmp_path / "elsewhere" / "my_keywords.json", "25.12")
    assert keyword_catalog_status().is_starter is True

    summary = install_keyword_catalog(source, settings)

    managed = managed_keywords_json_path()
    assert managed.name == "castep_keywords.json"
    assert managed != source and managed.read_text(encoding="utf-8") != ""
    assert (summary.castep_version, summary.cell_count, summary.param_count) == ("25.12", 48, 163)
    assert summary.describe() == "CASTEP 25.12: 48 cell and 163 param keywords"
    # The setup now points at MuFinder's own copy and other settings survive.
    assert load_castep_setup(settings) == CastepSetup(
        executable_path="/opt/castep/bin/castep",
        keywords_json_path=str(managed),
    )
    status = keyword_catalog_status()
    assert (status.origin, status.is_starter, status.castep_version) == ("setup", False, "25.12")

    # The copy is independent of the original file.
    source.unlink()
    schema_module.refresh_keyword_catalog(allow_fallback=True)
    assert keyword_catalog_status().origin == "setup"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("{not json", "is not valid JSON"),
        ("[]", "not a JSON object"),
        ('{"cell": {}}', "no 'param' keyword section"),
        ('{"cell": {"a": 1}, "param": {}}', "is not a keyword description"),
        ('{"cell": {"a": {"level": "Basic"}}, "param": {}}', "is missing option_type, docstring"),
        (
            '{"cell": {"a": {"option_type": "S", "level": "B", "docstring": "d"}}, "param": {}}',
            "does not describe LATTICE_CART, POSITIONS_FRAC, SPECIES_POT",
        ),
    ],
)
def test_install_keyword_catalog_rejects_unusable_files(
    no_catalog_env,
    tmp_path: Path,
    content: str,
    reason: str,
) -> None:
    settings = _FakeSettings()
    good = write_complete_catalog(tmp_path / "good.json")
    install_keyword_catalog(good, settings)
    installed = managed_keywords_json_path().read_text(encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")

    with pytest.raises(KeywordCatalogValidationError, match=reason):
        install_keyword_catalog(bad, settings)

    # A rejected file leaves the working catalog exactly as it was.
    assert managed_keywords_json_path().read_text(encoding="utf-8") == installed
    assert keyword_catalog_status().origin == "setup"


def test_install_keyword_catalog_rejects_truncated_catalog_and_missing_file(
    no_catalog_env,
    tmp_path: Path,
) -> None:
    import json

    from tests.support.keyword_catalogs import complete_catalog_payload

    payload = complete_catalog_payload()
    payload["param"] = dict(list(payload["param"].items())[:20])
    truncated = tmp_path / "truncated.json"
    truncated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeywordCatalogValidationError, match="48 cell and 20 param keywords"):
        install_keyword_catalog(truncated, _FakeSettings())
    with pytest.raises(KeywordCatalogValidationError, match="Could not read"):
        install_keyword_catalog(tmp_path / "missing.json", _FakeSettings())
    assert not managed_keywords_json_path().exists()


def test_app_data_directory_falls_back_when_qt_reports_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # drop the conftest redirection to exercise the real lookup
    monkeypatch.setattr(
        castep_setup_module.QStandardPaths, "writableLocation", lambda _location: ""
    )
    assert castep_setup_module._app_data_directory() == Path.home() / ".mufinder"

    monkeypatch.setattr(
        castep_setup_module.QStandardPaths, "writableLocation", lambda _location: "/data/MuFinder"
    )
    assert castep_setup_module._app_data_directory() == Path("/data/MuFinder")


def test_castep_setup_dialog_installs_generated_catalog_immediately(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _FakeSettings()
    dialog = CastepSetupDialog(settings=settings)
    assert "limited built-in keyword set" in dialog._catalog_status_label.text()

    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, message: infos.append((title, message)),
    )
    scratch_paths: list[Path] = []

    def fake_start_generation_process(executable: str, output_path: Path) -> None:
        scratch_paths.append(output_path)
        write_complete_catalog(output_path, "26.11")
        dialog._generation_output_path = output_path
        dialog._on_generation_finished(0, None)

    monkeypatch.setattr(dialog, "_start_generation_process", fake_start_generation_process)

    dialog._executable_edit.setText("/opt/castep/bin/castep")
    dialog._generate_keywords()

    # Installed and in use straight away, without the user choosing a path or
    # pressing OK, and the scratch copy is gone.
    managed = managed_keywords_json_path()
    assert infos == [
        (
            "Keyword catalog installed",
            "Keyword catalog generated: CASTEP 26.11: 48 cell and 163 param keywords.",
        )
    ]
    assert scratch_paths[0].parent != managed.parent
    assert not scratch_paths[0].parent.exists()
    assert settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) == str(managed)
    assert keyword_catalog_status().origin == "setup"
    assert "Keyword catalog for CASTEP 26.11" in dialog._catalog_status_label.text()
    assert "installed through CASTEP setup" in dialog._catalog_status_label.text()
    assert dialog._progress_label.text() == "CASTEP keyword generation complete."

    dialog._accept_and_save()
    assert settings.value(CASTEP_EXECUTABLE_KEY, "", str) == "/opt/castep/bin/castep"
    assert settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) == str(managed)
    assert keyword_catalog_status().origin == "setup"
    dialog.close()


def test_castep_setup_dialog_rejects_incomplete_generated_catalog(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _FakeSettings()
    dialog = CastepSetupDialog(settings=settings)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    def fake_start_generation_process(executable: str, output_path: Path) -> None:
        output_path.write_text("{}", encoding="utf-8")
        dialog._generation_output_path = output_path
        dialog._on_generation_finished(0, None)

    monkeypatch.setattr(dialog, "_start_generation_process", fake_start_generation_process)
    dialog._executable_edit.setText("/opt/castep/bin/castep")
    dialog._generate_keywords()

    assert [title for title, _message in warnings] == ["Keyword catalog not usable"]
    assert "the current keyword catalog is unchanged" in warnings[0][1]
    assert "no 'cell' keyword section" in warnings[0][1]
    assert settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) == ""
    assert keyword_catalog_status().is_starter is True
    assert dialog._progress_label.text() == "Generation failed."
    assert dialog._generation_scratch_dir is None
    dialog.close()


def test_castep_setup_dialog_imports_an_existing_catalog(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _FakeSettings()
    dialog = CastepSetupDialog(settings=settings)
    source = write_complete_catalog(tmp_path / "from_colleague.json", "24.1")
    infos: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, _title, message: infos.append(message),
    )
    monkeypatch.setattr(
        dialog_module.QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(source), "")
    )

    dialog._import_button.click()

    assert infos == ["Keyword catalog imported: CASTEP 24.1: 48 cell and 163 param keywords."]
    assert settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) == str(managed_keywords_json_path())
    assert "CASTEP 24.1" in dialog._catalog_status_label.text()

    # Cancelling the file chooser changes nothing.
    monkeypatch.setattr(
        dialog_module.QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: ("", "")
    )
    dialog._import_button.click()
    assert len(infos) == 1
    dialog.close()


def test_castep_setup_dialog_import_says_when_an_env_override_wins(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = write_complete_catalog(tmp_path / "override.json", "23.1")
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(override))
    dialog = CastepSetupDialog(settings=_FakeSettings())
    infos: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, _title, message: infos.append(message),
    )

    assert (
        dialog._install_catalog(
            write_complete_catalog(tmp_path / "new.json", "26.11"), action="imported"
        )
        is True
    )

    assert "It is not in use yet" in infos[0]
    assert "MUFINDER_CASTEP_KEYWORDS_JSON environment variable overrides CASTEP setup" in infos[0]
    assert keyword_catalog_status().castep_version == "23.1"
    assert "which overrides CASTEP setup" in dialog._catalog_status_label.text()
    dialog.close()


def test_castep_setup_dialog_reports_unwritable_data_directory(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    def refuse(*_args, **_kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(dialog_module, "install_keyword_catalog", refuse)

    assert dialog._install_catalog(tmp_path / "any.json", action="imported") is False
    assert warnings == ["Keyword catalog not saved"]
    dialog.close()


def test_castep_setup_dialog_warns_when_generating_without_executable(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._generate_keywords()

    assert warnings == ["Missing executable"]
    dialog.close()


def test_castep_setup_dialog_updates_progress_from_generator_output(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    dialog._update_progress_from_output("Processed: [##########          ] 50%")

    assert dialog._progress_bar.value() == 50
    assert "50%" in dialog._progress_label.text()
    dialog.close()


def test_castep_setup_dialog_local_generation_wraps_executable_in_mpi_launcher(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[str, list[str]]] = []

    class _FakeProcess:
        def __init__(self, _parent=None) -> None:
            self.readyReadStandardOutput = _FakeSignal()
            self.readyReadStandardError = _FakeSignal()
            self.finished = _FakeSignal()
            self.errorOccurred = _FakeSignal()

        def start(self, program: str, arguments: list[str]) -> None:
            captured.append((program, arguments))

    monkeypatch.setattr(dialog_module, "QProcess", _FakeProcess)

    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._executable_edit.setText(MPI_CASTEP_EXECUTABLE)
    dialog._mpirun_edit.setText(MPIRUN_LAUNCHER)

    dialog._start_generation_process(MPI_CASTEP_EXECUTABLE, tmp_path / "castep_keywords.json")

    assert len(captured) == 1
    program, arguments = captured[0]
    assert program == sys.executable
    # argv: ["-u", "-c", <script>, <executable>, <mpirun>, <output path>]
    assert arguments[:2] == ["-u", "-c"]
    assert arguments[3:] == [
        MPI_CASTEP_EXECUTABLE,
        MPIRUN_LAUNCHER,
        str(tmp_path / "castep_keywords.json"),
    ]

    # Running the script hands ASE the MPI-wrapped command and the output path.
    ase_calls: list[tuple[str, str, str]] = []

    def fake_create(
        command: str, filename: str, force_write: bool, path: str, fetch_only=None
    ) -> None:
        ase_calls.append((command, filename, path))
        (Path(path) / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("ase.calculators.castep.create_castep_keywords", fake_create)
    monkeypatch.setattr(sys, "argv", ["-c", *arguments[3:]])
    exec(arguments[2], {"__name__": "__main__"})

    assert ase_calls == [
        (f"{MPIRUN_LAUNCHER} -np 1 {MPI_CASTEP_EXECUTABLE}", "castep_keywords.json", str(tmp_path))
    ]
    dialog.close()


class _FakeSignal:
    def connect(self, _slot) -> None:  # pragma: no cover - trivial stub
        pass


def test_castep_setup_dialog_explains_a_missing_catalog(qapp, no_catalog_env) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    assert dialog.findChild(QLabel, "catalogStatusLabel").text() == (
        "No keyword catalog is loaded; MuFinder is using its limited built-in keyword set."
    )
    intro = next(
        label.text()
        for label in dialog.findChildren(QLabel)
        if "castep_keywords.json" in label.text()
    )
    assert "some results cannot be loaded" in intro
    assert "there is no location to choose" in intro
    dialog.close()


def test_castep_setup_dialog_reports_a_configured_catalog_that_went_missing(
    qapp,
    no_catalog_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _FakeSettings()
    missing = tmp_path / "deleted" / "castep_keywords.json"
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, str(missing))
    apply_saved_castep_environment(settings)
    schema_module.refresh_keyword_catalog(allow_fallback=True)

    dialog = CastepSetupDialog(settings=settings)

    text = dialog._catalog_status_label.text()
    assert f"The configured keyword catalog '{missing}' no longer exists." in text
    assert "No keyword catalog is loaded" in text
    dialog.close()


def test_castep_setup_dialog_lists_local_and_cluster_targets(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    targets = [
        dialog._generate_target_combo.itemText(index)
        for index in range(dialog._generate_target_combo.count())
    ]

    assert targets[:3] == ["This computer", "SCARF", "Redwood"]
    dialog.close()


def test_castep_setup_dialog_warns_on_incomplete_cluster_setup(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._generate_target_combo.setCurrentText("Redwood")

    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module, "validate_cluster_keyword_config", lambda _config: ["Missing shell init"]
    )
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._generate_keywords()

    assert warnings == ["Incomplete cluster setup"]
    dialog.close()


def test_castep_setup_dialog_starts_cluster_generation_for_selected_target(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._generate_target_combo.setCurrentText("Redwood")

    credentials = SSHCredentials(target_secret="glam", jump_secret="phys")
    captured: list[tuple[str, Path, SSHCredentials | None]] = []

    class _AlwaysConnected:
        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.connected = False

        @property
        def is_connected(self) -> bool:
            return self.connected

        def connect(self, password: str | None = None, **_kwargs) -> None:
            _ = password
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False

    monkeypatch.setattr(dialog_module, "validate_cluster_keyword_config", lambda _config: [])
    monkeypatch.setattr(dialog_module, "SSHConnection", _AlwaysConnected)
    monkeypatch.setattr(
        dialog_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: credentials,
    )
    monkeypatch.setattr(
        dialog,
        "_start_cluster_generation",
        lambda config, path, credentials=None: captured.append((config.name, path, credentials)),
    )

    dialog._generate_keywords()

    assert pump_until(lambda: bool(captured), timeout=5.0)
    assert [(name, path.name, creds) for name, path, creds in captured] == [
        ("Redwood", "castep_keywords.json", credentials)
    ]
    # Generation writes to scratch, never over the installed catalog.
    assert captured[0][1].parent == dialog._generation_scratch_dir
    assert captured[0][1] != managed_keywords_json_path()
    dialog.close()


def test_castep_setup_dialog_cluster_generation_trusts_unknown_host_key(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._generate_target_combo.setCurrentText("Redwood")

    prompts: list[str] = []
    credentials = SSHCredentials(target_secret="glam", jump_secret="phys")
    captured: list[tuple[str, Path, SSHCredentials | None]] = []

    class _UnknownHostConnection:
        attempts = 0

        def __init__(self, _config, *, settings=None) -> None:
            _ = settings
            self.connected = False

        @property
        def is_connected(self) -> bool:
            return self.connected

        def connect(self, password: str | None = None, **_kwargs) -> None:
            _ = password
            self.__class__.attempts += 1
            if self.__class__.attempts == 1:
                raise SSHUnknownHostKeyError(
                    hostname="glamdring.physics.ox.ac.uk",
                    port=22,
                    algorithm="ssh-ed25519",
                    fingerprint_sha256="SHA256:test",
                    key_base64="dGVzdA==",
                )
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False

    monkeypatch.setattr(dialog_module, "validate_cluster_keyword_config", lambda _config: [])
    monkeypatch.setattr(dialog_module, "SSHConnection", _UnknownHostConnection)
    monkeypatch.setattr(
        dialog_module,
        "prompt_to_trust_host_key",
        lambda *_args, **_kwargs: prompts.append("yes") or True,
    )
    monkeypatch.setattr(
        dialog_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: credentials,
    )
    monkeypatch.setattr(
        dialog,
        "_start_cluster_generation",
        lambda config, path, credentials=None: captured.append((config.name, path, credentials)),
    )

    dialog._generate_keywords()

    assert pump_until(lambda: bool(captured), timeout=5.0)
    assert prompts == ["yes"]
    assert [(name, path.name, creds) for name, path, creds in captured] == [
        ("Redwood", "castep_keywords.json", credentials)
    ]
    # Generation writes to scratch, never over the installed catalog.
    assert captured[0][1].parent == dialog._generation_scratch_dir
    assert captured[0][1] != managed_keywords_json_path()
    assert (
        dialog._settings.value(
            "trusted_host_keys/Redwood/glamdring.physics.ox.ac.uk/algorithm", "", str
        )
        == "ssh-ed25519"
    )
    dialog.close()


def test_castep_setup_dialog_text_fields_are_taller_by_default(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    assert dialog._executable_edit.minimumHeight() >= 34
    assert dialog._mpirun_edit.minimumHeight() >= 34
    assert dialog._executable_edit.minimumWidth() >= 740
    assert dialog._mpirun_edit.minimumWidth() >= 740
    dialog.close()


def test_castep_setup_dialog_prefers_a_set_up_cluster_without_a_local_executable(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster-only users should not have to discover the target selector."""

    # The built-in presets still carry their placeholder username, so they are
    # templates rather than clusters the user has set up.
    dialog = CastepSetupDialog(settings=_FakeSettings())
    assert dialog._generate_target_combo.currentData() == "local"
    dialog.close()

    presets = dialog_module.load_cluster_configs(_FakeSettings())
    monkeypatch.setattr(
        dialog_module,
        "load_cluster_configs",
        lambda _settings: [
            replace(config, username="alice") if config.name == "Redwood" else config
            for config in presets
        ],
    )
    configured = _FakeSettings()
    dialog = CastepSetupDialog(settings=configured)
    assert dialog._generate_target_combo.currentText() == "Redwood"
    dialog.close()

    configured.setValue(CASTEP_EXECUTABLE_KEY, "/opt/castep/bin/castep")
    dialog = CastepSetupDialog(settings=configured)
    assert dialog._generate_target_combo.currentData() == "local"
    dialog.close()


def test_castep_setup_dialog_says_which_settings_each_target_uses(qapp) -> None:
    """The local executable and MPI launcher apply to this computer only."""

    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog.show()
    hint = dialog.findChild(QLabel, "generateTargetHint")

    note = dialog.findChild(QLabel, "localCastepNote").text()
    assert "Only needed to run calculations on this computer" in note
    assert "Cluster jobs and cluster keyword generation never use these paths" in note
    assert "Setup > HPC Clusters" in note
    assert dialog._executable_edit.parent().parent().title() == "CASTEP on this computer (optional)"

    # On this computer there is no scheduler/login-node choice to make.
    assert dialog._generate_target_combo.currentData() == "local"
    assert hint.text().startswith("Runs the CASTEP executable set below on this computer")
    assert dialog._generate_method_combo.isVisible() is False

    dialog._generate_target_combo.setCurrentText("SCARF")
    assert dialog._generate_method_combo.isVisible() is True
    assert dialog._generate_method_combo.isEnabled() is True
    assert dialog._generate_method_combo.currentData() == "scheduler"
    assert hint.text().startswith(
        "Runs CASTEP on SCARF as a short scheduler job, using that cluster's settings"
    )
    assert "The paths below are not used." in hint.text()

    dialog._generate_method_combo.setCurrentText("Login node")
    assert hint.text().startswith(
        "Runs CASTEP on SCARF on its login node, using that cluster's settings"
    )
    assert "Only use this where the site allows CASTEP on its login nodes." in hint.text()

    # addqueue clusters can only generate through the scheduler.
    dialog._generate_target_combo.setCurrentText("Redwood")
    assert dialog._generate_method_combo.currentData() == "scheduler"
    assert dialog._generate_method_combo.isEnabled() is False
    assert hint.text().startswith("Runs CASTEP on Redwood as a short scheduler job")
    dialog.close()


def test_castep_setup_dialog_login_node_choice_is_used_and_remembered(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _FakeSettings()
    dialog = CastepSetupDialog(settings=settings)
    dialog._generate_target_combo.setCurrentText("SCARF")
    dialog._generate_method_combo.setCurrentText("Login node")

    started: list[str] = []
    monkeypatch.setattr(dialog_module, "validate_cluster_keyword_config", lambda _config: [])
    monkeypatch.setattr(
        dialog_module, "prompt_for_cluster_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(dialog_module.QMessageBox, "warning", lambda *_args, **_kwargs: None)
    real_validate = dialog_module.validate_cluster_keyword_config

    def record(config):
        started.append(config.keyword_execution_mode)
        return real_validate(config)

    monkeypatch.setattr(dialog_module, "validate_cluster_keyword_config", record)

    dialog._generate_keywords()

    # Generation runs with the chosen method, and the cluster keeps it.
    assert started == ["direct"]
    saved = {config.name: config for config in dialog_module.load_cluster_configs(settings)}
    assert saved["SCARF"].keyword_execution_mode == "direct"
    assert saved["Redwood"].keyword_execution_mode == "direct"  # stored preset value, untouched
    dialog.close()

    reopened = CastepSetupDialog(settings=settings)
    reopened._generate_target_combo.setCurrentText("SCARF")
    assert reopened._generate_method_combo.currentData() == "direct"

    # Choosing the stored method again writes nothing.
    monkeypatch.setattr(
        dialog_module,
        "save_cluster_configs",
        lambda *_args, **_kwargs: pytest.fail("unchanged method must not be re-saved"),
    )
    reopened._generate_keywords()
    reopened.close()
