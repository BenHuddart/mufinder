"""Tests for CASTEP setup dialog behavior."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

import pytest
from PySide6.QtWidgets import QApplication, QLabel

import mufinder.app.castep_setup_dialog as dialog_module
from mufinder.castep_setup import (
    CASTEP_EXECUTABLE_KEY,
    CASTEP_MPIRUN_KEY,
    CASTEP_KEYWORDS_JSON_KEY,
    CastepSetup,
    apply_castep_environment,
    apply_saved_castep_environment,
    discover_mpirun,
    generate_castep_keywords_json,
    load_castep_setup,
    save_castep_setup,
)
from mufinder.app.castep_setup_dialog import CastepSetupDialog
from mufinder.services.ssh_client import SSHCredentials


def _process_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    """Pump the GUI event loop until ``predicate`` holds or ``timeout`` elapses.

    The pre-generation cluster connection check runs on a worker thread, so
    its outcome arrives via queued signals that only deliver while the event
    loop spins.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


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
    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    apply_castep_environment(setup)

    assert "CASTEP_COMMAND" not in os.environ
    assert os.environ["MUFINDER_CASTEP_KEYWORDS_JSON"] == setup.keywords_json_path


def test_apply_saved_castep_environment_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _FakeSettings()
    settings.setValue(CASTEP_EXECUTABLE_KEY, "/usr/local/bin/castep")
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, "/tmp/generated/castep_keywords.json")
    settings.setValue(CASTEP_MPIRUN_KEY, "/opt/mpi/bin/mpirun")

    monkeypatch.delenv("CASTEP_COMMAND", raising=False)
    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)

    setup = apply_saved_castep_environment(settings)

    assert setup.executable_path == "/usr/local/bin/castep"
    assert setup.mpirun_path == "/opt/mpi/bin/mpirun"
    assert "CASTEP_COMMAND" not in os.environ
    assert os.environ["MUFINDER_CASTEP_KEYWORDS_JSON"] == "/tmp/generated/castep_keywords.json"


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

    def fake_create(command: str, filename: str, force_write: bool, path: str, fetch_only=None) -> None:
        calls.append((command, filename, path, force_write))
        (Path(path) / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("ase.calculators.castep.create_castep_keywords", fake_create)

    output = generate_castep_keywords_json(
        executable_path="/opt/castep/bin/castep",
        output_path=tmp_path / "castep_keywords.json",
    )

    assert calls == [("/opt/castep/bin/castep", "castep_keywords.json", str(tmp_path), True)]
    assert output == (tmp_path / "castep_keywords.json").resolve()


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
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", "/tmp/old.json")

    apply_castep_environment(CastepSetup(keywords_json_path="   "))

    assert "MUFINDER_CASTEP_KEYWORDS_JSON" not in os.environ


def test_castep_setup_dialog_generates_keywords_and_saves(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _FakeSettings()
    dialog = CastepSetupDialog(settings=settings)
    generated = tmp_path / "generated" / "castep_keywords.json"

    infos: list[str] = []
    refresh_calls: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )
    # Saving setup applies environment variables; keep this side effect isolated to this test.
    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    monkeypatch.setattr(dialog_module, "refresh_keyword_catalog", lambda: refresh_calls.append("refresh"))

    def fake_start_generation_process(executable: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        dialog._generation_output_path = output_path
        dialog._on_generation_finished(0, None)

    monkeypatch.setattr(dialog, "_start_generation_process", fake_start_generation_process)

    dialog._executable_edit.setText("/opt/castep/bin/castep")
    dialog._keywords_edit.setText(str(generated))
    dialog._generation_output_path = generated
    dialog._generate_keywords()
    dialog._accept_and_save()

    assert infos == ["Keywords generated"]
    assert refresh_calls == ["refresh"]
    assert settings.value(CASTEP_EXECUTABLE_KEY, "", str) == "/opt/castep/bin/castep"
    assert settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) == str(generated)
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


def test_castep_setup_dialog_shows_fallback_notice_when_json_not_configured(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dialog_module,
        "keyword_catalog_source_info",
        lambda: type("CatalogSource", (), {"path": Path("/tmp/discovered/castep_keywords.json"), "origin": "auto_discovery"})(),
    )

    dialog = CastepSetupDialog(settings=_FakeSettings())

    assert dialog.findChild(QLabel, "catalogFallbackNotice") is not None
    dialog.close()


def test_castep_setup_dialog_lists_local_and_cluster_targets(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    targets = [dialog._generate_target_combo.itemText(index) for index in range(dialog._generate_target_combo.count())]

    assert targets[:3] == ["Local", "SCARF", "Redwood"]
    dialog.close()


def test_castep_setup_dialog_warns_on_incomplete_cluster_setup(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._generate_target_combo.setCurrentText("Redwood")
    dialog._keywords_edit.setText("/tmp/generated/castep_keywords.json")

    warnings: list[str] = []
    monkeypatch.setattr(dialog_module, "validate_cluster_keyword_config", lambda _config: ["Missing shell init"])
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
    output_path = tmp_path / "castep_keywords.json"
    dialog._keywords_edit.setText(str(output_path))

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

    assert _process_until(lambda: bool(captured))
    assert captured == [("Redwood", output_path, credentials)]
    dialog.close()


def test_castep_setup_dialog_cluster_generation_trusts_unknown_host_key(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())
    dialog._generate_target_combo.setCurrentText("Redwood")
    output_path = tmp_path / "castep_keywords.json"
    dialog._keywords_edit.setText(str(output_path))

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
                raise dialog_module.SSHUnknownHostKeyError(
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
    monkeypatch.setattr(dialog_module, "prompt_to_trust_host_key", lambda *_args, **_kwargs: prompts.append("yes") or True)
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

    assert _process_until(lambda: bool(captured))
    assert prompts == ["yes"]
    assert captured == [("Redwood", output_path, credentials)]
    assert dialog._settings.value(
        "trusted_host_keys/Redwood/glamdring.physics.ox.ac.uk/algorithm", "", str
    ) == "ssh-ed25519"
    dialog.close()


def test_castep_setup_dialog_text_fields_are_taller_by_default(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    assert dialog._executable_edit.minimumHeight() >= 34
    assert dialog._keywords_edit.minimumHeight() >= 34
    assert dialog._mpirun_edit.minimumHeight() >= 34
    assert dialog._executable_edit.minimumWidth() >= 740
    assert dialog._keywords_edit.minimumWidth() >= 740
    assert dialog._mpirun_edit.minimumWidth() >= 740
    dialog.close()


def test_castep_setup_dialog_shows_workflow_guidance(qapp) -> None:
    dialog = CastepSetupDialog(settings=_FakeSettings())

    hint = dialog.findChild(QLabel, "castepSetupWorkflowHint")
    assert hint is not None
    text = hint.text()
    assert "Recommended workflow" in text
    assert "Set `Keywords JSON`" in text
    assert "Generate Keywords JSON" in text
    assert "click `OK`" in text
    dialog.close()
