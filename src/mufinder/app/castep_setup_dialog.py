"""Dialog for configuring CASTEP integration paths and keyword generation."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import os

from PySide6.QtCore import QProcess, QSettings, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.cluster_setup import ClusterConfig, load_cluster_configs
from mufinder.castep_setup import (
    CastepSetup,
    apply_castep_environment,
    discover_mpirun,
    generate_castep_keywords_json,
    load_castep_setup,
    save_castep_setup,
)
from mufinder.castep_io.schema import keyword_catalog_source_info, refresh_keyword_catalog
from mufinder.services.cluster_keyword_generator import (
    ClusterKeywordGenerationWorker,
    validate_cluster_keyword_config,
)
from mufinder.services.ssh_client import SSHConnection, SSHUnknownHostKeyError, save_trusted_host_key


_PROGRESS_RE = re.compile(r"Processed:\s*\[[# ]*\]\s*(\d+)%")
_GENERATOR_SCRIPT = """
from pathlib import Path
import sys
from ase.calculators.castep import create_castep_keywords

executable = sys.argv[1]
filename = sys.argv[2]
directory = sys.argv[3]
Path(directory).mkdir(parents=True, exist_ok=True)
create_castep_keywords(executable, filename=filename, path=directory, force_write=True)
""".strip()


class CastepSetupDialog(QDialog):
    """Configure CASTEP executable and keyword metadata locations."""

    def __init__(
        self,
        *,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else QSettings("MuFinder", "MuFinder")
        self._setup = load_castep_setup(self._settings)
        self._cluster_configs = {config.name: config for config in load_cluster_configs(self._settings)}
        self._generation_process: QProcess | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: ClusterKeywordGenerationWorker | None = None
        self._generation_cancelled = False
        self._generation_output_path: Path | None = None
        self._generation_log = ""
        self._text_field_min_height = 34
        self._path_field_min_width = 740

        self.setWindowTitle("CASTEP Setup")
        self.resize(720, 320)

        root = QVBoxLayout(self)

        intro = QLabel(
            "Tell MuFinder where CASTEP is installed and where keyword metadata should live. "
            "You can generate a `castep_keywords.json` file directly from CASTEP using ASE."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._workflow_hint = QLabel(
            "Recommended workflow: 1) Set `Keywords JSON` to the destination path, "
            "2) click `Generate Keywords JSON`, 3) click `OK` to save and apply. "
            "If the path is blank, MuFinder will ask where to write the file."
        )
        self._workflow_hint.setWordWrap(True)
        self._workflow_hint.setObjectName("castepSetupWorkflowHint")
        root.addWidget(self._workflow_hint)

        catalog_source = keyword_catalog_source_info()
        if not self._setup.keywords_json_path and catalog_source is not None and not catalog_source.origin.startswith("env_json:"):
            self._fallback_notice = QLabel(
                "No CASTEP keywords JSON is configured explicitly. The current session is using fallback "
                f"metadata discovery from `{catalog_source.path}`."
            )
            self._fallback_notice.setWordWrap(True)
            self._fallback_notice.setObjectName("catalogFallbackNotice")
            root.addWidget(self._fallback_notice)

        form = QFormLayout()

        executable_row = QWidget(self)
        executable_layout = QHBoxLayout(executable_row)
        executable_layout.setContentsMargins(0, 0, 0, 0)
        self._executable_edit = QLineEdit(self._setup.executable_path, executable_row)
        self._executable_edit.setMinimumHeight(self._text_field_min_height)
        self._executable_edit.setMinimumWidth(self._path_field_min_width)
        self._browse_executable_button = QPushButton("Browse…", executable_row)
        self._browse_executable_button.clicked.connect(self._browse_executable)
        executable_layout.addWidget(self._executable_edit, 1)
        executable_layout.addWidget(self._browse_executable_button)
        form.addRow("CASTEP executable", executable_row)

        keywords_row = QWidget(self)
        keywords_layout = QHBoxLayout(keywords_row)
        keywords_layout.setContentsMargins(0, 0, 0, 0)
        self._keywords_edit = QLineEdit(self._setup.keywords_json_path, keywords_row)
        self._keywords_edit.setMinimumHeight(self._text_field_min_height)
        self._keywords_edit.setMinimumWidth(self._path_field_min_width)
        self._browse_keywords_button = QPushButton("Browse…", keywords_row)
        self._browse_keywords_button.clicked.connect(self._browse_keywords_json)
        keywords_layout.addWidget(self._keywords_edit, 1)
        keywords_layout.addWidget(self._browse_keywords_button)
        form.addRow("Keywords JSON", keywords_row)

        mpirun_row = QWidget(self)
        mpirun_layout = QHBoxLayout(mpirun_row)
        mpirun_layout.setContentsMargins(0, 0, 0, 0)
        self._mpirun_edit = QLineEdit(self._setup.mpirun_path, mpirun_row)
        self._mpirun_edit.setMinimumHeight(self._text_field_min_height)
        self._mpirun_edit.setMinimumWidth(self._path_field_min_width)
        self._browse_mpirun_button = QPushButton("Browse…", mpirun_row)
        self._browse_mpirun_button.clicked.connect(self._browse_mpirun)
        self._detect_mpirun_button = QPushButton("Auto-detect", mpirun_row)
        self._detect_mpirun_button.clicked.connect(self._auto_detect_mpirun)
        mpirun_layout.addWidget(self._mpirun_edit, 1)
        mpirun_layout.addWidget(self._browse_mpirun_button)
        mpirun_layout.addWidget(self._detect_mpirun_button)
        form.addRow("MPI launcher (mpirun)", mpirun_row)

        root.addLayout(form)

        generate_row = QHBoxLayout()
        self._generate_target_combo = QComboBox(self)
        self._generate_target_combo.addItem("Local", "local")
        for cluster_name in self._cluster_configs:
            self._generate_target_combo.addItem(cluster_name, cluster_name)
        generate_row.addWidget(self._generate_target_combo)
        self._generate_button = QPushButton("Generate Keywords JSON", self)
        self._generate_button.clicked.connect(self._generate_keywords)
        generate_row.addWidget(self._generate_button)
        self._cancel_generation_button = QPushButton("Cancel", self)
        self._cancel_generation_button.clicked.connect(self._cancel_generation)
        self._cancel_generation_button.setVisible(False)
        generate_row.addWidget(self._cancel_generation_button)
        generate_row.addStretch(1)
        root.addLayout(generate_row)

        self._progress_label = QLabel("Idle", self)
        root.addWidget(self._progress_label)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        note = QLabel(
            "Local generation uses `ase.calculators.castep.create_castep_keywords(...)`. "
            "Cluster generation runs CASTEP help queries over SSH (or a scheduler job, depending on setup). "
            "The output is written to the `Keywords JSON` location above."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._button_box = buttons
        buttons.accepted.connect(self._accept_and_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def setup(self) -> CastepSetup:
        """Return the currently edited setup values."""

        return CastepSetup(
            executable_path=self._executable_edit.text().strip(),
            keywords_json_path=self._keywords_edit.text().strip(),
            mpirun_path=self._mpirun_edit.text().strip(),
        )

    def _browse_executable(self) -> None:
        start_dir = self._start_dir(self._executable_edit.text())
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select CASTEP Executable",
            start_dir,
            "All Files (*)",
        )
        if path:
            self._executable_edit.setText(path)

    def _browse_keywords_json(self) -> None:
        start_dir = self._start_dir(self._keywords_edit.text())
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Select CASTEP Keywords JSON",
            str(Path(start_dir) / "castep_keywords.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._keywords_edit.setText(path)

    def _browse_mpirun(self) -> None:
        start_dir = self._start_dir(self._mpirun_edit.text())
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select MPI launcher",
            start_dir,
            "All Files (*)",
        )
        if path:
            self._mpirun_edit.setText(path)

    def _auto_detect_mpirun(self) -> None:
        discovered = discover_mpirun()
        if not discovered:
            QMessageBox.information(
                self,
                "MPI launcher not found",
                "Could not locate mpirun on this machine. Set the path manually if needed.",
            )
            return
        self._mpirun_edit.setText(discovered)

    def _generate_keywords(self) -> None:
        target_key = str(self._generate_target_combo.currentData() or "local")
        if target_key == "local":
            executable = self._executable_edit.text().strip()
            if not executable:
                QMessageBox.warning(
                    self,
                    "Missing executable",
                    "Set the CASTEP executable path before generating castep_keywords.json.",
                )
                return
            output_path = self._ensure_output_path()
            if output_path is None:
                return
            self._start_generation_process(executable, output_path)
            return

        config = self._cluster_configs.get(target_key)
        if config is None:
            QMessageBox.warning(self, "Cluster not found", "The selected cluster is no longer configured.")
            return

        issues = validate_cluster_keyword_config(config)
        if issues:
            QMessageBox.warning(
                self,
                "Incomplete cluster setup",
                "\n".join(f"- {issue}" for issue in issues),
            )
            return

        output_path = self._ensure_output_path()
        if output_path is None:
            return

        password = self._prompt_for_cluster_secret(config)
        if config.auth_method == "password" and not password:
            QMessageBox.warning(self, "Missing password", "Password is required for this cluster.")
            return
        try:
            self._ensure_cluster_trusted_connection(config, password=password)
        except Exception as exc:  # noqa: BLE001
            show_detailed_warning(
                self,
                "Cluster connection failed",
                format_cluster_connection_error(config, exc),
                informative_text="Open Details for the full SSH connection error and recovery guidance.",
            )
            return
        self._start_cluster_generation(config, output_path, password=password)

    def _accept_and_save(self) -> None:
        if self._generation_in_progress:
            QMessageBox.information(
                self,
                "Generation in progress",
                "Wait for keyword generation to finish or cancel it before closing this dialog.",
            )
            return
        setup = self.setup()
        if setup.mpirun_path:
            mpirun_path = Path(setup.mpirun_path).expanduser()
            if not mpirun_path.exists() or not mpirun_path.is_file() or not os.access(mpirun_path, os.X_OK):
                QMessageBox.warning(
                    self,
                    "Invalid MPI launcher",
                    "The configured mpirun path does not point to an executable file.",
                )
                return
        save_castep_setup(self._settings, setup)
        apply_castep_environment(setup)
        try:
            refresh_keyword_catalog()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Keyword catalog reload failed",
                (
                    "Saved CASTEP setup but could not reload keyword metadata immediately. "
                    f"Details:\n{exc}"
                ),
            )
        self.accept()

    def reject(self) -> None:
        if self._generation_in_progress:
            self._cancel_generation()
            QMessageBox.information(
                self,
                "Cancelling generation",
                "CASTEP keyword generation is being cancelled. Close the dialog once cancellation finishes.",
            )
            return
        super().reject()

    @property
    def _generation_in_progress(self) -> bool:
        return self._generation_process is not None or self._generation_thread is not None

    @staticmethod
    def _start_dir(raw_path: str) -> str:
        if not raw_path.strip():
            return str(Path.home())
        path = Path(raw_path).expanduser()
        if path.exists():
            return str(path if path.is_dir() else path.parent)
        if path.parent.exists():
            return str(path.parent)
        return str(Path.home())

    def _ensure_output_path(self) -> Path | None:
        output_path = self._keywords_edit.text().strip()
        if output_path:
            return Path(output_path)

        executable = self._executable_edit.text().strip()
        suggested, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Choose CASTEP Keywords JSON Output",
            str(Path(self._start_dir(executable)) / "castep_keywords.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if not suggested:
            return None
        self._keywords_edit.setText(suggested)
        return Path(suggested)

    def _prompt_for_cluster_secret(self, config: ClusterConfig) -> str | None:
        if config.auth_method == "password":
            raw_password, accepted = QInputDialog.getText(
                self,
                "Cluster password",
                f"Enter the password for {config.name}:",
                QLineEdit.EchoMode.Password,
            )
            if not accepted:
                return None
            return raw_password.strip()

        raw_passphrase, accepted = QInputDialog.getText(
            self,
            "Key passphrase",
            f"Enter the key passphrase for {config.name} if needed:",
            QLineEdit.EchoMode.Password,
        )
        if not accepted or not raw_passphrase.strip():
            return None
        return raw_passphrase.strip()

    def _start_generation_process(self, executable: str, output_path: Path) -> None:
        if self._generation_in_progress:
            return

        self._generation_cancelled = False
        self._generation_output_path = output_path.expanduser()
        self._generation_log = ""
        self._set_generation_running(True)
        self._progress_label.setText("Starting CASTEP keyword extraction...")
        self._progress_bar.setValue(0)

        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._on_generation_output)
        process.readyReadStandardError.connect(self._on_generation_output)
        process.finished.connect(self._on_generation_finished)
        process.errorOccurred.connect(self._on_generation_error)
        self._generation_process = process
        process.start(
            sys.executable,
            [
                "-c",
                _GENERATOR_SCRIPT,
                executable,
                self._generation_output_path.name,
                str(self._generation_output_path.parent),
            ],
        )

    def _start_cluster_generation(
        self,
        config: ClusterConfig,
        output_path: Path,
        *,
        password: str | None,
    ) -> None:
        if self._generation_in_progress:
            return

        self._generation_cancelled = False
        self._generation_output_path = output_path.expanduser()
        self._generation_log = ""
        self._set_generation_running(True)
        self._progress_label.setText(f"Starting CASTEP keyword extraction on {config.name}...")
        self._progress_bar.setValue(0)

        worker = ClusterKeywordGenerationWorker(
            config=config,
            output_path=self._generation_output_path,
            password=password,
            settings=self._settings,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_changed.connect(self._on_cluster_generation_progress)
        worker.finished.connect(self._on_cluster_generation_finished)
        worker.failed.connect(self._on_cluster_generation_failed)
        worker.cancelled.connect(self._on_cluster_generation_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._on_cluster_generation_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._generation_worker = worker
        self._generation_thread = thread
        thread.start()

    def _set_generation_running(self, running: bool) -> None:
        self._generate_button.setEnabled(not running)
        self._generate_target_combo.setEnabled(not running)
        self._browse_executable_button.setEnabled(not running)
        self._browse_keywords_button.setEnabled(not running)
        self._browse_mpirun_button.setEnabled(not running)
        self._detect_mpirun_button.setEnabled(not running)
        self._executable_edit.setEnabled(not running)
        self._keywords_edit.setEnabled(not running)
        self._mpirun_edit.setEnabled(not running)
        self._cancel_generation_button.setVisible(running)
        self._button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not running)
        self._progress_bar.setVisible(running)

    def _cancel_generation(self) -> None:
        if self._generation_process is None and self._generation_worker is None:
            return
        self._generation_cancelled = True
        self._progress_label.setText("Cancelling CASTEP keyword generation...")
        if self._generation_process is not None:
            self._generation_process.kill()
        if self._generation_worker is not None:
            self._generation_worker.cancel()

    def _on_generation_output(self) -> None:
        if self._generation_process is None:
            return
        stdout = bytes(self._generation_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        stderr = bytes(self._generation_process.readAllStandardError()).decode("utf-8", errors="replace")
        text = stdout + stderr
        if not text:
            return
        self._generation_log += text
        self._update_progress_from_output(text)

    def _update_progress_from_output(self, text: str) -> None:
        normalized = text.replace("\r", "\n")
        for chunk in [part.strip() for part in normalized.split("\n") if part.strip()]:
            match = _PROGRESS_RE.search(chunk)
            if match:
                percent = max(0, min(100, int(match.group(1))))
                self._progress_bar.setValue(percent)
                self._progress_label.setText(f"Generating CASTEP keywords... {percent}%")
            else:
                self._progress_label.setText(chunk)

    def _on_generation_finished(self, exit_code: int, _exit_status) -> None:
        output_path = self._generation_output_path
        self._generation_process = None
        self._set_generation_running(False)

        if self._generation_cancelled:
            self._progress_label.setText("CASTEP keyword generation cancelled.")
            self._progress_bar.setValue(0)
            return

        if exit_code != 0 or output_path is None or not output_path.exists():
            self._show_generation_error(
                self._generation_log.strip() or "CASTEP keyword generation did not complete successfully."
            )
            self._progress_label.setText("Generation failed.")
            return

        self._complete_generation_success(output_path)

    def _on_generation_error(self, _error) -> None:
        if self._generation_process is None or self._generation_cancelled:
            return
        self._generation_log += bytes(self._generation_process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )

    def _on_cluster_generation_progress(self, message: str, percent: object) -> None:
        self._generation_log = f"{self._generation_log}\n{message}".strip()
        self._progress_label.setText(message)
        if isinstance(percent, int):
            self._progress_bar.setValue(max(0, min(100, percent)))

    def _on_cluster_generation_finished(self, output_path: str) -> None:
        resolved = Path(output_path).resolve()
        self._generation_output_path = resolved
        self._complete_generation_success(resolved)

    def _on_cluster_generation_failed(self, message: str) -> None:
        self._set_generation_running(False)
        self._generation_worker = None
        self._progress_label.setText("Generation failed.")
        self._show_generation_error(message.strip() or "CASTEP keyword generation did not complete successfully.")

    def _on_cluster_generation_cancelled(self) -> None:
        self._set_generation_running(False)
        self._generation_worker = None
        self._progress_bar.setValue(0)
        self._progress_label.setText("CASTEP keyword generation cancelled.")

    def _on_cluster_generation_thread_finished(self) -> None:
        self._generation_thread = None
        self._generation_worker = None

    def _complete_generation_success(self, output_path: Path) -> None:
        self._set_generation_running(False)
        self._keywords_edit.setText(str(output_path.resolve()))
        self._progress_bar.setValue(100)
        self._progress_label.setText("CASTEP keyword generation complete.")
        QMessageBox.information(
            self,
            "Keywords generated",
            f"Generated CASTEP keywords metadata at:\n{output_path.resolve()}",
        )

    def _show_generation_error(self, message: str) -> None:
        text = message.strip() or "CASTEP keyword generation did not complete successfully."
        lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
        summary = lines[0] if lines else text
        if len(summary) > 240:
            summary = f"{summary[:237]}..."

        if len(lines) <= 1 and len(text) <= 240:
            QMessageBox.warning(self, "Generation failed", summary)
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Generation failed")
        box.setText(summary)
        box.setInformativeText("Open Details to view the full cluster log and traceback.")
        box.setDetailedText(text)
        box.exec()

    def _ensure_cluster_trusted_connection(self, config: ClusterConfig, *, password: str | None) -> None:
        connection = SSHConnection(config, settings=self._settings)
        try:
            connection.connect(password=password)
        except SSHUnknownHostKeyError as exc:
            if not prompt_to_trust_host_key(self, config, exc):
                raise ConnectionError("Connection cancelled because the SSH host key was not trusted.") from exc
            save_trusted_host_key(
                self._settings,
                config,
                algorithm=exc.algorithm,
                key_base64=exc.key_base64,
            )
            connection = SSHConnection(config, settings=self._settings)
            connection.connect(password=password)
        finally:
            if connection.is_connected:
                connection.disconnect()
