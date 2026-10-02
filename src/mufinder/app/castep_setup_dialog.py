"""Dialog for configuring CASTEP integration paths and keyword generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil
import sys
import os
import tempfile

from PySide6.QtCore import QProcess, QSettings, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cluster_connection_check import ClusterConnectionCheck
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    make_message_box_resizable,
    prompt_for_cluster_credentials,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.app.keyword_catalog_notice import catalog_source_description
from mufinder.cluster_setup import ClusterConfig, load_cluster_configs, save_cluster_configs
from mufinder.castep_setup import (
    CastepSetup,
    apply_castep_environment,
    build_castep_keyword_command,
    discover_mpirun,
    install_keyword_catalog,
    load_castep_setup,
    save_castep_setup,
)
from mufinder.castep_io.keyword_catalog_file import KeywordCatalogValidationError
from mufinder.castep_io.schema import keyword_catalog_status, refresh_keyword_catalog
from mufinder.services.cluster_keyword_generator import (
    ClusterKeywordGenerationWorker,
    validate_cluster_keyword_config,
)
from mufinder.services.ssh_client import SSHConnection, SSHCredentials, SSHUnknownHostKeyError, save_trusted_host_key


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


def _cluster_is_set_up(config: ClusterConfig) -> bool:
    """Return whether ``config`` looks ready to generate keywords from.

    The built-in cluster presets ship with the placeholder username
    ``username``; until the user replaces it the preset is only a template.
    """

    if config.username.strip().lower() in {"", "username"}:
        return False
    return not validate_cluster_keyword_config(config)


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
        self._generation_scratch_dir: Path | None = None
        self._generation_log = ""
        self._text_field_min_height = 34
        self._path_field_min_width = 740

        self.setWindowTitle("CASTEP Setup")
        self.resize(720, 420)

        root = QVBoxLayout(self)

        # The keyword catalog comes first: it is the part every user needs,
        # whereas a CASTEP installation on this computer is optional.
        catalog_group = QGroupBox("Keyword catalog", self)
        catalog_layout = QVBoxLayout(catalog_group)

        intro = QLabel(
            "MuFinder needs a keyword catalog (castep_keywords.json) listing the keywords your CASTEP "
            "version understands. Without one, fewer parameters are offered and some results cannot be "
            "loaded. Generate it from a cluster or from CASTEP on this computer, or import a file you "
            "already have; MuFinder keeps its own copy, so there is no location to choose."
        )
        intro.setWordWrap(True)
        catalog_layout.addWidget(intro)

        self._catalog_status_label = QLabel(catalog_group)
        self._catalog_status_label.setWordWrap(True)
        self._catalog_status_label.setObjectName("catalogStatusLabel")
        status_font = self._catalog_status_label.font()
        status_font.setBold(True)
        self._catalog_status_label.setFont(status_font)
        catalog_layout.addWidget(self._catalog_status_label)

        generate_row = QHBoxLayout()
        generate_row.addWidget(QLabel("Generate from", catalog_group))
        self._generate_target_combo = QComboBox(catalog_group)
        self._generate_target_combo.addItem("This computer", "local")
        for cluster_name in self._cluster_configs:
            self._generate_target_combo.addItem(cluster_name, cluster_name)
        if not self._setup.executable_path:
            # Most users only have CASTEP on a cluster. Without a local
            # executable, preselect the first cluster that is set up well
            # enough to generate from.
            for cluster_name, config in self._cluster_configs.items():
                if _cluster_is_set_up(config):
                    self._generate_target_combo.setCurrentIndex(
                        self._generate_target_combo.findData(cluster_name)
                    )
                    break
        generate_row.addWidget(self._generate_target_combo)
        # How a cluster runs the queries. The same choice is offered in Setup >
        # HPC Clusters; it is repeated here because this is where the user
        # meets it.
        self._generate_method_label = QLabel("using", catalog_group)
        generate_row.addWidget(self._generate_method_label)
        self._generate_method_combo = QComboBox(catalog_group)
        self._generate_method_combo.addItem("Scheduler job (recommended)", "scheduler")
        self._generate_method_combo.addItem("Login node", "direct")
        self._generate_method_combo.setToolTip(
            "A scheduler job is safe where running CASTEP on the login node is forbidden or discouraged. "
            "The login node is quicker when it is permitted, because nothing waits in a queue. "
            "Your choice is remembered for this cluster."
        )
        generate_row.addWidget(self._generate_method_combo)
        self._generate_button = QPushButton("Generate Keyword Catalog", catalog_group)
        self._generate_button.clicked.connect(self._generate_keywords)
        generate_row.addWidget(self._generate_button)
        self._import_button = QPushButton("Use Existing File…", catalog_group)
        self._import_button.setToolTip(
            "Import a castep_keywords.json you already have. MuFinder checks it and keeps its own copy."
        )
        self._import_button.clicked.connect(self._import_keywords_json)
        generate_row.addWidget(self._import_button)
        self._cancel_generation_button = QPushButton("Cancel", catalog_group)
        self._cancel_generation_button.clicked.connect(self._cancel_generation)
        self._cancel_generation_button.setVisible(False)
        generate_row.addWidget(self._cancel_generation_button)
        generate_row.addStretch(1)
        catalog_layout.addLayout(generate_row)

        self._generate_target_hint = QLabel(catalog_group)
        self._generate_target_hint.setWordWrap(True)
        self._generate_target_hint.setObjectName("generateTargetHint")
        catalog_layout.addWidget(self._generate_target_hint)

        self._progress_label = QLabel("Idle", catalog_group)
        catalog_layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar(catalog_group)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        catalog_layout.addWidget(self._progress_bar)

        root.addWidget(catalog_group)

        local_group = QGroupBox("CASTEP on this computer (optional)", self)
        local_layout = QVBoxLayout(local_group)
        local_note = QLabel(
            "Only needed to run calculations on this computer, or to generate the keyword catalog from "
            "\"This computer\". Cluster jobs and cluster keyword generation never use these paths: each "
            "cluster's own CASTEP executable is set in Setup > HPC Clusters."
        )
        local_note.setWordWrap(True)
        local_note.setObjectName("localCastepNote")
        local_layout.addWidget(local_note)

        form = QFormLayout()

        executable_row = QWidget(local_group)
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

        mpirun_row = QWidget(local_group)
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

        local_layout.addLayout(form)
        root.addWidget(local_group)

        self._generate_target_combo.currentIndexChanged.connect(lambda *_: self._on_generate_target_changed())
        self._generate_method_combo.currentIndexChanged.connect(lambda *_: self._refresh_generate_target_hint())
        self._on_generate_target_changed()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._button_box = buttons
        buttons.accepted.connect(self._accept_and_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_catalog_status()

    def _selected_cluster_config(self) -> ClusterConfig | None:
        target_key = str(self._generate_target_combo.currentData() or "local")
        return self._cluster_configs.get(target_key)

    def _on_generate_target_changed(self) -> None:
        """Show the scheduler/login-node choice for cluster targets."""

        config = self._selected_cluster_config()
        is_cluster = config is not None
        self._generate_method_label.setVisible(is_cluster)
        self._generate_method_combo.setVisible(is_cluster)
        if config is not None:
            # addqueue clusters can only generate through the scheduler.
            forced = config.queue_system.strip().lower() == "addqueue"
            mode = "scheduler" if forced else config.keyword_execution_mode.strip().lower()
            index = self._generate_method_combo.findData(mode)
            self._generate_method_combo.blockSignals(True)
            self._generate_method_combo.setCurrentIndex(index if index >= 0 else 0)
            self._generate_method_combo.blockSignals(False)
            self._generate_method_combo.setEnabled(not forced)
        self._refresh_generate_target_hint()

    def _cluster_config_with_selected_method(self, config: ClusterConfig) -> ClusterConfig:
        """Return ``config`` with the method chosen here, remembering a change."""

        mode = str(self._generate_method_combo.currentData() or "scheduler")
        if config.queue_system.strip().lower() == "addqueue" or mode == config.keyword_execution_mode:
            return config
        updated = replace(config, keyword_execution_mode=mode)
        self._cluster_configs[config.name] = updated
        save_cluster_configs(self._settings, self._cluster_configs.values())
        return updated

    def _refresh_generate_target_hint(self) -> None:
        """Say which settings the selected generation target will use."""

        config = self._selected_cluster_config()
        if config is None:
            self._generate_target_hint.setText(
                "Runs the CASTEP executable set below on this computer, through the MPI launcher if one "
                "is set. Takes a minute or two."
            )
            return
        if str(self._generate_method_combo.currentData()) == "direct":
            how = "on its login node"
            timing = "Takes a minute or two. Only use this where the site allows CASTEP on its login nodes."
        else:
            how = "as a short scheduler job"
            timing = "Takes a few minutes, plus any time in the queue."
        self._generate_target_hint.setText(
            f"Runs CASTEP on {config.name} {how}, using that cluster's settings from Setup > HPC Clusters. "
            f"The paths below are not used. {timing}"
        )

    def _refresh_catalog_status(self) -> None:
        status = keyword_catalog_status()
        parts = [catalog_source_description(status)]
        if status.issue:
            parts.append(status.issue)
        self._catalog_status_label.setText("\n".join(parts))

    def setup(self) -> CastepSetup:
        """Return the currently edited setup values."""

        return CastepSetup(
            executable_path=self._executable_edit.text().strip(),
            keywords_json_path=load_castep_setup(self._settings).keywords_json_path,
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

    def _import_keywords_json(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select an existing CASTEP keyword catalog",
            self._start_dir(self._executable_edit.text()),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._install_catalog(Path(path), action="imported")

    def _install_catalog(self, source: Path, *, action: str) -> bool:
        """Validate ``source``, adopt it as MuFinder's catalog, and report the result."""

        try:
            summary = install_keyword_catalog(source, self._settings)
        except KeywordCatalogValidationError as exc:
            QMessageBox.warning(
                self,
                "Keyword catalog not usable",
                f"{source.name} was not installed, and the current keyword catalog is unchanged.\n\n{exc}",
            )
            return False
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Keyword catalog not saved",
                f"MuFinder could not save its copy of the keyword catalog.\n\n{exc}",
            )
            return False

        self._refresh_catalog_status()
        message = f"Keyword catalog {action}: {summary.describe()}."
        status = keyword_catalog_status()
        if status.origin.startswith("env_json:"):
            message += (
                "\n\nIt is not in use yet: the "
                f"{status.origin.split(':', maxsplit=1)[1]} environment variable overrides CASTEP setup."
            )
        QMessageBox.information(self, "Keyword catalog installed", message)
        return True

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
            self._start_generation_process(executable, self._new_generation_output_path())
            return

        config = self._cluster_configs.get(target_key)
        if config is None:
            QMessageBox.warning(self, "Cluster not found", "The selected cluster is no longer configured.")
            return
        config = self._cluster_config_with_selected_method(config)

        issues = validate_cluster_keyword_config(config)
        if issues:
            QMessageBox.warning(
                self,
                "Incomplete cluster setup",
                "\n".join(f"- {issue}" for issue in issues),
            )
            return

        output_path = self._new_generation_output_path()

        credentials = prompt_for_cluster_credentials(self, config)
        if credentials is None:
            QMessageBox.warning(self, "Missing credentials", "Credentials are required for this cluster.")
            return
        # Verify connectivity and host-key trust on a worker thread before
        # starting generation: the inline paramiko connect used to freeze the
        # GUI for over a minute against an unreachable cluster. The factory
        # and trust prompt are passed from this module's namespace so tests
        # can monkeypatch them.
        check = ClusterConnectionCheck(
            self,
            config,
            credentials,
            self._settings,
            title="Checking cluster connection",
            connection_factory=SSHConnection,
            trust_prompt=lambda parent, cfg, exc: prompt_to_trust_host_key(parent, cfg, exc),
        )

        def _on_succeeded() -> None:
            self._start_cluster_generation(config, output_path, credentials=credentials)

        def _on_failed(exc: object) -> None:
            show_detailed_warning(
                self,
                "Cluster connection failed",
                format_cluster_connection_error(config, exc),
                informative_text="Open Details for the full SSH connection error and recovery guidance.",
            )

        check.succeeded.connect(_on_succeeded)
        check.failed.connect(_on_failed)
        check.succeeded.connect(check.deleteLater)
        check.failed.connect(check.deleteLater)
        check.cancelled.connect(check.deleteLater)
        check.start()

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
        refresh_keyword_catalog(allow_fallback=True)
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

    def _new_generation_output_path(self) -> Path:
        """Return a scratch path for a freshly generated catalog.

        Generation never writes over the catalog in use: the result lands in a
        temporary directory and only replaces the installed copy once it has
        been validated.
        """

        self._discard_generation_scratch()
        self._generation_scratch_dir = Path(tempfile.mkdtemp(prefix="mufinder-keywords-"))
        return self._generation_scratch_dir / "castep_keywords.json"

    def _discard_generation_scratch(self) -> None:
        scratch = self._generation_scratch_dir
        self._generation_scratch_dir = None
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)

    def _start_generation_process(self, executable: str, output_path: Path) -> None:
        if self._generation_in_progress:
            return

        self._generation_cancelled = False
        self._generation_output_path = output_path.expanduser()
        self._generation_log = ""
        self._set_generation_running(True)
        self._progress_label.setText("Starting CASTEP keyword extraction...")
        self._progress_bar.setValue(0)

        # MPI-built CASTEP binaries hang when executed directly, so wrap the
        # executable in the configured MPI launcher (`mpirun -np 1`) for the
        # `-help` queries ASE runs. Without this the process stalls at 0%.
        castep_command = build_castep_keyword_command(executable, self._mpirun_edit.text().strip())

        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._on_generation_output)
        process.readyReadStandardError.connect(self._on_generation_output)
        process.finished.connect(self._on_generation_finished)
        process.errorOccurred.connect(self._on_generation_error)
        self._generation_process = process
        process.start(
            sys.executable,
            [
                "-u",
                "-c",
                _GENERATOR_SCRIPT,
                castep_command,
                self._generation_output_path.name,
                str(self._generation_output_path.parent),
            ],
        )

    def _start_cluster_generation(
        self,
        config: ClusterConfig,
        output_path: Path,
        *,
        credentials: SSHCredentials,
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
            credentials=credentials,
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
        self._generate_method_combo.setEnabled(
            not running
            and getattr(self._selected_cluster_config(), "queue_system", "").strip().lower() != "addqueue"
        )
        self._browse_executable_button.setEnabled(not running)
        self._import_button.setEnabled(not running)
        self._browse_mpirun_button.setEnabled(not running)
        self._detect_mpirun_button.setEnabled(not running)
        self._executable_edit.setEnabled(not running)
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
            self._discard_generation_scratch()
            self._progress_label.setText("CASTEP keyword generation cancelled.")
            self._progress_bar.setValue(0)
            return

        if exit_code != 0 or output_path is None or not output_path.exists():
            self._show_generation_error(
                self._generation_log.strip() or "CASTEP keyword generation did not complete successfully."
            )
            self._progress_label.setText("Generation failed.")
            self._discard_generation_scratch()
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
        self._discard_generation_scratch()
        self._progress_label.setText("Generation failed.")
        self._show_generation_error(message.strip() or "CASTEP keyword generation did not complete successfully.")

    def _on_cluster_generation_cancelled(self) -> None:
        self._set_generation_running(False)
        self._generation_worker = None
        self._discard_generation_scratch()
        self._progress_bar.setValue(0)
        self._progress_label.setText("CASTEP keyword generation cancelled.")

    def _on_cluster_generation_thread_finished(self) -> None:
        self._generation_thread = None
        self._generation_worker = None

    def _complete_generation_success(self, output_path: Path) -> None:
        self._set_generation_running(False)
        installed = self._install_catalog(output_path, action="generated")
        self._discard_generation_scratch()
        if installed:
            self._progress_bar.setValue(100)
            self._progress_label.setText("CASTEP keyword generation complete.")
        else:
            self._progress_bar.setValue(0)
            self._progress_label.setText("Generation failed.")

    def _show_generation_error(self, message: str) -> None:
        text = message.strip() or "CASTEP keyword generation did not complete successfully."
        lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
        summary = lines[0] if lines else text
        if len(summary) > 240:
            summary = f"{summary[:237]}..."

        if len(lines) <= 1 and len(text) <= 240:
            QMessageBox.warning(self, "Generation failed", summary)
            return

        box = make_message_box_resizable(QMessageBox(self))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Generation failed")
        box.setText(summary)
        box.setInformativeText("Open Details to view the full cluster log and traceback.")
        box.setDetailedText(text)
        box.exec()

