"""Helpers for persisted remote cluster configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import QSettings


_CLUSTER_NAMES_KEY = "clusters/names"
_SCARF_TEMPLATE_TOKEN = "builtin:scarf-slurm"
_DEFAULT_CASTEP_EXECUTABLE = "castep.mpi"
_REDWOOD_PARTITION = "redwood"
_REDWOOD_ADDQUEUE = "/bin/addqueue"
_REDWOOD_SQUEUE = "/usr/local/shared/slurm/bin//squeue"
_REDWOOD_SCANCEL = "/usr/local/shared/slurm/bin//scancel"
_REDWOOD_PRESET_NAME = "Redwood"
_DEFAULT_KEYWORD_EXECUTION_MODE = "direct"
_DEFAULT_KEYWORD_PYTHON_COMMAND = "python3"
_SCARF_KEYWORD_SHELL_INIT = "module load CASTEP"


def _default_clusters() -> list["ClusterConfig"]:
    return [
        ClusterConfig(
            name="SCARF",
            hostname="ui1.scarf.rl.ac.uk",
            port=22,
            username="username",
            auth_method="key",
            key_path="",
            default_working_directory="/work4/isis/username",
            queue_system="SLURM",
            cores_per_node=24,
            default_walltime="12:00:00",
            template_script_path=_SCARF_TEMPLATE_TOKEN,
            castep_executable=_DEFAULT_CASTEP_EXECUTABLE,
            default_queue_name="",
            default_memory_gb=None,
            submit_command_path="",
            queue_check_command_path="",
            cancel_command_path="",
            keyword_execution_mode="scheduler",
            keyword_shell_init=_SCARF_KEYWORD_SHELL_INIT,
            keyword_python_command=_DEFAULT_KEYWORD_PYTHON_COMMAND,
        ),
        ClusterConfig(
            name=_REDWOOD_PRESET_NAME,
            hostname="glamdring.physics.ox.ac.uk",
            port=22,
            username="username",
            auth_method="key",
            key_path="",
            default_working_directory="/mnt/redwoodfs/username",
            queue_system="ADDQUEUE",
            cores_per_node=32,
            default_walltime="",
            template_script_path="",
            castep_executable="/mnt/redwoodfs/muon/castep.mpi",
            default_queue_name=_REDWOOD_PARTITION,
            default_memory_gb=7.5,
            submit_command_path=_REDWOOD_ADDQUEUE,
            queue_check_command_path=_REDWOOD_SQUEUE,
            cancel_command_path=_REDWOOD_SCANCEL,
            keyword_execution_mode="direct",
            keyword_shell_init="",
            keyword_python_command=_DEFAULT_KEYWORD_PYTHON_COMMAND,
        )
    ]


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Persisted configuration for one remote HPC cluster."""

    name: str
    hostname: str
    port: int = 22
    username: str = ""
    auth_method: str = "key"
    key_path: str = ""
    default_working_directory: str = ""
    queue_system: str = "slurm"
    cores_per_node: int = 1
    default_walltime: str = "01:00:00"
    template_script_path: str = ""
    castep_executable: str = _DEFAULT_CASTEP_EXECUTABLE
    default_queue_name: str = ""
    default_memory_gb: float | None = None
    submit_command_path: str = ""
    queue_check_command_path: str = ""
    cancel_command_path: str = ""
    keyword_execution_mode: str = _DEFAULT_KEYWORD_EXECUTION_MODE
    keyword_shell_init: str = ""
    keyword_python_command: str = _DEFAULT_KEYWORD_PYTHON_COMMAND


def _builtin_cluster_by_name(name: str) -> ClusterConfig | None:
    target = name.strip().casefold()
    if not target:
        return None
    for config in _default_clusters():
        if config.name.casefold() == target:
            return config
    return None


def _ensure_redwood_preset(configs: list[ClusterConfig]) -> list[ClusterConfig]:
    if any(config.name.strip().casefold() == _REDWOOD_PRESET_NAME.casefold() for config in configs):
        return configs
    redwood = _builtin_cluster_by_name(_REDWOOD_PRESET_NAME)
    if redwood is None:
        return configs
    return [*configs, redwood]


def _cluster_prefix(name: str) -> str:
    return f"clusters/{name}"


def _normalise_cluster_names(value: object) -> list[str]:
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, (list, tuple)):
        names = [str(entry) for entry in value]
    else:
        names = []

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _cluster_from_settings(settings: QSettings, name: str) -> ClusterConfig | None:
    prefix = _cluster_prefix(name)
    builtin = _builtin_cluster_by_name(name)
    hostname = str(settings.value(f"{prefix}/hostname", "", str) or "").strip()
    if not hostname:
        return None

    username = str(settings.value(f"{prefix}/username", "", str) or "").strip()
    auth_method = str(settings.value(f"{prefix}/auth_method", "key", str) or "key").strip().lower()
    if auth_method not in {"key", "password"}:
        auth_method = "key"

    try:
        port = int(settings.value(f"{prefix}/port", 22) or 22)
    except (TypeError, ValueError):
        port = 22

    queue_system = str(settings.value(f"{prefix}/queue_system", "SLURM", str) or "SLURM").strip()
    if not queue_system:
        queue_system = "SLURM"

    try:
        cores_per_node = int(settings.value(f"{prefix}/cores_per_node", 1) or 1)
    except (TypeError, ValueError):
        cores_per_node = 1

    raw_walltime = settings.value(f"{prefix}/default_walltime", None, str)
    if raw_walltime is None:
        default_walltime = "" if queue_system.strip().lower() == "addqueue" else "01:00:00"
    else:
        default_walltime = str(raw_walltime).strip()
    if queue_system.strip().lower() != "addqueue" and not default_walltime:
        default_walltime = "01:00:00"

    default_memory_raw = settings.value(f"{prefix}/default_memory_gb", None)
    if default_memory_raw in {None, ""}:
        default_memory_gb: float | None = 7.5 if queue_system.strip().lower() == "addqueue" else None
    else:
        try:
            parsed_memory = float(default_memory_raw)
        except (TypeError, ValueError):
            parsed_memory = 7.5 if queue_system.strip().lower() == "addqueue" else 0.0
        default_memory_gb = parsed_memory if parsed_memory > 0 else None
        if queue_system.strip().lower() == "addqueue" and default_memory_gb is None:
            default_memory_gb = 7.5

    default_queue_name = str(settings.value(f"{prefix}/default_queue_name", "", str) or "").strip()
    if queue_system.strip().lower() == "addqueue" and not default_queue_name:
        default_queue_name = _REDWOOD_PARTITION

    keyword_execution_mode = str(
        settings.value(
            f"{prefix}/keyword_execution_mode",
            builtin.keyword_execution_mode if builtin is not None else _DEFAULT_KEYWORD_EXECUTION_MODE,
            str,
        )
        or _DEFAULT_KEYWORD_EXECUTION_MODE
    ).strip().lower()
    if keyword_execution_mode not in {"direct", "scheduler"}:
        keyword_execution_mode = (
            builtin.keyword_execution_mode if builtin is not None else _DEFAULT_KEYWORD_EXECUTION_MODE
        )
    keyword_shell_init = str(
        settings.value(
            f"{prefix}/keyword_shell_init",
            builtin.keyword_shell_init if builtin is not None else "",
            str,
        )
        or ""
    )

    keyword_python_command = str(
        settings.value(
            f"{prefix}/keyword_python_command",
            builtin.keyword_python_command if builtin is not None else _DEFAULT_KEYWORD_PYTHON_COMMAND,
            str,
        )
        or _DEFAULT_KEYWORD_PYTHON_COMMAND
    ).strip()
    if not keyword_python_command:
        keyword_python_command = _DEFAULT_KEYWORD_PYTHON_COMMAND

    return ClusterConfig(
        name=name,
        hostname=hostname,
        port=max(1, port),
        username=username,
        auth_method=auth_method,
        key_path=str(settings.value(f"{prefix}/key_path", "", str) or "").strip(),
        default_working_directory=str(
            settings.value(f"{prefix}/default_working_directory", "", str) or ""
        ).strip(),
        queue_system=queue_system,
        cores_per_node=max(1, cores_per_node),
        default_walltime=default_walltime,
        template_script_path=str(settings.value(f"{prefix}/template_script_path", "", str) or "").strip(),
        castep_executable=str(
            settings.value(f"{prefix}/castep_executable", _DEFAULT_CASTEP_EXECUTABLE, str)
            or _DEFAULT_CASTEP_EXECUTABLE
        ).strip()
        or _DEFAULT_CASTEP_EXECUTABLE,
        default_queue_name=default_queue_name,
        default_memory_gb=default_memory_gb,
        submit_command_path=str(settings.value(f"{prefix}/submit_command_path", "", str) or "").strip(),
        queue_check_command_path=str(
            settings.value(f"{prefix}/queue_check_command_path", "", str) or ""
        ).strip(),
        cancel_command_path=str(settings.value(f"{prefix}/cancel_command_path", "", str) or "").strip(),
        keyword_execution_mode=keyword_execution_mode,
        keyword_shell_init=keyword_shell_init,
        keyword_python_command=keyword_python_command,
    )


def load_cluster_configs(settings: QSettings) -> list[ClusterConfig]:
    """Load all configured clusters from QSettings."""

    names = _normalise_cluster_names(settings.value(_CLUSTER_NAMES_KEY, []))
    configs: list[ClusterConfig] = []
    for name in names:
        config = _cluster_from_settings(settings, name)
        if config is not None:
            configs.append(config)
    if not configs:
        return _default_clusters()
    return _ensure_redwood_preset(configs)


def load_cluster_config(settings: QSettings, name: str) -> ClusterConfig | None:
    """Load one configured cluster by name."""

    target = name.strip()
    if not target:
        return None

    names = _normalise_cluster_names(settings.value(_CLUSTER_NAMES_KEY, []))
    if not names:
        for config in _default_clusters():
            if config.name.casefold() == target.casefold():
                return config

    name_lookup = {entry.casefold(): entry for entry in names}
    stored_name = name_lookup.get(target.casefold())
    if stored_name is not None:
        return _cluster_from_settings(settings, stored_name)

    builtin = _builtin_cluster_by_name(target)
    if builtin is not None:
        return builtin
    return None


def save_cluster_configs(settings: QSettings, configs: Iterable[ClusterConfig]) -> None:
    """Persist cluster configurations to QSettings."""

    cleaned: list[ClusterConfig] = []
    seen: set[str] = set()
    for config in configs:
        name = config.name.strip()
        hostname = config.hostname.strip()
        if not name or not hostname or name in seen:
            continue
        seen.add(name)
        queue_system = config.queue_system.strip() or "SLURM"
        default_walltime = config.default_walltime.strip()
        if queue_system.lower() != "addqueue" and not default_walltime:
            default_walltime = "01:00:00"
        if config.default_memory_gb is None:
            default_memory_gb: float | None = 7.5 if queue_system.lower() == "addqueue" else None
        else:
            try:
                parsed_memory = float(config.default_memory_gb)
            except (TypeError, ValueError):
                parsed_memory = 7.5 if queue_system.lower() == "addqueue" else 0.0
            default_memory_gb = parsed_memory if parsed_memory > 0 else None
            if queue_system.lower() == "addqueue" and default_memory_gb is None:
                default_memory_gb = 7.5
        cleaned.append(
            ClusterConfig(
                name=name,
                hostname=hostname,
                port=max(1, int(config.port)),
                username=config.username.strip(),
                auth_method=(config.auth_method.strip().lower() or "key"),
                key_path=config.key_path.strip(),
                default_working_directory=config.default_working_directory.strip(),
                queue_system=queue_system,
                cores_per_node=max(1, int(config.cores_per_node)),
                default_walltime=default_walltime,
                template_script_path=config.template_script_path.strip(),
                castep_executable=config.castep_executable.strip() or _DEFAULT_CASTEP_EXECUTABLE,
                default_queue_name=config.default_queue_name.strip(),
                default_memory_gb=default_memory_gb,
                submit_command_path=config.submit_command_path.strip(),
                queue_check_command_path=config.queue_check_command_path.strip(),
                cancel_command_path=config.cancel_command_path.strip(),
                keyword_execution_mode=config.keyword_execution_mode.strip().lower() or _DEFAULT_KEYWORD_EXECUTION_MODE,
                keyword_shell_init=config.keyword_shell_init.rstrip(),
                keyword_python_command=(
                    config.keyword_python_command.strip() or _DEFAULT_KEYWORD_PYTHON_COMMAND
                ),
            )
        )

    settings.setValue(_CLUSTER_NAMES_KEY, [config.name for config in cleaned])

    for config in cleaned:
        prefix = _cluster_prefix(config.name)
        settings.setValue(f"{prefix}/hostname", config.hostname)
        settings.setValue(f"{prefix}/port", int(config.port))
        settings.setValue(f"{prefix}/username", config.username)
        settings.setValue(f"{prefix}/auth_method", config.auth_method)
        settings.setValue(f"{prefix}/key_path", config.key_path)
        settings.setValue(f"{prefix}/default_working_directory", config.default_working_directory)
        settings.setValue(f"{prefix}/queue_system", config.queue_system)
        settings.setValue(f"{prefix}/cores_per_node", int(config.cores_per_node))
        settings.setValue(f"{prefix}/default_walltime", config.default_walltime)
        settings.setValue(f"{prefix}/template_script_path", config.template_script_path)
        settings.setValue(f"{prefix}/castep_executable", config.castep_executable)
        settings.setValue(f"{prefix}/default_queue_name", config.default_queue_name)
        settings.setValue(
            f"{prefix}/default_memory_gb",
            "" if config.default_memory_gb is None else float(config.default_memory_gb),
        )
        settings.setValue(f"{prefix}/submit_command_path", config.submit_command_path)
        settings.setValue(f"{prefix}/queue_check_command_path", config.queue_check_command_path)
        settings.setValue(f"{prefix}/cancel_command_path", config.cancel_command_path)
        settings.setValue(f"{prefix}/keyword_execution_mode", config.keyword_execution_mode)
        settings.setValue(f"{prefix}/keyword_shell_init", config.keyword_shell_init)
        settings.setValue(f"{prefix}/keyword_python_command", config.keyword_python_command)
