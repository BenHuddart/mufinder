"""Helpers for persisted remote cluster configuration."""

from __future__ import annotations

import json
from typing import Iterable

from PySide6.QtCore import QSettings

# ``ClusterConfig`` lives in ``mufinder.domain.cluster``; this module re-exports
# it as a public import alias (see docs/compatibility_policy.rst).
from mufinder.domain.cluster import (
    DEFAULT_CASTEP_EXECUTABLE as _DEFAULT_CASTEP_EXECUTABLE,
    DEFAULT_KEYWORD_EXECUTION_MODE as _DEFAULT_KEYWORD_EXECUTION_MODE,
    PLACEHOLDER_JUMP_USERNAME,
    PLACEHOLDER_USERNAME,
    USERNAME_TOKEN,
    ClusterConfig,
    ClusterQueue,
)

__all__ = [
    "ClusterConfig",
    "ClusterQueue",
    "load_cluster_config",
    "load_cluster_configs",
    "save_cluster_configs",
]


_CLUSTER_NAMES_KEY = "clusters/names"
_SCARF_TEMPLATE_TOKEN = "builtin:scarf-slurm"
_REDWOOD_PARTITION = "redwood"
_REDWOOD_ADDQUEUE = "/usr/local/shared/bin/addqueue"
_REDWOOD_SQUEUE = "/usr/local/shared/slurm/bin//squeue"
_REDWOOD_SCANCEL = "/usr/local/shared/slurm/bin//scancel"
_REDWOOD_PRESET_NAME = "Redwood"
_REDWOOD_HOSTNAME = "glamdring.physics.ox.ac.uk"
_REDWOOD_JUMP_HOSTNAME = "bastion.physics.ox.ac.uk"
_SCARF_KEYWORD_SHELL_INIT = "module load CASTEP"


def _default_clusters() -> list[ClusterConfig]:
    return [
        ClusterConfig(
            name="SCARF",
            hostname="ui1.scarf.rl.ac.uk",
            port=22,
            username="",
            auth_method="key",
            key_path="",
            default_working_directory=f"/work4/isis/{USERNAME_TOKEN}",
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
        ),
        ClusterConfig(
            name=_REDWOOD_PRESET_NAME,
            hostname=_REDWOOD_HOSTNAME,
            port=22,
            username="",
            auth_method="password",
            key_path="",
            jump_host_enabled=True,
            jump_hostname=_REDWOOD_JUMP_HOSTNAME,
            jump_port=22,
            jump_username="",
            jump_auth_method="password",
            jump_key_path="",
            use_macos_keychain_passwords=False,
            default_working_directory=f"/mnt/redwoodfs/{USERNAME_TOKEN}",
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
        ),
    ]


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


def _working_directory_without_placeholder(directory: str) -> str:
    """Turn the shipped ``username`` path component back into ``{username}``."""

    return "/".join(
        USERNAME_TOKEN if part == PLACEHOLDER_USERNAME else part for part in directory.split("/")
    )


def _cluster_prefix(name: str) -> str:
    return f"clusters/{name}"


def _normalise_mem_per_cpu_gb(value, *, queue_system: str) -> float | None:  # noqa: ANN001
    """Coerce a stored per-CPU memory request to a positive float or ``None``.

    Unlike ``default_memory_gb`` (which addqueue always needs a value for), a
    per-CPU request is optional: ``None`` means "emit no ``#SBATCH`` memory
    directive" and is the default for every cluster. It is a SLURM batch-script
    directive, so it is dropped for any other queue system.
    """

    if queue_system.strip().lower() != "slurm":
        return None
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float(value) -> float | None:  # noqa: ANN001
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalise_queue(queue: ClusterQueue, *, queue_system: str) -> ClusterQueue | None:
    """Return ``queue`` with blank overrides cleared, or ``None`` if it has no name.

    A per-CPU memory override is a SLURM directive and a per-core memory
    override an addqueue argument, so each is dropped for the other scheduler,
    as is a template override for addqueue, which submits without a script.
    """

    name = queue.name.strip()
    if not name:
        return None
    is_slurm = queue_system.strip().lower() == "slurm"
    is_addqueue = queue_system.strip().lower() == "addqueue"
    try:
        cores = int(queue.cores_per_node) if queue.cores_per_node is not None else None
    except (TypeError, ValueError):
        cores = None
    return ClusterQueue(
        name=name,
        default_walltime=str(queue.default_walltime or "").strip(),
        max_walltime=str(queue.max_walltime or "").strip(),
        cores_per_node=cores if cores is not None and cores > 0 else None,
        default_memory_gb=_positive_float(queue.default_memory_gb) if is_addqueue else None,
        default_mem_per_cpu_gb=_positive_float(queue.default_mem_per_cpu_gb) if is_slurm else None,
        castep_executable=str(queue.castep_executable or "").strip(),
        template_script_path=str(queue.template_script_path or "").strip() if is_slurm else "",
    )


def _normalise_queues(
    queues: Iterable[ClusterQueue], *, queue_system: str
) -> tuple[ClusterQueue, ...]:
    cleaned: list[ClusterQueue] = []
    seen: set[str] = set()
    for queue in queues:
        normalised = _normalise_queue(queue, queue_system=queue_system)
        if normalised is None or normalised.name in seen:
            continue
        seen.add(normalised.name)
        cleaned.append(normalised)
    return tuple(cleaned)


def _resolve_default_queue_name(default_queue_name: str, queues: tuple[ClusterQueue, ...]) -> str:
    """Keep the default queue among the listed queues when there are any."""

    name = default_queue_name.strip()
    if not queues or any(queue.name == name for queue in queues):
        return name
    return queues[0].name


def _queues_from_settings(raw: object, *, queue_system: str) -> tuple[ClusterQueue, ...]:
    """Decode the persisted ``queues`` JSON list; anything unreadable is no queues."""

    if raw in {None, ""}:
        return ()
    try:
        entries = json.loads(str(raw))
    except (TypeError, ValueError):
        return ()
    if not isinstance(entries, list):
        return ()
    queues: list[ClusterQueue] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cores_raw = entry.get("cores_per_node")
        try:
            cores = int(cores_raw) if cores_raw not in {None, ""} else None
        except (TypeError, ValueError):
            cores = None
        queues.append(
            ClusterQueue(
                name=str(entry.get("name", "") or ""),
                default_walltime=str(entry.get("default_walltime", "") or ""),
                max_walltime=str(entry.get("max_walltime", "") or ""),
                cores_per_node=cores,
                default_memory_gb=_positive_float(entry.get("default_memory_gb")),
                default_mem_per_cpu_gb=_positive_float(entry.get("default_mem_per_cpu_gb")),
                castep_executable=str(entry.get("castep_executable", "") or ""),
                template_script_path=str(entry.get("template_script_path", "") or ""),
            )
        )
    return _normalise_queues(queues, queue_system=queue_system)


def _queues_to_settings(queues: tuple[ClusterQueue, ...]) -> str:
    """Encode queues as a JSON list, leaving out overrides that are not set."""

    entries: list[dict[str, object]] = []
    for queue in queues:
        entry: dict[str, object] = {"name": queue.name}
        if queue.default_walltime:
            entry["default_walltime"] = queue.default_walltime
        if queue.max_walltime:
            entry["max_walltime"] = queue.max_walltime
        if queue.cores_per_node is not None:
            entry["cores_per_node"] = int(queue.cores_per_node)
        if queue.default_memory_gb is not None:
            entry["default_memory_gb"] = float(queue.default_memory_gb)
        if queue.default_mem_per_cpu_gb is not None:
            entry["default_mem_per_cpu_gb"] = float(queue.default_mem_per_cpu_gb)
        if queue.castep_executable:
            entry["castep_executable"] = queue.castep_executable
        if queue.template_script_path:
            entry["template_script_path"] = queue.template_script_path
        entries.append(entry)
    return json.dumps(entries) if entries else ""


def _settings_value(settings: QSettings, key: str, default=None):  # noqa: ANN001
    value = settings.value(key, None)
    return default if value is None else value


def _settings_text(settings: QSettings, key: str, default: str = "") -> str:
    return str(_settings_value(settings, key, default) or "").strip()


def _settings_bool(settings: QSettings, key: str, default: bool = False) -> bool:
    value = _settings_value(settings, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _normalise_auth_method(value: str, *, default: str = "key") -> str:
    method = value.strip().lower()
    if method in {"key", "password"}:
        return method
    return default if default in {"key", "password"} else "key"


def _should_inherit_builtin_jump_defaults(builtin: ClusterConfig | None, hostname: str) -> bool:
    if builtin is None or not builtin.jump_host_enabled:
        return False
    return hostname.strip().casefold() == builtin.hostname.strip().casefold()


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
    hostname = _settings_text(settings, f"{prefix}/hostname")
    if not hostname:
        return None

    # A profile saved with the name an earlier preset shipped never had a real
    # username, so its directory held the placeholder too.
    username = _settings_text(settings, f"{prefix}/username")
    default_working_directory = str(
        settings.value(f"{prefix}/default_working_directory", "", str) or ""
    ).strip()
    if username == PLACEHOLDER_USERNAME:
        username = ""
        default_working_directory = _working_directory_without_placeholder(
            default_working_directory
        )
    auth_default = builtin.auth_method if builtin is not None else "key"
    auth_method = _normalise_auth_method(
        str(_settings_value(settings, f"{prefix}/auth_method", auth_default) or auth_default),
        default=auth_default,
    )

    try:
        port = int(_settings_value(settings, f"{prefix}/port", 22) or 22)
    except (TypeError, ValueError):
        port = 22

    jump_defaults = builtin if _should_inherit_builtin_jump_defaults(builtin, hostname) else None
    jump_host_enabled = _settings_bool(
        settings,
        f"{prefix}/jump_host_enabled",
        jump_defaults.jump_host_enabled if jump_defaults is not None else False,
    )
    try:
        jump_port = int(
            _settings_value(
                settings,
                f"{prefix}/jump_port",
                jump_defaults.jump_port if jump_defaults is not None else 22,
            )
            or 22
        )
    except (TypeError, ValueError):
        jump_port = jump_defaults.jump_port if jump_defaults is not None else 22
    jump_auth_default = jump_defaults.jump_auth_method if jump_defaults is not None else "key"
    jump_auth_method = _normalise_auth_method(
        str(
            _settings_value(
                settings,
                f"{prefix}/jump_auth_method",
                jump_auth_default,
            )
            or jump_auth_default
        ),
        default=jump_auth_default,
    )
    jump_hostname = _settings_text(
        settings,
        f"{prefix}/jump_hostname",
        jump_defaults.jump_hostname if jump_defaults is not None else "",
    )
    jump_username = _settings_text(
        settings,
        f"{prefix}/jump_username",
        jump_defaults.jump_username if jump_defaults is not None else "",
    )
    if jump_username == PLACEHOLDER_JUMP_USERNAME:
        jump_username = ""
    jump_key_path = _settings_text(
        settings,
        f"{prefix}/jump_key_path",
        jump_defaults.jump_key_path if jump_defaults is not None else "",
    )
    use_macos_keychain_passwords = _settings_bool(
        settings, f"{prefix}/use_macos_keychain_passwords", False
    )

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
        default_memory_gb: float | None = (
            7.5 if queue_system.strip().lower() == "addqueue" else None
        )
    else:
        try:
            parsed_memory = float(default_memory_raw)
        except (TypeError, ValueError):
            parsed_memory = 7.5 if queue_system.strip().lower() == "addqueue" else 0.0
        default_memory_gb = parsed_memory if parsed_memory > 0 else None
        if queue_system.strip().lower() == "addqueue" and default_memory_gb is None:
            default_memory_gb = 7.5

    default_mem_per_cpu_gb = _normalise_mem_per_cpu_gb(
        settings.value(f"{prefix}/default_mem_per_cpu_gb", None),
        queue_system=queue_system,
    )

    queues = _queues_from_settings(
        settings.value(f"{prefix}/queues", None), queue_system=queue_system
    )
    default_queue_name = str(settings.value(f"{prefix}/default_queue_name", "", str) or "").strip()
    if queue_system.strip().lower() == "addqueue" and not default_queue_name and not queues:
        default_queue_name = _REDWOOD_PARTITION
    default_queue_name = _resolve_default_queue_name(default_queue_name, queues)

    keyword_execution_mode = (
        str(
            settings.value(
                f"{prefix}/keyword_execution_mode",
                builtin.keyword_execution_mode
                if builtin is not None
                else _DEFAULT_KEYWORD_EXECUTION_MODE,
                str,
            )
            or _DEFAULT_KEYWORD_EXECUTION_MODE
        )
        .strip()
        .lower()
    )
    if keyword_execution_mode not in {"direct", "scheduler"}:
        keyword_execution_mode = (
            builtin.keyword_execution_mode
            if builtin is not None
            else _DEFAULT_KEYWORD_EXECUTION_MODE
        )
    keyword_shell_init = str(
        settings.value(
            f"{prefix}/keyword_shell_init",
            builtin.keyword_shell_init if builtin is not None else "",
            str,
        )
        or ""
    )

    return ClusterConfig(
        name=name,
        hostname=hostname,
        port=max(1, port),
        username=username,
        auth_method=auth_method,
        key_path=_settings_text(settings, f"{prefix}/key_path"),
        jump_host_enabled=jump_host_enabled,
        jump_hostname=jump_hostname,
        jump_port=max(1, jump_port),
        jump_username=jump_username,
        jump_auth_method=jump_auth_method,
        jump_key_path=jump_key_path,
        use_macos_keychain_passwords=use_macos_keychain_passwords,
        default_working_directory=default_working_directory,
        queue_system=queue_system,
        cores_per_node=max(1, cores_per_node),
        default_walltime=default_walltime,
        template_script_path=str(
            settings.value(f"{prefix}/template_script_path", "", str) or ""
        ).strip(),
        castep_executable=str(
            settings.value(f"{prefix}/castep_executable", _DEFAULT_CASTEP_EXECUTABLE, str)
            or _DEFAULT_CASTEP_EXECUTABLE
        ).strip()
        or _DEFAULT_CASTEP_EXECUTABLE,
        default_queue_name=default_queue_name,
        default_memory_gb=default_memory_gb,
        default_mem_per_cpu_gb=default_mem_per_cpu_gb,
        submit_command_path=str(
            settings.value(f"{prefix}/submit_command_path", "", str) or ""
        ).strip(),
        queue_check_command_path=str(
            settings.value(f"{prefix}/queue_check_command_path", "", str) or ""
        ).strip(),
        cancel_command_path=str(
            settings.value(f"{prefix}/cancel_command_path", "", str) or ""
        ).strip(),
        keyword_execution_mode=keyword_execution_mode,
        keyword_shell_init=keyword_shell_init,
        queues=queues,
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
        default_mem_per_cpu_gb = _normalise_mem_per_cpu_gb(
            config.default_mem_per_cpu_gb,
            queue_system=queue_system,
        )
        queues = _normalise_queues(config.queues, queue_system=queue_system)
        cleaned.append(
            ClusterConfig(
                name=name,
                hostname=hostname,
                port=max(1, int(config.port)),
                username=config.username.strip(),
                auth_method=_normalise_auth_method(config.auth_method),
                key_path=config.key_path.strip(),
                jump_host_enabled=bool(config.jump_host_enabled),
                jump_hostname=config.jump_hostname.strip(),
                jump_port=max(1, int(config.jump_port)),
                jump_username=config.jump_username.strip(),
                jump_auth_method=_normalise_auth_method(config.jump_auth_method),
                jump_key_path=config.jump_key_path.strip(),
                use_macos_keychain_passwords=bool(config.use_macos_keychain_passwords),
                default_working_directory=config.default_working_directory.strip(),
                queue_system=queue_system,
                cores_per_node=max(1, int(config.cores_per_node)),
                default_walltime=default_walltime,
                template_script_path=config.template_script_path.strip(),
                castep_executable=config.castep_executable.strip() or _DEFAULT_CASTEP_EXECUTABLE,
                default_queue_name=_resolve_default_queue_name(config.default_queue_name, queues),
                default_memory_gb=default_memory_gb,
                default_mem_per_cpu_gb=default_mem_per_cpu_gb,
                submit_command_path=config.submit_command_path.strip(),
                queue_check_command_path=config.queue_check_command_path.strip(),
                cancel_command_path=config.cancel_command_path.strip(),
                keyword_execution_mode=config.keyword_execution_mode.strip().lower()
                or _DEFAULT_KEYWORD_EXECUTION_MODE,
                keyword_shell_init=config.keyword_shell_init.rstrip(),
                queues=queues,
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
        settings.setValue(f"{prefix}/jump_host_enabled", bool(config.jump_host_enabled))
        settings.setValue(f"{prefix}/jump_hostname", config.jump_hostname)
        settings.setValue(f"{prefix}/jump_port", int(config.jump_port))
        settings.setValue(f"{prefix}/jump_username", config.jump_username)
        settings.setValue(f"{prefix}/jump_auth_method", config.jump_auth_method)
        settings.setValue(f"{prefix}/jump_key_path", config.jump_key_path)
        settings.setValue(
            f"{prefix}/use_macos_keychain_passwords", bool(config.use_macos_keychain_passwords)
        )
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
        settings.setValue(
            f"{prefix}/default_mem_per_cpu_gb",
            "" if config.default_mem_per_cpu_gb is None else float(config.default_mem_per_cpu_gb),
        )
        settings.setValue(f"{prefix}/submit_command_path", config.submit_command_path)
        settings.setValue(f"{prefix}/queue_check_command_path", config.queue_check_command_path)
        settings.setValue(f"{prefix}/cancel_command_path", config.cancel_command_path)
        settings.setValue(f"{prefix}/keyword_execution_mode", config.keyword_execution_mode)
        settings.setValue(f"{prefix}/keyword_shell_init", config.keyword_shell_init)
        settings.setValue(f"{prefix}/queues", _queues_to_settings(config.queues))
