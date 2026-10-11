"""The persisted configuration record for one remote HPC cluster.

``ClusterConfig`` is a plain frozen dataclass so that the services which talk
to a cluster (SSH, keychain lookups, job submission, keyword generation) can
depend on it without importing Qt. Loading and saving the records through
``QSettings``, and the built-in cluster presets, live in
``mufinder.cluster_setup``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: CASTEP executable used when a cluster does not name one.
DEFAULT_CASTEP_EXECUTABLE = "castep.mpi"

#: How keyword generation runs unless the user opts for the login node: many
#: sites forbid or discourage running CASTEP there, and a scheduler job can
#: reuse the cluster's own job script.
DEFAULT_KEYWORD_EXECUTION_MODE = "scheduler"


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Persisted configuration for one remote HPC cluster."""

    name: str
    hostname: str
    port: int = 22
    username: str = ""
    auth_method: str = "key"
    key_path: str = ""
    jump_host_enabled: bool = False
    jump_hostname: str = ""
    jump_port: int = 22
    jump_username: str = ""
    jump_auth_method: str = "key"
    jump_key_path: str = ""
    use_macos_keychain_passwords: bool = False
    default_working_directory: str = ""
    queue_system: str = "slurm"
    cores_per_node: int = 1
    default_walltime: str = "01:00:00"
    template_script_path: str = ""
    castep_executable: str = DEFAULT_CASTEP_EXECUTABLE
    default_queue_name: str = ""
    default_memory_gb: float | None = None
    default_mem_per_cpu_gb: float | None = None
    submit_command_path: str = ""
    queue_check_command_path: str = ""
    cancel_command_path: str = ""
    keyword_execution_mode: str = DEFAULT_KEYWORD_EXECUTION_MODE
    keyword_shell_init: str = ""
