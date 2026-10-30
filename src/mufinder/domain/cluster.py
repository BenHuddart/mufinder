"""The persisted configuration record for one remote HPC cluster.

``ClusterConfig`` is a plain frozen dataclass so that the services which talk
to a cluster (SSH, keychain lookups, job submission, keyword generation) can
depend on it without importing Qt. Loading and saving the records through
``QSettings``, and the built-in cluster presets, live in
``mufinder.cluster_setup``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: CASTEP executable used when a cluster does not name one.
DEFAULT_CASTEP_EXECUTABLE = "castep.mpi"

#: How keyword generation runs unless the user opts for the login node: many
#: sites forbid or discourage running CASTEP there, and a scheduler job can
#: reuse the cluster's own job script.
DEFAULT_KEYWORD_EXECUTION_MODE = "scheduler"

#: Stands for the profile's username in ``default_working_directory``.
USERNAME_TOKEN = "{username}"

#: Literal usernames an earlier build shipped in its presets. A saved profile
#: holding one reads as having no username (see docs/compatibility_policy.rst).
PLACEHOLDER_USERNAME = "username"
PLACEHOLDER_JUMP_USERNAME = "physics_username"


@dataclass(frozen=True, slots=True)
class ClusterQueue:
    """One named queue (SLURM partition or addqueue queue) of a cluster.

    Every setting other than ``name`` is an override: an empty string or
    ``None`` means "use the cluster profile's value". ``max_walltime`` has no
    profile-level counterpart; empty means the queue sets no limit MuFinder
    checks.
    """

    name: str
    default_walltime: str = ""
    max_walltime: str = ""
    cores_per_node: int | None = None
    default_memory_gb: float | None = None
    default_mem_per_cpu_gb: float | None = None
    castep_executable: str = ""
    template_script_path: str = ""


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
    #: Read Redwood passwords from the system keyring (macOS Keychain or Linux
    #: Secret Service); the name predates Linux support and is persisted.
    use_macos_keychain_passwords: bool = False
    default_working_directory: str = ""
    queue_system: str = "slurm"
    cores_per_node: int = 1
    default_walltime: str = "01:00:00"
    template_script_path: str = ""
    castep_executable: str = DEFAULT_CASTEP_EXECUTABLE
    #: The queue a submission starts on. With ``queues`` empty it is the
    #: profile's only queue and carries no overrides.
    default_queue_name: str = ""
    default_memory_gb: float | None = None
    default_mem_per_cpu_gb: float | None = None
    submit_command_path: str = ""
    queue_check_command_path: str = ""
    cancel_command_path: str = ""
    keyword_execution_mode: str = DEFAULT_KEYWORD_EXECUTION_MODE
    keyword_shell_init: str = ""
    queues: tuple[ClusterQueue, ...] = ()

    def working_directory_for(self, username: str | None = None) -> str:
        """Return the default working directory with ``{username}`` filled in.

        ``username`` replaces the profile's own. A directory that needs a
        username and has none is ``""``, so it never reaches a remote path.
        """

        directory = self.default_working_directory.strip()
        if USERNAME_TOKEN not in directory:
            return directory
        name = (self.username if username is None else username).strip()
        return directory.replace(USERNAME_TOKEN, name) if name else ""

    def missing_username_issue(self) -> str:
        """Say which username this profile still needs to connect, or ``""``."""

        if not self.username.strip():
            return f"The cluster '{self.name}' has no username."
        if self.jump_host_enabled and not self.jump_username.strip():
            return f"The cluster '{self.name}' has no jump host username."
        return ""

    def queue_names(self) -> list[str]:
        """Return the selectable queue names, the default queue first."""

        names = [queue.name for queue in self.queues if queue.name.strip()]
        default = self.default_queue_name.strip()
        if default:
            names = [default, *(name for name in names if name != default)]
        return names

    def queue(self, name: str) -> ClusterQueue | None:
        """Return the configured queue called ``name``, if there is one."""

        target = name.strip()
        if not target:
            return None
        return next((queue for queue in self.queues if queue.name == target), None)

    def for_queue(self, name: str) -> ClusterConfig:
        """Return this profile as it applies to submissions on queue ``name``.

        The queue's overrides replace the profile's values and
        ``default_queue_name`` becomes ``name``. A name the profile does not
        list (typed by hand, or a single-queue profile) changes only
        ``default_queue_name``; an empty name returns the profile unchanged.
        """

        target = name.strip()
        if not target:
            return self
        queue = self.queue(target)
        if queue is None:
            return replace(self, default_queue_name=target)
        return replace(
            self,
            default_queue_name=target,
            default_walltime=queue.default_walltime.strip() or self.default_walltime,
            cores_per_node=queue.cores_per_node if queue.cores_per_node else self.cores_per_node,
            default_memory_gb=(
                queue.default_memory_gb
                if queue.default_memory_gb is not None
                else self.default_memory_gb
            ),
            default_mem_per_cpu_gb=(
                queue.default_mem_per_cpu_gb
                if queue.default_mem_per_cpu_gb is not None
                else self.default_mem_per_cpu_gb
            ),
            castep_executable=queue.castep_executable.strip() or self.castep_executable,
            template_script_path=queue.template_script_path.strip() or self.template_script_path,
        )

    def max_walltime_for_queue(self, name: str) -> str:
        """Return the maximum walltime of queue ``name``, or ``""`` for none."""

        queue = self.queue(name)
        return queue.max_walltime.strip() if queue is not None else ""
