"""Tests for persisted cluster configuration helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest

import mufinder.cluster_setup as cluster_setup_module
from mufinder.cluster_setup import load_cluster_config, load_cluster_configs, save_cluster_configs
from mufinder.domain.cluster import ClusterConfig


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default=None, value_type=None):  # noqa: ANN001
        value = self._values.get(key, default)
        if value_type is str:
            return str(value)
        return value

    def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001
        self._values[key] = value


def test_cluster_setup_roundtrip_multiple_configs() -> None:
    settings = _FakeSettings()
    configs = [
        ClusterConfig(
            name="alpha",
            hostname="alpha.example",
            port=2222,
            username="alice",
            auth_method="key",
            key_path="/keys/alice",
            default_working_directory="/work/alice",
            queue_system="slurm",
            cores_per_node=128,
            default_walltime="02:00:00",
            template_script_path="/templates/slurm.sh",
            keyword_execution_mode="direct",
            keyword_shell_init="",
        ),
        ClusterConfig(
            name="beta",
            hostname="beta.example",
            port=22,
            username="bob",
            auth_method="password",
            key_path="",
            default_working_directory="/scratch/bob",
            queue_system="slurm",
            cores_per_node=64,
            default_walltime="01:30:00",
            template_script_path="/templates/beta.sh",
            keyword_execution_mode="scheduler",
            keyword_shell_init="module load CASTEP",
        ),
    ]

    save_cluster_configs(settings, configs)
    loaded = load_cluster_configs(settings)

    assert loaded[:2] == configs
    assert loaded[2].name == "Redwood"


def test_cluster_setup_load_by_name() -> None:
    settings = _FakeSettings()
    config = ClusterConfig(
        name="cluster-a",
        hostname="a.example",
        username="user",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/tmp/work",
        queue_system="slurm",
        cores_per_node=32,
        default_walltime="01:00:00",
        template_script_path="/tmp/template.sh",
    )

    save_cluster_configs(settings, [config])

    loaded = load_cluster_config(settings, "cluster-a")
    missing = load_cluster_config(settings, "cluster-missing")

    assert loaded == config
    assert missing is None


def test_cluster_setup_drops_invalid_entries() -> None:
    settings = _FakeSettings()
    save_cluster_configs(
        settings,
        [
            ClusterConfig(
                name="valid",
                hostname="valid.example",
                username="user",
                auth_method="key",
                key_path="/tmp/key",
                default_working_directory="/tmp/work",
                queue_system="slurm",
                cores_per_node=16,
                default_walltime="01:00:00",
                template_script_path="/tmp/template.sh",
            ),
            ClusterConfig(
                name="",
                hostname="invalid.example",
                username="user",
                auth_method="key",
                key_path="/tmp/key",
                default_working_directory="/tmp/work",
                queue_system="slurm",
                cores_per_node=16,
                default_walltime="01:00:00",
                template_script_path="/tmp/template.sh",
            ),
        ],
    )

    loaded = load_cluster_configs(settings)
    assert [config.name for config in loaded] == ["valid", "Redwood"]


def test_cluster_setup_returns_scarf_default_when_empty() -> None:
    settings = _FakeSettings()

    loaded = load_cluster_configs(settings)

    assert len(loaded) == 2
    assert loaded[0].name == "SCARF"
    assert loaded[0].hostname == "ui1.scarf.rl.ac.uk"
    assert loaded[0].username == ""
    assert loaded[0].default_working_directory == "/work4/isis/{username}"
    assert loaded[0].working_directory_for() == ""
    assert loaded[0].working_directory_for("alice") == "/work4/isis/alice"
    assert loaded[0].queue_system == "SLURM"
    assert loaded[0].cores_per_node == 24
    assert loaded[0].default_walltime == "12:00:00"
    assert loaded[0].template_script_path == "builtin:scarf-slurm"
    assert loaded[0].default_memory_gb is None
    assert loaded[0].keyword_execution_mode == "scheduler"
    assert loaded[0].keyword_shell_init == "module load CASTEP"

    assert loaded[1].name == "Redwood"
    assert loaded[1].hostname == "glamdring.physics.ox.ac.uk"
    assert loaded[1].auth_method == "password"
    assert loaded[1].jump_host_enabled is True
    assert loaded[1].jump_hostname == "bastion.physics.ox.ac.uk"
    assert loaded[1].jump_port == 22
    assert loaded[1].username == ""
    assert loaded[1].jump_username == ""
    assert loaded[1].jump_auth_method == "password"
    assert loaded[1].use_macos_keychain_passwords is False
    assert loaded[1].default_working_directory == "/mnt/redwoodfs/{username}"
    assert loaded[1].queue_system == "ADDQUEUE"
    assert loaded[1].default_walltime == ""
    assert loaded[1].default_queue_name == "redwood"
    assert loaded[1].default_memory_gb == 7.5
    assert loaded[1].castep_executable == "/mnt/redwoodfs/muon/castep.mpi"
    assert loaded[1].submit_command_path == "/usr/local/shared/bin/addqueue"
    assert loaded[1].keyword_execution_mode == "direct"
    assert loaded[1].keyword_shell_init == ""


def test_cluster_setup_load_by_name_uses_default_when_empty() -> None:
    settings = _FakeSettings()

    loaded = load_cluster_config(settings, "SCARF")

    assert loaded is not None
    assert loaded.hostname == "ui1.scarf.rl.ac.uk"


def test_cluster_setup_loads_redwood_default_when_empty() -> None:
    settings = _FakeSettings()

    loaded = load_cluster_config(settings, "Redwood")

    assert loaded is not None
    assert loaded.hostname == "glamdring.physics.ox.ac.uk"
    assert loaded.auth_method == "password"
    assert loaded.jump_host_enabled is True
    assert loaded.jump_hostname == "bastion.physics.ox.ac.uk"
    assert loaded.queue_system == "ADDQUEUE"
    assert loaded.default_walltime == ""


def test_cluster_setup_appends_redwood_for_legacy_saved_configs() -> None:
    settings = _FakeSettings()
    save_cluster_configs(
        settings,
        [
            ClusterConfig(
                name="legacy",
                hostname="legacy.example",
                username="legacy-user",
                auth_method="key",
                key_path="",
                default_working_directory="/legacy/work",
                queue_system="SLURM",
                cores_per_node=16,
                default_walltime="01:00:00",
                template_script_path="/templates/legacy.sh",
            )
        ],
    )

    loaded = load_cluster_configs(settings)

    assert [config.name for config in loaded] == ["legacy", "Redwood"]
    redwood = load_cluster_config(settings, "Redwood")
    assert redwood is not None
    assert redwood.hostname == "glamdring.physics.ox.ac.uk"
    assert redwood.jump_host_enabled is True
    assert redwood.jump_hostname == "bastion.physics.ox.ac.uk"


def test_cluster_setup_legacy_saved_configs_pick_keyword_defaults() -> None:
    settings = _FakeSettings()
    settings.setValue("clusters/names", ["SCARF", "Redwood"])
    settings.setValue("clusters/SCARF/hostname", "ui1.scarf.rl.ac.uk")
    settings.setValue("clusters/SCARF/username", "alice")
    settings.setValue("clusters/SCARF/template_script_path", "builtin:scarf-slurm")
    settings.setValue("clusters/SCARF/default_working_directory", "/work4/isis/alice")
    settings.setValue("clusters/Redwood/hostname", "glamdring.physics.ox.ac.uk")
    settings.setValue("clusters/Redwood/username", "alice")
    settings.setValue("clusters/Redwood/default_working_directory", "/mnt/redwoodfs/alice")

    loaded = load_cluster_configs(settings)

    scarf = next(config for config in loaded if config.name == "SCARF")
    redwood = next(config for config in loaded if config.name == "Redwood")
    assert scarf.keyword_execution_mode == "scheduler"
    assert scarf.keyword_shell_init == "module load CASTEP"
    assert redwood.keyword_execution_mode == "direct"
    assert redwood.auth_method == "password"
    assert redwood.jump_host_enabled is True
    assert redwood.jump_hostname == "bastion.physics.ox.ac.uk"
    assert redwood.jump_username == ""
    assert redwood.use_macos_keychain_passwords is False


def test_cluster_setup_reads_the_shipped_placeholder_usernames_as_empty() -> None:
    """A profile saved from the old presets holds the literal names, never a real one."""

    settings = _FakeSettings()
    settings.setValue("clusters/names", ["SCARF", "Redwood"])
    settings.setValue("clusters/SCARF/hostname", "ui1.scarf.rl.ac.uk")
    settings.setValue("clusters/SCARF/username", "username")
    settings.setValue("clusters/SCARF/default_working_directory", "/work4/isis/username")
    settings.setValue("clusters/Redwood/hostname", "glamdring.physics.ox.ac.uk")
    settings.setValue("clusters/Redwood/username", "username")
    settings.setValue("clusters/Redwood/jump_host_enabled", True)
    settings.setValue("clusters/Redwood/jump_hostname", "bastion.physics.ox.ac.uk")
    settings.setValue("clusters/Redwood/jump_username", "physics_username")
    settings.setValue("clusters/Redwood/default_working_directory", "/mnt/redwoodfs/username/x")

    scarf, redwood = load_cluster_configs(settings)

    assert (scarf.username, scarf.default_working_directory) == ("", "/work4/isis/{username}")
    assert (redwood.username, redwood.jump_username) == ("", "")
    assert redwood.default_working_directory == "/mnt/redwoodfs/{username}/x"
    assert redwood.working_directory_for("bob") == "/mnt/redwoodfs/bob/x"

    # Saving writes the values as read, so the placeholders do not come back.
    save_cluster_configs(settings, [scarf, redwood])
    assert settings.value("clusters/SCARF/username") == ""
    assert settings.value("clusters/Redwood/jump_username") == ""
    assert settings.value("clusters/SCARF/default_working_directory") == "/work4/isis/{username}"


def test_cluster_setup_keeps_saved_usernames_and_directories() -> None:
    settings = _FakeSettings()
    settings.setValue("clusters/names", ["SCARF", "Redwood"])
    settings.setValue("clusters/SCARF/hostname", "ui1.scarf.rl.ac.uk")
    settings.setValue("clusters/SCARF/username", "alice")
    settings.setValue("clusters/SCARF/default_working_directory", "/work4/isis/username")
    settings.setValue("clusters/SCARF/auth_method", "password")
    settings.setValue("clusters/SCARF/keyword_execution_mode", "direct")
    settings.setValue("clusters/SCARF/keyword_shell_init", "module load X")
    settings.setValue("clusters/SCARF/template_script_path", "/templates/site.sh")
    settings.setValue("clusters/Redwood/hostname", "glamdring.physics.ox.ac.uk")
    settings.setValue("clusters/Redwood/username", "bob")
    settings.setValue("clusters/Redwood/jump_username", "bob-physics")

    scarf, redwood = load_cluster_configs(settings)

    assert scarf.username == "alice"
    assert scarf.working_directory_for() == "/work4/isis/username"
    assert scarf.auth_method == "password"
    assert scarf.keyword_execution_mode == "direct"
    assert scarf.keyword_shell_init == "module load X"
    assert scarf.template_script_path == "/templates/site.sh"
    assert (redwood.username, redwood.jump_username) == ("bob", "bob-physics")


def test_cluster_config_says_which_username_is_missing() -> None:
    config = ClusterConfig(name="alpha", hostname="alpha.example")
    assert config.missing_username_issue() == "The cluster 'alpha' has no username."

    config = replace(config, username="alice")
    assert config.missing_username_issue() == ""

    config = replace(config, jump_host_enabled=True)
    assert config.missing_username_issue() == "The cluster 'alpha' has no jump host username."

    assert replace(config, jump_username="gw").missing_username_issue() == ""


def test_cluster_setup_roundtrips_jump_host_settings() -> None:
    settings = _FakeSettings()
    config = ClusterConfig(
        name="jumped",
        hostname="target.example",
        username="target-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.example",
        jump_port=2222,
        jump_username="physics-user",
        jump_auth_method="key",
        jump_key_path="/keys/physics",
        use_macos_keychain_passwords=True,
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=8,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    save_cluster_configs(settings, [config])

    loaded = load_cluster_config(settings, "jumped")
    assert loaded == config


def test_cluster_setup_saved_redwood_hostname_override_disables_builtin_jump_default() -> None:
    settings = _FakeSettings()
    settings.setValue("clusters/names", ["Redwood"])
    settings.setValue("clusters/Redwood/hostname", "redwood.internal.example")
    settings.setValue("clusters/Redwood/username", "alice")
    settings.setValue("clusters/Redwood/default_working_directory", "/mnt/redwoodfs/alice")

    redwood = load_cluster_config(settings, "Redwood")

    assert redwood is not None
    assert redwood.jump_host_enabled is False
    assert redwood.jump_hostname == ""


def test_cluster_setup_roundtrips_default_mem_per_cpu_for_slurm() -> None:
    settings = _FakeSettings()

    save_cluster_configs(
        settings,
        [
            ClusterConfig(
                name="scarf",
                hostname="ui1.scarf.rl.ac.uk",
                queue_system="SLURM",
                default_mem_per_cpu_gb=6.0,
            )
        ],
    )

    loaded = load_cluster_configs(settings)
    assert loaded[0].default_mem_per_cpu_gb == 6.0


def test_cluster_setup_default_mem_per_cpu_defaults_to_unset() -> None:
    settings = _FakeSettings()

    save_cluster_configs(
        settings,
        [ClusterConfig(name="scarf", hostname="ui1.scarf.rl.ac.uk", queue_system="SLURM")],
    )

    loaded = load_cluster_configs(settings)
    assert loaded[0].default_mem_per_cpu_gb is None


@pytest.mark.parametrize("stored", ["", "bad-memory", 0, -3.0])
def test_cluster_setup_rejects_non_positive_default_mem_per_cpu(stored: object) -> None:
    settings = _FakeSettings()
    save_cluster_configs(
        settings,
        [ClusterConfig(name="scarf", hostname="ui1.scarf.rl.ac.uk", queue_system="SLURM")],
    )
    settings.setValue("clusters/scarf/default_mem_per_cpu_gb", stored)

    loaded = load_cluster_configs(settings)
    assert loaded[0].default_mem_per_cpu_gb is None


def test_cluster_setup_drops_default_mem_per_cpu_for_addqueue() -> None:
    # --mem-per-cpu is a SLURM batch-script directive; addqueue has no use for
    # it, so a stale value must never survive a save/load cycle there.
    settings = _FakeSettings()

    save_cluster_configs(
        settings,
        [
            ClusterConfig(
                name="redwood",
                hostname="glamdring.physics.ox.ac.uk",
                queue_system="ADDQUEUE",
                default_walltime="",
                default_mem_per_cpu_gb=6.0,
            )
        ],
    )

    loaded = load_cluster_configs(settings)
    redwood = next(config for config in loaded if config.name == "redwood")
    assert redwood.default_mem_per_cpu_gb is None


def test_cluster_setup_load_falls_back_to_defaults_for_unusable_stored_values() -> None:
    settings = _FakeSettings()
    # A single stored name may come back from QSettings as a bare string.
    settings.setValue("clusters/names", "example")
    settings.setValue("clusters/example/hostname", "example.org")
    settings.setValue("clusters/example/auth_method", "token")
    settings.setValue("clusters/example/port", "bad-port")
    settings.setValue("clusters/example/queue_system", "ADDQUEUE")
    settings.setValue("clusters/example/cores_per_node", "bad-cores")
    settings.setValue("clusters/example/default_walltime", "")
    settings.setValue("clusters/example/default_memory_gb", "bad-memory")
    settings.setValue("clusters/example/default_queue_name", "")
    settings.setValue("clusters/example/keyword_execution_mode", "mystery")

    config = load_cluster_config(settings, "example")

    assert config is not None
    assert config.auth_method == "key"
    assert config.port == 22
    assert config.cores_per_node == 1
    assert config.default_walltime == ""
    assert config.default_memory_gb == 7.5
    assert config.default_queue_name == "redwood"
    # An unrecognised mode falls back to the default: a scheduler job.
    assert config.keyword_execution_mode == "scheduler"
    assert load_cluster_config(settings, "") is None


def test_cluster_setup_load_skips_blank_and_duplicate_stored_names() -> None:
    settings = _FakeSettings()
    settings.setValue("clusters/names", ["alpha", " beta ", "alpha", ""])
    settings.setValue("clusters/alpha/hostname", "alpha.example")
    settings.setValue("clusters/beta/hostname", "beta.example")

    loaded = load_cluster_configs(settings)

    assert [config.name for config in loaded] == ["alpha", "beta", "Redwood"]


def test_cluster_setup_save_skips_duplicate_names_and_fills_defaults() -> None:
    settings = _FakeSettings()

    save_cluster_configs(
        settings,
        [
            ClusterConfig(
                name="alpha", hostname="alpha.example", default_walltime="", queue_system="SLURM"
            ),
            ClusterConfig(name="alpha", hostname="duplicate.example"),
            ClusterConfig(
                name="beta", hostname="beta.example", queue_system="ADDQUEUE", default_memory_gb=0
            ),
        ],
    )

    assert settings.value("clusters/names") == ["alpha", "beta"]
    assert settings.value("clusters/alpha/hostname") == "alpha.example"
    assert settings.value("clusters/alpha/default_walltime") == "01:00:00"
    assert settings.value("clusters/beta/default_memory_gb") == 7.5


def test_cluster_setup_reexports_domain_cluster_config() -> None:
    # Public import alias kept for one minor release; see docs/compatibility_policy.rst.
    assert cluster_setup_module.ClusterConfig is ClusterConfig
    assert "ClusterConfig" in cluster_setup_module.__all__


def _psi_profile(**overrides) -> ClusterConfig:  # noqa: ANN003
    from mufinder.domain.cluster import ClusterQueue

    values = dict(
        name="PSI",
        hostname="login.psi.example",
        username="user",
        queue_system="SLURM",
        cores_per_node=44,
        default_walltime="01:00:00",
        default_queue_name="daily",
        queues=(
            ClusterQueue(name="hourly", max_walltime="01:00:00"),
            ClusterQueue(
                name="daily",
                default_walltime="12:00:00",
                max_walltime="1-00:00:00",
                cores_per_node=88,
                default_mem_per_cpu_gb=8.0,
                castep_executable="/opt/castep/castep.mpi",
                template_script_path="/templates/daily.sh",
            ),
        ),
    )
    values.update(overrides)
    return ClusterConfig(**values)


def test_cluster_setup_roundtrips_queues() -> None:
    settings = _FakeSettings()
    save_cluster_configs(settings, [_psi_profile()])

    loaded = load_cluster_config(settings, "PSI")

    assert loaded is not None
    assert loaded.default_queue_name == "daily"
    assert loaded.queues == _psi_profile().queues


def test_cluster_setup_profile_saved_without_queues_loads_as_before() -> None:
    settings = _FakeSettings()
    save_cluster_configs(settings, [_psi_profile(queues=(), default_queue_name="")])
    # A profile saved before queues existed has no queues key at all.
    del settings._values["clusters/PSI/queues"]

    loaded = load_cluster_config(settings, "PSI")

    assert loaded is not None
    assert loaded.queues == ()
    assert loaded.default_queue_name == ""
    assert loaded.queue_names() == []


def test_cluster_setup_legacy_addqueue_profile_keeps_its_single_queue() -> None:
    settings = _FakeSettings()
    redwood = load_cluster_config(settings, "Redwood")
    assert redwood is not None
    save_cluster_configs(settings, [redwood])
    del settings._values["clusters/Redwood/queues"]

    loaded = load_cluster_config(settings, "Redwood")

    assert loaded is not None
    assert loaded.queues == ()
    assert loaded.queue_names() == ["redwood"]


def test_cluster_setup_moves_the_default_queue_onto_a_listed_queue() -> None:
    settings = _FakeSettings()
    save_cluster_configs(settings, [_psi_profile(default_queue_name="gone")])

    loaded = load_cluster_config(settings, "PSI")

    assert loaded is not None
    assert loaded.default_queue_name == "hourly"


@pytest.mark.parametrize("stored", ["not json", "{}", "[1, 2]", '[{"name": ""}]'])
def test_cluster_setup_ignores_unreadable_queues(stored: str) -> None:
    settings = _FakeSettings()
    save_cluster_configs(settings, [_psi_profile(queues=(), default_queue_name="")])
    settings._values["clusters/PSI/queues"] = stored

    loaded = load_cluster_config(settings, "PSI")

    assert loaded is not None
    assert loaded.queues == ()


def test_cluster_setup_drops_queue_overrides_the_scheduler_cannot_use() -> None:
    from mufinder.domain.cluster import ClusterQueue

    settings = _FakeSettings()
    save_cluster_configs(
        settings,
        [
            _psi_profile(
                queue_system="ADDQUEUE",
                default_queue_name="redwood",
                queues=(
                    ClusterQueue(
                        name=" redwood ",
                        default_memory_gb=16.0,
                        default_mem_per_cpu_gb=8.0,
                        template_script_path="/templates/x.sh",
                        cores_per_node=0,
                    ),
                    ClusterQueue(name="redwood"),
                ),
            )
        ],
    )

    loaded = load_cluster_config(settings, "PSI")

    assert loaded is not None
    (queue,) = loaded.queues
    assert queue == ClusterQueue(name="redwood", default_memory_gb=16.0)
