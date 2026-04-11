"""Tests for persisted cluster configuration helpers."""

from __future__ import annotations

from mufinder.cluster_setup import ClusterConfig, load_cluster_config, load_cluster_configs, save_cluster_configs


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
            keyword_python_command="python3",
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
            keyword_python_command="python",
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
    assert loaded[0].username == "username"
    assert loaded[0].default_working_directory == "/work4/isis/username"
    assert loaded[0].queue_system == "SLURM"
    assert loaded[0].cores_per_node == 24
    assert loaded[0].default_walltime == "12:00:00"
    assert loaded[0].template_script_path == "builtin:scarf-slurm"
    assert loaded[0].default_memory_gb is None
    assert loaded[0].keyword_execution_mode == "scheduler"
    assert loaded[0].keyword_shell_init == "module load CASTEP"
    assert loaded[0].keyword_python_command == "python3"

    assert loaded[1].name == "Redwood"
    assert loaded[1].hostname == "glamdring.physics.ox.ac.uk"
    assert loaded[1].default_working_directory == "/mnt/redwoodfs/username"
    assert loaded[1].queue_system == "ADDQUEUE"
    assert loaded[1].default_walltime == ""
    assert loaded[1].default_queue_name == "redwood"
    assert loaded[1].default_memory_gb == 7.5
    assert loaded[1].castep_executable == "/mnt/redwoodfs/muon/castep.mpi"
    assert loaded[1].keyword_execution_mode == "direct"
    assert loaded[1].keyword_shell_init == ""
    assert loaded[1].keyword_python_command == "python3"


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
    assert scarf.keyword_python_command == "python3"
    assert redwood.keyword_execution_mode == "direct"
    assert redwood.keyword_python_command == "python3"
