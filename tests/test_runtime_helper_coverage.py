"""Additional coverage for runtime, scheduler and cluster-setup helpers."""

from __future__ import annotations

import pytest

import mufinder.main as main_module
from mufinder.cluster_setup import (
    ClusterConfig,
    _cluster_from_settings,
    _normalise_cluster_names,
    load_cluster_config,
    save_cluster_configs,
)
from mufinder.domain.enums import JobStatus
from mufinder.services.scheduler import (
    AddqueueScheduler,
    SchedulerSubmitRequest,
    SlurmScheduler,
    _parse_submitted_job_id,
)


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


class _FakeSSH:
    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self._responses = responses
        self.commands: list[str] = []

    def exec_command(self, command: str, timeout: int = 30) -> tuple[int, str, str]:
        self.commands.append(command)
        if not self._responses:
            return 0, "", ""
        return self._responses.pop(0)


def test_main_helpers_cover_process_and_visibility_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class AppWithFallback:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def processEvents(self, *args) -> None:  # noqa: ANN002
            self.calls.append(args)
            if args:
                raise TypeError("overload not available")

    class ExposedHandle:
        def isExposed(self) -> bool:  # noqa: N802
            return True

    class InvisibleSplash:
        def windowHandle(self):  # noqa: N802
            return None

        def isVisible(self) -> bool:  # noqa: N802
            return False

    class BrokenSplash:
        def windowHandle(self):  # noqa: N802
            class _Handle:
                def isExposed(self) -> bool:  # noqa: N802
                    raise RuntimeError("deleted")

            return _Handle()

        def isVisible(self) -> bool:  # noqa: N802
            raise RuntimeError("deleted")

    app = AppWithFallback()
    main_module._process_events(app, max_ms=10)
    main_module._process_events(app, max_ms=None)
    main_module._process_events(object(), max_ms=10)

    assert app.calls == [
        (main_module.QEventLoop.ProcessEventsFlag.AllEvents, 10),
        (),
        (),
    ]

    assert main_module._is_splash_exposed(type("Splash", (), {"windowHandle": lambda self: ExposedHandle()})()) is True
    assert main_module._is_splash_exposed(InvisibleSplash()) is False
    assert main_module._is_splash_exposed(BrokenSplash()) is False

    class FakeTimer:
        def __init__(self) -> None:
            self._elapsed = -10

        def start(self) -> None:
            self._elapsed = 0

        def elapsed(self) -> int:
            self._elapsed += 10
            return self._elapsed

    exposures = iter([False, False, True])
    splash = object()
    process_calls: list[int | None] = []

    monkeypatch.setattr(main_module, "QElapsedTimer", FakeTimer)
    monkeypatch.setattr(main_module.QThread, "msleep", lambda _ms: None)
    monkeypatch.setattr(main_module, "_is_splash_exposed", lambda _s: next(exposures))
    monkeypatch.setattr(main_module, "_process_events", lambda _app, *, max_ms=None: process_calls.append(max_ms))

    main_module._wait_for_splash_exposed(object(), splash, timeout_ms=40)

    assert process_calls == [25, 25, 10]


def test_scheduler_helpers_cover_status_resolution_and_error_paths() -> None:
    scheduler = SlurmScheduler()
    assert scheduler.status(_FakeSSH([]), []) == {}
    assert scheduler.is_terminal("completed") is True
    assert scheduler.is_terminal("running") is False
    assert scheduler.map_to_job_status("node_fail") == JobStatus.FAILED
    assert scheduler.map_to_job_status("mystery") == JobStatus.RUNNING

    ssh = _FakeSSH([(1, "", "permission denied")])
    with pytest.raises(RuntimeError, match="permission denied"):
        SlurmScheduler(cancel_command_path="/usr/bin/scancel").cancel(ssh, "123")

    status_ssh = _FakeSSH([(1, "", "scheduler offline")])
    with pytest.raises(RuntimeError, match="scheduler offline"):
        SlurmScheduler(queue_check_command_path="/usr/bin/squeue").status(status_ssh, ["123"])

    parsed = SlurmScheduler._parse_status_output("1|R\n2|PD|queued_job\nbad-line\n")
    assert parsed["1"].elapsed_runtime == ""
    assert parsed["1"].name == ""
    assert parsed["2"].name == "queued_job"

    class SSHWithResolver:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def resolve_command_path(self, command_name: str, configured: str) -> str:
            self.calls.append((command_name, configured))
            return f"/resolved/{command_name}"

    resolver = SSHWithResolver()
    resolved = scheduler._resolve_command(resolver, "submit", "sbatch", "")
    cached = scheduler._resolve_command(resolver, "submit", "sbatch", "")
    assert resolved == "/resolved/sbatch"
    assert cached == "/resolved/sbatch"
    assert resolver.calls == [("sbatch", "")]

    configured = scheduler._resolve_command(_FakeSSH([]), "status", "squeue", "/usr/bin/squeue")
    assert configured == "/usr/bin/squeue"

    addqueue = AddqueueScheduler(submit_command_path="/bin/addqueue")
    failing_ssh = _FakeSSH([(1, "", "queue unavailable")])
    with pytest.raises(RuntimeError, match="queue unavailable"):
        addqueue.submit(
            failing_ssh,
            SchedulerSubmitRequest(
                working_dir="/remote/work",
                seed_name="seed",
                num_processes=4,
                cores_per_node=32,
                walltime="02:00:00",
                castep_executable="castep.mpi",
            ),
        )

    assert _parse_submitted_job_id("job created: id=24680", "slurm") == "24680"


def test_cluster_setup_helpers_cover_normalization_and_defaults() -> None:
    settings = _FakeSettings()
    settings.setValue("clusters/example/hostname", "example.org")
    settings.setValue("clusters/example/auth_method", "token")
    settings.setValue("clusters/example/port", "bad-port")
    settings.setValue("clusters/example/queue_system", "ADDQUEUE")
    settings.setValue("clusters/example/cores_per_node", "bad-cores")
    settings.setValue("clusters/example/default_walltime", "")
    settings.setValue("clusters/example/default_memory_gb", "bad-memory")
    settings.setValue("clusters/example/default_queue_name", "")
    settings.setValue("clusters/example/keyword_execution_mode", "mystery")

    config = _cluster_from_settings(settings, "example")
    assert config is not None
    assert config.auth_method == "key"
    assert config.port == 22
    assert config.cores_per_node == 1
    assert config.default_walltime == ""
    assert config.default_memory_gb == 7.5
    assert config.default_queue_name == "redwood"
    # An unrecognised mode falls back to the default: a scheduler job.
    assert config.keyword_execution_mode == "scheduler"

    assert _normalise_cluster_names("alpha") == ["alpha"]
    assert _normalise_cluster_names(["alpha", " beta ", "alpha", ""]) == ["alpha", "beta"]
    assert load_cluster_config(settings, "") is None

    save_settings = _FakeSettings()
    save_cluster_configs(
        save_settings,
        [
            ClusterConfig(name="alpha", hostname="alpha.example", default_walltime="", queue_system="SLURM"),
            ClusterConfig(name="alpha", hostname="duplicate.example"),
            ClusterConfig(name="beta", hostname="beta.example", queue_system="ADDQUEUE", default_memory_gb=0),
        ],
    )

    assert save_settings.value("clusters/names") == ["alpha", "beta"]
    assert save_settings.value("clusters/alpha/default_walltime") == "01:00:00"
    assert save_settings.value("clusters/beta/default_memory_gb") == 7.5
