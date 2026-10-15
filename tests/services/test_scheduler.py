"""Tests for scheduler abstraction and SLURM backend."""

from __future__ import annotations

import pytest

from mufinder.domain.enums import JobStatus
from mufinder.services.scheduler import (
    AddqueueScheduler,
    SchedulerSubmitRequest,
    SlurmScheduler,
    create_scheduler,
)


class _FakeSSH:
    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self._responses = responses
        self.commands: list[str] = []

    def exec_command(self, command: str, timeout: int = 30) -> tuple[int, str, str]:
        self.commands.append(command)
        if not self._responses:
            return 0, "", ""
        return self._responses.pop(0)


def test_slurm_submit_parses_job_id() -> None:
    scheduler = SlurmScheduler(submit_command_path="/usr/local/bin/sbatch")
    ssh = _FakeSSH([(0, "Submitted batch job 12345\n", "")])

    job_id = scheduler.submit(
        ssh,
        SchedulerSubmitRequest(
            working_dir="/remote",
            seed_name="seed",
            num_processes=8,
            cores_per_node=32,
            walltime="01:00:00",
            castep_executable="castep.mpi",
            script_path="/remote/run.sh",
        ),
    )

    assert job_id == "12345"
    assert "/usr/local/bin/sbatch" in ssh.commands[0]


def test_slurm_submit_raises_on_bad_output() -> None:
    scheduler = SlurmScheduler(submit_command_path="/usr/local/bin/sbatch")
    ssh = _FakeSSH([(0, "unexpected output\n", "")])

    with pytest.raises(RuntimeError, match="Could not parse job id"):
        scheduler.submit(
            ssh,
            SchedulerSubmitRequest(
                working_dir="/remote",
                seed_name="seed",
                num_processes=8,
                cores_per_node=32,
                walltime="01:00:00",
                castep_executable="castep.mpi",
                script_path="/remote/run.sh",
            ),
        )


def test_slurm_cancel_calls_scancel() -> None:
    scheduler = SlurmScheduler(cancel_command_path="/usr/local/bin/scancel")
    ssh = _FakeSSH([(0, "", "")])

    scheduler.cancel(ssh, "991")

    assert "/usr/local/bin/scancel" in ssh.commands[0]
    assert "991" in ssh.commands[0]


def test_slurm_status_parses_squeue_output() -> None:
    scheduler = SlurmScheduler(queue_check_command_path="/usr/local/bin/squeue")
    ssh = _FakeSSH([(0, "123|PD|00:00|my_job\n124|R|01:23:45|other_job\n", "")])

    statuses = scheduler.status(ssh, ["123", "124"])

    assert statuses["123"].state == "PD"
    assert statuses["123"].elapsed_runtime == "00:00"
    assert statuses["123"].name == "my_job"
    assert statuses["124"].state == "R"
    assert statuses["124"].elapsed_runtime == "01:23:45"
    assert "/usr/local/bin/squeue" in ssh.commands[0]


def test_slurm_status_lists_jobs_in_hidden_partitions() -> None:
    scheduler = SlurmScheduler(queue_check_command_path="/usr/local/bin/squeue")
    ssh = _FakeSSH(
        [
            (1, "", "slurm_load_jobs error: Invalid job id specified"),
            (0, "123|PD|0:00|queued_job\n", ""),
        ]
    )

    scheduler.status(ssh, ["123"])

    # Without --all, squeue leaves out jobs in hidden or unavailable partitions,
    # so a pending job there would look finished. The per-id fallback keeps it.
    assert all(" --all " in command for command in ssh.commands)


def test_slurm_history_reads_sacct_allocation_records() -> None:
    scheduler = SlurmScheduler()
    ssh = _FakeSSH(
        [
            (0, "/usr/bin/sacct\n", ""),
            (0, "123|PENDING|00:00:00|queued_job\n124|CANCELLED by 1001|00:01:02|other_job\n", ""),
        ]
    )

    records = scheduler.history(ssh, ["123", "124", "125"])

    assert ssh.commands[-1] == (
        "/usr/bin/sacct --jobs=123,124,125 --allocations --noheader --parsable2 "
        "--format=JobID,State,Elapsed,JobName"
    )
    assert sorted(records) == ["123", "124"]
    assert scheduler.map_to_job_status(records["123"].state) == JobStatus.QUEUED
    assert scheduler.map_to_job_status(records["124"].state) == JobStatus.CANCELLED
    assert scheduler.state_name(records["124"].state) == "CANCELLED"
    assert records["124"].elapsed_runtime == "00:01:02"


def test_slurm_history_without_accounting_reports_no_records() -> None:
    scheduler = SlurmScheduler()
    ssh = _FakeSSH(
        [
            (0, "/usr/bin/sacct\n", ""),
            (1, "", "sacct: error: Slurm accounting storage is disabled"),
        ]
    )

    assert scheduler.history(ssh, ["123"]) == {}
    assert scheduler.history(_FakeSSH([]), []) == {}


def test_addqueue_history_reads_sacct() -> None:
    scheduler = AddqueueScheduler()
    ssh = _FakeSSH([(0, "/usr/bin/sacct\n", ""), (0, "7|RUNNING|00:10:00|job\n", "")])

    # addqueue polls through SLURM, so it shares the sacct lookup.
    assert scheduler.history(ssh, ["7"])["7"].state == "RUNNING"


def test_slurm_status_invalid_job_id_fallback_queries_per_job() -> None:
    scheduler = SlurmScheduler(queue_check_command_path="/usr/local/bin/squeue")
    ssh = _FakeSSH(
        [
            (1, "", "slurm_load_jobs error: Invalid job id specified"),
            (0, "123|R|00:05:00|valid_job\n", ""),
            (1, "", "slurm_load_jobs error: Invalid job id specified"),
        ]
    )

    statuses = scheduler.status(ssh, ["123", "999"])

    assert list(statuses) == ["123"]
    assert statuses["123"].state == "R"
    assert statuses["123"].elapsed_runtime == "00:05:00"
    assert len(ssh.commands) == 3
    assert "--jobs=123,999" in ssh.commands[0]
    assert "--jobs=123" in ssh.commands[1]
    assert "--jobs=999" in ssh.commands[2]


def test_slurm_state_mapping() -> None:
    scheduler = SlurmScheduler()

    assert scheduler.map_to_job_status("PD") == JobStatus.QUEUED
    assert scheduler.map_to_job_status("R") == JobStatus.RUNNING
    assert scheduler.map_to_job_status("CD") == JobStatus.COMPLETED
    assert scheduler.map_to_job_status("F") == JobStatus.FAILED
    assert scheduler.map_to_job_status("TO") == JobStatus.STOPPED
    assert scheduler.map_to_job_status("CA") == JobStatus.CANCELLED
    assert scheduler.map_to_job_status("node_fail") == JobStatus.FAILED
    # An unrecognised state keeps the job tracked, but never claims it runs.
    assert scheduler.map_to_job_status("mystery") == JobStatus.QUEUED
    assert scheduler.is_terminal("mystery") is False
    assert scheduler.is_terminal("completed") is True
    assert scheduler.is_terminal("running") is False


@pytest.mark.parametrize("state", ["CF", "S", "ST", "RH", "RQ", "RF", "RD", "SE", "SUSPENDED", "REQUEUE_HOLD"])
def test_slurm_waiting_and_held_states_map_to_queued(state: str) -> None:
    scheduler = SlurmScheduler()

    assert scheduler.map_to_job_status(state) == JobStatus.QUEUED
    assert scheduler.is_terminal(state) is False


@pytest.mark.parametrize("state", ["CG", "RS", "SI", "SO", "COMPLETING", "STAGE_OUT"])
def test_slurm_executing_states_map_to_running(state: str) -> None:
    scheduler = SlurmScheduler()

    assert scheduler.map_to_job_status(state) == JobStatus.RUNNING
    assert scheduler.is_terminal(state) is False


@pytest.mark.parametrize("state", ["BF", "DL", "PR", "RV", "BOOT_FAIL", "PREEMPTED", "OUT_OF_MEMORY"])
def test_slurm_terminal_failure_states_map_to_failed(state: str) -> None:
    scheduler = SlurmScheduler()

    assert scheduler.map_to_job_status(state) == JobStatus.FAILED
    assert scheduler.is_terminal(state) is True


def test_slurm_status_without_job_ids_does_not_query_the_queue() -> None:
    ssh = _FakeSSH([])

    assert SlurmScheduler().status(ssh, []) == {}
    assert ssh.commands == []


def test_slurm_status_tolerates_short_and_malformed_squeue_rows() -> None:
    scheduler = SlurmScheduler(queue_check_command_path="/usr/bin/squeue")
    ssh = _FakeSSH([(0, "1|R\n2|PD|queued_job\nbad-line\n", "")])

    statuses = scheduler.status(ssh, ["1", "2"])

    assert sorted(statuses) == ["1", "2"]
    assert statuses["1"].state == "R"
    assert statuses["1"].elapsed_runtime == ""
    assert statuses["1"].name == ""
    assert statuses["2"].elapsed_runtime == ""
    assert statuses["2"].name == "queued_job"


def test_slurm_cancel_and_status_raise_the_scheduler_error() -> None:
    with pytest.raises(RuntimeError, match="permission denied"):
        SlurmScheduler(cancel_command_path="/usr/bin/scancel").cancel(
            _FakeSSH([(1, "", "permission denied")]), "123"
        )
    with pytest.raises(RuntimeError, match="scheduler offline"):
        SlurmScheduler(queue_check_command_path="/usr/bin/squeue").status(
            _FakeSSH([(1, "", "scheduler offline")]), ["123"]
        )


def test_addqueue_submit_builds_expected_command() -> None:
    scheduler = AddqueueScheduler(submit_command_path="/bin/addqueue")
    ssh = _FakeSSH([(0, "Submitted batch job 98765\n", "")])

    job_id = scheduler.submit(
        ssh,
        SchedulerSubmitRequest(
            working_dir="/remote/work",
            seed_name="conv_ecut_300",
            num_processes=40,
            cores_per_node=32,
            walltime="",
            castep_executable="/mnt/redwoodfs/muon/castep.mpi",
            queue_name="redwood",
            memory_gb=7.5,
            comment="overnight run",
        ),
    )

    assert job_id == "98765"
    command = ssh.commands[0]
    assert "/bin/addqueue" in command
    assert "--sbatch" in command
    assert "-n 40" in command
    assert "-q redwood" in command
    assert "-m 7.5" in command
    assert "-c 'overnight run'" in command
    assert "-t" not in command


def _addqueue_request() -> SchedulerSubmitRequest:
    return SchedulerSubmitRequest(
        working_dir="/remote/work",
        seed_name="seed",
        num_processes=4,
        cores_per_node=32,
        walltime="02:00:00",
        castep_executable="castep.mpi",
    )


def test_addqueue_submit_reads_a_job_id_from_free_form_output() -> None:
    scheduler = AddqueueScheduler(submit_command_path="/bin/addqueue")

    job_id = scheduler.submit(_FakeSSH([(0, "job created: id=24680\n", "")]), _addqueue_request())

    assert job_id == "24680"


def test_addqueue_submit_raises_the_scheduler_error() -> None:
    scheduler = AddqueueScheduler(submit_command_path="/bin/addqueue")

    with pytest.raises(RuntimeError, match="queue unavailable"):
        scheduler.submit(_FakeSSH([(1, "", "queue unavailable")]), _addqueue_request())


def test_scheduler_command_resolution_uses_probe_then_fallback() -> None:
    scheduler = SlurmScheduler()
    ssh = _FakeSSH(
        [
            (1, "", "not found"),
            (0, "Submitted batch job 111\n", ""),
        ]
    )

    job_id = scheduler.submit(
        ssh,
        SchedulerSubmitRequest(
            working_dir="/remote",
            seed_name="seed",
            num_processes=2,
            cores_per_node=1,
            walltime="01:00:00",
            castep_executable="castep.mpi",
            script_path="/remote/run.sh",
        ),
    )

    assert job_id == "111"
    assert ssh.commands[0].startswith("command -v sbatch")
    assert " sbatch " in ssh.commands[1]


def test_scheduler_command_resolution_delegates_to_the_connection_once() -> None:
    class _ResolvingSSH(_FakeSSH):
        def __init__(self) -> None:
            super().__init__([(0, "Submitted batch job 1\n", ""), (0, "Submitted batch job 2\n", "")])
            self.resolved: list[tuple[str, str]] = []

        def resolve_command_path(self, command_name: str, configured: str) -> str:
            self.resolved.append((command_name, configured))
            return f"/resolved/{command_name}"

    scheduler = SlurmScheduler()
    ssh = _ResolvingSSH()
    request = SchedulerSubmitRequest(
        working_dir="/remote",
        seed_name="seed",
        num_processes=2,
        cores_per_node=1,
        walltime="01:00:00",
        castep_executable="castep.mpi",
        script_path="/remote/run.sh",
    )

    assert scheduler.submit(ssh, request) == "1"
    assert scheduler.submit(ssh, request) == "2"
    # The resolved path is cached per scheduler, so the connection is asked once.
    assert ssh.resolved == [("sbatch", "")]
    assert all("/resolved/sbatch" in command for command in ssh.commands)


def test_create_scheduler_factory() -> None:
    scheduler = create_scheduler("slurm")
    assert isinstance(scheduler, SlurmScheduler)

    addqueue_scheduler = create_scheduler("addqueue")
    assert isinstance(addqueue_scheduler, AddqueueScheduler)

    with pytest.raises(ValueError, match="Unsupported queue system"):
        create_scheduler("pbs")


@pytest.mark.parametrize(
    ("state", "name"),
    [("OOM", "OUT_OF_MEMORY"), ("TO", "TIMEOUT"), ("NF", "NODE_FAIL"), ("F", "FAILED"), ("timeout", "TIMEOUT"), ("BF", "BOOT_FAIL"), ("XX", "XX")],
)
def test_slurm_state_name_expands_short_codes(state: str, name: str) -> None:
    assert SlurmScheduler().state_name(state) == name
