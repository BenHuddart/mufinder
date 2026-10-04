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


def test_create_scheduler_factory() -> None:
    scheduler = create_scheduler("slurm")
    assert isinstance(scheduler, SlurmScheduler)

    addqueue_scheduler = create_scheduler("addqueue")
    assert isinstance(addqueue_scheduler, AddqueueScheduler)

    with pytest.raises(ValueError, match="Unsupported queue system"):
        create_scheduler("pbs")


@pytest.mark.parametrize(
    ("state", "name"),
    [("OOM", "OUT_OF_MEMORY"), ("TO", "TIMEOUT"), ("NF", "NODE_FAIL"), ("F", "FAILED"), ("timeout", "TIMEOUT"), ("BF", "BF")],
)
def test_slurm_state_name_expands_short_codes(state: str, name: str) -> None:
    assert SlurmScheduler().state_name(state) == name
