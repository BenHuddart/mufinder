"""Conformance and error-contract pins for the :class:`JobRunner` protocol.

These tests codify Phase 3 of the submission-consistency initiative. They pin:

- Both concrete runners structurally satisfy ``JobRunner`` (the five signals with
  their expected parameter counts, plus ``submit``/``cancel``).
- The deliberately-different submit error contracts documented on ``JobRunner``:
  the local runner never raises for bad input and reports via ``job_failed``;
  the cluster runner may raise and its callers convert the exception.

The cluster-side error conversions (design decision 1) are already pinned by
existing tests; this module cross-references rather than duplicates them:

- ``JobExecutionController.submit_cluster_job`` catching a raising submit ->
  ``tests/services/test_job_execution_controller.py``::
  ``test_cluster_submission_failure_marks_job_failed_and_records_terminal_event``
  and ``test_cluster_resubmission_failure_keeps_existing_remote_context``.
- ``ClusterController._run_submit_batch`` converting a raising submit into
  ``job_submit_finished(job_id, None, error)`` ->
  ``tests/app/test_cluster_controller.py``::
  ``test_submit_batch_reports_per_job_failure``.

The decision-2 queued-path pins (status_changed -> ProjectService.mark_job_queued)
live in ``tests/services/test_submission_harness.py``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtCore import QMetaMethod

from mufinder.domain.enums import JobStatus
from mufinder.services import JobRunner
from mufinder.services.cluster_runner import ClusterJobRunner
from mufinder.services.local_runner import LocalJobRunner

# Signal name -> expected parameter count of the emit signature.
_EXPECTED_SIGNALS = {
    "job_started": 1,
    "job_finished": 2,
    "job_failed": 2,
    "job_cancelled": 1,
    "status_changed": 2,
}


def _signal_param_count(runner: object, name: str) -> int:
    meta = QMetaMethod.fromSignal(getattr(runner, name))
    assert meta.isValid(), f"{name} is not a valid signal on {type(runner).__name__}"
    return meta.parameterCount()


@pytest.mark.parametrize("runner_factory", [LocalJobRunner, ClusterJobRunner])
def test_runner_satisfies_protocol_structurally(qapp, runner_factory) -> None:
    runner = runner_factory()

    # runtime_checkable only verifies attribute presence; it is the secondary
    # check. The explicit assertions below carry the load-bearing failure detail.
    assert isinstance(runner, JobRunner)

    for name, expected_params in _EXPECTED_SIGNALS.items():
        assert hasattr(runner, name), f"{type(runner).__name__} missing signal {name}"
        assert _signal_param_count(runner, name) == expected_params, (
            f"{type(runner).__name__}.{name} emit signature changed"
        )

    assert callable(runner.submit)
    assert callable(runner.cancel)


def test_local_submit_missing_executable_does_not_raise_and_emits_failed(
    qapp, tmp_path: Path
) -> None:
    # characterization / contract pin (JobRunner design decision 1): the local
    # runner reports validation failures via job_failed, it does not raise.
    runner = LocalJobRunner()
    failures: list[tuple[object, str]] = []
    runner.job_failed.connect(lambda job_id, message: failures.append((job_id, message)))

    job_id = uuid4()
    # No pytest.raises wrapper: reaching the assertions proves submit did not raise.
    runner.submit(
        job_id=job_id,
        job_dir=tmp_path,
        seed_name="job_a",
        castep_executable="",
        mpirun_path="",
        num_cores=1,
    )

    assert failures == [(job_id, "No CASTEP executable is configured.")]


def test_local_submit_missing_directory_does_not_raise_and_emits_failed(
    qapp, tmp_path: Path
) -> None:
    # characterization / contract pin: a non-existent job directory is reported
    # via job_failed rather than raised, exercising the full submit() path (not
    # just _validate_request).
    runner = LocalJobRunner()
    failures: list[tuple[object, str]] = []
    statuses: list[tuple[object, object]] = []
    runner.job_failed.connect(lambda job_id, message: failures.append((job_id, message)))
    runner.status_changed.connect(lambda job_id, status: statuses.append((job_id, status)))

    job_id = uuid4()
    missing_dir = tmp_path / "does-not-exist"
    runner.submit(
        job_id=job_id,
        job_dir=missing_dir,
        seed_name="job_a",
        castep_executable="/bin/castep",
        mpirun_path="",
        num_cores=1,
    )

    assert len(failures) == 1
    assert failures[0][0] == job_id
    assert "does not exist" in failures[0][1]
    assert statuses == [(job_id, JobStatus.FAILED.value)]


def test_cluster_submit_raises_without_connection(qapp, tmp_path: Path) -> None:
    # characterization / contract pin (JobRunner design decision 1): unlike the
    # local runner, ClusterJobRunner.submit *raises* rather than emitting
    # job_failed. Here it raises the connection guard (submit's first line is
    # _require_ssh) on an unconnected runner. Callers convert the exception --
    # see the module docstring for the tests that pin that conversion.
    runner = ClusterJobRunner()

    with pytest.raises(ConnectionError, match="connection is not active"):
        runner.submit(
            job_id=uuid4(),
            local_job_dir=tmp_path / "missing",
            seed_name="job_a",
            num_processes=1,
            walltime="01:00:00",
            working_directory="/remote",
            template_script_path="",
            cores_per_node=1,
        )
