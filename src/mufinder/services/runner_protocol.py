"""Structural contract of a runner that holds the jobs it runs.

:class:`mufinder.services.local_runner.LocalJobRunner` holds a process for each
job and reports its lifecycle through the signals below;
:class:`mufinder.services.job_execution_controller.JobExecutionController` and
the ``MainWindow`` signal wiring rely on that surface. The protocol adds no
runtime behavior and the runner does not inherit from it. Conformance is
structural and is verified in ``tests/services/test_runner_protocol.py``.

:class:`mufinder.services.cluster_runner.ClusterJobRunner` is deliberately not a
``JobRunner``. It holds no job: a check reports what the cluster shows as
evidence (``job_checked``), and
:func:`mufinder.services.job_reconcile.reconcile` decides the state from it.

State authority (who may mutate what)
-------------------------------------

Job state transitions are owned **exclusively** by ``ProjectService`` primitives
(``submit_job``, ``mark_job_active``, ``complete_job``, ``end_job``,
``reset_failed_job``, ``prepare_job_continuation``). ``JobExecutionController`` mutates
only bookkeeping fields on the ``Job`` (``execution_target``,
``num_processes``, remote metadata clears, monitor stamps) and routes every
status change through the service.

**Runners never touch ``Job`` objects.** They communicate solely by emitting
signals keyed by job id. The controller translates those signals into
``ProjectService`` calls. A runner that mutated a ``Job`` would violate this
contract.

Submit error contracts (deliberately different per runner)
----------------------------------------------------------

The two ``submit`` implementations have **deliberately different** error
contracts, and this difference is intentional -- it is not a bug to be unified:

- :meth:`LocalJobRunner.submit` **never raises** for bad input. It validates the
  request (CASTEP executable present, ``mpirun`` present when multiple cores are
  requested, job directory exists) and reports any problem by emitting
  ``job_failed`` (and ``status_changed`` -> ``failed``). Its single caller
  ``JobExecutionController.submit_local_jobs`` still wraps the call in a
  belt-and-suspenders ``try``/``except``, but the runner's own contract is
  "validation failures are reported via the ``job_failed`` signal, not raised".

- :meth:`ClusterJobRunner.submit` **may raise** (``ConnectionError`` and similar)
  because it performs blocking SSH/SFTP work, and returns the scheduler job ID.
  Its two callers convert the exception rather than letting it escape:

  * ``JobExecutionController.submit_cluster_job`` catches the exception and
    returns a failed ``ClusterJobSubmissionOutcome`` (via
    ``fail_cluster_submission``).
  * ``ClusterController._run_submit_batch`` catches per job and emits
    ``job_submit_finished(job_id, None, str(exc))``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from PySide6.QtCore import Signal


@runtime_checkable
class JobRunner(Protocol):
    """Signal-and-method surface of a runner that holds its jobs.

    The five signals below are what ``MainWindow`` connects and what the
    lifecycle depends on; see the module docstring for the state-authority and
    submit error contracts. ``runtime_checkable`` only verifies attribute
    *presence*, so the conformance tests additionally assert the signal
    parameter counts (1, 3, 2, 1, 2 respectively).
    """

    #: Emitted with ``(job_id)`` when a job's process/scheduler run begins.
    job_started: Signal
    #: Emitted with ``(job_id, CastepResult, ParsedJobArtifacts)`` on successful
    #: completion. The third payload carries the secondary outputs (relaxed cell,
    #: phonon, transition state) parsed once on the runner worker thread.
    job_finished: Signal
    #: Emitted with ``(job_id, message)`` on failure.
    job_failed: Signal
    #: Emitted with ``(job_id)`` when a job is cancelled.
    job_cancelled: Signal
    #: Emitted with ``(job_id, status_value)`` for lifecycle status updates.
    status_changed: Signal

    def submit(self, **kwargs: Any) -> Any:
        """Submit one job for execution.

        The local runner never raises and returns ``None`` (see the module
        docstring for the cluster runner's different contract).
        """
        ...

    def cancel(self, job_id: UUID) -> None:
        """Cancel a job the runner holds."""
        ...
