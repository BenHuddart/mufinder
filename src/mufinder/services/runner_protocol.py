"""Structural contract shared by the local and cluster job runners.

This module codifies -- for type-checking and documentation -- the surface that
callers (:class:`mufinder.services.job_execution_controller.JobExecutionController`
and the ``MainWindow`` signal wiring) rely on from both
:class:`mufinder.services.local_runner.LocalJobRunner` and
:class:`mufinder.services.cluster_runner.ClusterJobRunner`. It adds no runtime
behavior; the runner classes are *not* changed to inherit from it. Conformance is
structural and is verified in ``tests/services/test_runner_protocol.py``.

State authority (who may mutate what)
-------------------------------------

Job state transitions are owned **exclusively** by ``ProjectService`` primitives
(``submit_job``, ``mark_job_queued``, ``start_job``, ``mark_job_retrieving``,
``complete_job``, ``stop_job``, ``fail_job``, ``cancel_job``,
``reset_failed_job``). ``JobExecutionController`` mutates only bookkeeping fields
on the ``Job`` (``execution_target``, ``num_processes``, remote metadata clears)
and routes every status change through the service.

**Runners never touch ``Job`` objects.** They operate on their own in-memory
tracking state and communicate solely by emitting signals keyed by job id. The
controller translates those signals into ``ProjectService`` calls. A runner that
mutated a ``Job`` would violate this contract.

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
  because it performs blocking SSH/SFTP work. Its two callers convert the
  exception rather than letting it escape:

  * ``JobExecutionController.submit_cluster_job`` catches the exception and
    returns a failed ``ClusterJobSubmissionOutcome`` (via
    ``fail_cluster_submission``).
  * ``ClusterController._run_submit_batch`` catches per job and emits
    ``job_submit_finished(job_id, None, str(exc))``.

``cancel`` is identical on both runners: ``cancel(job_id)`` cancels a tracked
non-terminal job and returns ``None``.

Modeling note
-------------

``submit`` is modeled loosely as ``submit(self, **kwargs: Any) -> Any`` because
the two implementations take materially different keyword arguments and return
different types (``LocalJobRunner.submit`` returns ``None``;
``ClusterJobRunner.submit`` returns the remote scheduler id ``str``). Forcing a
single concrete signature would be dishonest; the per-implementation contracts
above are the real specification. ``cancel`` is modeled precisely because both
runners share the exact ``cancel(self, job_id: UUID) -> None`` signature.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from PySide6.QtCore import Signal


@runtime_checkable
class JobRunner(Protocol):
    """Common signal-and-method surface of the local and cluster runners.

    The five signals below are what ``MainWindow`` connects and what the
    lifecycle depends on; see the module docstring for the state-authority and
    submit error contracts. ``runtime_checkable`` only verifies attribute
    *presence*, so the conformance tests additionally assert the signal
    parameter counts (1, 2, 2, 1, 2 respectively).
    """

    #: Emitted with ``(job_id)`` when a job's process/scheduler run begins.
    job_started: Signal
    #: Emitted with ``(job_id, CastepResult)`` on successful completion.
    job_finished: Signal
    #: Emitted with ``(job_id, message)`` on failure.
    job_failed: Signal
    #: Emitted with ``(job_id)`` when a job is cancelled.
    job_cancelled: Signal
    #: Emitted with ``(job_id, status_value)`` for lifecycle status updates.
    status_changed: Signal

    def submit(self, **kwargs: Any) -> Any:
        """Submit one job for execution.

        Signature and return type are per-implementation (see module docstring):
        the local runner never raises and returns ``None``; the cluster runner
        may raise and returns the remote scheduler id.
        """
        ...

    def cancel(self, job_id: UUID) -> None:
        """Cancel a tracked, non-terminal job. Identical on both runners."""
        ...
