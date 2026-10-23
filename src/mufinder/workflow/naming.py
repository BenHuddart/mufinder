"""Deterministic, filesystem-safe job naming helpers.

:func:`job_name` names a job from a prefix and suffix. The remote helpers name
the cluster side of a submission: :func:`cluster_workflow_directory_fragment`
is the working-directory fragment for a batch of jobs (and the key the
last-used cluster submission settings are stored under),
:func:`default_cluster_job_name_prefix` the scheduler job-name prefix offered
for that fragment, and :func:`sanitize_remote_path_component` /
:func:`sanitize_remote_path_fragment` keep both to remote-safe characters.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES, is_muon_relaxation_calc_type
from mufinder.domain.models import Job


def _sanitize(token: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", token)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def job_name(prefix: str, suffix: str = "") -> str:
    """Create a deterministic filesystem-safe job name."""

    raw = f"{prefix}_{suffix}" if suffix else prefix
    normalized = _sanitize(raw)
    if not normalized:
        raise ValueError("Job name must contain at least one valid character.")
    return normalized


def sanitize_remote_path_component(value: str) -> str:
    """Return ``value`` as one remote path component.

    Whitespace runs become ``_``, every other character outside letters,
    digits, dot, underscore and hyphen becomes ``_``, and leading or trailing
    ``_`` are stripped. An empty result falls back to ``"workflow"``.
    """

    cleaned = re.sub(r"\s+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned.strip("_") or "workflow"


def sanitize_remote_path_fragment(value: str) -> str:
    """Sanitize each ``/``-separated component of ``value``, dropping empty ones.

    An empty result falls back to ``"workflow"``.
    """

    parts = [
        sanitize_remote_path_component(part) for part in value.strip().split("/") if part.strip()
    ]
    return "/".join(parts) or "workflow"


def default_cluster_job_name_prefix(workflow_type: str | None) -> str:
    """Return the scheduler job-name prefix offered for ``workflow_type``.

    ``pristine_relaxation`` gives ``pristine_relax``, ``uep`` gives ``uep``, a
    fragment ending in ``positivemuon`` or ``muonium`` gives that charge state,
    and anything else gives ``convergence``.
    """

    normalized = (workflow_type or "").strip().strip("/")
    workflow_defaults = {
        "pristine_relaxation": "pristine_relax",
        "uep": "uep",
    }
    if normalized in workflow_defaults:
        return workflow_defaults[normalized]
    last_component = normalized.rsplit("/", 1)[-1]
    if last_component in {"positivemuon", "muonium"}:
        return last_component
    return "convergence"


def cluster_workflow_directory_fragment(jobs: Sequence[Job]) -> str:
    """Return the remote workflow fragment for a batch of ``jobs``.

    Muon relaxations of one charge state give ``muon_relaxation/positivemuon``
    or ``muon_relaxation/muonium``, and of both charge states
    ``muon_relaxation_mixed_charge_states``. Otherwise jobs sharing one
    workflow stage give that stage's sanitized value, and a mixed batch gives
    ``mixed_workflows``.
    """

    muon_calc_types = {job.calc_type for job in jobs if is_muon_relaxation_calc_type(job.calc_type)}
    if muon_calc_types and len(muon_calc_types) == len({job.calc_type for job in jobs}):
        if len(muon_calc_types) == 1:
            calc_type = next(iter(muon_calc_types))
            prefix = (
                "positivemuon"
                if calc_type == MUON_RELAXATION_CALC_TYPES["positive_muon"]
                else "muonium"
            )
            return sanitize_remote_path_fragment(f"muon_relaxation/{prefix}")
        return "muon_relaxation_mixed_charge_states"
    workflow_values = {job.workflow_stage.value for job in jobs}
    if len(workflow_values) == 1:
        return sanitize_remote_path_component(next(iter(workflow_values)))
    return "mixed_workflows"
