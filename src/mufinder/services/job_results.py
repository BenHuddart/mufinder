"""Scalar readings from a job's stored result.

A completed run stores its parsed CASTEP result as ``Result.data``. The readers
here take the result of a job's latest run (highest ``run_index``) and pull
one scalar from it: the total energy, the maximum force or the maximum stress.
Each scalar may be stored under several keys, directly or in a nested group,
as a number or as a ``{"value": ...}`` mapping; booleans and NaN are not
readings.

Layering: this lives in ``mufinder.services`` and imports ``domain`` only.
"""

from __future__ import annotations

from math import isnan
from typing import TYPE_CHECKING
from uuid import UUID

from mufinder.domain.models import Result

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument

_TOTAL_ENERGY_KEYS = (
    "total_energy",
    "total_energy_ev",
    "final_energy",
    "final_energy_ev",
    "energy",
    "energy_ev",
)
_TOTAL_ENERGY_GROUP = ("energies", ("total", "total_energy", "final", "final_energy", "energy"))
_MAX_FORCE_KEYS = ("max_force", "maximum_force", "max_force_eva")
_MAX_FORCE_GROUP = ("forces", ("max", "max_force", "maximum"))
_MAX_STRESS_KEYS = ("max_stress", "maximum_stress", "max_stress_gpa")
_MAX_STRESS_GROUP = ("stress", ("max", "max_stress", "maximum"))


def parse_numeric(candidate: object) -> float | None:
    """Return ``candidate`` as a float, unwrapping ``{"value": ...}`` mappings.

    Booleans, NaN and anything non-numeric yield ``None``.
    """

    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, (int, float)):
        numeric = float(candidate)
        if isnan(numeric):
            return None
        return numeric
    if isinstance(candidate, dict):
        return parse_numeric(candidate.get("value"))
    return None


def _extract(
    payload: object,
    keys: tuple[str, ...],
    group: tuple[str, tuple[str, ...]],
) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = parse_numeric(payload.get(key))
        if value is not None:
            return value
    group_name, group_keys = group
    nested = payload.get(group_name)
    if isinstance(nested, dict):
        for key in group_keys:
            value = parse_numeric(nested.get(key))
            if value is not None:
                return value
    return None


def extract_total_energy(payload: object) -> float | None:
    """Return the total energy in a stored result payload, or ``None``."""

    return _extract(payload, _TOTAL_ENERGY_KEYS, _TOTAL_ENERGY_GROUP)


def extract_max_force(payload: object) -> float | None:
    """Return the maximum force in a stored result payload, or ``None``."""

    return _extract(payload, _MAX_FORCE_KEYS, _MAX_FORCE_GROUP)


def extract_max_stress(payload: object) -> float | None:
    """Return the maximum stress in a stored result payload, or ``None``."""

    return _extract(payload, _MAX_STRESS_KEYS, _MAX_STRESS_GROUP)


def latest_result_for_job(document: ProjectDocument | None, job_id: UUID) -> Result | None:
    """Return the result of ``job_id``'s highest-index run, or ``None``."""

    if document is None:
        return None
    runs = [run for run in document.runs.values() if run.job_id == job_id]
    if not runs:
        return None
    latest_run = max(runs, key=lambda run: run.run_index)
    if latest_run.result_id is None:
        return None
    return document.results.get(latest_run.result_id)


def total_energy_for_job(document: ProjectDocument | None, job_id: UUID) -> float | None:
    """Return the total energy of ``job_id``'s latest run, or ``None``."""

    result = latest_result_for_job(document, job_id)
    return None if result is None else extract_total_energy(result.data)


def max_force_for_job(document: ProjectDocument | None, job_id: UUID) -> float | None:
    """Return the maximum force of ``job_id``'s latest run, or ``None``."""

    result = latest_result_for_job(document, job_id)
    return None if result is None else extract_max_force(result.data)


def max_stress_for_job(document: ProjectDocument | None, job_id: UUID) -> float | None:
    """Return the maximum stress of ``job_id``'s latest run, or ``None``."""

    result = latest_result_for_job(document, job_id)
    return None if result is None else extract_max_stress(result.data)
