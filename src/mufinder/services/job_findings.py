"""Findings of completed jobs: a site's zero-point energy and a path's energy barrier.

:meth:`~mufinder.services.job_execution_controller.JobExecutionController.complete_job_with_result`
calls :func:`record_job_finding` once a job's result is stored. See
:mod:`mufinder.services.study_findings` for what a finding is. A job's fingerprint
is its row, the row's stored settings and what the row was calculated from, and its
subject is its row, so running a reset job again with the same inputs records nothing
new that day.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, ProvenanceRecord, Result
from mufinder.services.job_results import latest_result_for_job, parse_numeric
from mufinder.services.muon_sites_summary import pinned_source_selection
from mufinder.services.project_document import (
    ProjectDocument,
    definition_row_jobs,
    definitions_for_stage,
)
from mufinder.services.quantum_results import mode_widths_from_result_data, w1_from_result_data
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_for_job,
    eligible_quantum_source_candidates,
    energy_barrier_neb_sources_by_job_id,
    energy_barrier_source_site_labels,
    neb_barrier_energies,
    quantum_source_energy_group,
    quantum_source_relative_energies,
)
from mufinder.services.study_findings import (
    FINGERPRINT_KEY,
    SUBJECT_KEY,
    input_fingerprint,
    record_finding,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
)
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_SET_CALC_TYPE


def record_job_finding(document: ProjectDocument, job: Job, result: Result) -> bool:
    """Record the finding of ``job``'s stored ``result``, if its kind concludes anything."""

    record: ProvenanceRecord | None = None
    if result.result_type == QUANTUM_PHONON_CALC_TYPE:
        record = _quantum_corrected_record(document, job, result)
    elif result.result_type == NUDGED_ELASTIC_BAND_CALC_TYPE:
        record = _barrier_calculated_record(document, job, result)
    return record_finding(document, record) if record is not None else False


def _row_settings(definition: CalculationDefinition | None, key: str) -> object:
    """Return the settings stored for row ``key`` of ``definition``, as they were saved."""

    configuration = definition.parameters.get("configuration") if definition else None
    rows = configuration.get("row_settings") if isinstance(configuration, Mapping) else None
    return rows.get(key) if isinstance(rows, Mapping) else None


def _current_rows(
    definitions: Sequence[CalculationDefinition],
) -> dict[str, tuple[CalculationDefinition, str]]:
    """Each row's newest binding, with the job id's text; ``definitions`` run newest first."""

    rows: dict[str, tuple[CalculationDefinition, str]] = {}
    for definition in definitions:
        for key, job_id in definition_row_jobs(definition):
            rows.setdefault(key, (definition, job_id))
    return rows


def _quantum_corrected_record(
    document: ProjectDocument, job: Job, result: Result
) -> ProvenanceRecord | None:
    data = result.data if isinstance(result.data, Mapping) else {}
    zpe = parse_numeric(data.get("muon_zpe_total_eV"))
    if zpe is None:
        return None
    rows = _current_rows(
        definitions_for_stage(
            document, WorkflowStage.QUANTUM_CORRECTIONS, calc_type=QUANTUM_PHONON_SET_CALC_TYPE
        )
    )
    definition, key = next(
        ((row[0], row_key) for row_key, row in rows.items() if row[1] == str(job.id)),
        (None, job.label),
    )
    candidates = eligible_quantum_source_candidates(document)
    pinned = pinned_source_selection(document, candidates).labels_by_key()
    by_key = {candidate.key: candidate for candidate in candidates}
    candidate = by_key.get(key)
    charge_state = charge_state_for_job(candidate.job) if candidate is not None else None

    def label(item: QuantumSourceCandidate) -> str:
        return pinned.get(item.key) or item.site.label or item.job.label

    comparison = (
        _zpe_comparison(document, candidate, candidates, rows, label)
        if candidate is not None
        else []
    )
    modes = data.get("muon_zpe_modes")
    raw_warnings = data.get("phonon_warnings")
    return ProvenanceRecord(
        action="quantum_corrected",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        actor="system",
        description=f"Stored the zero-point energy of {job.label}",
        parameters={
            "job_label": job.label,
            "site_label": label(candidate) if candidate is not None else job.label,
            "charge_state": charge_state,
            "charge_state_label": muon_charge_state_label(charge_state) if charge_state else None,
            "muon_zpe_total_eV": zpe,
            "mode_frequencies_cm": [
                mode.get("frequency_cm") for mode in modes if isinstance(mode, Mapping)
            ]
            if isinstance(modes, list)
            else [],
            "mode_widths_A": list(mode_widths_from_result_data(data)),
            "w1": w1_from_result_data(data),
            "warnings": list(raw_warnings) if isinstance(raw_warnings, list) else [],
            "comparison": comparison,
            "ordering_changed": _ordering_changed(comparison),
            SUBJECT_KEY: key,
            FINGERPRINT_KEY: input_fingerprint(
                key,
                _row_settings(definition, key),
                str(candidate.run.id) if candidate is not None else None,
            ),
        },
        job_id=job.id,
    )


def _zpe_comparison(
    document: ProjectDocument,
    candidate: QuantumSourceCandidate,
    candidates: Sequence[QuantumSourceCandidate],
    rows: Mapping[str, tuple[CalculationDefinition, str]],
    label: Any,
) -> list[dict[str, object]]:
    """The sites comparable with ``candidate`` that have a ZPE, lowest ``ΔE + ZPE`` first.

    ``ΔE`` is relative to the lowest of the sites compared, and ``ΔE + ZPE`` to the
    lowest corrected energy, so each column starts from zero.
    """

    group = quantum_source_energy_group(candidate)
    relative = quantum_source_relative_energies(candidates)
    zpe_by_key: dict[str, float] = {}
    for key, (_definition, job_id) in rows.items():
        try:
            stored = latest_result_for_job(document, UUID(job_id))
        except ValueError:
            continue
        value = (
            parse_numeric(stored.data.get("muon_zpe_total_eV"))
            if stored is not None and isinstance(stored.data, Mapping)
            else None
        )
        if value is not None:
            zpe_by_key[key] = value
    compared = [
        (item, energy, zpe_by_key[item.key])
        for item in candidates
        if quantum_source_energy_group(item) == group
        and (energy := relative.get(item.key)) is not None
        and item.key in zpe_by_key
    ]
    if not compared:
        return []
    lowest_energy = min(energy for _item, energy, _zpe in compared)
    lowest_corrected = min(energy + zpe for _item, energy, zpe in compared)
    comparison = [
        {
            "site_label": label(item),
            "relative_energy_eV": energy - lowest_energy,
            "zpe_eV": zpe,
            "corrected_energy_eV": energy + zpe - lowest_corrected,
        }
        for item, energy, zpe in compared
    ]
    comparison.sort(key=lambda row: (row["corrected_energy_eV"], row["site_label"]))
    return comparison


def _ordering_changed(comparison: Sequence[Mapping[str, Any]]) -> bool | None:
    """Whether adding the ZPE reorders the sites; ``None`` with fewer than two to compare."""

    if len(comparison) < 2:
        return None
    by_energy = sorted(comparison, key=lambda row: (row["relative_energy_eV"], row["site_label"]))
    return [row["site_label"] for row in by_energy] != [row["site_label"] for row in comparison]


def _path_end_labels(
    document: ProjectDocument, source: Mapping[str, Any] | None
) -> tuple[str, str] | None:
    """A path's ends by their pinned summary labels, else their site labels."""

    labels = energy_barrier_source_site_labels(document, source)
    if source is None or labels is None:
        return None
    candidates = eligible_quantum_source_candidates(document)
    pinned = pinned_source_selection(document, candidates).labels_by_key()
    by_relaxation = {
        (str(candidate.site.id), str(candidate.job.id)): pinned.get(candidate.key)
        for candidate in candidates
    }
    start = by_relaxation.get(
        (str(source.get("reactant_site_id")), str(source.get("reactant_job_id")))
    )
    end = by_relaxation.get((str(source.get("product_site_id")), str(source.get("product_job_id"))))
    return start or labels[0], end or labels[1]


def _barrier_calculated_record(
    document: ProjectDocument, job: Job, result: Result
) -> ProvenanceRecord | None:
    data = result.data if isinstance(result.data, Mapping) else {}
    forward, reverse, delta = neb_barrier_energies(data)
    if forward is None:
        return None
    definitions = definitions_for_stage(
        document, WorkflowStage.ENERGY_BARRIERS, calc_type=NUDGED_ELASTIC_BAND_SET_CALC_TYPE
    )
    source = energy_barrier_neb_sources_by_job_id(document, definitions).get(job.id)
    labels = _path_end_labels(document, source)
    key = str(source.get("key", "")) if source is not None else job.label
    definition = next(
        (
            definition
            for definition in definitions
            if (key, str(job.id)) in definition_row_jobs(definition)
        ),
        None,
    )
    raw_charge = str(source.get("charge_state", "")).strip() if source is not None else ""
    return ProvenanceRecord(
        action="barrier_calculated",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        actor="system",
        description=f"Stored the energy barrier of {job.label}",
        parameters={
            "job_label": job.label,
            "start_label": labels[0] if labels else None,
            "end_label": labels[1] if labels else None,
            "charge_state": raw_charge or None,
            "charge_state_label": muon_charge_state_label(raw_charge) if raw_charge else None,
            "forward_barrier_eV": forward,
            "reverse_barrier_eV": reverse,
            "endpoint_delta_eV": delta,
            "num_images": data.get("num_images"),
            "path_length_ang": parse_numeric(source.get("path_length_ang")) if source else None,
            SUBJECT_KEY: key,
            FINGERPRINT_KEY: input_fingerprint(key, _row_settings(definition, key), source),
        },
        job_id=job.id,
    )


__all__ = ["record_job_finding"]
