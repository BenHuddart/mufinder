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

from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, ProvenanceRecord, Result
from mufinder.services.job_results import parse_numeric
from mufinder.services.muon_sites_summary import pinned_source_selection
from mufinder.services.project_document import (
    ProjectDocument,
    definition_row_jobs,
    definitions_for_stage,
)
from mufinder.services.quantum_results import (
    mode_widths_from_result_data,
    phonon_warnings_from_result_data,
    w1_from_result_data,
)
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
from mufinder.services.zpe_ranking import ZpeRanking, zpe_ranking
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

    ranking = zpe_ranking(document, candidates)
    comparison = _zpe_comparison(candidate, candidates, ranking, label) if candidate else []
    group = quantum_source_energy_group(candidate) if candidate else None
    modes = data.get("muon_zpe_modes")
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
            "warnings": list(phonon_warnings_from_result_data(data)),
            "zpe_mark": ranking.mark(key),
            "ranking_notes": list(ranking.notes(key)),
            "comparison": comparison,
            "not_ranked": [
                f"{label(item)} ({ranking.mark(item.key)})"
                for item in candidates
                if item.key != key
                and quantum_source_energy_group(item) == group
                and ranking.mark(item.key)
            ],
            # About a site outside the ranking, the other sites' order says nothing.
            "ordering_changed": (
                _ordering_changed(comparison) if ranking.corrected_eV(key) is not None else None
            ),
            SUBJECT_KEY: key,
            FINGERPRINT_KEY: input_fingerprint(
                key,
                _row_settings(definition, key),
                str(candidate.run.id) if candidate is not None else None,
                # A rerun that brings the ZPE into the ranking is news, whatever its inputs.
                ranking.mark(key),
            ),
        },
        job_id=job.id,
    )


def _zpe_comparison(
    candidate: QuantumSourceCandidate,
    candidates: Sequence[QuantumSourceCandidate],
    ranking: ZpeRanking,
    label: Any,
) -> list[dict[str, object]]:
    """The sites ranked with ``candidate``, lowest ``ΔE + ZPE`` first.

    ``ΔE`` is relative to the lowest of the sites compared, and ``ΔE + ZPE`` is the
    ranking's, so a flagged site can read below zero.
    """

    group = quantum_source_energy_group(candidate)
    relative = quantum_source_relative_energies(candidates)
    compared = [
        (item, energy, corrected)
        for item in candidates
        if quantum_source_energy_group(item) == group
        and (energy := relative.get(item.key)) is not None
        and (corrected := ranking.corrected_eV(item.key)) is not None
    ]
    if not compared:
        return []
    lowest_energy = min(energy for _item, energy, _corrected in compared)
    comparison = [
        {
            "site_label": label(item),
            "relative_energy_eV": energy - lowest_energy,
            "zpe_eV": ranking.zpe_eV(item.key),
            "corrected_energy_eV": corrected,
            "warned": ranking.warned(item.key),
        }
        for item, energy, corrected in compared
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
