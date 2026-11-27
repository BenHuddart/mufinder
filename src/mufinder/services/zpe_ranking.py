"""The zero-point-corrected ranking of a project's relaxed muon sites.

The Quantum Corrections table, the Journal's ZPE comparison and the project survey's
ZPE markers all read :func:`zpe_ranking`, so they agree. A site's result is the newest
of the phonon job its row binds in the current definition, while that job is completed.
A ZPE that :func:`~mufinder.services.source_candidates.phonon_zpe_source_state` cannot
tie to the site's current relaxation run is left out of the ranking, and a result with
a non-positive muon mode cannot set its group's zero
(:func:`~mufinder.services.source_candidates.quantum_source_corrected_relative_energies`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, Result, as_aware_utc
from mufinder.services.job_results import parse_numeric
from mufinder.services.project_document import (
    ProjectDocument,
    active_definition,
    job_for_mapping_key,
)
from mufinder.services.quantum_results import (
    has_non_positive_muon_mode,
    phonon_warnings_from_result_data,
)
from mufinder.services.source_candidates import (
    CorrectedEnergies,
    PhononZpeSource,
    QuantumSourceCandidate,
    phonon_zpe_source_state,
    quantum_source_corrected_relative_energies,
)
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_SET_CALC_TYPE

#: The definition mapping that binds each row to its phonon job.
PHONON_JOB_MAPPING = "phonon_job_ids_by_label"
#: The relaxation run each row's phonon inputs were last written from.
PHONON_SOURCE_RUN_MAPPING = "phonon_source_run_ids_by_label"


def phonon_job_for_row(
    document: ProjectDocument, definition: CalculationDefinition | None, key: str
) -> Job | None:
    """Return the row's phonon job, falling back to the ``job_ids_by_label`` alias."""

    job = job_for_mapping_key(
        document,
        definition,
        key,
        mapping_name=PHONON_JOB_MAPPING,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    if job is not None:
        return job
    return job_for_mapping_key(document, definition, key, calc_type=QUANTUM_PHONON_CALC_TYPE)


def newest_phonon_results(document: ProjectDocument) -> dict[UUID, Result]:
    """Return the newest phonon result of each job, or a result with no job under its own id.

    A result with no payload has nothing to show. A naive ``created_at`` from an
    older project is read as UTC.
    """

    newest: dict[UUID, Result] = {}
    for result in document.results.values():
        if result.result_type != QUANTUM_PHONON_CALC_TYPE or not isinstance(result.data, dict):
            continue
        run = (
            document.runs.get(result.calculation_run_id)
            if result.calculation_run_id is not None
            else None
        )
        owner = run.job_id if run is not None and run.job_id is not None else result.id
        current = newest.get(owner)
        if current is None or as_aware_utc(result.created_at) > as_aware_utc(current.created_at):
            newest[owner] = result
    return newest


@dataclass(frozen=True, slots=True)
class ZpeRanking:
    """Each relaxed site's phonon result and where it stands once its ZPE is counted.

    ``results`` holds only the sites with a result, by source key, and ``sources`` which
    relaxation each of those read. ``corrected`` gives ``dE + ZPE``, ``None`` for a site
    left out. ``flagged`` holds the sites with a non-positive muon mode.
    ``newest_results`` is :func:`newest_phonon_results`, including those of no listed site.
    """

    newest_results: Mapping[UUID, Result]
    results: Mapping[str, Result]
    sources: Mapping[str, PhononZpeSource]
    corrected: CorrectedEnergies
    flagged: frozenset[str]

    def zpe_eV(self, key: str) -> float | None:
        """Return the stored muon ZPE of a site's result, or ``None``."""

        result = self.results.get(key)
        return None if result is None else parse_numeric(result.data.get("muon_zpe_total_eV"))

    def corrected_eV(self, key: str) -> float | None:
        """Return a site's ``dE + ZPE``, or ``None`` when it is not ranked."""

        return self.corrected.values.get(key)

    def mark(self, key: str) -> str:
        """Return why a site's ZPE is not ranked (``earlier``, ``unconfirmed``), else nothing."""

        source = self.sources.get(key)
        return "" if source in (None, PhononZpeSource.CURRENT) else source.value

    def warned(self, key: str) -> bool:
        """Return whether a site's result was stored with a warning, so its numbers are marked."""

        return bool(self.warnings(key))

    def warnings(self, key: str) -> tuple[str, ...]:
        """Return the warnings stored with a site's result."""

        result = self.results.get(key)
        return () if result is None else phonon_warnings_from_result_data(result.data)

    def notes(self, key: str) -> tuple[str, ...]:
        """Return the lines on why a site's ``dE + ZPE`` reads as it does."""

        source = self.sources.get(key)
        if source is PhononZpeSource.EARLIER:
            return (
                "This ZPE is from an earlier geometry: the site's relaxation has run again since, "
                "so it is not ranked.",
            )
        if source is PhononZpeSource.UNCONFIRMED:
            return (
                "This project does not record which relaxation this ZPE came from, so it is not "
                "ranked.",
            )
        if key in self.corrected.zero_from_flagged:
            return (
                "Every ranked site in this group has a non-positive muon mode, so the zero is the "
                "lowest of them.",
            )
        return ()


def zpe_ranking(
    document: ProjectDocument, candidates: Sequence[QuantumSourceCandidate]
) -> ZpeRanking:
    """Return where each of ``candidates`` stands, from the current definition's phonon jobs.

    A job that is running again or has been reset no longer stands behind the numbers
    of an earlier run, so its site has no result.
    """

    definition = active_definition(
        document, WorkflowStage.QUANTUM_CORRECTIONS, calc_type=QUANTUM_PHONON_SET_CALC_TYPE
    )
    recorded = definition.parameters.get(PHONON_SOURCE_RUN_MAPPING) if definition else None
    newest = newest_phonon_results(document)
    results: dict[str, Result] = {}
    sources: dict[str, PhononZpeSource] = {}
    for candidate in candidates:
        job = phonon_job_for_row(document, definition, candidate.key)
        if job is None or job.status != JobStatus.COMPLETED or job.id not in newest:
            continue
        results[candidate.key] = newest[job.id]
        sources[candidate.key] = phonon_zpe_source_state(
            candidate,
            job,
            recorded.get(candidate.key) if isinstance(recorded, dict) else None,
        )
    flagged = frozenset(
        key for key, result in results.items() if has_non_positive_muon_mode(result.data)
    )
    corrected = quantum_source_corrected_relative_energies(
        candidates,
        {
            key: parse_numeric(result.data.get("muon_zpe_total_eV"))
            for key, result in results.items()
            if sources[key] is PhononZpeSource.CURRENT
        },
        flagged_keys=flagged,
    )
    return ZpeRanking(newest, results, sources, corrected, flagged)


__all__ = [
    "PHONON_JOB_MAPPING",
    "PHONON_SOURCE_RUN_MAPPING",
    "ZpeRanking",
    "newest_phonon_results",
    "phonon_job_for_row",
    "zpe_ranking",
]
