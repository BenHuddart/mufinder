"""Where a job's input and output files live on disk.

A job's directory is recorded on the calculation definition that owns it, in
one of several label or sweep-value mappings, so resolving it walks the
project's definitions. :func:`job_output_directory` returns that directory,
:func:`job_seed_name` the CASTEP seed inside it and :func:`job_output_files`
the standard output files present there.

Layering: this lives in ``mufinder.services`` and imports ``domain`` and
``workflow`` only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import posixpath
from typing import TYPE_CHECKING

from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES, is_muon_relaxation_calc_type
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job
from mufinder.workflow.naming import job_name

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument

#: Resolves the seed name a quantum-corrections row's phonon job uses from its
#: definition and source key.
QuantumSeedName = Callable[[CalculationDefinition, str], str]

#: Definition mapping binding a row to the singlepoint job older projects ran
#: before the phonon job. MuFinder no longer writes it; it is read so that job
#: keeps its directory.
QUANTUM_SINGLEPOINT_JOB_MAPPING = "singlepoint_job_ids_by_label"

#: Definition mapping recording the seed a row's legacy singlepoint ran under,
#: written when the row's phonon job is given a fresh seed of its own. A legacy
#: singlepoint without an entry shares the row's seed from
#: ``job_seed_names_by_label``.
QUANTUM_SINGLEPOINT_SEED_MAPPING = "singlepoint_seed_names_by_label"

_QUANTUM_JOB_MAPPINGS = (QUANTUM_SINGLEPOINT_JOB_MAPPING, "phonon_job_ids_by_label", "job_ids_by_label")

#: Standard output file suffixes, in the order :func:`job_output_files` lists them.
#: Task farming writes the band to ``<seed>_farm001.ts``; there is no unfarmed
#: ``.ts`` at all for a farmed NEB run.
_OUTPUT_FILE_SUFFIXES = (
    ".cell",
    ".param",
    ".castep",
    "_farm001.castep",
    ".den_fmt",
    "-out.cell",
    ".phonon",
    ".ts",
    "_farm001.ts",
)


def definition_output_directory(
    document: ProjectDocument | None,
    definition: CalculationDefinition,
) -> Path | None:
    """Return the output directory ``definition`` records, resolved against the project root.

    ``None`` when there is no document or the definition records no directory.
    """

    if document is None:
        return None
    raw_output_dir = definition.parameters.get("output_directory")
    if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
        return None
    base = Path(raw_output_dir)
    if not base.is_absolute():
        base = document.project_root / base
    return base


def remote_directory_seed_name(job: Job) -> str:
    """Return the last component of the job's remote directory, or ``""``."""

    remote_directory = (job.remote_directory or "").strip()
    if not remote_directory:
        return ""
    seed_name = posixpath.basename(posixpath.normpath(remote_directory)).strip()
    return seed_name if seed_name not in {"", ".", "/"} else ""


def _key_for_job(mapping: Mapping[object, object], job: Job) -> str:
    return next((str(key) for key, value in mapping.items() if str(value) == str(job.id)), "")


def _muon_relaxation_job_directory(base: Path, params: Mapping[str, object], job: Job) -> Path:
    job_ids_by_label = params.get("job_ids_by_label", {})
    matching_key = _key_for_job(job_ids_by_label, job) if isinstance(job_ids_by_label, dict) else ""
    prefix = "positivemuon" if job.calc_type == MUON_RELAXATION_CALC_TYPES["positive_muon"] else "muonium"
    seed_mapping = params.get("job_seed_names_by_label", {})
    seed_name = job.label
    if isinstance(seed_mapping, dict):
        raw_seed = seed_mapping.get(matching_key)
        if isinstance(raw_seed, str) and raw_seed.strip():
            seed_name = raw_seed.strip()
    if seed_name == job.label:
        remote_seed = remote_directory_seed_name(job)
        if remote_seed:
            seed_name = remote_seed
    if seed_name == job.label:
        inferred_seed = job_name(prefix, job.label)
        inferred_path = base / prefix / inferred_seed
        if inferred_path.exists() and inferred_path.is_dir():
            seed_name = inferred_seed
    return base / prefix / seed_name


def job_output_directory(
    document: ProjectDocument | None,
    job: Job,
    *,
    quantum_seed_name: QuantumSeedName,
) -> Path | None:
    """Return the directory holding ``job``'s input and output files, or ``None``.

    The definitions are searched in project order. A definition claims the job
    through ``job_ids_by_cutoff`` (``<output>/<label>``), a quantum-corrections
    job mapping (``<output>/<seed>``, the seed from ``quantum_seed_name``; a
    legacy singlepoint job uses its own seed from
    ``singlepoint_seed_names_by_label`` when the row records one),
    ``job_ids_by_label`` (``<output>/<label>``, or
    ``<output>/<positivemuon|muonium>/<seed>`` for a muon relaxation) or
    ``job_labels`` when ``<output>/<label>`` exists. A job no definition claims
    falls back to the first existing directory among
    ``files/<stage>/<label>``, ``files/convergence_cutoff/<label>`` and
    ``files/<label>`` under the project root.
    """

    if document is None:
        return None

    for definition in document.calculations.values():
        params = definition.parameters
        job_ids_by_cutoff = params.get("job_ids_by_cutoff", {})
        if isinstance(job_ids_by_cutoff, dict) and str(job.id) in {
            str(value) for value in job_ids_by_cutoff.values()
        }:
            base = definition_output_directory(document, definition)
            if base is not None:
                return base / job.label

        if definition.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS:
            base = definition_output_directory(document, definition)
            if base is None:
                continue
            for mapping_name in _QUANTUM_JOB_MAPPINGS:
                mapping = params.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                matching_key = _key_for_job(mapping, job)
                if not matching_key:
                    continue
                if mapping_name == QUANTUM_SINGLEPOINT_JOB_MAPPING:
                    legacy_seeds = params.get(QUANTUM_SINGLEPOINT_SEED_MAPPING, {})
                    legacy_seed = legacy_seeds.get(matching_key) if isinstance(legacy_seeds, dict) else None
                    if isinstance(legacy_seed, str) and legacy_seed.strip():
                        return base / legacy_seed.strip()
                return base / quantum_seed_name(definition, matching_key)

        job_ids_by_label = params.get("job_ids_by_label", {})
        if isinstance(job_ids_by_label, dict) and str(job.id) in {
            str(value) for value in job_ids_by_label.values()
        }:
            base = definition_output_directory(document, definition)
            if base is not None:
                if definition.workflow_stage == WorkflowStage.MUON_RELAXATION and is_muon_relaxation_calc_type(
                    job.calc_type
                ):
                    return _muon_relaxation_job_directory(base, params, job)
                return base / job.label

        labels = params.get("job_labels", [])
        if not isinstance(labels, list) or job.label not in labels:
            continue

        base = definition_output_directory(document, definition)
        if base is None:
            continue
        candidate = base / job.label
        if candidate.exists() and candidate.is_dir():
            return candidate

    fallback_candidates = (
        document.project_root / "files" / job.workflow_stage.value / job.label,
        document.project_root / "files" / "convergence_cutoff" / job.label,
        document.project_root / "files" / job.label,
    )
    for fallback in fallback_candidates:
        if fallback.exists() and fallback.is_dir():
            return fallback
    return None


def job_seed_name(job: Job, output_directory: Callable[[Job], Path | None]) -> str:
    """Return the CASTEP seed name ``job`` runs under.

    The remote directory's last component wins; otherwise the name of the
    directory ``output_directory`` resolves; otherwise the job label.
    ``output_directory`` is called only when the remote directory gives no seed.
    """

    remote_seed = remote_directory_seed_name(job)
    if remote_seed:
        return remote_seed
    output_dir = output_directory(job)
    if output_dir is not None:
        return output_dir.name
    return job.label


def job_output_files(project_root: Path, output_dir: Path | None, seed_name: str) -> list[str]:
    """Return the standard CASTEP files for ``seed_name`` present in ``output_dir``.

    Paths inside ``project_root`` are returned relative to it; others are
    returned absolute. An absent ``output_dir`` yields no files.
    """

    if output_dir is None:
        return []
    files: list[str] = []
    for suffix in _OUTPUT_FILE_SUFFIXES:
        path = output_dir / f"{seed_name}{suffix}"
        if not path.exists():
            continue
        try:
            files.append(str(path.relative_to(project_root)))
        except ValueError:
            files.append(str(path))
    return files
