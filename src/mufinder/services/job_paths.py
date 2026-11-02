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
import os
from pathlib import Path
import posixpath
from typing import TYPE_CHECKING

from mufinder.domain.charge_states import (
    MUON_RELAXATION_CALC_TYPES,
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    is_muon_relaxation_calc_type,
    muon_charge_state_prefix,
    muon_relaxation_charge_state,
)
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

_QUANTUM_JOB_MAPPINGS = (
    QUANTUM_SINGLEPOINT_JOB_MAPPING,
    "phonon_job_ids_by_label",
    "job_ids_by_label",
)

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
    ".magres",
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
    prefix = (
        "positivemuon"
        if job.calc_type == MUON_RELAXATION_CALC_TYPES["positive_muon"]
        else "muonium"
    )
    seed_mapping = params.get("job_seed_names_by_label", {})
    seed_name = job.label
    if isinstance(seed_mapping, dict):
        raw_seed = seed_mapping.get(matching_key)
        if isinstance(raw_seed, str) and raw_seed.strip():
            # The folder that was written, if one has exactly this name: the trimmed
            # name is a different folder, possibly another job's. Otherwise trimmed,
            # as a saved seed with stray spaces has always been read.
            exact = (base / prefix / raw_seed).is_dir()
            seed_name = raw_seed if exact else raw_seed.strip()
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
                    legacy_seed = (
                        legacy_seeds.get(matching_key) if isinstance(legacy_seeds, dict) else None
                    )
                    if isinstance(legacy_seed, str) and legacy_seed.strip():
                        return base / legacy_seed.strip()
                return base / quantum_seed_name(definition, matching_key)

        job_ids_by_label = params.get("job_ids_by_label", {})
        if isinstance(job_ids_by_label, dict) and str(job.id) in {
            str(value) for value in job_ids_by_label.values()
        }:
            base = definition_output_directory(document, definition)
            if base is not None:
                if (
                    definition.workflow_stage == WorkflowStage.MUON_RELAXATION
                    and is_muon_relaxation_calc_type(job.calc_type)
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


def resolved_path(path: Path) -> Path:
    """Return ``path`` resolved, or as written when it cannot be: a link cycle raises."""

    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path


def job_directory_link(project_root: Path, directory: Path) -> Path | None:
    """Return the first link on the way to ``directory`` below ``files/``, or ``None``.

    Read from the path as written, since resolving would look through the link. A path
    spelled from elsewhere that lands inside ``files/`` is walked back up to it. Links above
    ``files/``, such as a project opened by another spelling of its path, are the user's own
    arrangement of their disk and do not count.
    """

    files_root = project_root.resolve() / "files"
    for root in (project_root / "files", files_root):
        if directory.is_relative_to(root):
            current = root
            for part in directory.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    return current
            return None
    if not resolved_path(directory).is_relative_to(files_root):
        return None
    for path in (directory, *directory.parents):
        if resolved_path(path) == files_root:
            return None
        if path.is_symlink():
            return path
    return None


def job_directory_on_disk(project_root: Path, directory: Path) -> bool:
    """Return whether there is anything at ``directory`` for a clean-up to deal with.

    A dangling link counts, and so does a missing folder reached through one: neither
    exists, and both have to be refused rather than read as nothing being there.
    """

    return directory.exists() or job_directory_link(project_root, directory) is not None


def job_directory_link_refusal(project_root: Path, directory: Path) -> str | None:
    """Return why ``directory`` is a link or reached through one, or ``None``.

    Checked before resolving, which would look through it: a link is not the job's folder,
    whatever it points at, and deleting one neither removes nor refuses.
    """

    if directory.is_symlink():
        return f"its folder is a link to somewhere else ({directory.name!r})"
    if (link := job_directory_link(project_root, directory)) is not None:
        return f"its folder is reached through a link ({link.name!r})"
    return None


def job_directory_refusal(project_root: Path, job: Job | None, directory: Path) -> str | None:
    """Return why ``directory`` must not be deleted as a job's own folder, or ``None``.

    The last check before any job folder goes, on every route. A job's folder comes
    from names saved in the project: ``.`` or ``..`` resolves to a folder holding every
    other job, an absolute name to a folder that is not the project's at all, and a link
    below ``files/`` to anywhere at all. So links are looked for before resolving, the
    place is checked once resolved, and a name alone is never enough.
    """

    if ".." in directory.parts:
        return f"its folder name leads out of its own folder ({directory.name!r})"
    if (link_refusal := job_directory_link_refusal(project_root, directory)) is not None:
        return link_refusal
    resolved = resolved_path(directory)
    files_root = project_root.resolve() / "files"
    if not resolved.is_relative_to(files_root):
        return f"its folder is outside this project's job folders ({resolved})"
    if len(resolved.relative_to(files_root).parts) < 2:
        return f"its folder is the folder of a whole workflow ({resolved.name!r})"
    charge_folders = {
        muon_charge_state_prefix(POSITIVE_MUON_CHARGE_STATE),
        muon_charge_state_prefix(MUONIUM_CHARGE_STATE),
    }
    # By place, not name alone: a job inside a charge folder may have a seed of that name.
    if (
        job is not None
        and is_muon_relaxation_calc_type(job.calc_type)
        and resolved.name in charge_folders
        and resolved.parent.name not in charge_folders
    ):
        return f"its folder is the one shared by a whole charge state ({resolved.name!r})"
    return None


def job_directory_foreign(project_root: Path, job: Job | None, directory: Path) -> str | None:
    """Return why ``directory`` is not where this muon relaxation's folder belongs, or ``None``.

    A muon relaxation's folder is ``<workflow>/<its charge>/<name>``, or the
    ``<workflow>/<name>`` the fallback for an unclaimed job finds. Its seed is saved in the
    project, so it can name a folder elsewhere, which is then some other work's even when no
    job record says so. Call after :func:`job_directory_refusal` has passed.
    """

    if job is None or not is_muon_relaxation_calc_type(job.calc_type):
        return None
    place = resolved_path(directory).relative_to(project_root.resolve() / "files").parts
    charge_state = muon_relaxation_charge_state(job.calc_type)
    charge_folder = muon_charge_state_prefix(charge_state) if charge_state else None
    if place[0] != job.workflow_stage.value:
        return f"its folder is in another workflow's job folders ({directory.name!r})"
    if len(place) > 2 and (len(place) != 3 or place[1] != charge_folder):
        return f"its folder is not inside its charge folder ({directory.name!r})"
    return None


def job_directory_misplaced(project_root: Path, job: Job | None, directory: Path) -> bool:
    """Return whether a saved name leads ``directory`` out of the job's place, a link aside.

    Nothing in the app can correct a saved name, so a removal leaves that folder alone and
    lets the job's records go. A link can be removed, and is refused instead.
    """

    if job_directory_link_refusal(project_root, directory) is not None:
        return False
    return bool(
        job_directory_refusal(project_root, job, directory)
        or job_directory_foreign(project_root, job, directory)
    )


def job_directory_label(project_root: Path, directory: Path) -> str:
    """Return ``directory`` relative to the project, enough to tell any two folders apart."""

    try:
        return Path(os.path.relpath(directory, project_root)).as_posix()
    except ValueError:  # On another drive.
        return str(directory)


def free_job_seed(seed: str, parent: Path) -> str:
    """Return ``seed``, suffixed until nothing in ``parent`` has its name.

    A job's folder is named for its seed, so this is the name a job takes when the one it
    would have had is in use.
    """

    if not (parent / seed).exists():
        return seed
    index = 2
    while (parent / f"{seed}_{index}").exists():
        index += 1
    return f"{seed}_{index}"


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
