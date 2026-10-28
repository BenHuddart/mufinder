"""Off-GUI-thread parsing of a completed job's optional CASTEP outputs.

When a CASTEP job finishes, MuFinder parses several *secondary* output files
beyond the primary ``.castep`` result: the relaxed ``-out.cell`` (relaxation
workflows), the ``.phonon`` file plus its derived quantum quantities (quantum
corrections), the ``.ts`` transition-state file (energy barriers), and the
``.magres`` hyperfine tensors with the spin density at the muon from the
``.den_fmt`` (hyperfine coupling). These
parses used to run twice on the GUI thread per completed job — once as a
failure-detection guard in ``MainWindow._on_local_job_finished`` and again in
the owning workflow controller's ``build_result`` — serializing N jobs'
parse + quantum compute + re-parse work on the GUI thread during a completion
burst.

This module moves that work onto the runner worker threads and performs it
**exactly once** per job. Both the local and cluster runners call
:func:`collect_job_artifacts` on their own worker thread after the primary
``.castep`` parse succeeds, and deliver the resulting :class:`ParsedJobArtifacts`
alongside the ``CastepResult`` on the ``job_finished`` signal. The GUI handler
then decides failure purely from the carrier (never re-parsing), and the
workflow controllers consume the pre-parsed data instead of re-reading disk.

Errors are captured **as data** — never raised across the Qt signal boundary.
Each optional artifact is either present (parsed object) or accompanied by an
error string carrying the original exception message, so the GUI handler can
reproduce today's exact user-facing failure messages without touching disk.

Layering: this lives in ``mufinder.services`` and may import ``castep_io`` and
``science`` (both lower layers). It must not import ``app`` or ``workflow``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from mufinder.castep_io.cell_parser import ParsedCell, parse_cell_file
from mufinder.castep_io.charge_density import parse_formatted_density
from mufinder.castep_io.magres import (
    HyperfineTensor,
    hyperfine_tensor_for_species,
    parse_magres_hyperfine,
)
from mufinder.castep_io.ts_parser import TransitionStatePath, parse_transition_state_file
from mufinder.domain.charge_states import is_muon_relaxation_calc_type
from mufinder.science.hyperfine import spin_density_at
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.quantum_phonons import (
    PhononCalculation,
    muon_zpe,
    parse_phonon_path,
    w1_entanglement_witness,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE, HYPERFINE_MUON_SPECIES
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

_UEP_SOURCE_CALC_TYPE = "castep_uep_source"
_PRISTINE_RELAXATION_CALC_TYPE = "castep_pristine_relaxation"


@dataclass(frozen=True, slots=True)
class ParsedJobArtifacts:
    """Pre-parsed secondary outputs for one completed job.

    Every field is a plain Python object so the carrier can cross the Qt signal
    boundary from a runner worker thread to the GUI thread. Each optional
    artifact is populated only when the job's outputs called for it (driven by
    which retrieval flags the runner tracked); an unrequested artifact leaves
    both its parsed field and its error field ``None``.

    A ``*_error`` string is set (and the corresponding parsed field left
    ``None``) when the file was requested but was missing or failed to
    parse/compute. The owning workflow controller's ``validate_artifacts`` hook
    reads these flags and error strings and turns them into the job's failure
    message (see "Result dispatch" in ``docs/architecture.rst``).
    """

    #: Relaxed ``-out.cell`` parse (relaxation workflows). ``relaxed_cell`` is
    #: the parsed cell and ``lattice_abc`` its ``(a, b, c)`` lengths; both are
    #: ``None`` when ``cell_error`` is set or no ``-out.cell`` was requested.
    relaxed_cell: ParsedCell | None = None
    lattice_abc: tuple[float, float, float] | None = None
    cell_error: str | None = None
    #: ``True`` when the requested ``-out.cell`` was absent on disk (distinct
    #: from a present-but-unparseable file, which sets ``cell_error``).
    cell_missing: bool = False

    #: Parsed ``.phonon`` calculation (quantum-corrections workflow). ``None``
    #: when ``phonon_error`` is set or no phonon output was requested. The full
    #: ZPE / Gamma / witness computation is exercised during collection so a
    #: file that parses but fails downstream still surfaces via ``phonon_error``.
    phonon: PhononCalculation | None = None
    phonon_error: str | None = None
    #: ``True`` when the requested ``.phonon`` file was absent on disk.
    phonon_missing: bool = False

    #: Parsed ``.ts`` transition-state path (energy-barriers workflow). ``None``
    #: when ``ts_error`` is set or no ``.ts`` output was requested.
    transition_state: TransitionStatePath | None = None
    ts_error: str | None = None
    #: ``True`` when the requested ``.ts`` file was absent on disk.
    ts_missing: bool = False

    #: The muon's tensor from the ``.magres`` hyperfine output (hyperfine
    #: coupling), exactly as CASTEP prints it. ``None`` when ``magres_error`` is
    #: set or no magres output was requested.
    magres: HyperfineTensor | None = None
    magres_error: str | None = None
    #: ``True`` when the requested ``.magres`` file was absent on disk.
    magres_missing: bool = False
    #: Pseudo spin density at the muon, in Å^-3, from the job's ``.den_fmt``
    #: and the muon position in its ``.cell``. ``None`` with
    #: ``spin_density_error`` set when either could not be read; a missing
    #: density is a warning, not a failure.
    pseudo_spin_density_angstrom3: float | None = None
    spin_density_error: str | None = None

    #: Source file paths the collector inspected (for diagnostics / provenance).
    out_cell_path: Path | None = None
    phonon_path: Path | None = None
    ts_path: Path | None = None
    magres_path: Path | None = None


def effective_num_farms(task_farming_enabled: bool, num_farms: int | None) -> int:
    """Collapse a runner's two farming fields into a single farm count.

    Both runners carry ``task_farming_enabled`` alongside an optional
    ``num_farms``; farming is only actually in effect when the flag is set *and*
    the count exceeds one. Returns 1 whenever farming is off, so callers can
    treat the result as "the number of farms CASTEP was told to use".
    """

    if not task_farming_enabled:
        return 1
    return max(1, int(num_farms or 1))


@dataclass(frozen=True, slots=True)
class RetrievalFlags:
    """Which secondary outputs a job produces, and how CASTEP named them.

    One value per job, threaded from submission through both runners to
    :func:`collect_job_artifacts`. The ``retrieve_*`` flags select the optional
    outputs (``.den_fmt``, relaxed ``-out.cell``, ``.phonon``, ``.ts``); the
    farming pair says whether task farming renamed them to
    ``<seed>_farmNNN.<ext>``. The local runner reads but never downloads
    ``.den_fmt``, so it ignores ``retrieve_den_fmt``.

    The field names are also the runners' keyword names: ``submit``,
    ``track_existing_job`` and ``retrieve_outputs_for`` take the fields as
    keywords and rebuild the value with ``RetrievalFlags(**kwargs)``, and
    :meth:`as_kwargs` produces them. Construction normalises the values, so two
    equal requests always compare equal.
    """

    retrieve_den_fmt: bool = False
    retrieve_out_cell: bool = False
    retrieve_phonon: bool = False
    retrieve_ts: bool = False
    retrieve_magres: bool = False
    task_farming_enabled: bool = False
    num_farms: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "retrieve_den_fmt",
            "retrieve_out_cell",
            "retrieve_phonon",
            "retrieve_ts",
            "retrieve_magres",
            "task_farming_enabled",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.num_farms is not None:
            object.__setattr__(self, "num_farms", max(1, int(self.num_farms)))

    @classmethod
    def for_calc_type(
        cls,
        calc_type: str,
        *,
        task_farming_enabled: bool = False,
        num_farms: int | None = None,
    ) -> "RetrievalFlags":
        """Return the flags for a job of ``calc_type``.

        UEP source jobs keep their ``.den_fmt``; pristine and muon relaxations
        their ``-out.cell``; quantum-phonon jobs their ``.phonon``;
        transition-state and NEB jobs their ``.ts``; and hyperfine jobs their
        ``.magres`` and ``.den_fmt``. Every other calc type
        produces only the primary ``.castep``. The farm count is per job, not
        per calc type, so the caller supplies it.
        """

        return cls(
            retrieve_den_fmt=calc_type in {_UEP_SOURCE_CALC_TYPE, HYPERFINE_CALC_TYPE},
            retrieve_out_cell=calc_type == _PRISTINE_RELAXATION_CALC_TYPE
            or is_muon_relaxation_calc_type(calc_type),
            retrieve_phonon=calc_type == QUANTUM_PHONON_CALC_TYPE,
            retrieve_ts=calc_type
            in {TRANSITION_STATE_SEARCH_CALC_TYPE, NUDGED_ELASTIC_BAND_CALC_TYPE},
            retrieve_magres=calc_type == HYPERFINE_CALC_TYPE,
            task_farming_enabled=task_farming_enabled,
            num_farms=num_farms,
        )

    @property
    def farm_count(self) -> int:
        """Return the farm count in effect; see :func:`effective_num_farms`."""

        return effective_num_farms(self.task_farming_enabled, self.num_farms)

    def as_kwargs(self) -> dict[str, Any]:
        """Return the fields as the runner keyword arguments they travel as."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    def collect(self, output_dir: Path | None, seed_name: str) -> "ParsedJobArtifacts":
        """Parse the secondary outputs these flags request; see :func:`collect_job_artifacts`."""

        return collect_job_artifacts(
            output_dir,
            seed_name,
            want_out_cell=self.retrieve_out_cell,
            want_phonon=self.retrieve_phonon,
            want_ts=self.retrieve_ts,
            want_magres=self.retrieve_magres,
            num_farms=self.farm_count,
        )


def _resolve_ts_path(output_dir: Path, seed_name: str, num_farms: int) -> Path:
    """Return the ``.ts`` file a job with ``num_farms`` farms should have written.

    CASTEP appends ``_farmNNN`` to the seedname before the task runs, so a farmed
    run writes ``<seed>_farm001.ts`` … ``<seed>_farmNNN.ts`` and **no**
    ``<seed>.ts``. Every farm writes a complete, identical band (the images are
    synchronised across farms before each write), so farm 001 is the canonical
    copy — the same choice the runners already make for ``.castep``.

    Note there is deliberately **no** fallback to ``<seed>.ts`` when farming is
    on. Unlike ``.castep`` (where farmed runs still write a master file), a
    farmed run never writes an unfarmed ``.ts``, so any ``<seed>.ts`` present is
    a leftover from an earlier unfarmed run of the same seed. CASTEP *appends*
    to an existing ``.ts``, so reading it could splice two runs' bands together
    and report a barrier that was never computed. Reporting the farmed file as
    missing is the honest outcome.
    """

    if max(1, int(num_farms)) > 1:
        return output_dir / f"{seed_name}_farm001.ts"
    return output_dir / f"{seed_name}.ts"


def collect_job_artifacts(
    output_dir: Path | None,
    seed_name: str,
    *,
    want_out_cell: bool = False,
    want_phonon: bool = False,
    want_ts: bool = False,
    want_magres: bool = False,
    num_farms: int = 1,
) -> ParsedJobArtifacts:
    """Parse a completed job's requested secondary outputs, capturing errors.

    ``want_out_cell`` / ``want_phonon`` / ``want_ts`` / ``want_magres`` mirror
    the runner's per-calc-type retrieval flags (relaxation, quantum-phonon,
    TS/NEB and hyperfine respectively), so collection is driven by which outputs the job actually
    produced. This runs on a runner worker thread; it never raises — every
    failure is captured on the returned :class:`ParsedJobArtifacts` as data.

    ``output_dir`` may be ``None`` (the runner could not resolve the directory);
    in that case every requested artifact is reported missing.

    ``num_farms`` is the CASTEP task-farm count the job ran with. Task farming
    renames every *output* to ``<seed>_farmNNN.<ext>``, so a farmed NEB run
    writes no ``<seed>.ts`` at all — see :func:`_resolve_ts_path`.
    """

    out_cell_path: Path | None = None
    phonon_path: Path | None = None
    ts_path: Path | None = None
    magres_path: Path | None = None
    if output_dir is not None:
        if want_out_cell:
            out_cell_path = output_dir / f"{seed_name}-out.cell"
        if want_phonon:
            phonon_path = output_dir / f"{seed_name}.phonon"
        if want_ts:
            ts_path = _resolve_ts_path(output_dir, seed_name, num_farms)
        if want_magres:
            magres_path = output_dir / f"{seed_name}.magres"

    relaxed_cell: ParsedCell | None = None
    lattice_abc: tuple[float, float, float] | None = None
    cell_error: str | None = None
    cell_missing = False
    if want_out_cell:
        if out_cell_path is None or not out_cell_path.exists():
            cell_missing = True
        else:
            # Catch broadly, not just CellParseError: the parser reads the file
            # via Path.read_text, so OSError / UnicodeDecodeError can also
            # surface here, and this function must never raise on the worker
            # thread — an escaping exception would skip the job_finished emit
            # and leave the job stuck. Only the geometry is read: an -out.cell
            # echoes the run's settings, which may lie outside the loaded
            # keyword catalog (EXTERNAL_PRESSURE, for one).
            try:
                relaxed_cell = parse_cell_file(out_cell_path, geometry_only=True)
                a, b, c = lattice_parameters(relaxed_cell.lattice_vectors)[:3]
                lattice_abc = (a, b, c)
            except Exception as exc:
                relaxed_cell = None
                lattice_abc = None
                cell_error = str(exc)

    phonon: PhononCalculation | None = None
    phonon_error: str | None = None
    phonon_missing = False
    if want_phonon:
        if phonon_path is None or not phonon_path.exists():
            phonon_missing = True
        else:
            try:
                phonon = parse_phonon_path(phonon_path)
                # Exercise the full downstream computation so a phonon file that
                # parses but fails to yield ZPE / Gamma / witness (missing Gamma
                # point, wrong H:mu atom count) is still reported as a failure.
                muon_zpe(phonon)
                phonon.gamma_point()
                w1_entanglement_witness(phonon)
            except Exception as exc:
                # Broader than PhononParseError for the same never-raise reason
                # as the cell branch: read_text I/O errors and unexpected
                # failures in the downstream compute must surface as data.
                phonon = None
                phonon_error = str(exc)

    transition_state: TransitionStatePath | None = None
    ts_error: str | None = None
    ts_missing = False
    if want_ts:
        if ts_path is None or not ts_path.exists():
            ts_missing = True
        else:
            try:
                transition_state = parse_transition_state_file(ts_path)
            except Exception as exc:
                # Broader than TransitionStateParseError for the same
                # never-raise reason as the cell branch.
                transition_state = None
                ts_error = str(exc)

    magres: HyperfineTensor | None = None
    magres_error: str | None = None
    magres_missing = False
    pseudo_spin_density: float | None = None
    spin_density_error: str | None = None
    if want_magres:
        if magres_path is None or not magres_path.exists():
            magres_missing = True
        else:
            try:
                magres = hyperfine_tensor_for_species(
                    parse_magres_hyperfine(magres_path), HYPERFINE_MUON_SPECIES
                )
            except Exception as exc:
                # Broader than MagresInputError for the same never-raise reason
                # as the cell branch.
                magres = None
                magres_error = str(exc)
        # The density only matters beside a tensor; a job without one fails anyway.
        if output_dir is not None and magres is not None:
            pseudo_spin_density, spin_density_error = _muon_pseudo_spin_density(
                output_dir, seed_name
            )

    return ParsedJobArtifacts(
        relaxed_cell=relaxed_cell,
        lattice_abc=lattice_abc,
        cell_error=cell_error,
        cell_missing=cell_missing,
        phonon=phonon,
        phonon_error=phonon_error,
        phonon_missing=phonon_missing,
        transition_state=transition_state,
        ts_error=ts_error,
        ts_missing=ts_missing,
        magres=magres,
        magres_error=magres_error,
        magres_missing=magres_missing,
        pseudo_spin_density_angstrom3=pseudo_spin_density,
        spin_density_error=spin_density_error,
        out_cell_path=out_cell_path,
        phonon_path=phonon_path,
        ts_path=ts_path,
        magres_path=magres_path,
    )


def _muon_pseudo_spin_density(output_dir: Path, seed_name: str) -> tuple[float | None, str | None]:
    """Return the pseudo spin density at the muon in Å^-3, or ``None`` and why not.

    The muon position is read from the job's own ``.cell``, the input the run
    used; the density from its ``.den_fmt``.
    """

    density_path = output_dir / f"{seed_name}.den_fmt"
    if not density_path.exists():
        return None, f"Formatted density {density_path.name} was not found."
    try:
        cell = parse_cell_file(output_dir / f"{seed_name}.cell", geometry_only=True)
        muon_positions = [
            position
            for species, position in zip(cell.species, cell.fractional_positions, strict=True)
            if species == HYPERFINE_MUON_SPECIES
        ]
        if len(muon_positions) != 1:
            return None, f"Expected one {HYPERFINE_MUON_SPECIES} atom in {seed_name}.cell."
        return spin_density_at(parse_formatted_density(density_path), muon_positions[0]), None
    except Exception as exc:
        # Never raise on the worker thread: an unreadable density is a warning.
        return None, str(exc)


__all__ = [
    "ParsedJobArtifacts",
    "RetrievalFlags",
    "collect_job_artifacts",
    "effective_num_farms",
]
