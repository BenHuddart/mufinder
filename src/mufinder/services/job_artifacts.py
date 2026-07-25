"""Off-GUI-thread parsing of a completed job's optional CASTEP outputs.

When a CASTEP job finishes, MuFinder parses several *secondary* output files
beyond the primary ``.castep`` result: the relaxed ``-out.cell`` (relaxation
workflows), the ``.phonon`` file plus its derived quantum quantities (quantum
corrections), and the ``.ts`` transition-state file (energy barriers). These
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

from dataclasses import dataclass
from pathlib import Path

from mufinder.castep_io.cell_parser import ParsedCell, parse_cell_file
from mufinder.castep_io.ts_parser import TransitionStatePath, parse_transition_state_file
from mufinder.science.kpoint_grids import lattice_parameters
from mufinder.science.quantum_phonons import (
    PhononCalculation,
    muon_zpe,
    parse_phonon_path,
    w1_entanglement_witness,
)


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
    parse/compute. The GUI handler converts these into the same user-facing
    failure messages that the former inline guards produced.
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

    #: Source file paths the collector inspected (for diagnostics / provenance).
    out_cell_path: Path | None = None
    phonon_path: Path | None = None
    ts_path: Path | None = None


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
    num_farms: int = 1,
) -> ParsedJobArtifacts:
    """Parse a completed job's requested secondary outputs, capturing errors.

    ``want_out_cell`` / ``want_phonon`` / ``want_ts`` mirror the runner's
    per-calc-type retrieval flags (relaxation, quantum-phonon, and TS/NEB
    respectively), so collection is driven by which outputs the job actually
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
    if output_dir is not None:
        if want_out_cell:
            out_cell_path = output_dir / f"{seed_name}-out.cell"
        if want_phonon:
            phonon_path = output_dir / f"{seed_name}.phonon"
        if want_ts:
            ts_path = _resolve_ts_path(output_dir, seed_name, num_farms)

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
            # and leave the job stuck.
            try:
                relaxed_cell = parse_cell_file(out_cell_path)
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
        out_cell_path=out_cell_path,
        phonon_path=phonon_path,
        ts_path=ts_path,
    )


__all__ = [
    "ParsedJobArtifacts",
    "collect_job_artifacts",
]
