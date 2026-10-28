"""Tests for the off-thread job-artifact collector.

:func:`~mufinder.services.job_artifacts.collect_job_artifacts` parses a completed
job's secondary CASTEP outputs (relaxed ``-out.cell``, ``.phonon``, ``.ts``)
exactly once and captures every failure as data on the returned
:class:`~mufinder.services.job_artifacts.ParsedJobArtifacts`. These tests cover
each requested-output branch: success, missing file, and unparseable file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.services import job_artifacts as job_artifacts_module
from mufinder.services.job_artifacts import (
    ParsedJobArtifacts,
    RetrievalFlags,
    collect_job_artifacts,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE


_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.3 0.4 0.5
%ENDBLOCK POSITIONS_FRAC
"""


_PHONON_TEXT = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   1   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   2   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   2   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   3   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   3   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.630841000000  0.000000000000
   4   1 -0.501570000000  0.000000000000     -0.324961000000  0.000000000000     -0.339089000000  0.000000000000
   4   2  0.536891000000  0.000000000000      0.459183000000  0.000000000000      0.169545000000  0.000000000000
   5   1  0.583042000000  0.000000000000     -0.144467000000  0.000000000000     -0.701203000000  0.000000000000
   5   2 -0.145760000000  0.000000000000      0.057786700000  0.000000000000      0.350601000000  0.000000000000
   6   1 -0.349315000000  0.000000000000      0.764127000000  0.000000000000     -0.392979000000  0.000000000000
   6   2  0.087328800000  0.000000000000     -0.305651000000  0.000000000000      0.196490000000  0.000000000000
"""


_TS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "transition_state_path.ts"


# --- default / no-request behavior ---


def test_collect_requests_nothing_by_default(tmp_path: Path) -> None:
    artifacts = collect_job_artifacts(tmp_path, "seed")
    assert artifacts == ParsedJobArtifacts()


def test_collect_none_output_dir_reports_requested_outputs_missing() -> None:
    artifacts = collect_job_artifacts(
        None, "seed", want_out_cell=True, want_phonon=True, want_ts=True
    )
    assert artifacts.cell_missing is True
    assert artifacts.phonon_missing is True
    assert artifacts.ts_missing is True
    assert artifacts.relaxed_cell is None
    assert artifacts.phonon is None
    assert artifacts.transition_state is None


# --- relaxed -out.cell ---


def test_collect_out_cell_success_carries_cell_and_lattice(tmp_path: Path) -> None:
    (tmp_path / "seed-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.cell_missing is False
    assert artifacts.cell_error is None
    assert artifacts.relaxed_cell is not None
    assert artifacts.lattice_abc == pytest.approx((5.1, 5.2, 5.3))
    assert "H:mu" in artifacts.relaxed_cell.species
    assert artifacts.out_cell_path == tmp_path / "seed-out.cell"


def test_collect_out_cell_reads_the_geometry_past_calculation_keywords(tmp_path: Path) -> None:
    """A real ``-out.cell`` echoes the run's settings; only the geometry is read.

    ``EXTERNAL_PRESSURE`` is outside the built-in keyword set, so validating the
    whole file against the catalog would report a finished relaxation as failed.
    """

    text = (
        _OUT_CELL_TEXT
        + """
%BLOCK EXTERNAL_PRESSURE
GPa
0.5 0.0 0.0
    0.5 0.0
        0.5
%ENDBLOCK EXTERNAL_PRESSURE

FIX_COM : true
SYMMETRY_TOL :     0.001000
"""
    )
    (tmp_path / "seed-out.cell").write_text(text, encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.cell_error is None
    assert artifacts.relaxed_cell is not None
    assert list(artifacts.relaxed_cell.species) == ["Si", "H:mu"]
    assert artifacts.lattice_abc == pytest.approx((5.1, 5.2, 5.3))


def test_collect_out_cell_missing_sets_flag_not_error(tmp_path: Path) -> None:
    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.cell_missing is True
    assert artifacts.cell_error is None
    assert artifacts.relaxed_cell is None
    assert artifacts.lattice_abc is None


def test_collect_out_cell_unparseable_captures_error(tmp_path: Path) -> None:
    (tmp_path / "seed-out.cell").write_text("not a valid cell file", encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.cell_missing is False
    assert artifacts.cell_error is not None
    assert artifacts.relaxed_cell is None
    assert artifacts.lattice_abc is None


# --- .phonon ---


def test_collect_phonon_success_carries_calculation(tmp_path: Path) -> None:
    (tmp_path / "seed.phonon").write_text(_PHONON_TEXT, encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_phonon=True)

    assert artifacts.phonon_missing is False
    assert artifacts.phonon_error is None
    assert artifacts.phonon is not None
    assert artifacts.phonon.atoms[1].species == "H:mu"
    assert artifacts.phonon_path == tmp_path / "seed.phonon"


def test_collect_phonon_missing_sets_flag(tmp_path: Path) -> None:
    artifacts = collect_job_artifacts(tmp_path, "seed", want_phonon=True)

    assert artifacts.phonon_missing is True
    assert artifacts.phonon_error is None
    assert artifacts.phonon is None


def test_collect_phonon_unparseable_captures_error(tmp_path: Path) -> None:
    (tmp_path / "seed.phonon").write_text("garbage without a header", encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_phonon=True)

    assert artifacts.phonon_missing is False
    assert artifacts.phonon_error is not None
    assert artifacts.phonon is None


def test_collect_phonon_downstream_compute_failure_captured_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phonon file that parses but fails the ZPE/Gamma/witness computation must
    surface as ``phonon_error``, not raise — mirroring the former inline guard
    that exercised the full computation."""

    from mufinder.science.quantum_phonons import PhononParseError

    (tmp_path / "seed.phonon").write_text(_PHONON_TEXT, encoding="utf-8")

    def _boom(_calc):
        raise PhononParseError("no Gamma point")

    monkeypatch.setattr(job_artifacts_module, "muon_zpe", _boom)

    artifacts = collect_job_artifacts(tmp_path, "seed", want_phonon=True)

    assert artifacts.phonon is None
    assert artifacts.phonon_error == "no Gamma point"


# --- .ts ---


def test_collect_ts_success_carries_path(tmp_path: Path) -> None:
    (tmp_path / "seed.ts").write_text(_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True)

    assert artifacts.ts_missing is False
    assert artifacts.ts_error is None
    assert artifacts.transition_state is not None
    assert len(artifacts.transition_state.images) >= 1
    assert artifacts.ts_path == tmp_path / "seed.ts"


def test_collect_ts_missing_sets_flag(tmp_path: Path) -> None:
    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True)

    assert artifacts.ts_missing is True
    assert artifacts.ts_error is None
    assert artifacts.transition_state is None


def test_collect_ts_reads_farmed_file_when_task_farming(tmp_path: Path) -> None:
    # A task-farmed CASTEP run renames every output to <seed>_farmNNN.<ext>, so
    # the band lands in seed_farm001.ts and no seed.ts is written at all.
    (tmp_path / "seed_farm001.ts").write_text(
        _TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True, num_farms=4)

    assert artifacts.ts_missing is False
    assert artifacts.transition_state is not None
    assert artifacts.ts_path == tmp_path / "seed_farm001.ts"


def test_collect_ts_ignores_stale_unfarmed_file_when_task_farming(tmp_path: Path) -> None:
    # Only a leftover unfarmed .ts from an earlier run is present. CASTEP appends
    # to an existing .ts, so reading it could splice two runs' bands together and
    # report a barrier that was never computed -- report missing instead.
    (tmp_path / "seed.ts").write_text(_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True, num_farms=2)

    assert artifacts.ts_missing is True
    assert artifacts.transition_state is None
    assert artifacts.ts_path == tmp_path / "seed_farm001.ts"


def test_collect_ts_uses_unfarmed_file_for_single_farm(tmp_path: Path) -> None:
    # num_farms=1 is "farming off": CASTEP does not rename anything.
    (tmp_path / "seed.ts").write_text(_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True, num_farms=1)

    assert artifacts.transition_state is not None
    assert artifacts.ts_path == tmp_path / "seed.ts"


def test_collect_ts_unparseable_captures_error(tmp_path: Path) -> None:
    (tmp_path / "seed.ts").write_text("not a valid ts file", encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True)

    assert artifacts.ts_missing is False
    assert artifacts.ts_error is not None
    assert artifacts.transition_state is None


# --- I/O errors are captured as data, never raised (worker-thread contract) ---


def test_collect_out_cell_io_error_captured_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "seed-out.cell").write_text("irrelevant", encoding="utf-8")

    def _io_boom(_path, **_kwargs):
        raise OSError("disk unplugged")

    monkeypatch.setattr(job_artifacts_module, "parse_cell_file", _io_boom)

    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.relaxed_cell is None
    assert artifacts.cell_error == "disk unplugged"


def test_collect_phonon_io_error_captured_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "seed.phonon").write_text("irrelevant", encoding="utf-8")

    def _io_boom(_path):
        raise OSError("disk unplugged")

    monkeypatch.setattr(job_artifacts_module, "parse_phonon_path", _io_boom)

    artifacts = collect_job_artifacts(tmp_path, "seed", want_phonon=True)

    assert artifacts.phonon is None
    assert artifacts.phonon_error == "disk unplugged"


def test_collect_ts_io_error_captured_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "seed.ts").write_text("irrelevant", encoding="utf-8")

    def _io_boom(_path):
        raise OSError("disk unplugged")

    monkeypatch.setattr(job_artifacts_module, "parse_transition_state_file", _io_boom)

    artifacts = collect_job_artifacts(tmp_path, "seed", want_ts=True)

    assert artifacts.transition_state is None
    assert artifacts.ts_error == "disk unplugged"


# --- only requested outputs are read ---


def test_collect_only_reads_requested_outputs(tmp_path: Path) -> None:
    """A present ``.phonon`` is ignored when only ``want_out_cell`` is set."""

    (tmp_path / "seed-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")
    (tmp_path / "seed.phonon").write_text(_PHONON_TEXT, encoding="utf-8")

    artifacts = collect_job_artifacts(tmp_path, "seed", want_out_cell=True)

    assert artifacts.relaxed_cell is not None
    assert artifacts.phonon is None
    assert artifacts.phonon_missing is False
    assert artifacts.phonon_path is None


# --- RetrievalFlags ---


@pytest.mark.parametrize(
    ("calc_type", "expected"),
    [
        ("castep_uep_source", "retrieve_den_fmt"),
        ("castep_pristine_relaxation", "retrieve_out_cell"),
        ("castep_positivemuon_relaxation", "retrieve_out_cell"),
        ("castep_muonium_relaxation", "retrieve_out_cell"),
        (QUANTUM_PHONON_CALC_TYPE, "retrieve_phonon"),
        (TRANSITION_STATE_SEARCH_CALC_TYPE, "retrieve_ts"),
        (NUDGED_ELASTIC_BAND_CALC_TYPE, "retrieve_ts"),
        ("castep_singlepoint", None),
    ],
)
def test_retrieval_flags_for_calc_type_selects_one_secondary_output(
    calc_type: str, expected: str | None
) -> None:
    flags = RetrievalFlags.for_calc_type(calc_type)

    requested = {
        name
        for name in (
            "retrieve_den_fmt",
            "retrieve_out_cell",
            "retrieve_phonon",
            "retrieve_ts",
            "retrieve_magres",
        )
        if getattr(flags, name)
    }
    assert requested == ({expected} if expected else set())
    assert flags.farm_count == 1


def test_retrieval_flags_keep_the_magres_and_density_of_a_hyperfine_job() -> None:
    flags = RetrievalFlags.for_calc_type(HYPERFINE_CALC_TYPE)

    assert flags.retrieve_magres and flags.retrieve_den_fmt
    assert not (flags.retrieve_out_cell or flags.retrieve_phonon or flags.retrieve_ts)


_HYPERFINE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hyperfine"


def test_collect_magres_carries_the_muon_tensor_and_warns_without_density(
    tmp_path: Path,
) -> None:
    (tmp_path / "seed.magres").write_text(
        (_HYPERFINE_FIXTURES / "muonium_vacuum.magres").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    artifacts = RetrievalFlags(retrieve_magres=True).collect(tmp_path, "seed")

    assert artifacts.magres is not None
    assert artifacts.magres.isotropic_mhz == pytest.approx(-4628.4054)
    assert artifacts.magres_path == tmp_path / "seed.magres"
    assert artifacts.pseudo_spin_density_angstrom3 is None
    assert artifacts.spin_density_error == "Formatted density seed.den_fmt was not found."


def test_collect_magres_missing_or_unreadable(tmp_path: Path) -> None:
    missing = collect_job_artifacts(tmp_path, "seed", want_magres=True)
    (tmp_path / "bad.magres").write_text("[magres]\n[/magres]\n", encoding="utf-8")
    unreadable = collect_job_artifacts(tmp_path, "bad", want_magres=True)

    assert missing.magres_missing and missing.magres is None
    assert unreadable.magres is None and "no hyperfine tensors" in (unreadable.magres_error or "")


def test_retrieval_flags_normalise_and_round_trip_as_runner_keywords() -> None:
    flags = RetrievalFlags.for_calc_type(
        NUDGED_ELASTIC_BAND_CALC_TYPE, task_farming_enabled=1, num_farms="3"
    )

    assert flags.task_farming_enabled is True
    assert flags.num_farms == 3
    assert flags.farm_count == 3
    assert RetrievalFlags(**flags.as_kwargs()) == flags
    assert RetrievalFlags(num_farms=0).num_farms == 1
    # Farming is only in effect when the flag is set.
    assert RetrievalFlags(num_farms=4).farm_count == 1
    with pytest.raises(TypeError):
        RetrievalFlags(**{"retrieve_castep": True})


def test_retrieval_flags_collect_reads_the_farmed_ts(tmp_path: Path) -> None:
    (tmp_path / "seed_farm001.ts").write_text(
        _TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )

    flags = RetrievalFlags(retrieve_ts=True, task_farming_enabled=True, num_farms=2)
    artifacts = flags.collect(tmp_path, "seed")

    assert artifacts.transition_state is not None
    assert artifacts.ts_path == tmp_path / "seed_farm001.ts"
    assert artifacts.relaxed_cell is None
    assert artifacts.cell_missing is False
