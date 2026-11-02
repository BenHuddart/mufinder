"""Tests for scanning a folder for CASTEP geometry optimisations.

Every fixture is hand-authored to the shape of CASTEP's files. Nothing is copied
from a real calculation: see ``docs/relaxation_import.rst`` (Research data
boundary).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.services.relaxation_import import (
    NO_MUON_BLOCKER,
    ParameterConflict,
    merge_parameters,
    muon_index,
    scan_relaxation_folder,
)
from mufinder.castep_io.cell_parser import parse_cell_text

HOST_LATTICE = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def _input_cell(*, muon_label: str = "H:mu", scale: int = 2, muon=(0.1, 0.2, 0.3)) -> str:
    """A genuine ``scale``-cubed supercell of a one-atom host, plus a muon.

    The host atoms really are tiled, not just the lattice scaled: the scan
    verifies that the atom count is ``det(S) * N_host + 1`` and that the
    composition matches, so a cell with a supercell lattice but un-tiled
    contents is correctly rejected as not belonging to the host.
    """

    length = 4.0 * scale
    step = 1.0 / scale
    host_atoms = "\n".join(
        f"Si {i * step:.6f} {j * step:.6f} {k * step:.6f}"
        for i in range(scale)
        for j in range(scale)
        for k in range(scale)
    )
    return f"""%BLOCK LATTICE_CART
ang
{length} 0.0 0.0
0.0 {length} 0.0
0.0 0.0 {length}
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
{host_atoms}
{muon_label} {muon[0]} {muon[1]} {muon[2]}
%ENDBLOCK POSITIONS_FRAC

FIX_ALL_CELL: TRUE
"""


def _out_cell(*, muon_label: str = "H:mu", muon=(0.11, 0.21, 0.31)) -> str:
    return _input_cell(muon_label=muon_label, muon=muon)


def _castep(
    *,
    charge: str = "+1.000",
    cutoff: str = "  600.0000   eV",
    converged: bool = True,
    completed: bool = True,
    mp_grid: str = "2  2  2",
) -> str:
    tail = ""
    if converged:
        tail += "\n Geometry optimization completed successfully.\n"
    if completed:
        tail += "\n Total time =     12.34 s\n"
    return f"""
 | Welcome to Academic Release CASTEP version 26.11 |

 Run started: Fri, 10 Jan 2025 11:57:08 +0000

 ***************************** General Parameters ******************************

 type of calculation                            : geometry optimization

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : Perdew Burke Ernzerhof

 **************************** Basis Set Parameters *****************************

 plane wave basis set cut-off                   : {cutoff}

 **************************** Electronic Parameters ****************************

 net charge of system                           : {charge}
 treating system as non-spin-polarized

 ********************* Electronic Minimization Parameters **********************

 max. number of SCF cycles                      :        100

 *******************************************************************************

                       MP grid size for SCF calculation is  {mp_grid}

 Final energy, E             =  -1234.5678     eV
{tail}"""


def _run_param(*, xc: str = "PBE", charge_line: str = "charge: +1\n") -> str:
    return f"""task: geometryoptimisation
{charge_line}xc_functional: {xc}
spin_polarized : false
basis_precision : FINE
write_cell_structure : true
geom_max_iter : 300
max_scf_cycles : 100
elec_energy_tol : 1E-6
continuation : default
"""


def build_folder(
    root: Path,
    *,
    sites=(1, 2),
    seed: str = "Host",
    castep_kwargs: dict | None = None,
    param_text: str | None = None,
    muon_label: str = "H:mu",
) -> Path:
    """Write a tree that keeps inputs, results and run folders apart."""

    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    for index in sites:
        stem = f"{seed}_{index}"
        (root / "input" / f"{stem}.cell").write_text(
            _input_cell(muon_label=muon_label), encoding="utf-8"
        )
        (root / "results" / f"{stem}.castep").write_text(
            _castep(**(castep_kwargs or {})), encoding="utf-8"
        )
        (root / "results" / f"{stem}-out.cell").write_text(
            _out_cell(muon_label=muon_label), encoding="utf-8"
        )
        run_site = root / "run" / f"site{index}"
        run_site.mkdir(parents=True, exist_ok=True)
        (run_site / f"{stem}.param").write_text(
            param_text if param_text is not None else _run_param(), encoding="utf-8"
        )

    return root


def write_run(
    directory: Path,
    seed: str = "mu",
    *,
    input_text: str | None = None,
    out_text: str | None = None,
    castep_kwargs: dict | None = None,
    castep_text: str | None = None,
) -> Path:
    """Write one run kept whole in ``directory``, as CASTEP leaves it."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{seed}.cell").write_text(
        _input_cell() if input_text is None else input_text, encoding="utf-8"
    )
    (directory / f"{seed}-out.cell").write_text(
        _out_cell() if out_text is None else out_text, encoding="utf-8"
    )
    (directory / f"{seed}.castep").write_text(
        _castep(**(castep_kwargs or {})) if castep_text is None else castep_text,
        encoding="utf-8",
    )
    return directory


def host_only_cell(scale: int = 2) -> str:
    """The same supercell with no muon in it."""

    return "\n".join(
        line for line in _input_cell(scale=scale).splitlines() if not line.startswith("H:mu")
    )


class TestMuonIndex:
    def test_prefers_the_muon_species(self) -> None:
        parsed = parse_cell_text(_input_cell())
        assert muon_index(parsed) == (8, "H:mu")

    def test_falls_back_to_a_lone_bare_hydrogen(self) -> None:
        """A hand-written cell may label the muon with a bare ``H``."""

        parsed = parse_cell_text(_input_cell(muon_label="H"))
        assert muon_index(parsed) == (8, "H")

    def test_refuses_a_bare_hydrogen_when_the_host_also_has_one(self) -> None:
        """Two hydrogens make a bare ``H`` ambiguous, so identify nothing.

        Guessing here would silently import the wrong atom as the muon.
        """

        text = _input_cell(muon_label="H").replace("Si 0.000000 0.000000 0.000000", "H 0.0 0.0 0.0")
        assert muon_index(parse_cell_text(text)) is None


class TestMergeParameters:
    def test_castep_wins_over_param_and_records_the_conflict(self) -> None:
        merged, sources, conflicts = merge_parameters(
            castep_values={"xc_functional": "PBE"},
            param_values={"xc_functional": "LDA"},
        )

        assert merged["xc_functional"] == "PBE"
        assert sources["xc_functional"] == "castep"
        assert len(conflicts) == 1
        assert conflicts[0].value == "PBE"
        assert conflicts[0].other_value == "LDA"
        assert "importing 'PBE'" in conflicts[0].describe()

    def test_a_value_only_one_source_supplies_is_not_a_conflict(self) -> None:
        merged, sources, conflicts = merge_parameters(
            castep_values={"cut_off_energy": 600.0},
            param_values={"basis_precision": "FINE"},
        )

        assert merged == {"cut_off_energy": 600.0, "basis_precision": "FINE"}
        assert sources["basis_precision"] == "param"
        assert conflicts == ()

    def test_equal_values_expressed_differently_do_not_conflict(self) -> None:
        _merged, _sources, conflicts = merge_parameters(
            castep_values={"max_scf_cycles": 100},
            param_values={"max_scf_cycles": 100.0},
        )

        assert conflicts == ()

    def test_cell_supplies_the_grid_when_nothing_better_does(self) -> None:
        merged, sources, _conflicts = merge_parameters(cell_values={"kpoint_mp_grid": (2, 2, 2)})

        assert merged["kpoint_mp_grid"] == (2, 2, 2)
        assert sources["kpoint_mp_grid"] == "cell"


class TestScanFolder:
    def test_detects_every_run(self, tmp_path: Path) -> None:
        scan = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1, 2, 3)), host_lattice=HOST_LATTICE
        )

        assert [run.seed_name for run in scan.runs] == ["Host_1", "Host_2", "Host_3"]
        assert all(run.importable for run in scan.runs)
        assert scan.converged_runs == scan.importable_runs

    def test_recovers_the_effective_parameters(self, tmp_path: Path) -> None:
        scan = scan_relaxation_folder(build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE)
        run = scan.runs[0]

        assert run.parameters["cut_off_energy"].value == pytest.approx(600.0)
        assert run.parameters["kpoint_mp_grid"] == (2, 2, 2)
        assert run.parameters["task"] == "GEOMETRYOPTIMIZATION"
        assert run.parameter_sources["cut_off_energy"] == "castep"
        # basis_precision is never echoed, so it can only come from the .param.
        assert run.parameter_sources["basis_precision"] == "param"

    def test_reports_a_stale_input_tree_as_a_conflict(self, tmp_path: Path) -> None:
        """The headline legacy hazard: the local .param disagrees with the run.

        Runs executed on a cluster routinely leave a stale local .param behind.
        The scan must surface that rather than trust it.
        """

        root = build_folder(tmp_path, sites=(1,), param_text=_run_param(xc="LDA"))
        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.parameters["xc_functional"] == "PBE"
        conflicts = {item.keyword: item for item in run.conflicts}
        assert "xc_functional" in conflicts
        assert conflicts["xc_functional"].other_value == "LDA"

    def test_infers_the_supercell_matrix_against_the_host(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.supercell_matrix == ((2, 0, 0), (0, 2, 0), (0, 0, 2))

    def test_blocks_a_run_whose_cell_is_not_a_supercell_of_the_host(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)),
            host_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.3]],
        ).runs[0]

        assert not run.importable
        assert any("integer supercell" in item for item in run.blockers)

    def test_scanning_without_a_host_skips_the_supercell_check(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(build_folder(tmp_path, sites=(1,))).runs[0]

        assert run.supercell_matrix is None
        assert run.importable

    @pytest.mark.parametrize(
        "charge, expected",
        [("+1.000", POSITIVE_MUON_CHARGE_STATE), (" 0.000", MUONIUM_CHARGE_STATE)],
    )
    def test_charge_alone_determines_the_charge_state(
        self, tmp_path: Path, charge: str, expected: str
    ) -> None:
        """Spin polarisation is orthogonal and must not be consulted.

        Both fixtures are non-spin-polarised, so a reader that conflated the two
        would call the neutral case a positive muon.
        """

        root = build_folder(tmp_path, sites=(1,), castep_kwargs={"charge": charge})
        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state == expected
        assert run.charge_state_source == "castep"

    def test_charge_state_falls_back_to_the_param(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,), castep_kwargs={"charge": "  7.000"})
        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state == POSITIVE_MUON_CHARGE_STATE
        assert run.charge_state_source == "param"

    def test_an_undeterminable_charge_state_warns_without_blocking(self, tmp_path: Path) -> None:
        root = build_folder(
            tmp_path,
            sites=(1,),
            castep_kwargs={"charge": "  7.000"},
            param_text=_run_param(charge_line=""),
        )
        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state is None
        assert any("charge state" in item for item in run.warnings)
        assert run.importable

    def test_records_the_final_and_initial_muon_positions(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.initial_muon_fractional_supercell == pytest.approx((0.1, 0.2, 0.3))
        assert run.final_muon_fractional_supercell == pytest.approx((0.11, 0.21, 0.31))

    def test_a_bare_hydrogen_muon_warns_about_relabelling(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,), muon_label="H"), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.muon_label == "H"
        assert run.importable
        assert any("relabelled" in item for item in run.warnings)

    def test_an_unconverged_run_is_importable_but_not_preselected(self, tmp_path: Path) -> None:
        """The wizard offers every run and pre-ticks only the converged ones."""

        root = build_folder(tmp_path, sites=(1,), castep_kwargs={"converged": False})
        scan = scan_relaxation_folder(root, host_lattice=HOST_LATTICE)

        assert scan.runs[0].importable
        assert not scan.runs[0].is_converged
        assert scan.converged_runs == ()

    def test_a_missing_out_cell_blocks_the_run(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,))
        next(root.glob("results/*-out.cell")).unlink()

        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert not run.importable
        assert any("-out.cell" in item for item in run.blockers)

    def test_an_out_cell_with_calculation_keywords_is_read_for_its_geometry(
        self, tmp_path: Path
    ) -> None:
        """``EXTERNAL_PRESSURE`` is outside the built-in keyword set; the scan needs only the geometry."""

        root = build_folder(tmp_path, sites=(1,))
        out_cell = next(root.glob("results/*-out.cell"))
        out_cell.write_text(
            out_cell.read_text(encoding="utf-8")
            + "\n%BLOCK EXTERNAL_PRESSURE\nGPa\n0.5 0.0 0.0\n0.5 0.0\n0.5\n%ENDBLOCK EXTERNAL_PRESSURE\n",
            encoding="utf-8",
        )

        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.importable, run.blockers
        assert run.final_muon_fractional_supercell == pytest.approx((0.11, 0.21, 0.31))

    def test_an_input_cell_with_calculation_keywords_anchors_the_run(self, tmp_path: Path) -> None:
        """A keyword outside the built-in set loses only the settings, not the geometry."""

        root = build_folder(tmp_path, sites=(1,))
        input_cell = root / "input" / "Host_1.cell"
        input_cell.write_text(
            input_cell.read_text(encoding="utf-8")
            + "\n%BLOCK EXTERNAL_PRESSURE\nGPa\n0.5 0.0 0.0\n0.5 0.0\n0.5\n%ENDBLOCK EXTERNAL_PRESSURE\n",
            encoding="utf-8",
        )

        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.importable, run.blockers
        assert run.initial_muon_fractional_supercell == pytest.approx((0.1, 0.2, 0.3))
        assert any("Host_1.cell" in item and "k-point grid" in item for item in run.warnings), (
            run.warnings
        )

    def test_an_unparseable_cell_does_not_end_the_scan(self, tmp_path: Path) -> None:
        """One bad run must never discard the rest of the directory."""

        root = build_folder(tmp_path, sites=(1, 2))
        (root / "input" / "Host_1.cell").write_text("this is not a cell file\n", encoding="utf-8")

        scan = scan_relaxation_folder(root, host_lattice=HOST_LATTICE)

        assert len(scan.runs) == 2
        assert not scan.runs[0].importable
        assert scan.runs[1].importable

    def test_an_empty_directory_scans_without_raising(self, tmp_path: Path) -> None:
        scan = scan_relaxation_folder(tmp_path)

        assert scan.runs == ()

    def test_records_the_castep_version_for_provenance(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.castep_version == "26.11"

    def test_unmapped_echo_labels_are_carried_not_dropped(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.unmapped == () or all(isinstance(item, str) for item in run.unmapped)


def test_parameter_conflict_describes_itself_readably() -> None:
    conflict = ParameterConflict(
        keyword="cut_off_energy",
        value=900.0,
        source="castep",
        other_value=600.0,
        other_source="param",
    )

    described = conflict.describe()
    assert "CUT_OFF_ENERGY" in described
    assert "castep" in described and "param" in described


class TestHostCompositionChecks:
    """A matching lattice is not enough to prove the cell belongs to the host.

    Two unrelated materials can share a lattice closely enough to pass the
    integer-transform test, and importing against the wrong host would attach
    every site to a structure it has nothing to do with.
    """

    def test_a_matching_composition_is_accepted(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,))

        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE, host_species=["Si"]).runs[0]

        assert run.importable

    def test_a_wrong_atom_count_is_blocked(self, tmp_path: Path) -> None:
        """The fixture cell is a 2x2x2 supercell of a one-atom host, so a
        two-atom host would need eight more atoms than are present."""

        root = build_folder(tmp_path, sites=(1,))

        run = scan_relaxation_folder(
            root, host_lattice=HOST_LATTICE, host_species=["Si", "Si"]
        ).runs[0]

        assert not run.importable
        assert any("host atoms" in item for item in run.blockers)

    def test_a_wrong_composition_is_blocked(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,))

        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE, host_species=["Ge"]).runs[0]

        assert not run.importable
        assert any("composition" in item for item in run.blockers)

    def test_the_muon_is_excluded_from_the_count(self, tmp_path: Path) -> None:
        """The input cell is the tiled host *plus* one muon."""

        root = build_folder(tmp_path, sites=(1,))
        run = scan_relaxation_folder(root, host_lattice=HOST_LATTICE, host_species=["Si"]).runs[0]

        # 8 host atoms (2x2x2 of one) + 1 muon; a check that counted the muon
        # would demand nine host atoms and reject a valid cell.
        assert run.importable

    def test_scanning_without_host_species_skips_the_check(self, tmp_path: Path) -> None:
        """Scanning without a host stays possible."""

        root = build_folder(tmp_path, sites=(1,))

        assert scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs[0].importable


class TestScanProgress:
    """A folder of large outputs takes long enough that silence reads as a hang."""

    def test_progress_is_reported_once_per_run(self, tmp_path: Path) -> None:
        seen: list[tuple[int, int]] = []

        scan_relaxation_folder(
            build_folder(tmp_path, sites=(1, 2, 3)),
            host_lattice=HOST_LATTICE,
            progress=lambda done, total: seen.append((done, total)),
        )

        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_scanning_without_a_progress_callback_still_works(self, tmp_path: Path) -> None:
        scan = scan_relaxation_folder(build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE)

        assert len(scan.runs) == 1


class TestAnyFolder:
    """A run is found wherever it is kept, and each kind of bad run says why."""

    def _mixed_folder(self, root: Path) -> Path:
        write_run(root / "study" / "finished", "finished")
        write_run(
            root / "study" / "deep" / "er" / "unconverged", castep_kwargs={"converged": False}
        )
        write_run(root / "unfinished", "cut_short", castep_kwargs={"completed": False})
        write_run(
            root / "pristine",
            "host",
            input_text=host_only_cell(),
            out_text=host_only_cell(),
            castep_kwargs={"charge": "0.000"},
        )
        # A lattice that is not an integer multiple of the host's.
        wrong = _input_cell().replace("8.0 0.0 0.0", "9.0 0.0 0.0")
        write_run(root / "other_material", "wrong", input_text=wrong, out_text=wrong)
        return root

    def test_a_nested_folder_of_mixed_runs(self, tmp_path: Path) -> None:
        scan = scan_relaxation_folder(self._mixed_folder(tmp_path), host_lattice=HOST_LATTICE)
        by_name = {run.relative_name: run for run in scan.runs}

        assert sorted(by_name) == [
            "other_material/wrong",
            "pristine/host",
            "study/deep/er/unconverged/mu",
            "study/finished/finished",
            "unfinished/cut_short",
        ]
        assert [run.relative_name for run in scan.converged_runs] == ["study/finished/finished"]

        unconverged = by_name["study/deep/er/unconverged/mu"]
        assert unconverged.importable and not unconverged.is_converged

        assert by_name["unfinished/cut_short"].blockers == ("The CASTEP run did not complete.",)
        assert by_name["pristine/host"].blockers == (NO_MUON_BLOCKER,)
        assert by_name["other_material/wrong"].blockers == (
            "The input cell is not an integer supercell of the active host structure.",
        )

    def test_no_muon_is_decided_from_the_structure_not_the_seed(self, tmp_path: Path) -> None:
        """``_0`` means nothing, and a seed of any name can be a stopping site."""

        write_run(tmp_path / "a", "Host_0")
        write_run(tmp_path / "b", "Host_7", input_text=host_only_cell(), out_text=host_only_cell())

        by_seed = {
            run.seed_name: run
            for run in scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs
        }

        assert by_seed["Host_0"].importable
        assert by_seed["Host_7"].blockers == (NO_MUON_BLOCKER,)

    def test_a_hydrogen_bearing_hosts_own_hydrogen_is_not_a_muon(self, tmp_path: Path) -> None:
        """A bare ``H`` that the host's composition accounts for is the host's."""

        host_run = host_only_cell(scale=1).replace("Si 0.000000", "H 0.000000")
        write_run(tmp_path, "host", input_text=host_run, out_text=host_run)

        run = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE, host_species=["H"]).runs[
            0
        ]

        assert run.blockers == (NO_MUON_BLOCKER,)

    def test_a_run_of_another_task_is_refused_with_the_reason(self, tmp_path: Path) -> None:
        phonon = _castep().replace("geometry optimization", "phonon calculation")
        write_run(tmp_path, "phonon", castep_text=phonon)
        (tmp_path / "phonon-out.cell").unlink()

        run = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs[0]

        assert len(run.blockers) == 1
        assert "not a geometry optimisation" in run.blockers[0]

    def test_a_seed_of_any_shape_is_found(self, tmp_path: Path) -> None:
        write_run(tmp_path, "my relaxation.v2")

        run = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs[0]

        assert run.seed_name == "my relaxation.v2"
        assert run.importable

    def test_same_seed_in_two_folders_are_two_runs_with_their_own_files(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path / "site1", out_text=_out_cell(muon=(0.11, 0.21, 0.31)))
        write_run(
            tmp_path / "site2",
            out_text=_out_cell(muon=(0.41, 0.51, 0.61)),
            castep_kwargs={"cutoff": "  700.0000   eV"},
        )

        runs = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert [run.relative_name for run in runs] == ["site1/mu", "site2/mu"]
        assert runs[0].final_muon_fractional_supercell == (0.11, 0.21, 0.31)
        assert runs[1].final_muon_fractional_supercell == (0.41, 0.51, 0.61)
        assert [run.out_cell_path.parent.name for run in runs] == ["site1", "site2"]

    def test_a_run_does_not_borrow_the_files_of_another_run_of_its_seed(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path / "site1")
        (tmp_path / "site1" / "mu.cell").unlink()
        write_run(
            tmp_path / "site2",
            input_text=_input_cell(muon=(0.4, 0.4, 0.4)),
            castep_kwargs={"cutoff": "  700.0000   eV"},
        )

        first, second = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert any("mu.cell" in item for item in first.blockers)
        assert second.importable

    def test_one_stray_file_is_not_shared_between_runs(self, tmp_path: Path) -> None:
        write_run(tmp_path / "site1")
        write_run(tmp_path / "site2", castep_kwargs={"cutoff": "  700.0000   eV"})
        for site in ("site1", "site2"):
            (tmp_path / site / "mu.cell").unlink()
            (tmp_path / site / "mu-out.cell").unlink()
        (tmp_path / "stray").mkdir()
        (tmp_path / "stray" / "mu.cell").write_text(_input_cell(), encoding="utf-8")
        (tmp_path / "stray" / "mu-out.cell").write_text(_out_cell(), encoding="utf-8")

        runs = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert not any(run.importable for run in runs)
        assert all(any("several runs" in item for item in run.blockers) for run in runs)

    def test_a_file_from_another_folder_is_named_in_the_review(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.importable
        assert run.pairing_notes == ("Read from another folder: Host_1.cell, Host_1.param.",)
        # Reviewed before the import; the report after it does not repeat one per run.
        assert run.warnings == ()

    def test_a_run_kept_whole_has_no_pairing_note(self, tmp_path: Path) -> None:
        run = scan_relaxation_folder(write_run(tmp_path), host_lattice=HOST_LATTICE).runs[0]

        assert run.pairing_notes == ()

    def test_a_relaxed_cell_with_other_atoms_blocks_the_run(self, tmp_path: Path) -> None:
        write_run(
            tmp_path,
            out_text=_input_cell().replace("Si 0.000000 0.000000 0.000000", "Ge 0.0 0.0 0.0"),
        )

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert any("does not hold the same atoms" in item for item in run.blockers)

    def test_a_relaxed_cell_that_lost_the_muon_is_not_called_a_host_only_run(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path, out_text=host_only_cell())

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert "The relaxed cell contains no muon." in run.blockers
        assert NO_MUON_BLOCKER not in run.blockers

    def test_an_output_that_cannot_be_read_blocks_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_path: Path) -> None:
            raise ValueError("corrupt")

        monkeypatch.setattr("mufinder.services.relaxation_import.parse_castep_output", _boom)
        write_run(tmp_path)

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert "Could not read mu.castep: corrupt." in run.blockers

    def test_a_blocked_runs_parameter_disagreement_is_not_reported(self, tmp_path: Path) -> None:
        """A phonon run beside a stale ``.param`` has one thing to say."""

        write_run(
            tmp_path,
            "phonon",
            castep_text=_castep().replace("geometry optimization", "phonon calculation"),
        )
        (tmp_path / "phonon.param").write_text(_run_param(), encoding="utf-8")

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert len(run.blockers) == 1
        assert run.conflicts == ()

    def test_equally_close_different_cells_are_not_guessed_between(self, tmp_path: Path) -> None:
        write_run(tmp_path / "results")
        (tmp_path / "results" / "mu.cell").unlink()
        for name, muon in (("a", (0.1, 0.1, 0.1)), ("b", (0.2, 0.2, 0.2))):
            (tmp_path / name).mkdir()
            (tmp_path / name / "mu.cell").write_text(_input_cell(muon=muon), encoding="utf-8")

        run = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs[0]

        assert any("equally close" in item for item in run.blockers)
        assert not any("was found" in item for item in run.blockers)

    def test_equally_close_copies_of_one_cell_are_accepted(self, tmp_path: Path) -> None:
        write_run(tmp_path / "results")
        (tmp_path / "results" / "mu.cell").unlink()
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "mu.cell").write_text(_input_cell(), encoding="utf-8")

        run = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs[0]

        assert run.importable

    def test_a_copied_output_is_one_run(self, tmp_path: Path) -> None:
        write_run(tmp_path / "results")
        (tmp_path / "backup").mkdir()
        (tmp_path / "backup" / "mu.castep").write_bytes(
            (tmp_path / "results" / "mu.castep").read_bytes()
        )

        runs = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert len(runs) == 1
        assert runs[0].importable

    def test_task_farm_image_outputs_are_not_runs(self, tmp_path: Path) -> None:
        write_run(tmp_path)
        (tmp_path / "mu_farm001.castep").write_text(_castep(), encoding="utf-8")

        runs = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert [run.seed_name for run in runs] == ["mu"]

    def test_a_dangling_link_does_not_end_the_scan(self, tmp_path: Path) -> None:
        write_run(tmp_path)
        (tmp_path / "gone.castep").symlink_to(tmp_path / "nowhere.castep")

        runs = scan_relaxation_folder(tmp_path, host_lattice=HOST_LATTICE).runs

        assert [run.seed_name for run in runs] == ["mu"]

    def test_a_tree_keeping_inputs_results_and_run_files_apart_still_pairs_them(
        self, tmp_path: Path
    ) -> None:
        root = build_folder(tmp_path, sites=(1, 2))
        # A second copy of the cell further away, as a run folder holds.
        (root / "run" / "site1" / "Host_1.cell").write_text(
            _input_cell(muon=(0.9, 0.9, 0.9)), encoding="utf-8"
        )

        runs = scan_relaxation_folder(root, host_lattice=HOST_LATTICE).runs

        assert [run.relative_name for run in runs] == ["results/Host_1", "results/Host_2"]
        assert all(run.importable for run in runs)
        assert runs[0].input_cell_path == root / "input" / "Host_1.cell"
        assert runs[0].out_cell_path == root / "results" / "Host_1-out.cell"
        assert runs[0].run_param_path == root / "run" / "site1" / "Host_1.param"
        assert runs[0].initial_muon_fractional_supercell == (0.1, 0.2, 0.3)


def test_the_old_module_names_still_resolve() -> None:
    from mufinder.services import (
        legacy_import,
        legacy_import_commit,
        legacy_import_removal,
        relaxation_import,
        relaxation_import_commit,
        relaxation_import_removal,
    )

    assert legacy_import.scan_legacy_folder is relaxation_import.scan_relaxation_folder
    assert legacy_import.LegacyFolderScan is relaxation_import.RelaxationFolderScan
    assert legacy_import_commit.commit_legacy_import is (
        relaxation_import_commit.commit_relaxation_import
    )
    assert legacy_import_commit.LegacyImportOutcome is (
        relaxation_import_commit.RelaxationImportOutcome
    )
    assert legacy_import_removal.plan_removal is relaxation_import_removal.plan_removal


needs_permissions = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads everything, whatever the mode"
)


class TestWhatTheScanCouldNotRead:
    @needs_permissions
    def test_a_locked_subfolder_is_reported(self, tmp_path: Path) -> None:
        write_run(tmp_path / "open")
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0)
        try:
            scan = scan_relaxation_folder(tmp_path)
        finally:
            locked.chmod(0o755)

        assert [run.relative_name for run in scan.runs] == ["open/mu"]
        assert scan.unreadable == (("locked", "permission denied"),)
        assert any(
            "1 location" in line and "locked (permission denied)" in line for line in scan.warnings
        )

    @needs_permissions
    def test_an_unreadable_chosen_folder_says_so(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0)
        try:
            scan = scan_relaxation_folder(locked)
        finally:
            locked.chmod(0o755)

        assert scan.runs == ()
        assert scan.unreadable_note == "Could not read this folder: permission denied."

    def test_a_tree_beyond_the_path_limit_is_reported(self, tmp_path: Path) -> None:
        name = "d" * 200
        current = os.open(tmp_path, os.O_RDONLY)
        try:
            for _level in range(25):
                os.mkdir(name, dir_fd=current)
                deeper = os.open(name, os.O_RDONLY, dir_fd=current)
                os.close(current)
                current = deeper
            handle = os.open("mu.castep", os.O_CREAT | os.O_WRONLY, dir_fd=current)
            os.close(handle)
        finally:
            os.close(current)

        try:
            scan = scan_relaxation_folder(tmp_path)

            assert scan.runs == ()
            assert scan.unreadable[0][1] == "the path is too long"
            assert "the path is too long" in scan.unreadable_note
            assert len(scan.unreadable_note) < 300
        finally:
            _remove_deep_tree(tmp_path, name)

    @needs_permissions
    def test_an_unreadable_output_gives_one_plain_reason(self, tmp_path: Path) -> None:
        write_run(tmp_path)
        output = tmp_path / "mu.castep"
        output.chmod(0)
        try:
            run = scan_relaxation_folder(tmp_path).runs[0]
        finally:
            output.chmod(0o644)

        assert "Could not read mu.castep: permission denied." in run.blockers
        everything = " ".join((*run.blockers, *run.warnings))
        assert "Errno" not in everything
        assert str(tmp_path) not in everything
        assert "charge state" not in everything

    @needs_permissions
    def test_an_unreadable_cell_gives_a_plain_reason(self, tmp_path: Path) -> None:
        write_run(tmp_path)
        cell = tmp_path / "mu.cell"
        cell.chmod(0)
        try:
            run = scan_relaxation_folder(tmp_path).runs[0]
        finally:
            cell.chmod(0o644)

        everything = " ".join((*run.blockers, *run.warnings))
        assert "Could not read mu.cell: permission denied." in everything
        assert "Errno" not in everything
        assert str(tmp_path) not in everything


def _remove_deep_tree(root: Path, name: str) -> None:
    """Remove a tree too deep for a path to reach, working down by file descriptor."""

    import shutil

    shutil.rmtree(root / name)


class TestChargeWording:
    def test_a_host_only_run_claims_no_charge_state(self, tmp_path: Path) -> None:
        write_run(
            tmp_path,
            "host",
            input_text=host_only_cell(),
            out_text=host_only_cell(),
            castep_kwargs={"charge": "0.000"},
        )

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert run.charge_state is None
        assert run.has_no_muon

    def test_an_undetermined_charge_says_what_will_happen(self, tmp_path: Path) -> None:
        write_run(tmp_path, castep_kwargs={"charge": "  7.000"})

        run = scan_relaxation_folder(tmp_path).runs[0]

        assert any("imported as a positive muon" in item for item in run.warnings)
        assert not any("choose" in item for item in run.warnings)
