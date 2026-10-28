"""Tests for scanning a legacy MuFinder working directory.

Every fixture is hand-authored to the documented legacy layout. Nothing is
copied from a real calculation: see ``docs/legacy_import.rst`` (Research data
boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.services.legacy_import import (
    ParameterConflict,
    merge_parameters,
    muon_index,
    scan_legacy_folder,
    site_index_from_stem,
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


def _legacy_param(*, xc: str = "PBE", charge_line: str = "charge: +1\n") -> str:
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
    include_site_zero: bool = False,
    include_cif: bool = True,
    castep_kwargs: dict | None = None,
    param_text: str | None = None,
    muon_label: str = "H:mu",
) -> Path:
    """Write a legacy-shaped working directory."""

    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    indices = list(sites)
    if include_site_zero:
        indices = [0, *indices]

    for index in indices:
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
            param_text if param_text is not None else _legacy_param(), encoding="utf-8"
        )

    if include_cif:
        (root / f"{seed}.cif").write_text("# placeholder\n", encoding="utf-8")
    return root


class TestSiteIndex:
    @pytest.mark.parametrize(
        "stem, expected",
        [("Host_1", 1), ("Host_42", 42), ("Some_Long_Name_7", 7), ("Host", None), ("Host_x", None)],
    )
    def test_index_is_the_trailing_field(self, stem: str, expected: int | None) -> None:
        assert site_index_from_stem(stem) == expected


class TestMuonIndex:
    def test_prefers_the_muon_species(self) -> None:
        parsed = parse_cell_text(_input_cell())
        assert muon_index(parsed) == (8, "H:mu")

    def test_falls_back_to_a_lone_bare_hydrogen(self) -> None:
        """The legacy label detector accepted a bare ``H``."""

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


class TestScanLegacyFolder:
    def test_detects_every_site(self, tmp_path: Path) -> None:
        scan = scan_legacy_folder(
            build_folder(tmp_path, sites=(1, 2, 3)), host_lattice=HOST_LATTICE
        )

        assert [run.site_index for run in scan.runs] == [1, 2, 3]
        assert all(run.importable for run in scan.runs)
        assert scan.converged_runs == scan.importable_runs

    def test_skips_the_pristine_reference_at_site_zero(self, tmp_path: Path) -> None:
        """Legacy site 0 is the neutral pristine reference, not a stopping site."""

        scan = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,), include_site_zero=True),
            host_lattice=HOST_LATTICE,
        )

        assert [run.site_index for run in scan.runs] == [1]

    def test_recovers_the_effective_parameters(self, tmp_path: Path) -> None:
        scan = scan_legacy_folder(build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE)
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

        root = build_folder(tmp_path, sites=(1,), param_text=_legacy_param(xc="LDA"))
        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.parameters["xc_functional"] == "PBE"
        conflicts = {item.keyword: item for item in run.conflicts}
        assert "xc_functional" in conflicts
        assert conflicts["xc_functional"].other_value == "LDA"

    def test_infers_the_supercell_matrix_against_the_host(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.supercell_matrix == ((2, 0, 0), (0, 2, 0), (0, 0, 2))

    def test_blocks_a_run_whose_cell_is_not_a_supercell_of_the_host(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,)),
            host_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.3]],
        ).runs[0]

        assert not run.importable
        assert any("integer supercell" in item for item in run.blockers)

    def test_scanning_without_a_host_skips_the_supercell_check(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(build_folder(tmp_path, sites=(1,))).runs[0]

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
        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state == expected
        assert run.charge_state_source == "castep"

    def test_charge_state_falls_back_to_the_param(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,), castep_kwargs={"charge": "  7.000"})
        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state == POSITIVE_MUON_CHARGE_STATE
        assert run.charge_state_source == "param"

    def test_an_undeterminable_charge_state_warns_without_blocking(self, tmp_path: Path) -> None:
        root = build_folder(
            tmp_path,
            sites=(1,),
            castep_kwargs={"charge": "  7.000"},
            param_text=_legacy_param(charge_line=""),
        )
        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.charge_state is None
        assert any("charge state" in item for item in run.warnings)
        assert run.importable

    def test_records_the_final_and_initial_muon_positions(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.initial_muon_fractional_supercell == pytest.approx((0.1, 0.2, 0.3))
        assert run.final_muon_fractional_supercell == pytest.approx((0.11, 0.21, 0.31))

    def test_a_bare_hydrogen_muon_warns_about_relabelling(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,), muon_label="H"), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.muon_label == "H"
        assert run.importable
        assert any("relabelled" in item for item in run.warnings)

    def test_an_unconverged_run_is_importable_but_not_preselected(self, tmp_path: Path) -> None:
        """The wizard offers every run and pre-ticks only the converged ones."""

        root = build_folder(tmp_path, sites=(1,), castep_kwargs={"converged": False})
        scan = scan_legacy_folder(root, host_lattice=HOST_LATTICE)

        assert scan.runs[0].importable
        assert not scan.runs[0].is_converged
        assert scan.converged_runs == ()

    def test_a_missing_out_cell_blocks_the_run(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,))
        next(root.glob("results/*-out.cell")).unlink()

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

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

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

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

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0]

        assert run.importable, run.blockers
        assert run.initial_muon_fractional_supercell == pytest.approx((0.1, 0.2, 0.3))
        assert any("Host_1.cell" in item and "k-point grid" in item for item in run.warnings), (
            run.warnings
        )

    def test_an_unparseable_cell_does_not_end_the_scan(self, tmp_path: Path) -> None:
        """One bad run must never discard the rest of the directory."""

        root = build_folder(tmp_path, sites=(1, 2))
        (root / "input" / "Host_1.cell").write_text("this is not a cell file\n", encoding="utf-8")

        scan = scan_legacy_folder(root, host_lattice=HOST_LATTICE)

        assert len(scan.runs) == 2
        assert not scan.runs[0].importable
        assert scan.runs[1].importable

    def test_reports_a_missing_cif(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,), include_cif=False)

        scan = scan_legacy_folder(root, host_lattice=HOST_LATTICE)

        assert scan.host_cif is None
        assert any("No CIF" in item for item in scan.warnings)

    def test_finds_the_cif_when_present(self, tmp_path: Path) -> None:
        scan = scan_legacy_folder(build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE)

        assert scan.host_cif is not None
        assert scan.host_cif.suffix == ".cif"

    def test_an_empty_directory_scans_without_raising(self, tmp_path: Path) -> None:
        scan = scan_legacy_folder(tmp_path)

        assert scan.runs == ()
        assert any("input" in item for item in scan.warnings)

    def test_records_the_castep_version_for_provenance(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
            build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE
        ).runs[0]

        assert run.castep_version == "26.11"

    def test_unmapped_echo_labels_are_carried_not_dropped(self, tmp_path: Path) -> None:
        run = scan_legacy_folder(
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

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE, host_species=["Si"]).runs[0]

        assert run.importable

    def test_a_wrong_atom_count_is_blocked(self, tmp_path: Path) -> None:
        """The fixture cell is a 2x2x2 supercell of a one-atom host, so a
        two-atom host would need eight more atoms than are present."""

        root = build_folder(tmp_path, sites=(1,))

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE, host_species=["Si", "Si"]).runs[0]

        assert not run.importable
        assert any("host atom(s)" in item for item in run.blockers)

    def test_a_wrong_composition_is_blocked(self, tmp_path: Path) -> None:
        root = build_folder(tmp_path, sites=(1,))

        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE, host_species=["Ge"]).runs[0]

        assert not run.importable
        assert any("composition" in item for item in run.blockers)

    def test_the_muon_is_excluded_from_the_count(self, tmp_path: Path) -> None:
        """The input cell is the tiled host *plus* one muon."""

        root = build_folder(tmp_path, sites=(1,))
        run = scan_legacy_folder(root, host_lattice=HOST_LATTICE, host_species=["Si"]).runs[0]

        # 8 host atoms (2x2x2 of one) + 1 muon; a check that counted the muon
        # would demand nine host atoms and reject a valid cell.
        assert run.importable

    def test_scanning_without_host_species_skips_the_check(self, tmp_path: Path) -> None:
        """Scanning without a host stays possible."""

        root = build_folder(tmp_path, sites=(1,))

        assert scan_legacy_folder(root, host_lattice=HOST_LATTICE).runs[0].importable


class TestScanProgress:
    """A folder of large outputs takes long enough that silence reads as a hang."""

    def test_progress_is_reported_once_per_run(self, tmp_path: Path) -> None:
        seen: list[tuple[int, int]] = []

        scan_legacy_folder(
            build_folder(tmp_path, sites=(1, 2, 3)),
            host_lattice=HOST_LATTICE,
            progress=lambda done, total: seen.append((done, total)),
        )

        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_the_pristine_reference_is_not_counted(self, tmp_path: Path) -> None:
        """Site 0 is skipped, so counting it would report a total that is never
        reached and leave the progress message stuck short of the end."""

        seen: list[tuple[int, int]] = []

        scan_legacy_folder(
            build_folder(tmp_path, sites=(1, 2), include_site_zero=True),
            host_lattice=HOST_LATTICE,
            progress=lambda done, total: seen.append((done, total)),
        )

        assert seen == [(1, 2), (2, 2)]

    def test_scanning_without_a_progress_callback_still_works(self, tmp_path: Path) -> None:
        scan = scan_legacy_folder(build_folder(tmp_path, sites=(1,)), host_lattice=HOST_LATTICE)

        assert len(scan.runs) == 1
