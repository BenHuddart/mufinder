from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.charge_density import ChargeDensityInputError, parse_formatted_density

_SPIN_HEADER = """ BEGIN header

           Real Lattice(A)               Lattice parameters(A)    Cell Angles
     2.0000000     0.0000000     0.0000000     a =    2.000000  alpha =   90.000000
     0.0000000     2.0000000     0.0000000     b =    2.000000  beta  =   90.000000
     0.0000000     0.0000000     2.0000000     c =    2.000000  gamma =   90.000000

   2   F                        ! nspins, non-collinear spin
   2     1     1                ! fine FFT grid along <a,b,c>
 END header: data is "<a b c> charge spin" in units of electrons/grid_point * number of grid_points

"""


def test_spin_polarised_density_keeps_the_spin_column(tmp_path: Path) -> None:
    path = tmp_path / "spin.den_fmt"
    path.write_text(
        _SPIN_HEADER + "     1     1     1   3.0   1.5\n     2     1     1   5.0  -0.5\n",
        encoding="utf-8",
    )

    density = parse_formatted_density(path)

    assert density.nspins == 2
    assert density.values.shape == (2, 1, 1, 2)
    assert density.values[:, 0, 0, 0].tolist() == [3.0, 5.0]
    assert density.values[:, 0, 0, 1].tolist() == [1.5, -0.5]


def test_spin_polarised_density_without_spin_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "no_spin.den_fmt"
    path.write_text(_SPIN_HEADER + "     1     1     1   3.0\n", encoding="utf-8")

    with pytest.raises(ChargeDensityInputError, match="has no spin column"):
        parse_formatted_density(path)
