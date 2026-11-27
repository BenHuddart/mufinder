"""Tests for :mod:`mufinder.services.principal_zpes`, on the paper's GdRu2Si2 outputs.

Huddart et al., Phys. Rev. B 111, 054440 (2025): the site 1 phonon run holds the muon at
another symmetry image of the site than its relaxed cell does, so its axes must be turned.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mufinder.castep_io.cell_parser import parse_cell_file
from mufinder.domain.models import Structure
from mufinder.science.quantum_phonons import parse_phonon_path
from mufinder.services.principal_zpes import site_principal_zpes
from tests.support.symmetry_hosts import gdru2si2_host

FIXTURES = Path(__file__).parents[1] / "fixtures" / "gdru2si2"


def _relaxed(number: int) -> Structure:
    cell = parse_cell_file(FIXTURES / f"GdRu2Si2_site{number}-out.cell")
    return Structure(
        label=f"site {number} relaxed",
        lattice_vectors=[list(row) for row in cell.lattice_vectors],
        species=list(cell.species),
        fractional_positions=[list(row) for row in cell.fractional_positions],
    )


def _principal(number: int, **overrides):
    arguments = {"host": gdru2si2_host(), "relaxed_site": _relaxed(number)} | overrides
    return site_principal_zpes(
        parse_phonon_path(FIXTURES / f"GdRu2Si2_site{number}.phonon"), **arguments
    )


def test_site_2s_249_mev_axis_is_1_bar_1_0_in_the_host_frame() -> None:
    principal = _principal(2)

    assert principal.at_relaxed_site
    assert [round(entry.zpe_eV * 1000) for entry in principal.axes] == [132, 249, 272]
    assert abs(float(np.dot(principal.axes[1].axis, (-1 / math.sqrt(2), 1 / math.sqrt(2), 0)))) == (
        pytest.approx(1.0, abs=1e-4)
    )


def test_site_1s_axes_are_turned_onto_its_relaxed_cell() -> None:
    """The phonons' soft axis is x; on the relaxed cell, which swaps x and y, it is y."""

    turned = _principal(1)
    unturned = _principal(1, host=None)

    assert turned.at_relaxed_site and not unturned.at_relaxed_site
    assert [round(entry.zpe_eV * 1000) for entry in turned.axes] == [70, 200, 287]
    assert np.abs(unturned.axes[0].axis) == pytest.approx((1.0, 0.0, 0.0), abs=1e-4)
    assert np.abs(turned.axes[0].axis) == pytest.approx((0.0, 1.0, 0.0), abs=1e-4)
    assert np.abs(turned.axes[1].axis) == pytest.approx((1.0, 0.0, 0.0), abs=1e-4)
    assert np.abs(turned.axes[2].axis) == pytest.approx((0.0, 0.0, 1.0), abs=1e-4)


def test_a_relaxed_site_whose_muon_is_no_image_keeps_the_phonon_frame() -> None:
    other = _relaxed(2)
    principal = _principal(1, relaxed_site=other)

    assert not principal.at_relaxed_site
    assert np.abs(principal.axes[0].axis) == pytest.approx((1.0, 0.0, 0.0), abs=1e-4)
