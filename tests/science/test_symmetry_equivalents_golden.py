"""Golden-value tests pinning symmetry-equivalent positions and symmetry distances.

``tests/golden/symmetry_equivalents.json`` was recorded from the original
pure-Python implementation of the symmetry-equivalent helpers in
:mod:`mufinder.science.site_generation` and the clustering helpers built on
them in :mod:`mufinder.science.relaxation_analysis`, so performance work on
either module has to reproduce it: the order of the equivalents, which near
duplicates are merged, the chosen closest representative, every distance, and
the tolerance-scan cluster counts.

Regenerate the file (only for a deliberate behaviour change) with::

    python tests/science/test_symmetry_equivalents_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from uuid import UUID

import pytest

# Lets the regeneration script import ``tests.support``.
if __name__ == "__main__":  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    anchored_sites,
    cluster_connected_components,
    scan_connected_component_tolerances,
    symmetry_distance_matrix,
)
from mufinder.science.site_generation import (
    closest_symmetry_equivalent_position,
    lattice_frac_to_cart,
    symmetry_equivalent_positions,
)
from tests.support.symmetry_hosts import (
    bcc_host,
    fcc_primitive_host,
    hcp_host,
    rutile_host,
    triclinic_host,
)

_GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "symmetry_equivalents.json"

# Host factory, candidate fractional positions, and the connected-components
# tolerance used to label sites for anchoring.  The positions mix general
# positions, special positions, points a few 1e-7 off a special position (whose
# images straddle the 1e-5 Å deduplication tolerance), and unwrapped inputs.
_CASES = {
    "bcc": (
        bcc_host,
        [
            (0.10, 0.10, 0.10),
            (0.12, 0.10, 0.10),
            (0.60, 0.20, 0.40),
            (0.61, 0.21, 0.40),
            (0.13, 0.27, 0.41),
            (0.25, 0.75, 0.50),
            (0.25, 0.5000005, 0.0),
            (0.25, 0.5000015, 0.0),
            (0.49, 0.01, 0.00),
            (-1e-17, 0.5, 0.25),
            (1.25, -0.40, 0.70),
        ],
        0.3,
    ),
    "triclinic": (
        triclinic_host,
        [
            (0.10, 0.10, 0.10),
            (0.112, 0.104, 0.100),
            (0.5, 0.5, 0.5),
            (0.8, 0.8, 0.8),
            (-0.05, 1.02, 0.99),
        ],
        0.2,
    ),
    "hcp": (
        hcp_host,
        [
            (0.01, 0.00, 0.01),
            (0.0, 0.0, 0.5),
            (1.0 / 3.0, 2.0 / 3.0, 0.62),
            (0.34, 0.66, 0.63),
            (0.5, 0.0, 0.0),
            (0.2, 0.1, 0.3),
            (0.5000004, 0.0, 0.0),
        ],
        0.3,
    ),
    "rutile": (
        rutile_host,
        [
            (0.5, 0.0, 0.0),
            (0.5, 0.01, 0.49),
            (0.0, 0.5, 0.25),
            (0.3, 0.0, 0.5),
            (0.12, 0.27, 0.41),
            (0.0, 0.5, 0.2500006),
        ],
        0.3,
    ),
    "fcc": (
        fcc_primitive_host,
        [
            (0.5, 0.5, 0.5),
            (0.25, 0.25, 0.25),
            (0.26, 0.25, 0.24),
            (0.1, 0.3, 0.2),
            (0.7500003, 0.75, 0.75),
        ],
        0.3,
    ),
}

_SCAN_POINTS = 11


def _site(host: Structure, index: int, fractional: tuple[float, float, float]) -> RelaxedMuonSite:
    return RelaxedMuonSite(
        job_id=UUID(int=index + 1),
        job_label=f"site-{index}",
        charge_state="positive_muon",
        structure_id=UUID(int=1000 + index),
        site_id=None,
        folded_fractional_position=tuple(value % 1.0 for value in fractional),
        cartesian_position_ang=tuple(lattice_frac_to_cart(fractional, host.lattice_vectors)),
        total_energy_ev=-10.0 + 0.05 * index,
    )


def _host_kwargs(host: Structure) -> dict[str, Any]:
    return {
        "lattice_vectors": host.lattice_vectors,
        "host_species": host.species,
        "host_positions_frac": host.fractional_positions,
    }


def _collect(host_name: str) -> dict[str, Any]:
    factory, fractions, tolerance = _CASES[host_name]
    host = factory()
    sites = [_site(host, index, fraction) for index, fraction in enumerate(fractions)]
    # Each candidate is anchored to the next one, so most anchors sit in a different orbit.
    anchors = fractions[1:] + fractions[:1]
    labels = cluster_connected_components(sites, host, tolerance_ang=tolerance).cluster_labels
    return {
        "equivalents": [
            symmetry_equivalent_positions(fraction, **_host_kwargs(host)) for fraction in fractions
        ],
        "closest": [
            closest_symmetry_equivalent_position(fraction, anchor_frac=anchor, **_host_kwargs(host))
            for fraction, anchor in zip(fractions, anchors, strict=True)
        ],
        "distance_matrix": symmetry_distance_matrix(sites, host),
        "scan": scan_connected_component_tolerances(
            sites, host, start_ang=0.0, stop_ang=1.0, points=_SCAN_POINTS
        ),
        "anchored": [
            [list(site.shifted_fractional_position), site.shifted, site.cluster_index]
            for site in anchored_sites(sites, host, labels)
        ],
    }


def _as_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    return float(value)


def _golden() -> dict[str, Any]:
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("host_name", sorted(_CASES))
def test_symmetry_helpers_match_recorded_golden_values(host_name: str) -> None:
    expected = _golden()[host_name]
    actual = _as_json(_collect(host_name))

    assert [len(group) for group in actual["equivalents"]] == [
        len(group) for group in expected["equivalents"]
    ]
    for actual_group, expected_group in zip(
        actual["equivalents"], expected["equivalents"], strict=True
    ):
        for actual_position, expected_position in zip(actual_group, expected_group, strict=True):
            assert actual_position == pytest.approx(expected_position, abs=1e-12)
    for actual_position, expected_position in zip(
        actual["closest"], expected["closest"], strict=True
    ):
        assert actual_position == pytest.approx(expected_position, abs=1e-12)
    for actual_row, expected_row in zip(
        actual["distance_matrix"], expected["distance_matrix"], strict=True
    ):
        assert actual_row == pytest.approx(expected_row, abs=1e-12)
    assert [tuple(row) for row in actual["scan"]] == [tuple(row) for row in expected["scan"]]
    assert len(actual["anchored"]) == len(expected["anchored"])
    for (position, shifted, cluster), (
        expected_position,
        expected_shifted,
        expected_cluster,
    ) in zip(actual["anchored"], expected["anchored"], strict=True):
        assert position == pytest.approx(expected_position, abs=1e-12)
        assert (shifted, cluster) == (expected_shifted, expected_cluster)


def test_golden_cases_exercise_deduplication_tolerance() -> None:
    expected = _golden()
    bcc = expected["bcc"]["equivalents"]
    # Sanity checks on the recorded data itself, so a regeneration cannot silently lose coverage.
    assert len(bcc[0]) == 16  # 16f
    assert len(bcc[4]) == 96  # general position
    assert len(bcc[6]) == 12  # 5e-7 off 12d: images 5 µÅ apart merge
    assert len(bcc[7]) == 48  # 1.5e-6 off 12d: images 15 µÅ apart stay distinct
    assert len(expected["triclinic"]["equivalents"][0]) == 1


if __name__ == "__main__":  # pragma: no cover - regeneration entry point.
    payload = {name: _as_json(_collect(name)) for name in sorted(_CASES)}
    _GOLDEN_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {_GOLDEN_PATH}")
