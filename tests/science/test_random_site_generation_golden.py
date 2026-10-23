"""Golden-value tests pinning seeded random candidate-site generation.

``tests/golden/random_site_generation.json`` was recorded before the
random-filling loops in :mod:`mufinder.science.site_generation` were optimised
(rejecting on the bare candidate before building its orbit, and broadcasting
the distance checks), so the optimised loops must reproduce every accepted
site, in order, and the number of attempts used.

Regenerate the file (only for a deliberate behaviour change) with::

    python tests/science/test_random_site_generation_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

if (
    __name__ == "__main__"
):  # pragma: no cover - lets the regeneration script import ``tests.support``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mufinder.science.site_generation import (
    generate_random_sites,
    generate_random_sites_by_distance,
)
from tests.support.symmetry_hosts import (
    bcc_host,
    diamond_si_host,
    fcc_primitive_host,
    hcp_host,
    rutile_host,
    triclinic_host,
)

_GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "random_site_generation.json"

_HOST_FACTORIES = {
    "bcc": bcc_host,
    "diamond": diamond_si_host,
    "fcc": fcc_primitive_host,
    "hcp": hcp_host,
    "rutile": rutile_host,
    "triclinic": triclinic_host,
}

# Case name -> (host, generator, keyword arguments).  The distance-mode cases use
# the dialog's default spacings; attempts are trimmed to keep the suite fast.
_CASES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "bcc-distance": (
        "bcc",
        "distance",
        {"min_muon_muon": 0.8, "min_muon_atom": 1.0, "max_attempts": 1500, "seed": 3},
    ),
    "diamond-distance": (
        "diamond",
        "distance",
        {"min_muon_muon": 0.8, "min_muon_atom": 1.0, "max_attempts": 1500, "seed": 1},
    ),
    "fcc-distance": (
        "fcc",
        "distance",
        {"min_muon_muon": 0.6, "min_muon_atom": 0.9, "max_attempts": 1500, "seed": 2},
    ),
    "hcp-distance": (
        "hcp",
        "distance",
        {"min_muon_muon": 0.8, "min_muon_atom": 1.0, "max_attempts": 1500, "seed": 4},
    ),
    "rutile-distance": (
        "rutile",
        "distance",
        {"min_muon_muon": 0.8, "min_muon_atom": 1.0, "max_attempts": 3000, "seed": 0},
    ),
    "triclinic-distance": (
        "triclinic",
        "distance",
        {"min_muon_muon": 2.0, "min_muon_atom": 1.0, "max_attempts": 600, "seed": 5},
    ),
    "hcp-distance-no-symmetry": (
        "hcp",
        "distance",
        {
            "min_muon_muon": 0.8,
            "min_muon_atom": 1.0,
            "max_attempts": 400,
            "seed": 6,
            "include_symmetry_equivalents": False,
        },
    ),
    "rutile-count": (
        "rutile",
        "count",
        {"count": 4, "min_muon_muon": 0.8, "min_muon_atom": 1.0, "seed": 7},
    ),
    "diamond-count": (
        "diamond",
        "count",
        {"count": 3, "min_muon_muon": 0.8, "min_muon_atom": 1.0, "seed": 8},
    ),
    "bcc-count-partial": (
        "bcc",
        "count",
        {
            "count": 5,
            "min_muon_muon": 0.8,
            "min_muon_atom": 1.0,
            "max_attempts": 800,
            "seed": 9,
            "allow_partial": True,
        },
    ),
}


def _generate(case: str) -> dict[str, Any]:
    host_name, mode, options = _CASES[case]
    host = _HOST_FACTORIES[host_name]()
    kwargs = {
        "lattice_vectors": host.lattice_vectors,
        "host_positions_frac": host.fractional_positions,
        "host_species": host.species,
        "include_symmetry_equivalents": True,
        **options,
    }
    generator = generate_random_sites_by_distance if mode == "distance" else generate_random_sites
    points, attempts = generator(**kwargs)
    return {
        "attempts": int(attempts),
        "fractional": [[float(value) for value in point.fractional] for point in points],
        "cartesian": [[float(value) for value in point.cartesian] for point in points],
    }


@pytest.mark.parametrize("case", sorted(_CASES))
def test_random_site_generation_matches_recorded_golden_values(case: str) -> None:
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))[case]

    actual = _generate(case)

    assert actual["attempts"] == expected["attempts"]
    assert len(actual["fractional"]) == len(expected["fractional"])
    for key in ("fractional", "cartesian"):
        for actual_point, expected_point in zip(actual[key], expected[key], strict=True):
            assert actual_point == pytest.approx(expected_point, abs=1e-12)


if __name__ == "__main__":  # pragma: no cover - regeneration entry point.
    payload = {case: _generate(case) for case in sorted(_CASES)}
    _GOLDEN_PATH.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {_GOLDEN_PATH}")
    for case, data in payload.items():
        print(f"  {case}: {len(data['fractional'])} sites in {data['attempts']} attempts")
