from __future__ import annotations

from pathlib import Path

from mufinder.castep_io import parse_transition_state_file

_FIXTURE = Path(__file__).parent / "fixtures" / "transition_state_path.ts"


def test_parse_transition_state_file_returns_final_path() -> None:
    parsed = parse_transition_state_file(_FIXTURE)

    assert len(parsed.images) >= 2
    assert parsed.images[0].label == "REA"
    assert parsed.images[-1].label == "PRO"
    assert parsed.barrier_height_eV is not None
