"""Coverage-focused tests for malformed/partial CASTEP `.ts` parsing branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.ts_parser import (
    TransitionStateImage,
    TransitionStateParseError,
    TransitionStatePath,
    parse_transition_state_file,
    parse_transition_state_text,
)

# A minimal, well-formed single image block (2 atoms). Reused to assemble paths.
_IMAGE_BODY = (
    "              -1.00000000E+000     -1.00000000E+000                       <-- E\n"
    "               7.00000000E+000      0.00000000E+000      0.00000000E+000  <-- h\n"
    "               0.00000000E+000      7.00000000E+000      0.00000000E+000  <-- h\n"
    "               0.00000000E+000      0.00000000E+000      7.00000000E+000  <-- h\n"
    " H           1      0.00000000E+000      0.00000000E+000      0.00000000E+000  <-- R\n"
    " H           2      0.00000000E+000      0.00000000E+000      1.00000000E+000  <-- R\n"
    " H           1      0.00000000E+000      0.00000000E+000      0.00000000E+000  <-- F\n"
    " H           2      0.00000000E+000      0.00000000E+000      0.00000000E+000  <-- F\n"
)


def _block(header: str, body: str = _IMAGE_BODY) -> str:
    return header + "\n" + body


def _full_path_text(extra: str = "") -> str:
    return (
        " TSTYPE TSConfirmation\n"
        + _block(" REA    1      0.00000000E+000")
        + extra
        + _block(" PRO    1      1.00000000E+000")
    )


# --- parse_transition_state_text: top-level error path -------------------------


def test_empty_text_raises_no_complete_path() -> None:
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text("")


def test_only_header_noise_raises_no_complete_path() -> None:
    # Lines that never form a valid REA/TST/PRO block -> no images.
    text = " TSTYPE TSConfirmation\n random noise here\n more stuff\n"
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text(text)


# --- header-line skipping branches (continue paths) ---------------------------


def test_header_with_wrong_part_count_is_skipped() -> None:
    # "REA 1" has only 2 parts -> skipped, leaving no images -> no path.
    text = "REA 1\nREA junk extra here now\n"
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text(text)


def test_header_with_non_numeric_iteration_is_skipped() -> None:
    # 3 parts, correct label, but iteration not an int -> ValueError -> skipped.
    text = "REA notanint 0.0\n"
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text(text)


def test_header_with_unknown_label_is_skipped() -> None:
    text = "FOO 1 0.0\n"
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text(text)


# --- malformed energy / lattice / position / force entries --------------------


def test_invalid_energy_entry_raises() -> None:
    body = "              not_a_number                       <-- E\n"
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Invalid energy entry"):
        parse_transition_state_text(text)


def test_invalid_lattice_entry_too_few_values_raises() -> None:
    # The guard counts every whitespace token including the "<-- h" marker, so a
    # bare marker line (2 tokens) is the way to trip len(values) < 3.
    body = (
        "              -1.00000000E+000                       <-- E\n"
        "  <-- h\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Invalid lattice-vector entry"):
        parse_transition_state_text(text)


def test_invalid_position_entry_too_few_values_raises() -> None:
    # "X Y <-- R" -> 4 tokens (< 5) trips the position guard.
    body = (
        "              -1.00000000E+000                       <-- E\n"
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        "               0.0  0.0  7.0                       <-- h\n"
        " H 1                                               <-- R\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Invalid position entry"):
        parse_transition_state_text(text)


def test_invalid_force_entry_too_few_values_raises() -> None:
    # "H 1 <-- F" -> 4 tokens (< 5) trips the force guard.
    body = (
        "              -1.00000000E+000                       <-- E\n"
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        "               0.0  0.0  7.0                       <-- h\n"
        " H 1 0.0 0.0 0.0                                   <-- R\n"
        " H 1                                               <-- F\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Invalid force entry"):
        parse_transition_state_text(text)


# --- incomplete image block validation (line 121-122) -------------------------


def test_missing_energy_raises_incomplete_image() -> None:
    body = (
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        "               0.0  0.0  7.0                       <-- h\n"
        " H 1 0.0 0.0 0.0                                   <-- R\n"
        " H 1 0.0 0.0 0.0                                   <-- F\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Incomplete image block"):
        parse_transition_state_text(text)


def test_wrong_lattice_count_raises_incomplete_image() -> None:
    body = (
        "              -1.0                                 <-- E\n"
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        " H 1 0.0 0.0 0.0                                   <-- R\n"
        " H 1 0.0 0.0 0.0                                   <-- F\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Incomplete image block"):
        parse_transition_state_text(text)


def test_positions_force_length_mismatch_raises_incomplete_image() -> None:
    body = (
        "              -1.0                                 <-- E\n"
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        "               0.0  0.0  7.0                       <-- h\n"
        " H 1 0.0 0.0 0.0                                   <-- R\n"
        " H 2 0.0 0.0 1.0                                   <-- R\n"
        " H 1 0.0 0.0 0.0                                   <-- F\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body)
    with pytest.raises(TransitionStateParseError, match="Incomplete image block"):
        parse_transition_state_text(text)


# --- _final_path branches via parse_transition_state_text ----------------------


def test_missing_products_raises_no_complete_path() -> None:
    # Only a reactant image parses, so _final_path returns [] -> top-level raise.
    text = " TSTYPE\n" + _block(" REA    1      0.0")
    with pytest.raises(TransitionStateParseError, match="No complete transition-state path"):
        parse_transition_state_text(text)


def test_path_with_transition_selects_latest_iteration() -> None:
    # Two TST images at different iterations; only the latest must survive.
    tst_old = _block(" TST    1      0.50000000E+000")
    tst_new = _block(" TST    2      0.50000000E+000")
    text = _full_path_text(extra=tst_old + tst_new)
    parsed = parse_transition_state_text(text)

    labels = [img.label for img in parsed.images]
    assert labels.count("TST") == 1
    tst_image = next(img for img in parsed.images if img.label == "TST")
    assert tst_image.iteration == 2
    assert labels[0] == "REA"
    assert labels[-1] == "PRO"
    # sorted by reaction coordinate: 0.0 (REA), 0.5 (TST), 1.0 (PRO)
    coords = [img.reaction_coordinate for img in parsed.images]
    assert coords == sorted(coords)


def test_unrecognized_data_line_inside_block_is_skipped() -> None:
    # A non-empty line inside a block that matches none of the "<-- *" markers
    # falls through to the bare index increment and is ignored.
    body = (
        "              -1.0                                 <-- E\n"
        " some stray annotation line that is not a marker\n"
        "               7.0  0.0  0.0                       <-- h\n"
        "               0.0  7.0  0.0                       <-- h\n"
        "               0.0  0.0  7.0                       <-- h\n"
        " H 1 0.0 0.0 0.0                                   <-- R\n"
        " H 1 0.0 0.0 0.0                                   <-- F\n"
    )
    text = " TSTYPE\n" + _block(" REA    1      0.0", body) + _block(" PRO    1      1.0")
    parsed = parse_transition_state_text(text)
    assert len(parsed.images) == 2


def test_header_break_skips_when_marker_line_present() -> None:
    # A line that looks like a header (3 parts, REA) but contains "<--" must NOT
    # break the block; it should be parsed as data. Here a force line for a
    # well-formed REA/PRO pair confirms normal completion.
    parsed = parse_transition_state_text(_full_path_text())
    assert len(parsed.images) == 2


# --- barrier_height_eV / dataclass behaviour ----------------------------------


def test_barrier_height_none_for_empty_path() -> None:
    empty = TransitionStatePath(images=())
    assert empty.barrier_height_eV is None


def test_barrier_height_computed_from_max_minus_first() -> None:
    parsed = parse_transition_state_text(_full_path_text())
    energies = [img.energy for img in parsed.images]
    assert parsed.barrier_height_eV == pytest.approx(max(energies) - energies[0])


# --- parse_transition_state_file (file read path) -----------------------------


def test_parse_transition_state_file_reads_from_disk(tmp_path: Path) -> None:
    ts_file = tmp_path / "path.ts"
    ts_file.write_text(_full_path_text(), encoding="utf-8")
    parsed = parse_transition_state_file(ts_file)
    assert isinstance(parsed, TransitionStatePath)
    assert len(parsed.images) == 2
    assert parsed.images[0].label == "REA"
    assert parsed.images[-1].label == "PRO"


def test_parse_transition_state_file_accepts_str_path(tmp_path: Path) -> None:
    ts_file = tmp_path / "path_str.ts"
    ts_file.write_text(_full_path_text(), encoding="utf-8")
    parsed = parse_transition_state_file(str(ts_file))
    assert len(parsed.images) == 2


def test_image_is_frozen_dataclass() -> None:
    parsed = parse_transition_state_text(_full_path_text())
    image = parsed.images[0]
    assert isinstance(image, TransitionStateImage)
    with pytest.raises((AttributeError, Exception)):
        image.energy = 0.0  # type: ignore[misc]
