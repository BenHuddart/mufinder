"""Reading a CASTEP output in pieces, so a stop waits for one piece, not the file.

A parser given ``should_stop`` must read exactly what it reads without one, and
must ask often enough that a stop never waits for the rest of a large file.
The pieces are cut tiny here so every cut lands inside lines and matches.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from mufinder.castep_io import output_parser, run_parameters
from mufinder.castep_io.output_parser import parse_castep_output
from mufinder.castep_io.run_parameters import parse_run_parameters
from mufinder.castep_io.stoppable_read import (
    ReadStopped,
    checked_lines,
    finditer,
    next_line_index,
    read_text,
    search,
    split_lines,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUTS = sorted(FIXTURES.rglob("*.castep"))


def _never() -> bool:
    return False


class _Asks:
    """A ``should_stop`` that counts its asks and answers true from the ``stop_at``-th on."""

    def __init__(self, stop_at: int | None = None) -> None:
        self.count = 0
        self.stop_at = stop_at

    def __call__(self) -> bool:
        self.count += 1
        return self.stop_at is not None and self.count >= self.stop_at


def test_there_are_outputs_to_compare() -> None:
    assert len(OUTPUTS) >= 5


@pytest.mark.usefixtures("tiny_read_pieces")
class TestAReadThatIsNotStoppedReadsTheSame:
    @pytest.mark.parametrize("output", OUTPUTS, ids=lambda path: path.name)
    def test_the_output_parse(self, output: Path) -> None:
        assert parse_castep_output(output, should_stop=_never) == parse_castep_output(output)

    @pytest.mark.parametrize("output", OUTPUTS, ids=lambda path: path.name)
    def test_the_parameter_echo(self, output: Path) -> None:
        assert parse_run_parameters(output, should_stop=_never) == parse_run_parameters(output)

    def test_an_appended_output_of_several_restarts(self, tmp_path: Path) -> None:
        output = tmp_path / "appended.castep"
        output.write_text("".join(path.read_text() for path in OUTPUTS[:3]))

        assert parse_castep_output(output, should_stop=_never) == parse_castep_output(output)
        assert parse_run_parameters(output, should_stop=_never) == parse_run_parameters(output)

    def test_text_with_every_line_ending_and_bad_bytes(self, tmp_path: Path) -> None:
        output = tmp_path / "mixed.castep"
        body = "".join(
            f"line {index} é\r\n Total time = {index}.5 s\rnext\x0bvertical\x1crecord\n"
            for index in range(40)
        )
        output.write_bytes(body.encode() + b"\xff\xfe tail without a newline")

        text = read_text(output, _never)

        assert text == output.read_text(encoding="utf-8", errors="replace")
        assert split_lines(text, _never) == text.splitlines()

    def test_a_match_across_a_cut_is_found_whole(self) -> None:
        pattern = re.compile(r"Total\s+energy[^\n=]*=\s*([-+]?\d+(?:\.\d+)?)\s*eV")
        text = "".join(
            f"{'x' * index}Total\n   energy = -{index}.123456789    eV\n" for index in range(60)
        )

        found = [(match.span(), match.group(1)) for match in finditer(pattern, text, _never)]

        assert found == [(match.span(), match.group(1)) for match in pattern.finditer(text)]
        assert len(found) == 60
        assert search(pattern, text, _never).span() == pattern.search(text).span()

    def test_lines_and_the_next_wanted_line(self) -> None:
        lines = [f"line {index}" for index in range(20)]

        assert list(checked_lines(lines, _never)) == lines
        assert next_line_index(lines, 4, lambda line: line.endswith("7"), _never) == 7
        assert next_line_index(lines, 8, lambda line: line == "none", _never) == 20


#: Every pattern the parsers search window by window. A new one belongs here.
_WINDOWED_PATTERNS = (
    *output_parser._ENERGY_PATTERNS,
    *output_parser._ENTHALPY_PATTERNS,
    *output_parser._MAX_FORCE_PATTERNS,
    output_parser._WALL_SECONDS_RE,
    output_parser._WALL_HMS_RE,
    output_parser._GEOM_SUCCESS_RE,
    output_parser._GEOM_FAILURE_RE,
    run_parameters._CASTEP_VERSION_RE,
)


@pytest.mark.parametrize("pattern", _WINDOWED_PATTERNS, ids=lambda pattern: pattern.pattern[:40])
def test_a_windowed_pattern_meets_the_windows_condition(pattern: re.Pattern[str]) -> None:
    """A window ends where the text does not, so a pattern that looks past its
    match, or matches nothing, would read differently in pieces (see ``finditer``)."""

    source = pattern.pattern.replace("\\\\", "")
    for construct in ("$", "\\Z", "(?=", "(?!", "\\b", "\\B"):
        assert construct not in source, f"{construct} in {pattern.pattern}"
    assert pattern.search("") is None


@pytest.mark.usefixtures("tiny_read_pieces")
class TestStoppingARead:
    def test_a_parse_stops_at_the_first_ask_answered_true(self) -> None:
        output = max(OUTPUTS, key=lambda path: path.stat().st_size)
        asks = _Asks(stop_at=40)

        with pytest.raises(ReadStopped):
            parse_castep_output(output, should_stop=asks)

        assert asks.count == 40

    def test_the_parameter_echo_stops_too(self) -> None:
        asks = _Asks(stop_at=10)

        with pytest.raises(ReadStopped):
            parse_run_parameters(OUTPUTS[0], should_stop=asks)

        assert asks.count == 10

    def test_a_parse_asks_throughout_a_file_not_once(self, tmp_path: Path) -> None:
        """The asks grow with the file, so no part of a large file goes unasked."""

        source = OUTPUTS[0].read_text()
        once, eight_times = tmp_path / "once.castep", tmp_path / "eight.castep"
        once.write_text(source)
        eight_times.write_text(source * 8)
        asked_once, asked_eight_times = _Asks(), _Asks()

        parse_castep_output(once, should_stop=asked_once)
        parse_castep_output(eight_times, should_stop=asked_eight_times)

        assert asked_once.count > 100
        assert asked_eight_times.count > 6 * asked_once.count
