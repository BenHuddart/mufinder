"""The documentation screenshots that ``scripts/docs_screenshots.py`` renders.

The screenshots are not committed, so the only check that a page can build is
that the script still writes every image the pages reference. The script's
``SCREENSHOTS`` table is read from its source, and each referenced screenshot
is rendered in its own process: the script configures Qt, VTK and the settings
directories before importing them, which must not leak into the test session,
and one process per screenshot lets ``pytest-xdist`` spread the renders.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPT = PROJECT_ROOT / "scripts" / "docs_screenshots.py"

# One screenshot opens one project in one window; a few seconds is typical.
RENDER_TIMEOUT_S = 60

_SCREENSHOT_REFERENCE = re.compile(r"_generated/screenshots/([\w.-]+)\.png")


def _referenced_screenshots() -> set[str]:
    return {
        name
        for path in DOCS_ROOT.rglob("*.rst")
        if "_build" not in path.relative_to(DOCS_ROOT).parts
        for name in _SCREENSHOT_REFERENCE.findall(path.read_text(encoding="utf-8"))
    }


def _script_screenshots() -> set[str]:
    """The keys of the script's ``SCREENSHOTS`` dict literal."""

    for node in ast.parse(SCRIPT.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SCREENSHOTS"
            and isinstance(node.value, ast.Dict)
        ):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(f"{SCRIPT.name} has no SCREENSHOTS dict literal")


def test_the_pages_reference_exactly_the_screenshots_the_script_renders() -> None:
    referenced = _referenced_screenshots()
    rendered = _script_screenshots()

    assert referenced, "no page references a generated screenshot"
    assert referenced <= rendered, f"the pages reference screenshots the script does not render: {sorted(referenced - rendered)}"
    assert rendered <= referenced, f"the script renders screenshots no page uses: {sorted(rendered - referenced)}"


@pytest.mark.gui
@pytest.mark.parametrize("name", sorted(_referenced_screenshots()))
def test_the_screenshot_script_renders_a_referenced_screenshot(name: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path), name],
        capture_output=True,
        text=True,
        timeout=RENDER_TIMEOUT_S,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [f"{name}.png"]
    assert (tmp_path / f"{name}.png").stat().st_size > 10_000, f"{name}.png is suspiciously small"
