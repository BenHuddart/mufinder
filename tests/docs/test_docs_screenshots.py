"""The user-guide screenshots that ``scripts/docs_screenshots.py`` renders.

The screenshots are not committed, so the only check that a page can build is
that the script still writes every image the pages reference. The script runs
in its own process: it configures Qt, VTK and the settings directories before
importing them, which must not leak into the test session.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPT = PROJECT_ROOT / "scripts" / "docs_screenshots.py"

_SCREENSHOT_REFERENCE = re.compile(r"_generated/screenshots/([\w.-]+\.png)")


def _referenced_screenshots() -> set[str]:
    return {
        name
        for path in DOCS_ROOT.rglob("*.rst")
        if "_build" not in path.relative_to(DOCS_ROOT).parts
        for name in _SCREENSHOT_REFERENCE.findall(path.read_text(encoding="utf-8"))
    }


@pytest.mark.gui
def test_the_screenshot_script_renders_every_image_the_docs_reference(tmp_path: Path) -> None:
    referenced = _referenced_screenshots()
    assert referenced, "no page references a generated screenshot"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    written = {path.name for path in tmp_path.glob("*.png")}
    assert referenced <= written, f"the pages reference screenshots the script does not render: {sorted(referenced - written)}"
    assert written <= referenced, f"the script renders screenshots no page uses: {sorted(written - referenced)}"
    for name in written:
        assert (tmp_path / name).stat().st_size > 10_000, f"{name} is suspiciously small"
