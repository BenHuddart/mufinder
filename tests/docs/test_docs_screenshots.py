"""The documentation screenshots that ``scripts/render_docs_screenshots.py`` renders.

The screenshots are not committed, so the only check that a page can build is
that the renderer still writes every image the pages reference. The renderer
runs in subprocesses: importing ``docs_screenshots.render`` configures Qt, VTK
and the settings directories before importing them, which must not leak into
the test session. ``--list`` gives the registry's screenshot names, and each
example module is rendered in one process with ``--example``, so its projects
are built once and shared by its scenes, and ``pytest-xdist`` spreads the
examples across workers.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPT = PROJECT_ROOT / "scripts" / "render_docs_screenshots.py"
EXAMPLES_ROOT = PROJECT_ROOT / "scripts" / "docs_screenshots" / "examples"
IMAGE_SUFFIX = ".png"

# Listing imports Qt and the application but opens no window.
LIST_TIMEOUT_S = 60
# Building an example's projects, the largest of which drives a MainWindow
# through every results workflow, then one window per scene.
PROJECT_BUILD_TIMEOUT_S = 60
SCENE_TIMEOUT_S = 30

_SCREENSHOT_REFERENCE = re.compile(r"_generated/screenshots/([\w.-]+)" + re.escape(IMAGE_SUFFIX))


def _referenced_screenshots() -> set[str]:
    return {
        name
        for path in DOCS_ROOT.rglob("*.rst")
        if "_build" not in path.relative_to(DOCS_ROOT).parts
        for name in _SCREENSHOT_REFERENCE.findall(path.read_text(encoding="utf-8"))
    }


def _examples() -> list[str]:
    return sorted(path.stem for path in EXAMPLES_ROOT.glob("*.py") if path.stem != "__init__")


def _listed(*arguments: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list", *arguments],
        capture_output=True,
        text=True,
        timeout=LIST_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.split()


def test_the_pages_reference_exactly_the_screenshots_the_registry_renders() -> None:
    referenced = _referenced_screenshots()
    rendered = set(_listed())

    assert referenced, "no page references a generated screenshot"
    assert referenced <= rendered, f"the pages reference screenshots the registry does not render: {sorted(referenced - rendered)}"
    assert rendered <= referenced, f"the registry renders screenshots no page uses: {sorted(rendered - referenced)}"


def test_every_example_module_is_registered() -> None:
    per_example = [name for example in _examples() for name in _listed("--example", example)]

    assert sorted(per_example) == sorted(_listed()), "an example module is missing from the registry"


@pytest.mark.gui
@pytest.mark.timeout(0)  # The subprocess timeout below scales with the scene count.
@pytest.mark.parametrize("example", _examples())
def test_the_renderer_renders_every_scene_of_an_example(example: str, tmp_path: Path) -> None:
    names = _listed("--example", example)
    timeout = PROJECT_BUILD_TIMEOUT_S + SCENE_TIMEOUT_S * len(names)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path), "--example", example],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(f"{name}{IMAGE_SUFFIX}" for name in names)
    for name in names:
        assert (tmp_path / f"{name}{IMAGE_SUFFIX}").stat().st_size > 10_000, f"{name}{IMAGE_SUFFIX} is suspiciously small"
