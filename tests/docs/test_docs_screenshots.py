"""The documentation screenshots that ``scripts/render_docs_screenshots.py`` renders.

The screenshots are not committed, so the only check that a page can build is
that the renderer still writes every image the pages reference. The renderer
runs in subprocesses: importing ``docs_screenshots.render`` configures Qt, VTK
and the settings directories before importing them, which must not leak into
the test session. ``--list`` gives the registry's screenshot names, and each
example module is rendered in one process with ``--example``, so its projects
are built once per render worker and shared by that worker's scenes, and
``pytest-xdist`` spreads the examples across test workers.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPT = PROJECT_ROOT / "scripts" / "render_docs_screenshots.py"
EXAMPLES_ROOT = PROJECT_ROOT / "scripts" / "docs_screenshots" / "examples"
DURATIONS = PROJECT_ROOT / "scripts" / "docs_screenshots" / "durations.json"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
IMAGE_SUFFIX = ".webp"

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


def test_the_recorded_durations_cover_exactly_the_registry() -> None:
    recorded = json.loads(DURATIONS.read_text(encoding="utf-8"))

    assert list(recorded) == _listed(), (
        "durations.json is out of step with the registry; record the new screenshots with "
        "python scripts/render_docs_screenshots.py --store-durations NAME ..."
    )


def _ci_screenshot_shards() -> int:
    """The ``docs-screenshots`` matrix size, checked against the ``SHARDS`` its render step passes."""

    job = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["docs-screenshots"]
    shards = job["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1)), "the docs-screenshots matrix must number its shards 1..N"
    render = next(step["run"] for step in job["steps"] if "screenshots-shard" in step.get("run", ""))
    assert f"SHARDS={len(shards)}" in render, "the render step's SHARDS differs from the matrix size"
    return len(shards)


def test_the_ci_shards_render_every_screenshot_exactly_once() -> None:
    count = _ci_screenshot_shards()
    shards = [_listed("--shard", f"{shard}/{count}") for shard in range(1, count + 1)]

    assert all(shards), "a CI shard renders no screenshot"
    assert sorted(name for shard in shards for name in shard) == sorted(_listed())


def test_a_shard_out_of_range_is_refused() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list", "--shard", "5/4"],
        capture_output=True,
        text=True,
        timeout=LIST_TIMEOUT_S,
        check=False,
    )

    assert completed.returncode == 2
    assert "--shard: shard 5 is not between 1 and 4" in completed.stderr


@pytest.mark.gui
@pytest.mark.screenshots
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
