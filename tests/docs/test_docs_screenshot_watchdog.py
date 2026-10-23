"""The time limit ``docs_screenshots.watchdog`` gives each screenshot scene.

A scene that runs past its limit ends the process, so each case runs in a
subprocess. The module imports no Qt.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
RENDERER = SCRIPTS / "render_docs_screenshots.py"
SUBPROCESS_TIMEOUT_S = 60


def _run_scene(
    body: str, *, seconds: float, cleanup: str = "None"
) -> subprocess.CompletedProcess[str]:
    program = "\n".join(
        [
            "import pathlib, sys, time",
            f"sys.path.insert(0, {str(SCRIPTS)!r})",
            "from docs_screenshots.watchdog import scene_deadline",
            f"cleanup = {cleanup}",
            f"with scene_deadline('slow_scene', {seconds}, cleanup=cleanup):",
            f"    {body}",
            "print('finished')",
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )


def test_a_scene_past_its_limit_is_named_and_every_stack_is_dumped() -> None:
    completed = _run_scene("time.sleep(30)", seconds=0.5)

    assert completed.returncode == 3
    assert (
        "Scene 'slow_scene' did not finish within 0.5 s; the stack of every thread follows."
        in completed.stderr
    )
    # The timer's own stack, then the stuck scene's, at the line it is stuck on.
    assert "Current thread 0x" in completed.stderr
    assert "\nThread 0x" in completed.stderr
    assert 'File "<string>", line 6 in <module>' in completed.stderr
    assert completed.stdout == ""


def test_a_scene_past_its_limit_runs_the_cleanup_before_exiting(tmp_path: Path) -> None:
    marker = tmp_path / "cleaned"

    completed = _run_scene(
        "time.sleep(30)", seconds=0.5, cleanup=f"lambda: pathlib.Path({str(marker)!r}).touch()"
    )

    assert completed.returncode == 3
    assert marker.exists()


def test_a_failing_cleanup_is_reported_and_the_exit_goes_ahead() -> None:
    completed = _run_scene("time.sleep(30)", seconds=0.5, cleanup="lambda: 1 / 0")

    assert completed.returncode == 3
    assert "Cleaning up after 'slow_scene' failed: division by zero" in completed.stderr


def test_a_scene_within_its_limit_finishes_quietly() -> None:
    completed = _run_scene("time.sleep(0.1)", seconds=10)

    assert completed.returncode == 0
    assert completed.stdout == "finished\n"
    assert completed.stderr == ""


def test_the_limit_does_not_outlive_its_scene() -> None:
    # Past the scene's block, neither the timer nor faulthandler's backstop fires.
    later = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys, time",
                    f"sys.path.insert(0, {str(SCRIPTS)!r})",
                    "from docs_screenshots import watchdog",
                    "watchdog.GIL_BACKSTOP_S = 0.2",
                    "with watchdog.scene_deadline('quick_scene', 0.3):",
                    "    pass",
                    "time.sleep(1.5)",
                    "print('finished')",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )

    assert later.returncode == 0
    assert later.stdout == "finished\n"
    assert later.stderr == ""


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "86401"])
def test_a_scene_timeout_out_of_range_is_refused(value: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(RENDERER), "--list", "--scene-timeout", value],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )

    assert completed.returncode == 2
    assert (
        f"--scene-timeout: must be more than 0 and at most 86400 seconds, not {value}"
        in completed.stderr
    )


def test_a_limit_that_runs_out_as_its_scene_finishes_does_nothing() -> None:
    # The timer is captured so it can be fired by hand after the block has
    # finished, which is the race a real timer can lose.
    program = "\n".join(
        [
            "import sys, threading",
            f"sys.path.insert(0, {str(SCRIPTS)!r})",
            "from docs_screenshots import watchdog",
            "timers = []",
            "class CapturedTimer(threading.Timer):",
            "    def start(self):",
            "        timers.append(self)",
            "watchdog.threading.Timer = CapturedTimer",
            "with watchdog.scene_deadline('quick_scene', 60):",
            "    pass",
            "timers[0].function()",
            "print('finished')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "finished\n"
    assert completed.stderr == ""


_REMOVE_WHILE_WRITING = """
import sys, threading, time
sys.path.insert(0, {scripts!r})
from docs_screenshots import render
{patch}
project = render.scratch_directory("project-")
stop = threading.Event()

def write():
    count = 0
    while not stop.is_set():
        try:
            (project / f"file{{count}}.json").write_text("{{}}")
        except OSError:
            pass
        count += 1

writer = threading.Thread(target=write)
writer.start()
time.sleep(0.2)
render.remove_scratch()
stop.set()
writer.join()
left = sorted(path.name for path in render._SCRATCH.parent.glob(render._SCRATCH.name + "*"))
print(left)
"""


def _remove_scratch_while_writing(patch: str = "") -> subprocess.CompletedProcess[str]:
    # The render module configures Qt as it is imported, so it runs in a subprocess.
    return subprocess.run(
        [sys.executable, "-c", _REMOVE_WHILE_WRITING.format(scripts=str(SCRIPTS), patch=patch)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )


def test_the_scratch_directory_is_removed_while_a_scene_still_writes_into_it() -> None:
    # A timed-out scene goes on writing its project while the time limit's
    # thread removes the scratch directory.
    completed = _remove_scratch_while_writing()

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_a_renamed_scratch_tree_that_resists_deletion_is_tried_again() -> None:
    # The first deletion leaves everything in place, as an open file on
    # Windows makes it; the renamed tree must not be forgotten.
    patch = "\n".join(
        [
            "import shutil",
            "_real_rmtree = shutil.rmtree",
            "_failures = [1]",
            "def _rmtree(path, ignore_errors=False):",
            "    if _failures:",
            "        _failures.pop()",
            "        return",
            "    _real_rmtree(path, ignore_errors=ignore_errors)",
            "render.shutil.rmtree = _rmtree",
        ]
    )

    completed = _remove_scratch_while_writing(patch)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
