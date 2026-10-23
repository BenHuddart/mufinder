"""The time limit ``docs_screenshots.watchdog`` gives each screenshot scene.

A scene that runs past its limit ends the process, so each case runs in a
subprocess. The module imports no Qt.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
RENDERER = SCRIPTS / "render_docs_screenshots.py"
SUBPROCESS_TIMEOUT_S = 60


def _run_scene(body: str, *, seconds: float) -> subprocess.CompletedProcess[str]:
    program = "\n".join(
        [
            "import sys, time",
            f"sys.path.insert(0, {str(SCRIPTS)!r})",
            "from docs_screenshots.watchdog import scene_deadline",
            f"with scene_deadline('slow_scene', {seconds}):",
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
    assert "Scene 'slow_scene' did not finish within 0.5 s; the stack of every thread follows." in completed.stderr
    # The timer's own stack, then the stuck scene's, at the line it is stuck on.
    assert "Current thread 0x" in completed.stderr
    assert "\nThread 0x" in completed.stderr
    assert 'File "<string>", line 5 in <module>' in completed.stderr
    assert completed.stdout == ""


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


def test_a_scene_timeout_of_zero_is_refused() -> None:
    completed = subprocess.run(
        [sys.executable, str(RENDERER), "--list", "--scene-timeout", "0"],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
        check=False,
    )

    assert completed.returncode == 2
    assert "--scene-timeout: must be more than 0 seconds, not 0" in completed.stderr
