"""How ``docs_screenshots.workers`` runs render workers, reports a failure and re-runs a crash.

The module imports no Qt, so it is imported here directly. The workers are a
stand-in script that takes the renderer's done-file option and scene names:
a scene named ``crash-once-*`` ends its worker with ``SIGSEGV`` the first time
it is rendered, ``crash-always-*`` every time, ``term-*`` ends it with
``SIGTERM``, ``raise-*`` exits with status 1, ``wait-for-rerun-*`` waits until
some scene has been started twice (exiting with status 1 after 30 s), and any
other scene is written.
"""

from __future__ import annotations

from pathlib import Path
import signal
import sys
import textwrap

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from docs_screenshots.workers import DONE_FILE_OPTION, describe_exit, render_shares  # noqa: E402

FAKE_WORKER = textwrap.dedent(
    """
    import faulthandler, os, pathlib, signal, sys, time

    faulthandler.enable(all_threads=True)
    assert sys.argv[1] == {option!r}
    done_file = pathlib.Path(sys.argv[2])
    log = pathlib.Path({log!r})
    for name in sys.argv[3:]:
        with log.open("a") as stream:
            stream.write(name + "\\n")
        crashed_before = pathlib.Path({log!r} + "." + name)
        if name.startswith("crash-always-") or (
            name.startswith("crash-once-") and not crashed_before.exists()
        ):
            crashed_before.touch()
            os.kill(os.getpid(), signal.SIGSEGV)
        if name.startswith("term-"):
            os.kill(os.getpid(), signal.SIGTERM)
        if name.startswith("raise-"):
            sys.exit(1)
        if name.startswith("wait-for-rerun-"):
            deadline = time.monotonic() + 30
            while True:
                started = log.read_text().split()
                if len(started) != len(set(started)):
                    break
                if time.monotonic() > deadline:
                    sys.exit(1)
                time.sleep(0.05)
        with done_file.open("a") as stream:
            stream.write(name + "\\n")
    """
)


@pytest.fixture
def worker_command(tmp_path: Path) -> tuple[list[str], Path]:
    """The stand-in worker's command and the log of every scene it started, in order."""

    log = tmp_path / "started.log"
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER.format(option=DONE_FILE_OPTION, log=str(log)), encoding="utf-8")
    return [sys.executable, str(script)], log


def _started(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").split() if log.exists() else []


def test_workers_that_all_succeed_report_nothing(worker_command, capfd) -> None:
    command, log = worker_command

    assert render_shares([["a", "b"], ["c"], []], command)

    assert sorted(_started(log)) == ["a", "b", "c"]
    assert "exited with" not in capfd.readouterr().err


def test_a_crashed_worker_is_reported_and_its_unwritten_scenes_rerun_once(
    worker_command, capfd
) -> None:
    command, log = worker_command

    assert render_shares([["a", "crash-once-b", "c"], ["d"]], command)

    started = _started(log)
    # "a" was written before the crash, so the re-run renders "crash-once-b" and "c".
    assert sorted(started) == sorted(["a", "crash-once-b", "d", "crash-once-b", "c"])
    err = capfd.readouterr().err
    assert "Render worker 1 exited with signal SIGSEGV (-11)." in err
    assert "Its scenes: a, crash-once-b, c." in err
    assert "It was on: crash-once-b." in err
    assert "Rendered before it ended: a." in err
    assert "Re-running its 2 unrendered scenes in a fresh process." in err
    # The fault handler in the worker printed the Python stack of the crash.
    assert "Fatal Python error: Segmentation fault" in err
    assert "Render worker 2" not in err


def test_a_second_crash_fails_the_render(worker_command, capfd) -> None:
    command, log = worker_command

    assert not render_shares([["a", "crash-always-b", "c"]], command)

    assert _started(log) == ["a", "crash-always-b", "crash-always-b"]
    err = capfd.readouterr().err
    assert "Render worker 1 exited with signal SIGSEGV (-11)." in err
    assert "Re-run worker 1 exited with signal SIGSEGV (-11)." in err
    assert "Its scenes: crash-always-b, c." in err


def test_a_worker_that_exits_with_a_status_fails_at_once(worker_command, capfd) -> None:
    command, log = worker_command

    assert not render_shares([["a", "raise-b", "c"], ["d"]], command)

    assert sorted(_started(log)) == ["a", "d", "raise-b"]
    err = capfd.readouterr().err
    assert "Render worker 1 exited with status 1." in err
    assert "It was on: raise-b." in err
    assert "Re-running" not in err


def test_a_rerun_starts_while_the_other_workers_still_run(worker_command, capfd) -> None:
    command, log = worker_command

    # The second worker finishes only once the first worker's crashed scene has
    # been started again, so the render passes only if the re-run does not wait
    # for it.
    assert render_shares([["crash-once-a"], ["wait-for-rerun-b"]], command)

    err = capfd.readouterr().err
    assert "Render worker 1 exited with signal SIGSEGV (-11)." in err
    assert "Render worker 2" not in err


def test_a_worker_stopped_by_sigterm_is_not_rerun(worker_command, capfd) -> None:
    command, log = worker_command

    assert not render_shares([["a", "term-b", "c"]], command)

    assert _started(log) == ["a", "term-b"]
    err = capfd.readouterr().err
    assert "Render worker 1 exited with signal SIGTERM (-15)." in err
    assert "Re-running" not in err


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (-signal.SIGSEGV, "signal SIGSEGV (-11)"),
        (-signal.SIGABRT, "signal SIGABRT (-6)"),
        (3, "status 3"),
        (-200, "signal 200 (-200)"),
    ],
)
def test_an_exit_code_is_described_by_its_signal_or_status(code: int, expected: str) -> None:
    assert describe_exit(code) == expected
