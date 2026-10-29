"""Tests for the startup cap on the native maths thread pools."""

from __future__ import annotations

import mufinder.native_threads as native_threads


def test_cap_sets_every_unsized_pool_to_four() -> None:
    environ: dict[str, str] = {"PATH": "/usr/bin"}

    capped = native_threads.cap_native_thread_pools(environ)

    assert capped == native_threads.THREAD_POOL_VARIABLES
    assert environ == {
        "PATH": "/usr/bin",
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }


def test_cap_keeps_the_users_values_and_treats_an_empty_value_as_unset() -> None:
    environ = {"OPENBLAS_NUM_THREADS": "16", "OMP_NUM_THREADS": ""}

    capped = native_threads.cap_native_thread_pools(environ, threads=2)

    assert capped == ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    assert environ == {
        "OPENBLAS_NUM_THREADS": "16",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }


def test_only_the_process_environment_is_recorded_for_child_processes(monkeypatch) -> None:
    monkeypatch.setattr(native_threads, "_capped_variables", ())

    native_threads.cap_native_thread_pools({})

    assert native_threads.capped_thread_variables() == ()
