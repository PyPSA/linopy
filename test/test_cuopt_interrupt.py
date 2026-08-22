from __future__ import annotations

import _thread
import threading
import time

import pytest

from linopy.solvers import _run_cuopt_with_keyboard_interrupt


class DummySolve:
    """
    Stand-in for cuOpt's ``Solve``: blocking, and with no cancel API.

    cuOpt defers SIGINT for the whole duration of its C++ solve -- measured at
    52.9 s on a model that takes that long -- so linopy runs the call in a
    worker thread and waits in the main one. The GPU work keeps going in the
    background afterwards, which this dummy reproduces.
    """

    def __init__(self, duration: float = 0.5) -> None:
        self.duration = duration
        self.started = threading.Event()
        self.finished = threading.Event()

    def __call__(self) -> str:
        self.started.set()
        time.sleep(self.duration)
        self.finished.set()
        return "solution"


def test_run_cuopt_interrupt_reaches_the_main_thread() -> None:
    dummy = DummySolve()

    def interrupter() -> None:
        assert dummy.started.wait(timeout=1)
        _thread.interrupt_main()

    threading.Thread(target=interrupter, daemon=True).start()

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_cuopt_with_keyboard_interrupt(dummy)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert dummy.finished.wait(timeout=5)


def test_run_cuopt_returns_the_solve_result() -> None:
    assert _run_cuopt_with_keyboard_interrupt(lambda: "solution") == "solution"


def test_run_cuopt_reraises_solver_errors() -> None:
    def boom() -> None:
        raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="solver failed"):
        _run_cuopt_with_keyboard_interrupt(boom)
