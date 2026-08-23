from __future__ import annotations

import _thread
import contextlib
import os
import select
import signal
import threading
import time

import pytest

from linopy.solvers import _cuopt_solve_queue, _run_cuopt_with_keyboard_interrupt


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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
def test_solve_queue_starts_a_fresh_worker_after_fork() -> None:
    """
    A forked child inherits the cached queue but not its daemon worker.

    Without the at-fork ``cache_clear`` the child hands its job to a queue
    nobody reads and waits on ``job.done`` forever; it then never writes to the
    pipe and the parent's ``select`` timeout below fails the test.
    """
    assert _run_cuopt_with_keyboard_interrupt(lambda: "parent") == "parent"

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child -- never returns
        exit_code = 1
        try:
            os.close(read_fd)
            cleared = _cuopt_solve_queue.cache_info().currsize == 0
            solved = _run_cuopt_with_keyboard_interrupt(lambda: "child") == "child"
            if cleared and solved:
                os.write(write_fd, b"ok")
                exit_code = 0
        finally:
            os._exit(exit_code)

    os.close(write_fd)
    reaped = False
    try:
        ready, _, _ = select.select([read_fd], [], [], 30)
        assert ready, "the forked child never finished its solve"
        assert os.read(read_fd, 8) == b"ok"
        wait_status = os.waitpid(pid, 0)[1]
        reaped = True
        assert os.waitstatus_to_exitcode(wait_status) == 0
    finally:
        os.close(read_fd)
        # Once the child is reaped its pid is free for reuse, so signalling it
        # again could land on an unrelated process.
        if not reaped:
            with contextlib.suppress(ChildProcessError, ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
