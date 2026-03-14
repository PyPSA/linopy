from __future__ import annotations

import _thread
import threading
import time

import pytest

from linopy.solvers import _run_highs_with_keyboard_interrupt


class DummyHighs:
    def __init__(self) -> None:
        self.HandleKeyboardInterrupt = False
        self.HandleUserInterrupt = False
        self._cancel_event = threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.cancel_calls = 0

    def run(self) -> None:
        self.started.set()
        self._cancel_event.wait(timeout=5)
        self.finished.set()

    def cancelSolve(self) -> None:
        self.cancel_calls += 1
        self._cancel_event.set()


def test_run_highs_cancels_on_keyboard_interrupt() -> None:
    dummy = DummyHighs()

    def interrupter() -> None:
        assert dummy.started.wait(timeout=1)
        time.sleep(0.05)
        _thread.interrupt_main()

    threading.Thread(target=interrupter, daemon=True).start()

    with pytest.raises(KeyboardInterrupt):
        _run_highs_with_keyboard_interrupt(dummy)

    assert dummy.cancel_calls >= 1
    assert dummy.finished.wait(timeout=1)
    assert dummy.HandleKeyboardInterrupt is False
    assert dummy.HandleUserInterrupt is False
