"""Tests for ``linopy.remote.ssh.RemoteHandler.solve_on_remote``."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("paramiko")

from linopy import Model  # noqa: E402
from linopy.remote.ssh import RemoteHandler  # noqa: E402


class _FakeSFTPClient:
    """In-memory SFTP stand-in: ``put`` / ``get`` round-trip file bytes."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def open(self, path: str, mode: str) -> Any:
        store = self.store

        @contextmanager
        def _writer() -> Iterator[Any]:
            class _Writer:
                def write(self_inner, data: str | bytes) -> None:
                    store[path] = data.encode() if isinstance(data, str) else data

            yield _Writer()

        return _writer()

    def put(self, local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as fh:
            self.store[remote_path] = fh.read()

    def get(self, remote_path: str, local_path: str) -> None:
        with open(local_path, "wb") as fh:
            fh.write(self.store[remote_path])

    def remove(self, path: str) -> None:
        self.store.pop(path, None)


def _make_sos_model() -> Model:
    m = Model()
    idx = pd.Index([0, 1, 2], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")
    m.add_objective(x * np.array([1.0, 2.0, 3.0]), sense="max")
    return m


@pytest.fixture
def handler() -> RemoteHandler:
    """``RemoteHandler`` wired to an in-memory SFTP and a no-op shell."""
    client = MagicMock()
    client.invoke_shell.return_value.makefile.return_value = MagicMock()
    sftp = _FakeSFTPClient()
    client.open_sftp.return_value = sftp

    h = RemoteHandler(hostname="fake", client=client)
    # The unsolved model gets put() into sftp.store under model_unsolved_file;
    # serve it back as the "solved" model so read_netcdf has something valid.
    h.sftp_client = sftp  # type: ignore[assignment]
    h.execute = MagicMock()  # type: ignore[method-assign]

    original_put = sftp.put

    def put_and_mirror(local_path: str, remote_path: str) -> None:
        original_put(local_path, remote_path)
        if remote_path == h.model_unsolved_file:
            sftp.store[h.model_solved_file] = sftp.store[remote_path]

    sftp.put = put_and_mirror  # type: ignore[method-assign]
    return h


class TestSolveOnRemoteSosBracket:
    """``solve_on_remote`` must bracket SOS reformulation around transfer."""

    def test_reformulates_before_transfer_and_restores_after(
        self, handler: RemoteHandler
    ) -> None:
        m = _make_sos_model()

        observed: dict[str, bool] = {}
        real_write = handler.write_model_on_remote

        def spy_write(model: Model) -> None:
            observed["state_active"] = model._sos_reformulation_state is not None
            observed["has_aux_var"] = "_sos_reform_x_y" in model.variables
            real_write(model)

        handler.write_model_on_remote = spy_write  # type: ignore[method-assign]

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            handler.solve_on_remote(m, reformulate_sos=True, solver_name="highs")

        assert observed["state_active"] is True
        assert observed["has_aux_var"] is True
        assert not any("active SOS reformulation" in str(w.message) for w in captured)
        assert m._sos_reformulation_state is None
        assert "_sos_reform_x_y" not in m.variables
        assert list(m.variables.sos) == ["x"]

    def test_skips_bracket_when_reformulate_sos_false(
        self, handler: RemoteHandler
    ) -> None:
        m = _make_sos_model()

        observed: dict[str, bool] = {}
        real_write = handler.write_model_on_remote

        def spy_write(model: Model) -> None:
            observed["state_active"] = model._sos_reformulation_state is not None
            real_write(model)

        handler.write_model_on_remote = spy_write  # type: ignore[method-assign]

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            handler.solve_on_remote(m, reformulate_sos=False)

        assert observed["state_active"] is False
        assert not any("active SOS reformulation" in str(w.message) for w in captured)
        assert m._sos_reformulation_state is None

    def test_auto_without_solver_name_raises_on_sos_model(
        self, handler: RemoteHandler
    ) -> None:
        m = _make_sos_model()
        with pytest.raises(ValueError, match="requires an explicit `solver_name`"):
            handler.solve_on_remote(m, reformulate_sos="auto")

    def test_no_sos_model_passes_through_unchanged(
        self, handler: RemoteHandler, tmp_path: Path
    ) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=1, name="x")
        m.add_objective(1.0 * x, sense="max")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            handler.solve_on_remote(m, reformulate_sos="auto")

        assert m._sos_reformulation_state is None
        assert not any("active SOS reformulation" in str(w.message) for w in captured)
