"""
Tests for the standalone remote classes (``Oetc`` / ``SSH``) and the
``Model.solve(remote=<Settings>)`` entry point.

The deprecated ``OetcHandler`` / ``RemoteHandler`` are covered by
``test_oetc.py`` and ``test_ssh.py`` separately; this file focuses on
the *new* public surface and its deprecation warnings.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from linopy import Model
from linopy.constants import (
    Result,
    Solution,
    SolverReport,
    Status,
)
from linopy.remote import (
    Oetc,
    OetcCredentials,
    OetcHandler,
    OetcSettings,
    RemoteHandler,
    SshSettings,
)

pytest.importorskip("paramiko")
from linopy.remote.ssh import SSH  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers


def _build_model() -> Model:
    m = Model()
    idx = pd.Index([0, 1, 2], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_constraints(x >= 0, name="c")
    m.add_objective(1.0 * x.sum())
    return m


def _settings_oetc() -> OetcSettings:
    return OetcSettings(
        email="a@b.com",
        password="pw",
        name="test-job",
        authentication_server_url="https://auth",
        orchestrator_server_url="https://orch",
    )


def _settings_ssh() -> SshSettings:
    return SshSettings(hostname="example.org", username="me")


def _fake_oetc_handler() -> MagicMock:
    """A MagicMock(spec=OetcHandler) with the methods Oetc.upload/submit/collect call."""
    h = MagicMock(spec=OetcHandler)
    h._upload_file_to_gcp = MagicMock(return_value="model.nc.gz")
    h._submit_job_to_compute_service = MagicMock(return_value="job-uuid")
    job_result = MagicMock()
    job_result.output_files = [{"name": "result.nc.gz"}]
    job_result.duration_in_seconds = 42
    h.wait_and_get_job_data = MagicMock(return_value=job_result)
    h._download_file_from_gcp = MagicMock(return_value="/tmp/fake-result.nc")
    return h


def _solved_model_like(m: Model) -> Model:
    """Build a Model with the same labels as ``m`` plus dummy solution data."""
    solved = Model()
    for name, var in m.variables.items():
        solved_var = solved.add_variables(
            lower=var.lower, upper=var.upper, coords=var.coords, name=name
        )
        solved_var.solution = solved_var.lower * 0  # zeros, real DataArray
    for name, con in m.constraints.items():
        solved.add_constraints(con.lhs >= con.rhs, name=name)
    solved.add_objective(m.objective.expression)
    solved.objective._value = 0.0
    solved.termination_condition = "optimal"
    solved.status = "ok"
    return solved


# ---------------------------------------------------------------------------
# Oetc class


class TestOetcClass:
    def test_solve_runs_upload_submit_collect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        oetc = Oetc(settings=_settings_oetc(), solver_name="highs")
        oetc._handler = _fake_oetc_handler()  # bypass auth

        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(m),
        )

        result = oetc.solve(m)

        assert isinstance(result, Result)
        assert result.solver_name == "highs"
        oetc._handler._upload_file_to_gcp.assert_called_once()
        oetc._handler._submit_job_to_compute_service.assert_called_once()
        oetc._handler.wait_and_get_job_data.assert_called_once_with("job-uuid")
        oetc._handler._download_file_from_gcp.assert_called_once_with("result.nc.gz")

    def test_validates_unknown_solver_name(self) -> None:
        m = _build_model()
        oetc = Oetc(settings=_settings_oetc(), solver_name="not-a-solver")
        oetc._handler = _fake_oetc_handler()
        with pytest.raises(ValueError, match="Unknown solver"):
            oetc.solve(m)

    def test_upload_submit_collect_separable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three-step lifecycle can be driven manually, e.g. for async work."""
        m = _build_model()
        oetc = Oetc(settings=_settings_oetc(), solver_name="highs")
        oetc._handler = _fake_oetc_handler()
        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(m),
        )

        oetc.upload(m)
        assert oetc._input_file_name == "model.nc.gz"
        assert oetc._handler._upload_file_to_gcp.call_count == 1

        job_id = oetc.submit()
        assert job_id == "job-uuid"
        assert oetc._handler._submit_job_to_compute_service.call_count == 1

        result = oetc.collect(m)
        assert isinstance(result, Result)
        assert oetc._handler.wait_and_get_job_data.call_count == 1

    def test_submit_before_upload_raises(self) -> None:
        oetc = Oetc(settings=_settings_oetc(), solver_name="highs")
        oetc._handler = _fake_oetc_handler()
        with pytest.raises(RuntimeError, match="upload"):
            oetc.submit()

    def test_collect_before_submit_raises(self) -> None:
        m = _build_model()
        oetc = Oetc(settings=_settings_oetc(), solver_name="highs")
        oetc._handler = _fake_oetc_handler()
        with pytest.raises(RuntimeError, match="upload.*submit"):
            oetc.collect(m)


# ---------------------------------------------------------------------------
# SSH class


class TestSSHClass:
    def test_solve_runs_setup_commands_then_delegates(self) -> None:
        m = _build_model()
        ssh = SSH(
            settings=SshSettings(
                hostname="example.org",
                setup_commands=["conda activate linopy-env", "export FOO=bar"],
            ),
            solver_name="highs",
        )
        fake_handler = MagicMock(spec=RemoteHandler)
        fake_handler.execute = MagicMock()
        fake_handler.solve_on_remote = MagicMock(return_value=_solved_model_like(m))
        ssh._handler = fake_handler

        result = ssh.solve(m)

        assert isinstance(result, Result)
        # solve_on_remote is the public surface from the deprecated handler
        fake_handler.solve_on_remote.assert_called_once()
        # setup_commands run only on first handler construction; here _handler
        # was injected, so they shouldn't run automatically:
        fake_handler.execute.assert_not_called()

    def test_setup_commands_run_when_handler_is_built_internally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First .solve() with a fresh SSH builds a RemoteHandler and runs setup."""
        m = _build_model()
        ssh = SSH(
            settings=SshSettings(
                hostname="example.org",
                setup_commands=["conda activate linopy-env"],
            ),
            solver_name="highs",
        )

        built: list[Any] = []

        class FakeRemoteHandler:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.execute = MagicMock()
                self.solve_on_remote = MagicMock(return_value=_solved_model_like(m))
                built.append(self)

        monkeypatch.setattr("linopy.remote.ssh.RemoteHandler", FakeRemoteHandler)
        ssh.solve(m)

        assert len(built) == 1
        built[0].execute.assert_called_once_with("conda activate linopy-env")
        assert built[0].kwargs.get("_internal") is True

    def test_validates_unknown_solver_name(self) -> None:
        m = _build_model()
        ssh = SSH(settings=_settings_ssh(), solver_name="not-a-solver")
        ssh._handler = MagicMock(spec=RemoteHandler)
        with pytest.raises(ValueError, match="Unknown solver"):
            ssh.solve(m)


# ---------------------------------------------------------------------------
# Model.solve(remote=<Settings>) end-to-end


class TestModelSolveRemote:
    def test_oetc_settings_dispatches_to_oetc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        captured: dict[str, Any] = {}

        def fake_solve(self: Oetc, model: Model) -> Result:
            captured["solver_name"] = self.solver_name
            captured["options"] = self.options
            captured["instance"] = self
            return Result(
                status=Status.from_termination_condition("optimal"),
                solution=Solution(
                    primal=np.zeros(model._xCounter, dtype=float),
                    dual=np.full(model._cCounter, np.nan, dtype=float),
                    objective=0.0,
                ),
                solver_name=self.solver_name,
                report=SolverReport(runtime=1.0),
            )

        monkeypatch.setattr(Oetc, "solve", fake_solve)

        m.solve("gurobi", remote=_settings_oetc(), Method=2)

        assert captured["solver_name"] == "gurobi"
        assert captured["options"] == {"Method": 2}
        assert m.remote is captured["instance"]
        assert m.solver is None  # remote-solve clears any prior local solver

    def test_ssh_settings_dispatches_to_ssh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        captured: dict[str, Any] = {}

        def fake_solve(self: SSH, model: Model) -> Result:
            captured["solver_name"] = self.solver_name
            captured["options"] = self.options
            captured["instance"] = self
            return Result(
                status=Status.from_termination_condition("optimal"),
                solution=Solution(
                    primal=np.zeros(model._xCounter, dtype=float),
                    dual=np.full(model._cCounter, np.nan, dtype=float),
                    objective=0.0,
                ),
                solver_name=self.solver_name,
            )

        monkeypatch.setattr(SSH, "solve", fake_solve)

        m.solve("highs", remote=_settings_ssh(), presolve="on")

        assert captured["solver_name"] == "highs"
        assert captured["options"] == {"presolve": "on"}
        assert m.remote is captured["instance"]


# ---------------------------------------------------------------------------
# Deprecation warnings


class TestDeprecations:
    def test_oetc_credentials_construction_warns(self) -> None:
        with pytest.warns(DeprecationWarning, match="OetcCredentials"):
            OetcCredentials(email="a@b.com", password="pw")

    def test_oetc_settings_credentials_kwarg_carries_values_through(self) -> None:
        # Constructing OetcCredentials warns (its own __post_init__).
        with pytest.warns(DeprecationWarning, match="OetcCredentials"):
            creds = OetcCredentials(email="a@b.com", password="pw")

        s = OetcSettings(
            credentials=creds,
            name="n",
            authentication_server_url="https://a",
            orchestrator_server_url="https://o",
        )
        assert s.email == "a@b.com"
        assert s.password == "pw"
        # `credentials` is consumed and cleared.
        assert s.credentials is None

    def test_oetc_settings_requires_email_and_password(self) -> None:
        with pytest.raises(ValueError, match="email.*password"):
            OetcSettings(
                name="n",
                authentication_server_url="https://a",
                orchestrator_server_url="https://o",
            )

    def test_oetc_handler_construction_warns(self) -> None:
        with (
            patch.object(OetcHandler, "_OetcHandler__sign_in"),
            patch.object(OetcHandler, "_OetcHandler__get_cloud_provider_credentials"),
        ):
            with pytest.warns(DeprecationWarning, match="OetcHandler"):
                OetcHandler(_settings_oetc())

    def test_oetc_handler_internal_construction_silent(self) -> None:
        with (
            patch.object(OetcHandler, "_OetcHandler__sign_in"),
            patch.object(OetcHandler, "_OetcHandler__get_cloud_provider_credentials"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                OetcHandler(_settings_oetc(), _internal=True)

    def test_remote_handler_construction_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = MagicMock()
        fake_client.invoke_shell.return_value.makefile.return_value = MagicMock()
        fake_client.open_sftp.return_value = MagicMock()

        with pytest.warns(DeprecationWarning, match="RemoteHandler"):
            RemoteHandler(hostname="x", client=fake_client)

    def test_remote_handler_internal_construction_silent(self) -> None:
        fake_client = MagicMock()
        fake_client.invoke_shell.return_value.makefile.return_value = MagicMock()
        fake_client.open_sftp.return_value = MagicMock()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RemoteHandler(hostname="x", client=fake_client, _internal=True)

    def test_model_solve_remote_handler_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        handler = MagicMock(spec=OetcHandler)
        handler.settings = _settings_oetc()
        handler.solve_on_oetc = MagicMock(return_value=_solved_model_like(m))
        with pytest.warns(DeprecationWarning, match="OetcHandler.*remote="):
            m.solve(solver_name="highs", remote=handler)
