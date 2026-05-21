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

import pandas as pd
import pytest

from linopy import Model
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
    """A MagicMock(spec=OetcHandler) with the methods Oetc.submit/status/collect call."""
    h = MagicMock(spec=OetcHandler)
    h.jwt = MagicMock(is_expired=False)  # a freshly authenticated handler
    h._upload_file_to_gcp = MagicMock(return_value="model.nc.gz")
    h._submit_job_to_compute_service = MagicMock(return_value="job-uuid")
    job_result = MagicMock()
    job_result.output_files = [{"name": "result.nc.gz"}]
    job_result.duration_in_seconds = 42
    h.wait_and_get_job_data = MagicMock(return_value=job_result)
    h._get_job = MagicMock(return_value=MagicMock(status="RUNNING"))
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
    def test_solve_runs_submit_and_collect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        oetc = Oetc(_settings_oetc())
        oetc._handler = _fake_oetc_handler()  # bypass auth

        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(m),
        )

        result = oetc.solve(m, "highs")

        assert isinstance(result, Model)
        oetc._handler._upload_file_to_gcp.assert_called_once()
        oetc._handler._submit_job_to_compute_service.assert_called_once()
        oetc._handler.wait_and_get_job_data.assert_called_once_with("job-uuid")
        oetc._handler._download_file_from_gcp.assert_called_once_with("result.nc.gz")

    def test_validates_unknown_solver_name(self) -> None:
        m = _build_model()
        oetc = Oetc(_settings_oetc())
        oetc._handler = _fake_oetc_handler()
        with pytest.raises(ValueError, match="Unknown solver"):
            oetc.solve(m, "not-a-solver")

    def test_submit_collect_separable_by_uuid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The submit/collect seam can be driven manually for async work."""
        m = _build_model()
        oetc = Oetc(_settings_oetc())
        oetc._handler = _fake_oetc_handler()
        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(m),
        )

        job_uuid = oetc.submit(m, "highs")
        assert job_uuid == "job-uuid"
        assert oetc._handler._upload_file_to_gcp.call_count == 1
        assert oetc._handler._submit_job_to_compute_service.call_count == 1

        result = oetc.collect(job_uuid)
        assert isinstance(result, Model)
        oetc._handler.wait_and_get_job_data.assert_called_once_with("job-uuid")

    def test_status_returns_job_state(self) -> None:
        oetc = Oetc(_settings_oetc())
        oetc._handler = _fake_oetc_handler()
        assert oetc.status("job-uuid") == "RUNNING"
        oetc._handler._get_job.assert_called_once_with("job-uuid")

    def test_one_connection_drives_multiple_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single Oetc connection submits and collects many models."""
        models = [_build_model() for _ in range(3)]
        oetc = Oetc(_settings_oetc())
        oetc._handler = _fake_oetc_handler()
        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(models[0]),
        )

        uuids = [oetc.submit(m, "highs") for m in models]
        assert len(uuids) == 3
        solved = [oetc.collect(u) for u in uuids]
        assert all(isinstance(s, Model) for s in solved)
        assert oetc._handler._submit_job_to_compute_service.call_count == 3
        assert oetc._handler.wait_and_get_job_data.call_count == 3

    def test_collect_by_uuid_from_a_fresh_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job uuid can be collected by an Oetc that never submitted it."""
        m = _build_model()
        submitter = Oetc(_settings_oetc())
        submitter._handler = _fake_oetc_handler()
        job_uuid = submitter.submit(m, "highs")

        # Simulate a separate process: a brand-new Oetc, given only the uuid.
        collector = Oetc(_settings_oetc())
        collector._handler = _fake_oetc_handler()
        monkeypatch.setattr(
            "linopy.remote.oetc.linopy.read_netcdf",
            lambda path: _solved_model_like(m),
        )
        result = collector.collect(job_uuid)
        assert isinstance(result, Model)

    def test_expired_token_triggers_reauth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale auth token makes the next call rebuild the handler."""
        oetc = Oetc(_settings_oetc())
        stale = _fake_oetc_handler()
        stale.jwt = MagicMock(is_expired=True)
        oetc._handler = stale

        rebuilt = _fake_oetc_handler()
        monkeypatch.setattr(
            "linopy.remote.oetc.OetcHandler",
            lambda settings, _internal=False: rebuilt,
        )

        assert oetc.status("job-uuid") == "RUNNING"
        assert oetc._handler is rebuilt  # expired token -> reconnected


# ---------------------------------------------------------------------------
# SSH class


class TestSSHClass:
    def test_solve_runs_setup_commands_then_delegates(self) -> None:
        m = _build_model()
        ssh = SSH(
            SshSettings(
                hostname="example.org",
                setup_commands=["conda activate linopy-env", "export FOO=bar"],
            )
        )
        fake_handler = MagicMock(spec=RemoteHandler)
        fake_handler.execute = MagicMock()
        fake_handler.solve_on_remote = MagicMock(return_value=_solved_model_like(m))
        ssh._handler = fake_handler

        result = ssh.solve(m, "highs")

        assert isinstance(result, Model)
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
            SshSettings(
                hostname="example.org",
                setup_commands=["conda activate linopy-env"],
            )
        )

        built: list[Any] = []

        class FakeRemoteHandler:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.execute = MagicMock()
                self.solve_on_remote = MagicMock(return_value=_solved_model_like(m))
                built.append(self)

        monkeypatch.setattr("linopy.remote.ssh.RemoteHandler", FakeRemoteHandler)
        ssh.solve(m, "highs")

        assert len(built) == 1
        built[0].execute.assert_called_once_with("conda activate linopy-env")
        assert built[0].kwargs.get("_internal") is True

    def test_validates_unknown_solver_name(self) -> None:
        m = _build_model()
        ssh = SSH(_settings_ssh())
        ssh._handler = MagicMock(spec=RemoteHandler)
        with pytest.raises(ValueError, match="Unknown solver"):
            ssh.solve(m, "not-a-solver")


# ---------------------------------------------------------------------------
# Model.solve(remote=<Settings>) end-to-end


class TestModelSolveRemote:
    def test_oetc_settings_dispatches_to_oetc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        captured: dict[str, Any] = {}

        def fake_solve(
            self: Oetc, model: Model, solver_name: str, **options: Any
        ) -> Model:
            captured["solver_name"] = solver_name
            captured["options"] = options
            captured["instance"] = self
            return _solved_model_like(model)

        monkeypatch.setattr(Oetc, "solve", fake_solve)

        m.solve("gurobi", remote=_settings_oetc(), Method=2)

        assert captured["solver_name"] == "gurobi"
        assert captured["options"] == {"Method": 2}
        assert m.remote is captured["instance"]
        assert m.solver is None  # remote-solve clears any prior local solver

    def test_oetc_settings_solver_used_when_no_solver_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The deprecated `OetcSettings.solver` is the fallback when
        `Model.solve(remote=...)` is called without a `solver_name`.
        """
        m = _build_model()
        captured: dict[str, Any] = {}

        def fake_solve(
            self: Oetc, model: Model, solver_name: str, **options: Any
        ) -> Model:
            captured["solver_name"] = solver_name
            captured["options"] = options
            return _solved_model_like(model)

        monkeypatch.setattr(Oetc, "solve", fake_solve)

        with pytest.warns(DeprecationWarning, match=r"OetcSettings\.solver"):
            settings = OetcSettings(
                email="a@b.com",
                password="pw",
                name="test-job",
                authentication_server_url="https://auth",
                orchestrator_server_url="https://orch",
                solver="cplex",
                solver_options={"TimeLimit": 10},
            )
        m.solve(remote=settings)

        assert captured["solver_name"] == "cplex"
        assert captured["options"] == {"TimeLimit": 10}

    def test_ssh_settings_dispatches_to_ssh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        m = _build_model()
        captured: dict[str, Any] = {}

        def fake_solve(
            self: SSH, model: Model, solver_name: str, **options: Any
        ) -> Model:
            captured["solver_name"] = solver_name
            captured["options"] = options
            captured["instance"] = self
            return _solved_model_like(model)

        monkeypatch.setattr(SSH, "solve", fake_solve)

        m.solve("highs", remote=_settings_ssh(), presolve="on")

        assert captured["solver_name"] == "highs"
        assert captured["options"] == {"presolve": "on"}
        assert m.remote is captured["instance"]

    @pytest.mark.parametrize(
        ("remote_cls", "settings_factory", "solver"),
        [(Oetc, _settings_oetc, "gurobi"), (SSH, _settings_ssh, "highs")],
    )
    def test_remote_solve_writes_solution_onto_caller_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        remote_cls: type,
        settings_factory: Any,
        solver: str,
    ) -> None:
        """
        `Model.solve(remote=...)` folds the solved model into the caller's
        own model in place and returns the (status, termination_condition)
        tuple — it never hands back the round-tripped model object.
        """
        m = _build_model()

        def fake_solve(
            self: Any, model: Model, solver_name: str, **options: Any
        ) -> Model:
            return _solved_model_like(model)

        monkeypatch.setattr(remote_cls, "solve", fake_solve)

        result = m.solve(solver, remote=settings_factory())

        assert result == ("ok", "optimal")
        assert m.status == "ok"
        assert m.termination_condition == "optimal"
        assert m.objective.value == 0.0
        assert float(m.variables["x"].solution.sum()) == 0.0


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
