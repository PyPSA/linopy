from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from linopy.remote.oetc import (
    ComputeProvider,
    OetcCredentials,
    OetcHandler,
    OetcSettings,
)

REQUIRED_ENV = {
    "OETC_EMAIL": "test@example.com",
    "OETC_PASSWORD": "secret",
    "OETC_NAME": "test-job",
    "OETC_AUTH_URL": "https://auth.example.com",
    "OETC_ORCHESTRATOR_URL": "https://orch.example.com",
}


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)


def _clear_oetc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "OETC_EMAIL",
        "OETC_PASSWORD",
        "OETC_NAME",
        "OETC_AUTH_URL",
        "OETC_ORCHESTRATOR_URL",
        "OETC_CPU_CORES",
        "OETC_DISK_SPACE_GB",
        "OETC_DELETE_WORKER_ON_ERROR",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_from_env_all_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_CPU_CORES", "8")
    monkeypatch.setenv("OETC_DISK_SPACE_GB", "20")
    monkeypatch.setenv("OETC_DELETE_WORKER_ON_ERROR", "true")

    s = OetcSettings.from_env()
    assert s.credentials.email == "test@example.com"
    assert s.credentials.password == "secret"
    assert s.name == "test-job"
    assert s.cpu_cores == 8
    assert s.disk_space_gb == 20
    assert s.compute_provider == ComputeProvider.GCP
    assert s.delete_worker_on_error is True


def test_from_env_kwargs_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)

    s = OetcSettings.from_env(email="override@example.com")
    assert s.credentials.email == "override@example.com"


def test_from_env_missing_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    with pytest.raises(
        ValueError,
        match="OETC_EMAIL.*OETC_PASSWORD.*OETC_NAME.*OETC_AUTH_URL.*OETC_ORCHESTRATOR_URL",
    ):
        OetcSettings.from_env()


def test_from_env_empty_string_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    monkeypatch.setenv("OETC_EMAIL", "")
    monkeypatch.setenv("OETC_PASSWORD", "   ")
    monkeypatch.setenv("OETC_NAME", "valid")
    monkeypatch.setenv("OETC_AUTH_URL", "https://auth.example.com")
    monkeypatch.setenv("OETC_ORCHESTRATOR_URL", "https://orch.example.com")

    with pytest.raises(ValueError, match="OETC_EMAIL.*OETC_PASSWORD"):
        OetcSettings.from_env()


def test_from_env_partial_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    monkeypatch.setenv("OETC_NAME", "env-name")
    monkeypatch.setenv("OETC_AUTH_URL", "https://auth.example.com")
    monkeypatch.setenv("OETC_ORCHESTRATOR_URL", "https://orch.example.com")

    s = OetcSettings.from_env(email="a@b.com", password="pw")
    assert s.credentials.email == "a@b.com"
    assert s.name == "env-name"


def test_from_env_defaults_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)

    s = OetcSettings.from_env()
    assert s.solver == "highs"
    assert s.solver_options == {}
    assert s.cpu_cores == 2
    assert s.disk_space_gb == 10
    assert s.compute_provider == ComputeProvider.GCP
    assert s.delete_worker_on_error is False


def test_from_env_cpu_cores_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_CPU_CORES", "4")

    assert OetcSettings.from_env().cpu_cores == 4


def test_from_env_cpu_cores_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_CPU_CORES", "abc")

    with pytest.raises(ValueError, match="OETC_CPU_CORES"):
        OetcSettings.from_env()


@pytest.mark.parametrize("val", ["true", "1", "yes"])
def test_from_env_bool_true_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_DELETE_WORKER_ON_ERROR", val)

    assert OetcSettings.from_env().delete_worker_on_error is True


@pytest.mark.parametrize("val", ["false", "0", "no"])
def test_from_env_bool_false_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_DELETE_WORKER_ON_ERROR", val)

    assert OetcSettings.from_env().delete_worker_on_error is False


def test_from_env_bool_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_DELETE_WORKER_ON_ERROR", "maybe")

    with pytest.raises(ValueError, match="OETC_DELETE_WORKER_ON_ERROR"):
        OetcSettings.from_env()


def _make_handler(settings: OetcSettings) -> OetcHandler:
    with (
        patch.object(OetcHandler, "_OetcHandler__sign_in", return_value=MagicMock()),
        patch.object(
            OetcHandler,
            "_OetcHandler__get_cloud_provider_credentials",
            return_value=MagicMock(),
        ),
    ):
        return OetcHandler(settings)


def _default_settings(**overrides: Any) -> OetcSettings:
    defaults: dict[str, Any] = dict(
        credentials=OetcCredentials(email="a@b.com", password="pw"),
        name="test",
        authentication_server_url="https://auth",
        orchestrator_server_url="https://orch",
        solver="highs",
        solver_options={"TimeLimit": 100},
    )
    defaults.update(overrides)
    return OetcSettings(**defaults)


def test_solve_on_oetc_mutation_safety() -> None:
    settings = _default_settings()
    handler = _make_handler(settings)
    original_opts = dict(settings.solver_options)

    mock_model = MagicMock()
    mock_solved = MagicMock()
    mock_solved.objective.value = 42.0
    mock_solved.status = "ok"

    with (
        patch.object(handler, "_upload_file_to_gcp", return_value="file.nc.gz"),
        patch.object(handler, "_submit_job_to_compute_service", return_value="uuid"),
        patch.object(handler, "wait_and_get_job_data") as mock_wait,
        patch.object(handler, "_download_file_from_gcp", return_value="/tmp/sol.nc"),
        patch("linopy.read_netcdf", return_value=mock_solved),
        patch("os.remove"),
    ):
        mock_wait.return_value = MagicMock(output_files=["out.nc.gz"])

        handler.solve_on_oetc(mock_model, Extra=999)
        handler.solve_on_oetc(mock_model, Other=1)

    assert settings.solver_options == original_opts


def test_solve_on_oetc_solver_name_override() -> None:
    settings = _default_settings()
    handler = _make_handler(settings)

    mock_model = MagicMock()
    mock_solved = MagicMock()
    mock_solved.objective.value = 1.0
    mock_solved.status = "ok"

    with (
        patch.object(handler, "_upload_file_to_gcp", return_value="file.nc.gz"),
        patch.object(
            handler, "_submit_job_to_compute_service", return_value="uuid"
        ) as mock_submit,
        patch.object(handler, "wait_and_get_job_data") as mock_wait,
        patch.object(handler, "_download_file_from_gcp", return_value="/tmp/sol.nc"),
        patch("linopy.read_netcdf", return_value=mock_solved),
        patch("os.remove"),
    ):
        mock_wait.return_value = MagicMock(output_files=["out.nc.gz"])

        handler.solve_on_oetc(mock_model, solver_name="gurobi")

    mock_submit.assert_called_once()
    assert mock_submit.call_args[0][1] == "gurobi"


def test_solve_on_oetc_solver_options_merge_precedence() -> None:
    settings = _default_settings(solver_options={"TimeLimit": 100})
    handler = _make_handler(settings)

    mock_model = MagicMock()
    mock_solved = MagicMock()
    mock_solved.objective.value = 1.0
    mock_solved.status = "ok"

    with (
        patch.object(handler, "_upload_file_to_gcp", return_value="file.nc.gz"),
        patch.object(
            handler, "_submit_job_to_compute_service", return_value="uuid"
        ) as mock_submit,
        patch.object(handler, "wait_and_get_job_data") as mock_wait,
        patch.object(handler, "_download_file_from_gcp", return_value="/tmp/sol.nc"),
        patch("linopy.read_netcdf", return_value=mock_solved),
        patch("os.remove"),
    ):
        mock_wait.return_value = MagicMock(output_files=["out.nc.gz"])

        handler.solve_on_oetc(mock_model, TimeLimit=200)

    mock_submit.assert_called_once()
    assert mock_submit.call_args[0][2] == {"TimeLimit": 200}


def test_solve_on_oetc_solver_name_default_fallback() -> None:
    settings = _default_settings(solver="cplex")
    handler = _make_handler(settings)

    mock_model = MagicMock()
    mock_solved = MagicMock()
    mock_solved.objective.value = 1.0
    mock_solved.status = "ok"

    with (
        patch.object(handler, "_upload_file_to_gcp", return_value="file.nc.gz"),
        patch.object(
            handler, "_submit_job_to_compute_service", return_value="uuid"
        ) as mock_submit,
        patch.object(handler, "wait_and_get_job_data") as mock_wait,
        patch.object(handler, "_download_file_from_gcp", return_value="/tmp/sol.nc"),
        patch("linopy.read_netcdf", return_value=mock_solved),
        patch("os.remove"),
    ):
        mock_wait.return_value = MagicMock(output_files=["out.nc.gz"])

        handler.solve_on_oetc(mock_model)

    mock_submit.assert_called_once()
    assert mock_submit.call_args[0][1] == "cplex"


def test_from_env_disk_space_gb_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oetc_env(monkeypatch)
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OETC_DISK_SPACE_GB", "abc")

    with pytest.raises(ValueError, match="OETC_DISK_SPACE_GB"):
        OetcSettings.from_env()


def test_model_solve_forwards_to_oetc() -> None:
    from linopy import Model

    m = Model()
    m.add_variables(lower=0, name="x")

    handler = MagicMock(spec=OetcHandler)
    mock_solved = MagicMock()
    mock_solved.status = "ok"
    mock_solved.termination_condition = "optimal"
    mock_solved.objective.value = 10.0
    mock_solved.variables.items.return_value = [(k, v) for k, v in m.variables.items()]
    mock_solved.constraints.items.return_value = []
    for k in m.variables:
        mock_solved.variables[k].solution = 0.0
    handler.solve_on_oetc.return_value = mock_solved

    m.solve(solver_name="gurobi", remote=handler, TimeLimit=100)

    handler.solve_on_oetc.assert_called_once_with(
        m, solver_name="gurobi", TimeLimit=100
    )
