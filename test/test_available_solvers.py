"""Tests for the lazy ``available_solvers`` collection and ``license_status``."""

from __future__ import annotations

import subprocess
import sys

import pytest

import linopy
from linopy import solvers as solvers_mod
from linopy.solvers import (
    LicenseStatus,
    SolverName,
    _solver_class_for,
    available_solvers,
    check_solver_licenses,
    quadratic_solvers,
)


def test_import_does_not_load_license_managed_packages() -> None:
    """
    Importing linopy must not import packages whose ``__init__`` runs license logic.

    Verified in a subprocess so the test isn't fooled by modules other tests
    have already imported.
    """
    code = (
        "import sys, linopy;"
        "loaded = [m for m in ('mindoptpy', 'coptpy') if m in sys.modules];"
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


def test_is_available_matches_membership() -> None:
    for sn in SolverName:
        cls = _solver_class_for(sn.value)
        if cls is None:
            continue
        assert cls.is_available() == (sn.value in available_solvers)


def test_available_solvers_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = _solver_class_for("highs")
    assert cls is not None
    counter = {"n": 0}

    def probe() -> bool:
        counter["n"] += 1
        return True

    monkeypatch.setattr(cls, "is_available", classmethod(lambda c: probe()))
    fresh = solvers_mod._AvailableSolvers()
    list(fresh)
    list(fresh)
    assert counter["n"] == 1


def test_available_solvers_refresh_reprobes() -> None:
    fresh = solvers_mod._AvailableSolvers()
    first = list(fresh)
    fresh.refresh()
    second = list(fresh)
    assert first == second


def test_quadratic_solvers_is_subset_of_available() -> None:
    assert set(quadratic_solvers).issubset(set(available_solvers))


def test_license_status_on_uninstalled_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _solver_class_for("gurobi")
    assert cls is not None
    monkeypatch.setattr(cls, "is_available", classmethod(lambda c: False))
    probe_called = {"n": 0}

    def _probe() -> None:
        probe_called["n"] += 1

    monkeypatch.setattr(cls, "_license_probe", classmethod(lambda c: _probe()))
    status = cls.license_status()
    assert status.ok is False
    assert status.message == "package not installed"
    assert probe_called["n"] == 0


def test_license_status_wraps_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _solver_class_for("gurobi")
    assert cls is not None
    monkeypatch.setattr(cls, "is_available", classmethod(lambda c: True))

    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cls, "_license_probe", classmethod(lambda c: _boom()))
    status = cls.license_status()
    assert status.ok is False
    assert "boom" in (status.message or "")
    assert bool(status) is False


def test_license_status_ok_when_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _solver_class_for("highs")
    assert cls is not None
    monkeypatch.setattr(cls, "is_available", classmethod(lambda c: True))
    monkeypatch.setattr(cls, "_license_probe", classmethod(lambda c: None))
    status = cls.license_status()
    assert status.ok is True
    assert bool(status) is True
    assert isinstance(status, LicenseStatus)


def test_check_solver_licenses_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        check_solver_licenses("not-a-solver")


def test_check_solver_licenses_returns_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = _solver_class_for("highs")
    assert cls is not None
    monkeypatch.setattr(cls, "is_available", classmethod(lambda c: True))
    monkeypatch.setattr(cls, "_license_probe", classmethod(lambda c: None))
    result = check_solver_licenses("highs")
    assert set(result) == {"highs"}
    assert result["highs"].ok is True


def test_available_solvers_reexported_from_top_level() -> None:
    assert linopy.available_solvers is available_solvers
