"""Tests for int32 default label dtype."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from linopy import Model


def test_default_label_dtype_is_int32() -> None:
    assert Model().label_dtype == np.int32


def test_variable_labels_are_int32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    assert x.labels.dtype == np.int32


def test_constraint_labels_are_int32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(x >= 1, name="c")
    assert m.constraints["c"].labels.dtype == np.int32


def test_expression_vars_are_int32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    expr = 2 * x + 1
    assert expr.vars.dtype == np.int32


@pytest.mark.skipif(
    not pytest.importorskip("highspy", reason="highspy not installed"),
    reason="highspy not installed",
)
def test_solve_with_int32_labels() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y <= 15, name="c1")
    m.add_objective(x + 2 * y, sense="max")
    m.solve("highs")
    assert m.objective.value == pytest.approx(25.0)


def test_auto_widen_variables() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    assert x.labels.dtype == np.int64
    assert m.label_dtype == np.int64
    # the widening is per-model: other models are untouched
    other = Model()
    y = other.add_variables(lower=0, upper=1, coords=[range(5)], name="y")
    assert y.labels.dtype == np.int32


def test_auto_widen_constraints() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m._cCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        m.add_constraints(x >= 0, name="c")
    assert m.constraints["c"].labels.dtype == np.int64
    assert m.label_dtype == np.int64


def test_widen_applies_to_expressions() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    assert (2 * x + 1).vars.dtype == np.int64


def test_label_dtype_init_arg() -> None:
    m = Model(label_dtype=np.int64)
    assert m.label_dtype == np.int64
    x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    assert x.labels.dtype == np.int64
    assert (2 * x + 1).vars.dtype == np.int64


def test_label_dtype_init_arg_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="label_dtype must be"):
        Model(label_dtype=np.float64)  # type: ignore[arg-type]


def test_auto_widen_survives_netcdf(tmp_path: Path) -> None:
    from linopy import read_netcdf

    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m.add_constraints(x >= 0, name="c")
    path = tmp_path / "widened.nc"
    m.to_netcdf(path)

    loaded = read_netcdf(path)

    assert loaded.label_dtype == np.int64
    assert loaded.variables["x"].labels.dtype == np.int64
    assert (2 * loaded.variables["x"]).vars.dtype == np.int64


def test_auto_widen_survives_pickle() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m.add_constraints(x >= 0, name="c")

    loaded = pickle.loads(pickle.dumps(m))

    assert loaded.label_dtype == np.int64
    assert (2 * loaded.variables["x"]).vars.dtype == np.int64
