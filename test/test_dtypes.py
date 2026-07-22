"""Tests for int32 default label dtype."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from linopy import Model


def test_default_label_dtype_is_int32() -> None:
    assert Model().dtypes["labels"] == np.int32


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
    assert m.dtypes["labels"] == np.int64
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
    assert m.dtypes["labels"] == np.int64


def test_widen_applies_to_expressions() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    assert (2 * x + 1).vars.dtype == np.int64


def test_label_dtype_init_arg() -> None:
    m = Model(dtypes={"labels": np.int64})
    assert m.dtypes["labels"] == np.int64
    x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    assert x.labels.dtype == np.int64
    assert (2 * x + 1).vars.dtype == np.int64


def test_label_dtype_init_arg_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="dtypes\\['labels'\\] must be"):
        Model(dtypes={"labels": np.float64})


def test_dtypes_init_arg_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="only supports the keys"):
        Model(dtypes={"coeffs": np.int64})  # type: ignore[dict-item]


def test_dtypes_is_read_only_mapping() -> None:
    dtypes = Model().dtypes
    assert set(dtypes) == {"labels", "values"}
    with pytest.raises(TypeError):
        dtypes["labels"] = np.int64  # type: ignore[index]


def test_default_values_dtype_is_float32() -> None:
    assert Model().dtypes["values"] == np.float32


def test_default_bounds_are_float32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    assert x.lower.dtype == np.float32
    assert x.upper.dtype == np.float32


def test_default_coeffs_and_const_are_float32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    expr = 2 * x + 1
    assert expr.coeffs.dtype == np.float32
    assert expr.const.dtype == np.float32


def test_default_rhs_and_constraint_coeffs_are_float32() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(2 * x + 1 <= 5, name="c")
    con = m.constraints["c"]
    assert con.rhs.dtype == np.float32
    assert con.coeffs.dtype == np.float32


def test_values_dtype_init_arg_float64() -> None:
    m = Model(dtypes={"values": np.float64})
    assert m.dtypes["values"] == np.float64
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    assert x.lower.dtype == np.float64
    assert x.upper.dtype == np.float64
    expr = 2 * x + 1
    assert expr.coeffs.dtype == np.float64
    assert expr.const.dtype == np.float64
    m.add_constraints(2 * x + 1 <= 5, name="c")
    assert m.constraints["c"].rhs.dtype == np.float64


def test_values_dtype_init_arg_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="dtypes\\['values'\\] must be"):
        Model(dtypes={"values": np.float16})


@pytest.mark.skipif(
    not pytest.importorskip("highspy", reason="highspy not installed"),
    reason="highspy not installed",
)
def test_solve_with_float32_values() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y <= 15, name="c1")
    m.add_objective(x + 2 * y, sense="max")
    assert m.constraints["c1"].coeffs.dtype == np.float32
    m.solve("highs")
    assert m.objective.value == pytest.approx(25.0, abs=1e-4)


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

    assert loaded.dtypes["labels"] == np.int64
    assert loaded.variables["x"].labels.dtype == np.int64
    assert (2 * loaded.variables["x"]).vars.dtype == np.int64


def test_auto_widen_survives_pickle() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.warns(UserWarning, match="widened to int64"):
        x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m.add_constraints(x >= 0, name="c")

    loaded = pickle.loads(pickle.dumps(m))

    assert loaded.dtypes["labels"] == np.int64
    assert (2 * loaded.variables["x"]).vars.dtype == np.int64
