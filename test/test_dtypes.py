"""Tests for int32 default label dtype."""

import numpy as np
import pytest

from linopy import Model
from linopy.config import options


def test_default_label_dtype_is_int32() -> None:
    assert options["label_dtype"] == np.int32


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


def test_overflow_guard_variables() -> None:
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")


def test_overflow_guard_constraints() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m._cCounter = np.iinfo(np.int32).max - 1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        m.add_constraints(x >= 0, name="c")


def test_label_dtype_option_int64() -> None:
    with options:
        options["label_dtype"] = np.int64
        m = Model()
        x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
        assert x.labels.dtype == np.int64
        expr = 2 * x + 1
        assert expr.vars.dtype == np.int64


def test_label_dtype_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="label_dtype must be one of"):
        options["label_dtype"] = np.float64
