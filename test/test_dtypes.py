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
        Model(dtypes={"labels": np.float64})  # type: ignore[dict-item]


def test_dtypes_init_arg_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="only supports the keys"):
        Model(dtypes={"coeffs": np.int64})  # type: ignore[dict-item]


def test_dtypes_is_read_only_mapping() -> None:
    dtypes = Model().dtypes
    assert set(dtypes) == {"labels", "sign"}
    with pytest.raises(TypeError):
        dtypes["labels"] = np.int64  # type: ignore[index]


def test_default_sign_dtype_is_int8() -> None:
    assert Model().dtypes["sign"] == np.int8


def test_sign_stored_as_int8_but_read_as_strings() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(x <= 5, name="c")
    con = m.constraints["c"]
    # Stored compactly as int8 category codes ...
    assert con.data["sign"].dtype == np.int8
    # ... but decoded to canonical strings at the public boundary.
    assert con.sign.dtype.kind == "U"
    assert set(np.unique(con.sign.values)) == {"<="}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda x: x <= 5, "<="),
        (lambda x: x >= 1, ">="),
        (lambda x: x == 3, "="),
    ],
)
def test_all_three_senses_round_trip(build, expected) -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(build(x), name="c")
    con = m.constraints["c"]
    assert con.data["sign"].dtype == np.int8
    assert set(np.unique(con.sign.values)) == {expected}


def test_sign_int8_is_eightfold_smaller() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=1, coords=[range(100_000)], name="x")
    m.add_constraints(x <= 1, name="c")
    compact = m.constraints["c"].data["sign"]
    assert compact.dtype == np.int8
    legacy_nbytes = m.constraints["c"].sign.astype("<U2").nbytes
    assert compact.nbytes * 8 == legacy_nbytes


def test_sign_dtype_str_reproduces_legacy_storage() -> None:
    m = Model(dtypes={"sign": np.str_})
    assert m.dtypes["sign"] == np.str_
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(x <= 5, name="c")
    con = m.constraints["c"]
    assert con.data["sign"].dtype.kind == "U"
    assert set(np.unique(con.sign.values)) == {"<="}


def test_sign_dtype_init_arg_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="dtypes\\['sign'\\] must be"):
        Model(dtypes={"sign": np.float64})  # type: ignore[dict-item]


def test_sign_update_round_trips() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(x <= 5, name="c")
    con = m.constraints["c"]
    con.update(sign=">=")
    assert con.data["sign"].dtype == np.int8
    assert set(np.unique(con.sign.values)) == {">="}


@pytest.mark.skipif(
    not pytest.importorskip("highspy", reason="highspy not installed"),
    reason="highspy not installed",
)
def test_mixed_senses_solve_correctly() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y <= 15, name="c1")
    m.add_constraints(x >= 2, name="c2")
    m.add_constraints(y == 4, name="c3")
    m.add_objective(x + 2 * y, sense="max")
    m.solve("highs")
    # x = 10, y = 4  ->  10 + 8 = 18
    assert m.objective.value == pytest.approx(18.0)
    assert m.solution["x"].item() == pytest.approx(10.0)
    assert m.solution["y"].item() == pytest.approx(4.0)


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
