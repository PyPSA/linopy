#!/usr/bin/env python3
"""
This module aims at testing the correct behavior of the Expressions class.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, Variable
from linopy.expressions import (
    Expressions,
    LazyExpression,
    LinearExpression,
    QuadraticExpression,
)
from linopy.solvers import available_solvers
from linopy.testing import assert_linequal, assert_quadequal


@pytest.fixture
def m() -> Model:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    y = m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_expressions(x + 1, name="expr_x")
    m.add_expressions(x * y, name="expr_xy")
    return m


def test_expressions_repr(m: Model) -> None:
    m.expressions.__repr__()
    repr(Model())


def test_expressions_getitem(m: Model) -> None:
    assert isinstance(m.expressions["expr_x"], LinearExpression)

    subset = m.expressions[["expr_x"]]
    assert isinstance(subset, Expressions)
    assert len(subset) == 1


def test_expressions_getattr(m: Model) -> None:
    assert_linequal(m.expressions.expr_x, m.expressions["expr_x"])

    with pytest.raises(AttributeError):
        m.expressions.does_not_exist


def test_expressions_getattr_formatted() -> None:
    m = Model()
    x = m.add_variables(name="x")
    m.add_expressions(x + 1, name="e-0")
    assert_linequal(m.expressions.e_0, m.expressions["e-0"])


def test_expressions_dict_protocol(m: Model) -> None:
    assert len(m.expressions) == 2
    assert set(iter(m.expressions)) == {"expr_x", "expr_xy"}
    assert set(dict(m.expressions.items())) == {"expr_x", "expr_xy"}
    assert "expr_x" in m.expressions
    assert m.expressions._ipython_key_completions_() == list(m.expressions)
    assert "expr_x" in dir(m.expressions)


def test_expressions_name_counter() -> None:
    m = Model()
    x = m.add_variables(name="x")
    m.add_expressions(x + 1)
    m.add_expressions(x + 1)
    assert "expr0" in m.expressions
    assert "expr1" in m.expressions


def test_expressions_duplicate_name_raises(m: Model) -> None:
    x = m.variables["x"]
    with pytest.raises(ValueError, match="already assigned"):
        m.add_expressions(x + 1, name="expr_x")


def test_add_expressions_from_variable_and_tuples() -> None:
    m = Model()
    x = m.add_variables(name="x")

    expr = m.add_expressions(x, name="from_var")
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, x.to_linexpr())

    expr = m.add_expressions([(2, x)], name="from_tuples")
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, 2 * x)


def test_add_expressions_quadratic(m: Model) -> None:
    assert isinstance(m.expressions["expr_xy"], QuadraticExpression)


def test_add_expressions_mask() -> None:
    m = Model()
    idx = pd.RangeIndex(10, name="first")
    x = m.add_variables(coords=[idx], name="x")
    mask = xr.DataArray([True] * 5 + [False] * 5, coords=[idx])

    expr = m.add_expressions(x + 1, name="masked", mask=mask)
    assert_linequal(expr, (x + 1).where(mask))


def test_expressions_remove(m: Model) -> None:
    m.expressions.remove("expr_x")
    assert "expr_x" not in m.expressions

    with pytest.raises(KeyError):
        m.expressions.remove("expr_x")


def test_remove_expressions(m: Model) -> None:
    m.remove_expressions("expr_x")
    assert "expr_x" not in m.expressions
    assert "expr_xy" in m.expressions


def test_remove_expressions_with_list(m: Model) -> None:
    m.remove_expressions(["expr_x", "expr_xy"])
    assert len(m.expressions) == 0


def test_model_repr_contains_expressions(m: Model) -> None:
    r = repr(m)
    assert "Expressions:" in r
    assert "* expr_x" in r


@pytest.mark.skipif(not available_solvers, reason="No solver available")
def test_expressions_solution() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="first")], name="x")
    m.add_constraints(x >= 2)
    m.add_expressions(2 * x, name="double_x")
    m.add_objective(x.sum())
    m.solve(available_solvers[0])

    sol = m.expressions.solution
    assert isinstance(sol, xr.Dataset)
    assert "double_x" in sol
    assert (sol["double_x"] == 4).all()


class TestLazyExpression:
    """Tests for the callable-`data` (lazy) path of `Model.add_expressions`."""

    def test_add_expressions_with_callable_places_placeholder(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        assert isinstance(lazy, LazyExpression)
        assert m.expressions["lazy"] is lazy

    def test_evaluate_equals_eager_and_stays_lazy(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        assert_linequal(lazy.evaluate(), x + y)
        assert m.expressions["lazy"] is lazy

    def test_promote_swaps_carries_attrs_and_is_idempotent(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        lazy.attrs["references"] = ["x", "y"]
        promoted = lazy.promote()
        assert isinstance(promoted, LinearExpression)
        assert m.expressions["lazy"] is promoted
        assert promoted.attrs["references"] == ["x", "y"]
        assert promoted.attrs["name"] == "lazy"
        assert_linequal(promoted, x + y)
        # A second promote (from the stale placeholder) returns the existing entry.
        assert lazy.promote() is promoted

    def test_promote_derived_expression_raises(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        derived = lazy * 2
        assert derived.name is None
        with pytest.raises(ValueError, match="derived"):
            derived.promote()

    def test_duplicate_name_raises(self, m: Model, x: Variable) -> None:
        m.add_expressions(lambda m: 1 * x, name="lazy")
        with pytest.raises(ValueError, match="already assigned to model"):
            m.add_expressions(lambda m: 2 * x, name="lazy")
        with pytest.raises(ValueError, match="already assigned to model"):
            m.add_expressions(1 * x, name="lazy")

    def test_auto_naming(self, m: Model, x: Variable) -> None:
        lazy = m.add_expressions(lambda m: 1 * x)
        assert lazy.name.startswith("expr")
        assert lazy.name in m.expressions

    def test_metadata_without_evaluation(self, m: Model, x: Variable) -> None:
        calls = 0

        def evaluator(model: Model) -> LinearExpression:
            nonlocal calls
            calls += 1
            return 1 * x

        ds = xr.Dataset({"const": ("dim_0", [1.0, 2.0])})
        lazy = m.add_expressions(evaluator, name="lazy", dims=("dim_0",), input_data=ds)
        # The input data is shared by pointer, never copied.
        assert lazy.input_data is ds
        assert lazy.dims == ("dim_0",)
        assert "not yet evaluated" in repr(lazy)
        assert calls == 0
        lazy.evaluate()
        assert calls == 1
        # Nothing is cached: a second evaluation runs the evaluator again.
        lazy.evaluate()
        assert calls == 2

    def test_params_forwarded_to_evaluator(self, m: Model, x: Variable) -> None:
        lazy = m.add_expressions(
            lambda model, factor: factor * x, name="lazy", factor=3
        )
        assert_linequal(lazy.evaluate(), 3 * x)

    def test_params_forwarded_to_callable_mask(self, m: Model, x: Variable) -> None:
        mask_calls = 0

        def mask(model: Model, threshold: int) -> xr.DataArray:
            nonlocal mask_calls
            mask_calls += 1
            return x.coords["first"] >= threshold

        lazy = m.add_expressions(
            lambda model, threshold: x + 1, name="lazy", mask=mask, threshold=1
        )
        assert mask_calls == 0
        result = lazy.evaluate()
        assert mask_calls == 1
        expected = (x + 1).where(x.coords["first"] >= 1)
        assert_linequal(result, expected)

    def test_concrete_mask_matches_eager(self, x: Variable) -> None:
        m = x.model
        mask = x.coords["first"] < 1
        lazy = m.add_expressions(lambda model: x + 1, name="lazy", mask=mask)
        eager = m.add_expressions(x + 1, name="eager", mask=mask)
        assert_linequal(lazy.evaluate(), eager)

    def test_all_false_mask_yields_empty_expression_no_error(self, x: Variable) -> None:
        m = x.model
        mask = xr.zeros_like(x.coords["first"], dtype=bool)
        lazy = m.add_expressions(lambda model: x + 1, name="lazy", mask=mask)
        result = lazy.evaluate()
        assert result.const.isnull().all()

    def test_callable_mask_requires_callable_data(self, m: Model, x: Variable) -> None:
        with pytest.raises(TypeError, match="callable mask"):
            m.add_expressions(x + 1, name="lazy", mask=lambda model: True)

    def test_lazy_algebra_stays_lazy_and_matches_eager(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        eager = x + y

        combos = [
            (lazy + lazy, eager + eager),
            (lazy - lazy, eager - eager),
            (lazy * 2, eager * 2),
            (2 * lazy, 2 * eager),
            (-lazy, -eager),
            (eager + lazy, eager + eager),
            (eager - lazy, eager - eager),
            (np.array(2) * lazy, eager * 2),
        ]
        for result, expected in combos:
            assert isinstance(result, LazyExpression)
            assert_linequal(result.evaluate(), expected)

    def test_lazy_pow_and_matmul(self, m: Model, x: Variable) -> None:
        lazy = m.add_expressions(lambda m: 1 * x, name="lazy")

        squared = lazy**2
        assert isinstance(squared, LazyExpression)
        assert_quadequal(squared.evaluate(), (1 * x) ** 2)

        arr = xr.DataArray(
            np.ones(x.coords["first"].size), coords=x.coords, dims=x.dims
        )
        matmul_result = lazy @ arr
        assert isinstance(matmul_result, LazyExpression)
        assert_linequal(matmul_result.evaluate(), (1 * x) @ arr)

    def test_evaluator_called_once_per_leaf_per_evaluate(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        calls = {"a": 0, "b": 0}

        def eval_a(model: Model) -> LinearExpression:
            calls["a"] += 1
            return 1 * x

        def eval_b(model: Model) -> LinearExpression:
            calls["b"] += 1
            return 1 * y

        lazy_a = m.add_expressions(eval_a, name="a")
        lazy_b = m.add_expressions(eval_b, name="b")
        chain = (lazy_a + lazy_b) * 2

        assert calls == {"a": 0, "b": 0}
        chain.evaluate()
        assert calls == {"a": 1, "b": 1}

    def test_solution_raises_before_solve(
        self, m: Model, x: Variable, y: Variable
    ) -> None:
        lazy = m.add_expressions(lambda m: x + y, name="lazy")
        with pytest.raises(AttributeError, match="not optimized"):
            lazy.solution

    @pytest.mark.skipif(not available_solvers, reason="No solver installed")
    def test_solution_matches_eager_after_solve(self) -> None:
        m = Model()
        time = pd.RangeIndex(3, name="time")
        x = m.add_variables(lower=1, coords=[time], name="x")
        eager = m.add_expressions(2 * x, name="eager")
        lazy = m.add_expressions(lambda m: 2 * m.variables["x"], name="lazy")
        m.add_objective(x.sum())
        m.solve(available_solvers[0])
        xr.testing.assert_allclose(lazy.solution, eager.solution)
        # Requesting the solution must not promote the placeholder.
        assert m.expressions["lazy"] is lazy

    @pytest.mark.skipif(not available_solvers, reason="No solver installed")
    def test_expressions_solution_with_lazy_member(self) -> None:
        m = Model()
        time = pd.RangeIndex(3, name="time")
        x = m.add_variables(lower=2, coords=[time], name="x")
        m.add_expressions(lambda m: 2 * m.variables["x"], name="lazy")
        m.add_objective(x.sum())
        m.solve(available_solvers[0])

        sol = m.expressions.solution
        assert isinstance(sol, xr.Dataset)
        assert "lazy" in sol
        assert (sol["lazy"] == 4).all()
