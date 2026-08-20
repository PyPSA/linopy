"""
Tests for automatic dependency ordering of declarative math variables and expressions.
"""

from __future__ import annotations

import pytest
import xarray as xr

from linopy import Model
from linopy.declarative.build import declarative_model
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.testing import assert_linequal, assert_varequal


def _reordered(mapping: dict, order: list[str]) -> dict:
    """Return a copy of `mapping` with keys in the given order."""
    assert set(order) == set(mapping)
    return {name: mapping[name] for name in order}


def _expr(model: Model, name: str) -> LinearExpression | QuadraticExpression:
    expr = model.expressions[name]
    assert isinstance(expr, LinearExpression | QuadraticExpression)
    return expr


class TestExpressionOrdering:
    def test_forward_reference_builds(self, math: dict, inputs: xr.Dataset) -> None:
        """An expression referencing one declared below it builds correctly."""
        math["expressions"]["scaled_cost"] = {
            "foreach": ["node"],
            "equations": [{"expression": "2 * total_cost"}],
        }
        # `scaled_cost` references `total_cost`; place it first in the YAML order.
        forward = dict(math)
        forward["expressions"] = _reordered(
            math["expressions"],
            ["scaled_cost", "total_cost", "cost_plus_one", "sub_expr_test"],
        )
        backward = dict(math)
        backward["expressions"] = _reordered(
            math["expressions"],
            ["total_cost", "cost_plus_one", "sub_expr_test", "scaled_cost"],
        )
        model_forward = declarative_model(forward, inputs, {})
        model_backward = declarative_model(backward, inputs, {})
        assert_linequal(
            _expr(model_forward, "scaled_cost"), _expr(model_backward, "scaled_cost")
        )

    def test_three_long_chain_in_reverse_order(
        self, math: dict, inputs: xr.Dataset
    ) -> None:
        """A chain c -> b -> a declared in reverse (c first) builds."""
        math["expressions"] = {
            "c": {"foreach": ["node"], "equations": [{"expression": "b * 3"}]},
            "b": {"foreach": ["node"], "equations": [{"expression": "a * 2"}]},
            "a": {"foreach": ["node"], "equations": [{"expression": "flow * cost"}]},
        }
        math["objectives"]["obj"]["equations"] = [{"expression": "sum(c, over=node)"}]
        model = declarative_model(math, inputs, {})
        assert_linequal(_expr(model, "c"), _expr(model, "a") * 6)

    def test_independent_expressions_keep_definition_order(
        self, math: dict, inputs: xr.Dataset
    ) -> None:
        """Expressions with no cross-references keep their YAML order."""
        model = declarative_model(math, inputs, {})
        assert list(model.expressions) == [
            "total_cost",
            "cost_plus_one",
            "sub_expr_test",
        ]

    def test_dependent_expression_moves_after_its_dependency(
        self, math: dict, inputs: xr.Dataset
    ) -> None:
        """Only the forward-referencing expression is moved; the rest keep YAML order."""
        math["expressions"] = _reordered(
            {
                **math["expressions"],
                "scaled_cost": {
                    "foreach": ["node"],
                    "equations": [{"expression": "2 * total_cost"}],
                },
            },
            ["scaled_cost", "total_cost", "cost_plus_one", "sub_expr_test"],
        )
        model = declarative_model(math, inputs, {})
        assert list(model.expressions) == [
            "total_cost",
            "cost_plus_one",
            "sub_expr_test",
            "scaled_cost",
        ]

    def test_cycle_raises(self, math: dict, inputs: xr.Dataset) -> None:
        """A reference cycle between two expressions raises a ValueError."""
        math["expressions"]["a"] = {
            "foreach": ["node"],
            "equations": [{"expression": "b + 1"}],
        }
        math["expressions"]["b"] = {
            "foreach": ["node"],
            "equations": [{"expression": "a + 1"}],
        }
        with pytest.raises(ValueError, match="expressions | .*cycle or self-reference"):
            declarative_model(math, inputs, {})

    def test_self_reference_raises(self, math: dict, inputs: xr.Dataset) -> None:
        """An expression referencing itself raises a ValueError naming it."""
        math["expressions"]["recursive"] = {
            "foreach": ["node"],
            "equations": [{"expression": "recursive + 1"}],
        }
        with pytest.raises(ValueError, match="recursive"):
            declarative_model(math, inputs, {})


class TestVariableOrdering:
    def test_variable_mask_forward_reference_builds(
        self, math: dict, inputs: xr.Dataset
    ) -> None:
        """A variable whose mask references a variable declared below it builds."""
        flow_dependent = {
            "foreach": ["node"],
            "mask": "flow",
            "bounds": {"lower": 0, "upper": float("inf")},
        }
        forward = dict(math)
        forward["variables"] = {
            "flow_dependent": flow_dependent,
            **math["variables"],
        }
        backward = dict(math)
        backward["variables"] = {
            **math["variables"],
            "flow_dependent": flow_dependent,
        }
        model_forward = declarative_model(forward, inputs, {})
        model_backward = declarative_model(backward, inputs, {})
        assert_varequal(
            model_forward.variables["flow_dependent"],
            model_backward.variables["flow_dependent"],
        )

    def test_variable_mask_cycle_raises(self, math: dict, inputs: xr.Dataset) -> None:
        """Mutually mask-referencing variables raise a ValueError."""
        math["variables"]["var_a"] = {
            "foreach": ["node"],
            "mask": "var_b",
            "bounds": {"lower": 0, "upper": float("inf")},
        }
        math["variables"]["var_b"] = {
            "foreach": ["node"],
            "mask": "var_a",
            "bounds": {"lower": 0, "upper": float("inf")},
        }
        with pytest.raises(ValueError, match="variables | .*cycle or self-reference"):
            declarative_model(math, inputs, {})
