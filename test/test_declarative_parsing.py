# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""
Tests for the declarative math parser route separation (mask / expr / raw).

These tests guard the contracts established by the parser-route refactor:

- mask evaluation always returns a boolean ``xr.DataArray``;
- expression evaluation always returns a linopy expression;
- equation (comparison) evaluation returns a ``(lhs, sign, rhs)`` tuple;
- helper-function arguments are always evaluated in ``raw`` mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from linopy.declarative import helper_functions
from linopy.declarative.build import DeclarativeModelBuilder, declarative_model
from linopy.declarative.parsing import ParsedBackendComponent
from linopy.expressions import LinearExpression
from linopy.variables import Variable

NODES = ["a", "b", "c"]


def _math() -> dict:
    """Minimal but representative math definition exercising every route."""
    return {
        "dimensions": {"node": {"dtype": "string", "iterator": "n"}},
        "parameters": {
            "cost": {"default": 0},
            "cap_max": {"default": float("inf")},
        },
        "variables": {
            "flow": {
                "foreach": ["node"],
                "bounds": {"lower": 0, "upper": float("inf")},
            },
        },
        "expressions": {
            "total_cost": {
                "foreach": ["node"],
                "equations": [{"expression": "flow * cost"}],
            },
            # Pure-parameter expression: evaluates to a DataArray, must be coerced.
            "cost_plus_one": {
                "foreach": ["node"],
                "equations": [{"expression": "cost + 1"}],
            },
            # Sub-expression reference to a variable (regression for the expr route).
            "sub_expr_test": {
                "foreach": ["node"],
                "equations": [{"expression": "$foo * cost"}],
                "sub_expressions": {"foo": [{"expression": "flow"}]},
            },
        },
        "constraints": {
            "cap": {
                "foreach": ["node"],
                "equations": [{"expression": "flow <= cap_max"}],
            },
        },
        "objectives": {
            "obj": {
                "equations": [{"expression": "sum(total_cost, over=node)"}],
                "sense": "min",
            },
        },
    }


def _inputs() -> xr.Dataset:
    return xr.Dataset(
        {
            "cost": ("node", [1.0, 2.0, 3.0]),
            "cap_max": ("node", [10.0, 20.0, 30.0]),
        },
        coords={"node": NODES},
    )


@pytest.fixture
def builder_with_flow() -> DeclarativeModelBuilder:
    """A builder with the ``flow`` variable already added to the model."""
    builder = DeclarativeModelBuilder(_math(), _inputs(), {})
    builder.add_variable("flow", builder.math.variables["flow"])
    return builder


def _first_equation(builder: DeclarativeModelBuilder, group: str, name: str):
    """Parse a component and return (component, first_equation, foreach_sub_mask)."""
    definition = getattr(builder.math, group)[name]
    component = ParsedBackendComponent(
        group, name, definition, builder.math.parsing_components
    )
    mask = component.generate_top_level_mask(
        builder.input_data,
        builder.model,
        builder.math,
        builder.config,
        references=set(),
    )
    equation = component.parse_equations()[0]
    sub_mask = equation.evaluate_mask(
        builder.input_data,
        builder.model,
        builder.math,
        builder.config,
        initial_mask=mask,
    )
    sub_mask = component.drop_dims_not_in_foreach(sub_mask)
    return component, equation, sub_mask


# --------------------------------------------------------------------------- #
# Mask route
# --------------------------------------------------------------------------- #


def test_top_level_mask_returns_boolean_dataarray(builder_with_flow):
    component, _, sub_mask = _first_equation(builder_with_flow, "constraints", "cap")
    assert isinstance(sub_mask, xr.DataArray)
    assert sub_mask.dtype == bool
    assert bool(sub_mask.all())


def test_mask_comparison_and_subset_and_helper_return_bool():
    math = _math()
    math["constraints"]["cap"]["equations"][0]["mask"] = (
        "cost > 1 and [a, b] in node and any(cap_max, over=node)"
    )
    builder = DeclarativeModelBuilder(math, _inputs(), {})
    builder.add_variable("flow", builder.math.variables["flow"])
    component = ParsedBackendComponent(
        "constraints",
        "cap",
        builder.math.constraints["cap"],
        builder.math.parsing_components,
    )
    mask = component.generate_top_level_mask(
        builder.input_data,
        builder.model,
        builder.math,
        builder.config,
        references=set(),
    )
    equation = component.parse_equations()[0]
    result = equation.evaluate_mask(
        builder.input_data,
        builder.model,
        builder.math,
        builder.config,
        initial_mask=mask,
    )
    assert isinstance(result, xr.DataArray)
    assert result.dtype == bool


# --------------------------------------------------------------------------- #
# Expression route
# --------------------------------------------------------------------------- #


def test_expression_with_variable_returns_linexpr(builder_with_flow):
    _, equation, sub_mask = _first_equation(
        builder_with_flow, "expressions", "total_cost"
    )
    result = equation.evaluate_expression(
        builder_with_flow.input_data,
        builder_with_flow.model,
        builder_with_flow.math,
        mask=sub_mask,
    )
    assert isinstance(result, LinearExpression)


def test_pure_parameter_expression_coerced_to_linexpr(builder_with_flow):
    _, equation, sub_mask = _first_equation(
        builder_with_flow, "expressions", "cost_plus_one"
    )
    result = equation.evaluate_expression(
        builder_with_flow.input_data,
        builder_with_flow.model,
        builder_with_flow.math,
        mask=sub_mask,
    )
    # No decision variable is involved, but the contract is still LinearExpression.
    assert isinstance(result, LinearExpression)


def test_sub_expression_reference_returns_linexpr(builder_with_flow):
    _, equation, sub_mask = _first_equation(
        builder_with_flow, "expressions", "sub_expr_test"
    )
    result = equation.evaluate_expression(
        builder_with_flow.input_data,
        builder_with_flow.model,
        builder_with_flow.math,
        mask=sub_mask,
    )
    assert isinstance(result, LinearExpression)


# --------------------------------------------------------------------------- #
# Equation (constraint) route
# --------------------------------------------------------------------------- #


def test_equation_returns_lhs_sign_rhs_tuple(builder_with_flow):
    _, equation, sub_mask = _first_equation(builder_with_flow, "constraints", "cap")
    lhs, sign, rhs = equation.evaluate_equation(
        builder_with_flow.input_data,
        builder_with_flow.model,
        builder_with_flow.math,
        mask=sub_mask,
    )
    assert isinstance(lhs, LinearExpression)  # decision variable side
    assert isinstance(rhs, LinearExpression)  # pure-parameter side, coerced
    assert isinstance(sign, xr.DataArray)
    assert set(np.unique(sign.values)) <= {"<="}


# --------------------------------------------------------------------------- #
# Helper-function argument evaluation (raw mode)
# --------------------------------------------------------------------------- #


class _RecordArgs(helper_functions.ParsingHelperFunction):
    """Test-only helper that records the types of the arguments it receives."""

    NAME = "record_args"
    ALLOWED_IN = ["expression"]
    received: list[type] = []

    def as_math_string(self, *args, **kwargs):  # noqa: D102
        return "record_args"

    def as_raw(self, *args, **kwargs):  # noqa: D102
        type(self).received.extend(type(a) for a in args)
        # Return the first argument so the enclosing expression stays valid.
        return args[0]


def test_helper_arguments_are_evaluated_raw(builder_with_flow):
    math = _math()
    # flow is a variable, cost is a parameter -> raw mode must preserve both types.
    math["expressions"]["total_cost"]["equations"][0]["expression"] = (
        "record_args(flow, cost)"
    )
    builder = DeclarativeModelBuilder(math, _inputs(), {})
    builder.add_variable("flow", builder.math.variables["flow"])
    _, equation, sub_mask = _first_equation(builder, "expressions", "total_cost")

    _RecordArgs.received = []
    equation.evaluate_expression(
        builder.input_data, builder.model, builder.math, mask=sub_mask
    )
    assert _RecordArgs.received, "helper was not called"
    # The variable arrives un-normalised (raw Variable, not LinearExpression);
    # the parameter arrives as a raw DataArray (not coerced/masked to booleans).
    assert Variable in _RecordArgs.received
    assert xr.DataArray in _RecordArgs.received
    assert LinearExpression not in _RecordArgs.received


def test_invalid_helper_function_rejected(builder_with_flow):
    """A registry entry that is not a ParsingHelperFunction subclass is rejected."""
    registry = helper_functions._registry["expression"]
    registry["not_a_helper"] = str  # type: ignore[assignment]
    try:
        math = _math()
        math["expressions"]["total_cost"]["equations"][0]["expression"] = (
            "not_a_helper(flow)"
        )
        builder = DeclarativeModelBuilder(math, _inputs(), {})
        builder.add_variable("flow", builder.math.variables["flow"])
        _, equation, sub_mask = _first_equation(builder, "expressions", "total_cost")
        with pytest.raises(ValueError, match="must be subclassed"):
            equation.evaluate_expression(
                builder.input_data, builder.model, builder.math, mask=sub_mask
            )
    finally:
        registry.pop("not_a_helper", None)


# --------------------------------------------------------------------------- #
# Input-data checks
# --------------------------------------------------------------------------- #


def test_checks_run_without_active_variable():
    """`_check_inputs` must not require an `active` variable in the input data."""
    math = _math()
    math["checks"] = {
        "too_expensive": {
            "mask": "cost > 100",
            "message": "cost too high",
            "errors": "raise",
        }
    }
    # No `active` variable in the inputs, and the check does not trigger.
    declarative_model(math, _inputs(), {})


def test_check_raises_when_triggered_without_active():
    math = _math()
    math["checks"] = {
        "too_expensive": {
            "mask": "cost > 0",
            "message": "cost too high",
            "errors": "raise",
        }
    }
    with pytest.raises(ValueError, match="cost too high"):
        declarative_model(math, _inputs(), {})


# --------------------------------------------------------------------------- #
# End-to-end build
# --------------------------------------------------------------------------- #


def test_declarative_model_end_to_end():
    model = declarative_model(_math(), _inputs(), {})
    assert "flow" in model.variables
    assert "total_cost" in model.expressions
    assert "cap" in model.constraints
    assert model.objective is not None
    # flow is indexed over the node dimension.
    assert set(model.variables["flow"].dims) == {"node"}
