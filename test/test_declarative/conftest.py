"""Shared fixtures for the declarative math interface tests."""

from __future__ import annotations

import pytest
import xarray as xr

NODES = ["a", "b", "c"]


@pytest.fixture
def math() -> dict:
    """Minimal but representative math definition exercising every route."""
    return {
        "dimensions": {"node": {"dtype": "string", "iterator": "n"}},
        "parameters": {
            "cost": {"default": 0, "dims": ["node"]},
            "cap_max": {"default": float("inf")},
            "param_inactive": {"default": 0, "active": False},
        },
        "lookups": {
            "active": {"default": True, "dtype": "bool", "dims": ["node"]},
        },
        "variables": {
            "flow": {
                "foreach": ["node"],
                "bounds": {"lower": 0, "upper": float("inf")},
            },
            "flow_inactive": {
                "foreach": ["node"],
                "bounds": {"lower": 0, "upper": float("inf")},
                "active": False,
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


@pytest.fixture
def inputs() -> xr.Dataset:
    return xr.Dataset(
        {
            "cost": ("node", [1.0, 2.0, 3.0]),
            "cap_max": ("node", [10.0, 20.0, 30.0]),
        },
        coords={"node": NODES},
    )
