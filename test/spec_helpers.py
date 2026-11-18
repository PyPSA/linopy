"""Specs and data the spec tests share; imported under an ``importorskip("mathspec")`` guard."""

from __future__ import annotations

import functools
import os
import unittest.mock
import warnings
from collections.abc import Callable
from typing import Any

import mathspec
import pandas as pd
import pytest
import xarray as xr
import yaml

from linopy import Model, PerformanceWarning

EXAMPLES_DIR = os.environ.get("MATHSPEC_EXAMPLES")
needs_examples = pytest.mark.skipif(
    EXAMPLES_DIR is None,
    reason="set MATHSPEC_EXAMPLES to a mathspec examples directory",
)
EXAMPLE_DISPATCH = """
description: Least-cost dispatch of a generator fleet against an hourly load.

dimensions:
  snapshot: { dtype: int, description: dispatch periods, ordered: true }
  generator: { description: generating units }

parameters:
  p_max: { dims: [generator], description: installed capacity }
  load: { dims: [snapshot], description: demand to be met }
  cost: { dims: [generator], description: marginal cost }

variables:
  p:
    description: output of a generator in a snapshot
    dims: [snapshot, generator]
    where: "p_max > 0"
    bounds: { lower: 0, upper: p_max }

constraints:
  power_balance:
    dims: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(p * cost)

expressions:
  spend: sum(p * cost, over=generator)
  usage:
    dims: [snapshot, generator]
    expression: p / p_max
"""

GENERATOR = pd.Index(["wind", "gas"], name="generator")
SNAPSHOT = pd.Index([0, 1, 2], name="snapshot")
DISPATCH_DATA: dict[str, Any] = {
    "snapshot": SNAPSHOT,
    "generator": GENERATOR,
    "p_max": pd.DataFrame({"generator": GENERATOR, "p_max": [100.0, 200.0]}),
    "load": pd.DataFrame({"snapshot": SNAPSHOT, "load": [80.0, 150.0, 50.0]}),
    "cost": pd.DataFrame({"generator": GENERATOR, "cost": [0.0, 50.0]}),
}
DISPATCH_P = xr.DataArray(
    [[80.0, 0.0], [100.0, 50.0], [50.0, 0.0]],
    coords={"snapshot": SNAPSHOT, "generator": GENERATOR},
)


def table(series: pd.Series, column: str) -> pd.DataFrame:
    """*series* as a spec table: its named index levels as label columns, its values under *column*."""
    return series.rename(column).reset_index()


def solved(spec: Any, sources: Any, store: bool = True, **kwargs: Any) -> Model:
    m = Model.from_spec(spec, sources, store=store, **kwargs)
    m.solve(solver_name="highs", output_flag=False, reformulate_sos=True)
    return m


def yaml_dict() -> dict[str, Any]:
    return mathspec.to_spec(yaml.safe_load(EXAMPLE_DISPATCH)).to_dict()


def with_(spec: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    for section, entries in sections.items():
        out[section] = {**spec.get(section, {}), **entries}
    return out


TT = pd.Index([0, 1, 2, 3], name="t")
S = pd.Index(["a", "b"], name="s")
DAYS = pd.date_range("2030-01-01", periods=4, freq="D", name="d")

WHERE_SPEC: dict[str, Any] = {
    "dimensions": {
        "t": {"dtype": "int", "ordered": True},
        "s": {"dtype": "str"},
        "d": {"dtype": "datetime", "ordered": True},
    },
    "relations": {
        "season_of": {"key": "t", "values": "s", "missing": "absent"},
        "other_of": {"key": "t", "values": "s"},
    },
    "parameters": {
        "flag": {"dims": ["t"], "dtype": "bool", "missing": "absent"},
        "cost": {"dims": ["t"], "missing": "absent"},
        "label": {"dims": ["t"], "dtype": "str", "missing": "absent"},
        "day_cost": {"dims": ["d"]},
    },
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 1}},
        "y": {"dims": ["d"], "bounds": {"lower": 0, "upper": 1}},
    },
    "conditions": {"late": {"condition": "t > 1"}},
    "objective": {"sense": "minimize", "expression": "sum(x) + sum(y)"},
}
WHERE_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "d": DAYS,
    "season_of": pd.DataFrame({"t": TT[:3], "s": ["a", "a", "b"]}),
    "other_of": pd.DataFrame({"t": TT, "s": ["a", "b", "b", "a"]}),
    "flag": pd.DataFrame({"t": TT[:2], "flag": [True, False]}),
    "cost": pd.DataFrame({"t": TT[:3], "cost": [1.0, float("inf"), 3.0]}),
    "label": pd.DataFrame({"t": TT[1:3], "label": ["u", "v"]}),
    "day_cost": pd.DataFrame({"d": DAYS, "day_cost": [1.0, 2.0, 3.0, 4.0]}),
}

BP = pd.Index([0, 1, 2, 3], name="bp")
UNITS = pd.Index(["hydro", "gas"], name="generator")
CURVE_SPEC: dict[str, Any] = {
    "dimensions": {
        "snapshot": {"dtype": "int"},
        "generator": {"dtype": "str"},
        "bp": {"dtype": "int", "ordered": True},
    },
    "parameters": {
        "p_max": {"dims": ["generator"]},
        "load": {"dims": ["snapshot"]},
        "bp_x": {"dims": ["generator", "bp"]},
        "bp_y": {"dims": ["generator", "bp"]},
    },
    "variables": {
        "p": {
            "dims": ["snapshot", "generator"],
            "bounds": {"lower": 0, "upper": "p_max"},
        },
        "op_cost": {"dims": ["snapshot", "generator"], "bounds": {"lower": 0}},
    },
    "piecewise": {
        "cost_curve": {
            "along": "bp",
            "dims": ["snapshot", "generator"],
            "links": {"p": ["p", "bp_x"], "op_cost": ["op_cost", "bp_y", ">="]},
            "method": "lp",
        }
    },
    "expressions": {"spend": "sum(op_cost, over=generator)"},
    "constraints": {
        "balance": {
            "dims": ["snapshot"],
            "expression": "sum(p, over=generator) == load",
        }
    },
    "objective": {"sense": "minimize", "expression": "sum(op_cost)"},
}
MASKED_CURVE_SPEC = with_(
    CURVE_SPEC,
    parameters={
        "bp_x": {"dims": ["generator", "bp"], "missing": "absent"},
        "bp_y": {"dims": ["generator", "bp"], "missing": "absent"},
    },
    piecewise={
        "cost_curve": {**CURVE_SPEC["piecewise"]["cost_curve"], "where": "bp_x"}
    },
)


def curve(points: dict[tuple[str, int], float]) -> xr.DataArray:
    """Breakpoints as a grid over (generator, bp), NaN where a curve has none."""
    index = pd.MultiIndex.from_tuples(list(points), names=["generator", "bp"])
    return xr.DataArray.from_series(pd.Series(list(points.values()), index=index))


FULL_X = curve(
    {(g, k): x for g in UNITS for k, x in enumerate([0.0, 20.0, 50.0, 80.0])}
)
FULL_Y = curve(
    {(g, k): y for g in UNITS for k, y in enumerate([0.0, 150.0, 450.0, 900.0])}
)
RAGGED_X = curve(
    {
        ("hydro", 0): 0.0,
        ("hydro", 1): 40.0,
        **{("gas", k): x for k, x in enumerate([0.0, 20.0, 50.0, 80.0])},
    }
)
RAGGED_Y = curve(
    {
        ("hydro", 0): 0.0,
        ("hydro", 1): 200.0,
        **{("gas", k): y for k, y in enumerate([0.0, 150.0, 450.0, 900.0])},
    }
)
CURVE_DATA: dict[str, Any] = {
    "snapshot": [0],
    "generator": UNITS,
    "bp": BP,
    "p_max": pd.DataFrame({"generator": UNITS, "p_max": [40.0, 80.0]}),
    "load": pd.DataFrame({"snapshot": [0], "load": [50.0]}),
}


def _rows(flat: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("labels", "vars") if c in flat.columns]
    return flat.sort_values(keys).reset_index(drop=True)


def _same_terms(a: Any, b: Any) -> None:
    """*a* and *b*, two expressions, hold the same terms and constants, however their term slots are laid out."""
    pd.testing.assert_frame_equal(_rows(a.flat), _rows(b.flat), check_exact=True)
    assert a.const.transpose(*b.const.dims).equals(b.const)


def assert_same_model(a: Model, b: Model) -> None:
    """*a* and *b* hold the same variables, rows, objective and named expressions, term slots aside."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        _assert_same_model(a, b)


def _assert_same_model(a: Model, b: Model) -> None:
    assert list(a.variables) == list(b.variables)
    for name in a.variables:
        for field in ("labels", "lower", "upper"):
            assert getattr(a.variables[name], field).equals(
                getattr(b.variables[name], field)
            ), (name, field)
    assert list(a.constraints) == list(b.constraints)
    for name in a.constraints:
        ca, cb = a.constraints[name], b.constraints[name]
        assert ca.labels.equals(cb.labels), name
        pd.testing.assert_frame_equal(_rows(ca._flat()), _rows(cb._flat()))
    _same_terms(a.objective.expression, b.objective.expression)
    assert list(a.expressions) == list(b.expressions)
    for name in a.expressions:
        _same_terms(a.expressions[name], b.expressions[name])


def checked_against_unpruned(build_into: Callable[..., None]) -> Callable[..., None]:
    """*build_into* that builds every model again without pruning and asserts the two builds agree."""
    from linopy.spec import api
    from linopy.spec.context import Context

    def checked(model: Model, *args: Any, **kwargs: Any) -> None:
        twin = Model(sparse=model.sparse, dtypes=model.dtypes)
        unpruned = functools.partial(Context, prune=False)
        error: Exception | None = None
        with unittest.mock.patch.object(api, "Context", unpruned):
            try:
                build_into(twin, *args, **kwargs)
            except Exception as e:
                error = e
        try:
            build_into(model, *args, **kwargs)
        except Exception as e:
            assert (type(e), str(e)) == (type(error), str(error))
            raise
        assert error is None, f"only the unpruned build raised: {error!r}"
        assert_same_model(model, twin)

    return checked
