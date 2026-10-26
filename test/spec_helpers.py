"""Specs and data the spec tests share; imported under an ``importorskip("mathspec")`` guard."""

from __future__ import annotations

import os
from typing import Any

import mathspec
import pandas as pd
import xarray as xr
import yaml

from linopy import Model

EXAMPLES_DIR = os.environ.get("MATHSPEC_EXAMPLES")
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
    "p_max": pd.Series([100.0, 200.0], index=GENERATOR),
    "load": pd.Series([80.0, 150.0, 50.0], index=SNAPSHOT),
    "cost": pd.Series([0.0, 50.0], index=GENERATOR),
}
DISPATCH_P = xr.DataArray(
    [[80.0, 0.0], [100.0, 50.0], [50.0, 0.0]],
    coords={"snapshot": SNAPSHOT, "generator": GENERATOR},
)


def solved(spec: Any, sources: Any, **kwargs: Any) -> Model:
    m = Model.from_spec(spec, sources, **kwargs)
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
    "objective": {"sense": "minimize", "expression": "sum(x) + sum(y)"},
}
WHERE_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "d": DAYS,
    "season_of": pd.Series(["a", "a", "b"], index=TT[:3]),
    "other_of": pd.Series(["a", "b", "b", "a"], index=TT),
    "flag": pd.Series([True, False], index=TT[:2]),
    "cost": pd.Series([1.0, float("inf"), 3.0], index=TT[:3]),
    "label": pd.Series(["u", "v"], index=TT[1:3]),
    "day_cost": pd.Series([1.0, 2.0, 3.0, 4.0], index=DAYS),
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
            "over": "bp",
            "links": [["p", "bp_x"], ["op_cost", "bp_y", ">="]],
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
        "cost_curve": {**CURVE_SPEC["piecewise"]["cost_curve"], "points": "bp_x"}
    },
)


def curve(points: dict[tuple[str, int], float]) -> pd.Series:
    index = pd.MultiIndex.from_tuples(list(points), names=["generator", "bp"])
    return pd.Series(list(points.values()), index=index)


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
    "p_max": pd.Series([40.0, 80.0], index=UNITS),
    "load": pd.Series([50.0], index=pd.Index([0], name="snapshot")),
}
