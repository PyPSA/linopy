"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

if TYPE_CHECKING:
    from linopy import Model, Variable

# ``linopy`` is intentionally NOT imported at module level — doing so
# loads it from site-packages before pytest's ``--doctest-modules``
# collection walks the ``linopy/`` source tree, and the resulting
# __file__ mismatch breaks the whole run on Windows CI (and elsewhere).
# Same reasoning as the ``filterwarnings`` comment in ``pyproject.toml``.
# Values mirror ``linopy.config.LEGACY_SEMANTICS`` / ``V1_SEMANTICS``.
_LEGACY_SEMANTICS = "legacy"
_V1_SEMANTICS = "v1"
_VALID_SEMANTICS = {_LEGACY_SEMANTICS, _V1_SEMANTICS}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that require GPU hardware",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and behavior."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU hardware")
    for sem in sorted(_VALID_SEMANTICS):
        config.addinivalue_line(
            "markers", f"{sem}: run this test only under the {sem} semantics"
        )

    # Set environment variable so test modules can check if GPU tests are enabled
    # This is needed because parametrize happens at import time
    if config.getoption("--run-gpu", default=False):
        os.environ["LINOPY_RUN_GPU_TESTS"] = "1"
    else:
        os.environ.pop("LINOPY_RUN_GPU_TESTS", None)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Auto-skip GPU-only solvers (e.g. cuPDLPx) unless --run-gpu is passed.

    Solvers that *also* have a CPU mode (e.g. xpress, which carries
    ``GPU_ACCELERATION`` from version 9.8) are not skipped — their default
    pathway is CPU and they should run in normal CI.
    """
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run GPU tests")
    for item in items:
        # Check if this is a parametrized test with a GPU-only solver
        if hasattr(item, "callspec") and "solver" in item.callspec.params:
            solver = item.callspec.params["solver"]
            # Import here to avoid circular dependency
            from linopy.solver_capabilities import (
                SolverFeature,
                solver_supports,
            )

            if solver_supports(solver, SolverFeature.GPU_ONLY):
                item.add_marker(skip_gpu)
                item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True, params=[_LEGACY_SEMANTICS, _V1_SEMANTICS])
def semantics(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Run every test under both arithmetic semantics by default.

    A test marked with a semantics name (``@pytest.mark.legacy`` or
    ``@pytest.mark.v1``) runs only under that semantics. Under ``legacy``,
    ``LinopySemanticsWarning`` is suppressed so test output stays clean;
    ``test_convention.py`` verifies the warnings are actually emitted.
    """
    from linopy.config import LinopySemanticsWarning, options

    item = request.node
    for sem in _VALID_SEMANTICS:
        if item.get_closest_marker(sem) and request.param != sem:
            pytest.skip(f"{sem}-only test")

    old = options["semantics"]
    options["semantics"] = request.param
    if request.param == _LEGACY_SEMANTICS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopySemanticsWarning)
            yield request.param
    else:
        yield request.param
    options["semantics"] = old


@pytest.fixture
def m() -> Model:
    from linopy import Model

    m = Model()
    m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_variables(4, pd.Series([8, 10]), name="y")
    m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
    m.add_variables(coords=[pd.RangeIndex(20, name="dim_2")], name="v")
    return m


@pytest.fixture
def x(m: Model) -> Variable:
    return m.variables["x"]


@pytest.fixture
def y(m: Model) -> Variable:
    return m.variables["y"]


@pytest.fixture
def z(m: Model) -> Variable:
    return m.variables["z"]


@pytest.fixture
def v(m: Model) -> Variable:
    return m.variables["v"]


@pytest.fixture
def u(m: Model) -> Variable:
    """
    `dim_3` variable: a (level1, level2) MultiIndex under legacy, the flat dim +
    level1/level2 aux coords equivalent under v1 (where MultiIndex is disallowed).
    """
    from linopy.semantics import is_v1

    if is_v1():
        m.add_variables(coords=[pd.RangeIndex(4, name="dim_3")], name="u")
        return m.variables["u"].assign_coords(
            level1=("dim_3", [1, 1, 2, 2]),
            level2=("dim_3", ["a", "b", "a", "b"]),
        )
    idx = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("level1", "level2"))
    idx.name = "dim_3"
    m.add_variables(coords=[idx], name="u")
    return m.variables["u"]


if find_spec("math_spec") is not None:
    import math_spec
    import xarray as xr
    import yaml

    from linopy import Model

    EXAMPLE_DISPATCH = """
description: Least-cost dispatch of a generator fleet against an hourly load.

dimensions:
  snapshot: { dtype: int, description: dispatch periods }
  generator: { description: generating units }

parameters:
  p_max: { dims: [generator], description: installed capacity }
  load: { dims: [snapshot], description: demand to be met }
  cost: { dims: [generator], description: marginal cost }

variables:
  p:
    description: output of a generator in a snapshot
    foreach: [snapshot, generator]
    where: "p_max > 0"
    bounds: { lower: 0, upper: p_max }

constraints:
  power_balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(p * cost)

expressions:
  spend: sum(p * cost, over=generator)
  usage: p / p_max
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
        return math_spec.to_spec(yaml.safe_load(EXAMPLE_DISPATCH)).to_dict()

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
            "t": {"dtype": "int"},
            "s": {"dtype": "str"},
            "d": {"dtype": "datetime"},
        },
        "lookups": {
            "season_of": {"over": "t", "into": "s"},
            "other_of": {"over": "t", "into": "s"},
            "tag": {"over": "t", "dtype": "str"},
        },
        "parameters": {
            "flag": {"dims": ["t"], "dtype": "bool"},
            "cost": {"dims": ["t"]},
            "label": {"dims": ["t"], "dtype": "str"},
            "day_cost": {"dims": ["d"]},
        },
        "variables": {
            "x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": 1}},
            "y": {"foreach": ["d"], "bounds": {"lower": 0, "upper": 1}},
        },
        "objective": {"sense": "minimize", "expression": "sum(x) + sum(y)"},
    }
    WHERE_DATA: dict[str, Any] = {
        "t": TT,
        "s": S,
        "d": DAYS,
        "season_of": pd.Series(["a", "a", "b"], index=TT[:3]),
        "other_of": pd.Series(["a", "b", "b", "a"], index=TT),
        "tag": pd.Series(["p", "q"], index=TT[:2]),
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
            "bp": {"dtype": "int"},
        },
        "parameters": {
            "p_max": {"dims": ["generator"]},
            "load": {"dims": ["snapshot"]},
            "bp_x": {"dims": ["generator", "bp"]},
            "bp_y": {"dims": ["generator", "bp"]},
        },
        "variables": {
            "p": {
                "foreach": ["snapshot", "generator"],
                "bounds": {"lower": 0, "upper": "p_max"},
            },
            "op_cost": {"foreach": ["snapshot", "generator"], "bounds": {"lower": 0}},
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
                "foreach": ["snapshot"],
                "expression": "sum(p, over=generator) == load",
            }
        },
        "objective": {"sense": "minimize", "expression": "sum(op_cost)"},
    }
    MASKED_CURVE_SPEC = with_(
        CURVE_SPEC,
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
