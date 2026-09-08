"""
Building declarations from math-spec programs: variables, constraints, the
objective, SOS-constrained curves, coverage refusals and side-swapped
expressions.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from conftest import (  # noqa: E402, F401
    CURVE_DATA,
    CURVE_SPEC,
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    FULL_X,
    FULL_Y,
    GENERATOR,
    WHERE_DATA,
    WHERE_SPEC,
    solved,
    with_,
    yaml_dict,
)
from linopy import Model  # noqa: E402
from linopy.spec import SpecDataError  # noqa: E402
from linopy.spec.testing import synthetic_sources  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

EXAMPLES_DIR = os.environ.get("MATH_SPEC_EXAMPLES")
EXAMPLES = (
    sorted(glob.glob(f"{EXAMPLES_DIR}/*.yaml") + glob.glob(f"{EXAMPLES_DIR}/*/*.yaml"))
    if EXAMPLES_DIR
    else []
)


@pytest.mark.skipif(
    not EXAMPLES, reason="set MATH_SPEC_EXAMPLES to a math-spec examples directory"
)
@pytest.mark.parametrize(
    "path", EXAMPLES, ids=lambda p: str(Path(p).relative_to(EXAMPLES_DIR or ""))
)
def test_every_math_spec_example_builds_and_solves(path: str) -> None:
    if "/symbols/" in path:
        pytest.skip("typesetting input, not a spec")
    program = math_spec.to_program(path)
    m = solved(path, synthetic_sources(program), retain="all")
    assert m.nvars == sum(int(m.variables[v].labels.count()) for v in program.variables)
    assert m.termination_condition in ("optimal", "infeasible")


def test_the_dispatch_example_solves_and_its_expressions_fold() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    assert m.objective.value == pytest.approx(2500.0)
    xr.testing.assert_allclose(m.solution["p"], DISPATCH_P)
    spend = m.spec.expressions["spend"].solution
    xr.testing.assert_allclose(
        spend, (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )
    usage = m.spec.expressions["usage"].solution
    xr.testing.assert_allclose(usage, (DISPATCH_P / [100.0, 200.0]).rename("usage"))
    assert (
        set(m.spec.expressions) == {"spend", "usage"} and len(m.spec.expressions) == 2
    )
    assert set(m.spec.parameters.data_vars) == {"cost", "p_max"}
    assert m.spec.coords["generator"].equals(GENERATOR)


# ---------------------------------------------------------------------------
# absence: a missing row by position
# ---------------------------------------------------------------------------

T = pd.Index([0, 1, 2], name="t")
SPARSE_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int"}},
    "parameters": {"c": {"dims": ["t"]}, "w": {"dims": ["t"]}},
    "variables": {"x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": 10}}},
    "constraints": {"cap": {"foreach": ["t"], "expression": "w * x <= c"}},
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}
FULL_W = pd.Series([1.0, 1.0, 1.0], index=T)
FULL_C = pd.Series([0.0, 4.0, 5.0], index=T)
HOLE_AT_0 = pd.Series([4.0, 5.0], index=T[1:])
W_HOLE_AT_0 = pd.Series([1.0, 1.0], index=T[1:])

NO_W_CONSTRAINT = {"cap": {"foreach": ["t"], "expression": "x <= c"}}


@pytest.mark.parametrize(
    ("spec", "data", "match"),
    [
        pytest.param(
            SPARSE_SPEC,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "constraint 'cap'.*parameter 'w' is used as a coefficient",
            id="coefficient-in-a-constraint",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints=NO_W_CONSTRAINT,
                objective={"sense": "maximize", "expression": "sum(w * x, over=t)"},
            ),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "the objective.*parameter 'w' is used as a coefficient",
            id="coefficient-in-the-objective",
        ),
        pytest.param(
            with_(SPARSE_SPEC, constraints=NO_W_CONSTRAINT, expressions={"e": "w * x"}),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "expression 'e'.*parameter 'w' is used as a coefficient",
            id="coefficient-in-a-named-expression",
        ),
        pytest.param(
            SPARSE_SPEC,
            {"w": FULL_W, "c": HOLE_AT_0},
            "constraint 'cap'.*covers 1 fewer",
            id="constant-side",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                variables={
                    "x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": "c"}}
                },
            ),
            {"w": FULL_W, "c": HOLE_AT_0},
            "variable 'x': 1 rows have NULL bounds",
            id="bound",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints={"cap": {"foreach": ["t"], "expression": "x / w <= c"}},
            ),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "constraint 'cap'.*divisor",
            id="divisor-in-a-constraint",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints=NO_W_CONSTRAINT,
                objective={"sense": "maximize", "expression": "sum(x / w, over=t)"},
            ),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "the objective.*divisor",
            id="divisor-in-the-objective",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints=NO_W_CONSTRAINT,
                expressions={"ratio": "x / w"},
            ),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "expression 'ratio'.*divisor",
            id="divisor-in-a-named-expression",
        ),
    ],
)
def test_a_missing_row_is_refused_wherever_it_is_used(
    spec: dict[str, Any], data: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {"t": T, **data})


def test_a_masked_variable_bound_needs_no_row_where_it_is_masked() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={
            **SPARSE_SPEC["parameters"],
            "live": {"dims": ["t"], "dtype": "bool"},
        },
        variables={
            "x": {
                "foreach": ["t"],
                "where": "live",
                "bounds": {"lower": 0, "upper": "c"},
            }
        },
        constraints={
            "cap": {"foreach": ["t"], "where": "live", "expression": "w * x <= c"}
        },
    )
    live = pd.Series([True, True], index=T[1:])
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": HOLE_AT_0, "live": live})
    assert int(m.variables["x"].labels.count()) == 3
    assert int((m.variables["x"].labels != -1).sum()) == 2


WHERE_MASKS = with_(
    SPARSE_SPEC,
    constraints={"cap": {**SPARSE_SPEC["constraints"]["cap"], "where": "w"}},
)


@pytest.mark.parametrize(
    ("spec", "data", "objective"),
    [
        pytest.param(SPARSE_SPEC, {"w": FULL_W, "c": FULL_C}, 9.0, id="fully-covered"),
        pytest.param(
            WHERE_MASKS,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            19.0,
            id="a-where-masks-a-coefficient-hole",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints={
                    "cap": {**SPARSE_SPEC["constraints"]["cap"], "where": "c"}
                },
            ),
            {"w": FULL_W, "c": HOLE_AT_0},
            19.0,
            id="a-where-masks-a-constant-side-hole",
        ),
    ],
)
def test_a_covered_or_masked_row_builds(
    spec: dict[str, Any], data: dict[str, Any], objective: float
) -> None:
    m = solved(spec, {"t": T, **data})
    assert m.objective.value == pytest.approx(objective)


F = pd.Index(["a", "b"], name="f")
ENVELOPE_SPEC: dict[str, Any] = {
    "dimensions": {"f": {"dtype": "str"}},
    "parameters": {"gate": {"dims": ["f"], "dtype": "bool"}, "relmax": {"dims": ["f"]}},
    "variables": {
        "x": {"foreach": ["f"], "bounds": {"lower": 0, "upper": 100}},
        "size": {
            "foreach": ["f"],
            "where": "gate",
            "bounds": {"lower": 0, "upper": 50},
        },
    },
    "constraints": {
        "envelope": {"foreach": ["f"], "expression": "x - relmax * size <= 0"}
    },
    "objective": {"sense": "maximize", "expression": "sum(x, over=f)"},
}
ENVELOPE_DATA: dict[str, Any] = {
    "f": F,
    "gate": pd.Series([True], index=F[:1]),
    "relmax": pd.Series([0.5, 0.5], index=F),
}
DEFINED_SPEC = with_(
    ENVELOPE_SPEC,
    constraints={
        "envelope": {
            "foreach": ["f"],
            "where": "size",
            "expression": "x - relmax * size <= 0",
        },
        "pinned": {"foreach": ["f"], "where": "NOT size", "expression": "x <= 0"},
    },
)


@pytest.mark.parametrize(
    ("spec", "unsized"),
    [
        pytest.param(ENVELOPE_SPEC, 100.0, id="an-absent-term-drops-the-row"),
        pytest.param(
            DEFINED_SPEC, 0.0, id="a-bare-variable-in-a-where-asks-whether-it-exists"
        ),
    ],
)
def test_an_absent_variable_takes_its_row_unless_a_where_says_otherwise(
    spec: dict[str, Any], unsized: float
) -> None:
    m = solved(spec, ENVELOPE_DATA)
    x = m.solution["x"]
    assert float(x.sel(f="a")) == pytest.approx(25.0)
    assert float(x.sel(f="b")) == pytest.approx(unsized)


SCALAR_SWITCH: dict[str, Any] = {
    "dimensions": {"i": {"dtype": "int"}},
    "parameters": {"on": {"dims": [], "dtype": "bool"}},
    "variables": {
        "x": {"foreach": ["i"], "bounds": {"lower": 1, "upper": 5}, "where": "on"},
        "y": {"foreach": ["i"], "bounds": {"lower": 2, "upper": 5}},
    },
    "objective": {"sense": "minimize", "expression": "sum(x) + sum(y)"},
}


@pytest.mark.parametrize(("on", "objective"), [(True, 6.0), (False, 4.0)])
def test_a_scalar_where_gates_a_whole_variable(on: bool, objective: float) -> None:
    m = solved(SCALAR_SWITCH, {"i": [1, 2], "on": on})
    assert m.objective.value == pytest.approx(objective)


def test_a_dimension_with_no_members_builds_no_row() -> None:
    spec = with_(
        SPARSE_SPEC,
        constraints={"budget": {"foreach": [], "expression": "sum(x, over=t) <= 10"}},
    )
    empty = pd.Index([], name="t", dtype=int)
    m = Model.from_spec(
        spec,
        {
            "t": empty,
            "w": pd.Series([], index=empty, dtype=float),
            "c": pd.Series([], index=empty, dtype=float),
        },
    )
    assert "budget" not in m.constraints


@pytest.mark.parametrize(
    ("absence", "masked_reads_nan"),
    [("undefined", True), ("zero", False)],
    ids=["undefined-leaves-a-masked-slot-nan", "zero-fills-a-masked-slot"],
)
def test_a_fold_reads_a_masked_slot_the_way_its_absence_says(
    absence: str, masked_reads_nan: bool
) -> None:
    spec = yaml_dict()
    spec["variables"]["p"]["absence"] = absence
    spec["expressions"] = {"spend_by_unit": "p * cost"}
    data = {**DISPATCH_DATA, "p_max": pd.Series([200.0, 0.0], index=GENERATOR)}
    spend = solved(spec, data).spec.expressions["spend_by_unit"].solution
    masked = spend.sel(generator="gas")
    assert bool(masked.isnull().all()) is masked_reads_nan
    if not masked_reads_nan:
        assert float(masked.max()) == pytest.approx(0.0)
    assert not bool(spend.sel(generator="wind").isnull().any())


# ---------------------------------------------------------------------------
# piecewise curves as SOS constraints
# ---------------------------------------------------------------------------


def test_a_sos2_curve_is_built_as_a_special_ordered_set() -> None:
    spec = with_(
        CURVE_SPEC,
        piecewise={
            "cost_curve": {
                "over": "bp",
                "links": [["p", "bp_x"], ["op_cost", "bp_y"]],
                "method": "sos2",
            }
        },
    )
    m = Model.from_spec(spec, {**CURVE_DATA, "bp_x": FULL_X, "bp_y": FULL_Y})
    assert m.variables["cost_curve_lam"].attrs["sos_type"] == 2
    # math-spec lowers the block into ordinary declarations, so none of it is drift.
    assert not m.spec.unspecified


# ---------------------------------------------------------------------------
# a constant on the left is the same row, a power hides nothing
# ---------------------------------------------------------------------------


def test_a_constant_on_the_left_is_the_same_row() -> None:
    flipped = with_(
        SPARSE_SPEC, constraints={"cap": {"foreach": ["t"], "expression": "c >= w * x"}}
    )
    m = solved(flipped, {"t": T, "w": FULL_W, "c": FULL_C})
    assert m.objective.value == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        pytest.param(
            "x <= c ** 2",
            "constraint 'cap'.*covers 1 fewer",
            id="constant-side-under-a-power",
        ),
        pytest.param(
            "x / (c ** 2) <= 1", "constraint 'cap'.*divisor", id="divisor-under-a-power"
        ),
    ],
)
def test_a_parameter_under_a_power_is_still_checked_for_coverage(
    expression: str, match: str
) -> None:
    spec = with_(
        SPARSE_SPEC, constraints={"cap": {"foreach": ["t"], "expression": expression}}
    )
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {"t": T, "w": FULL_W, "c": HOLE_AT_0})


def test_an_operator_under_a_power_keeps_its_parameters_retained() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={**SPARSE_SPEC["parameters"], "lag": {"dims": [], "dtype": "int"}},
        expressions={"e": "shift(c, over=t, offset=lag, edge=0) ** 1"},
    )
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": FULL_C, "lag": 1})
    assert {"c", "lag"} <= set(m.spec.parameters.data_vars)
    xr.testing.assert_allclose(
        m.spec.expressions["e"].solution,
        xr.DataArray([0.0, 0.0, 4.0], coords={"t": T}, name="e"),
    )
