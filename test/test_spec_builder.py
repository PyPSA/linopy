"""
Building declarations from mathspec programs: variables, constraints, the
objective, SOS-constrained curves, coverage refusals, side-swapped
expressions and the assumptions checked against the data.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")
pytest.importorskip("highspy")
yaml = pytest.importorskip("yaml")

from spec_helpers import (  # noqa: E402
    CURVE_DATA,
    CURVE_SPEC,
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLES_DIR,
    FULL_X,
    FULL_Y,
    GENERATOR,
    MASKED_CURVE_SPEC,
    RAGGED_X,
    RAGGED_Y,
    UNITS,
    curve,
    solved,
    with_,
    yaml_dict,
)

import linopy  # noqa: E402
from linopy import Model  # noqa: E402
from linopy.spec import SpecDataError, evaluate, parameter  # noqa: E402
from linopy.spec.api import lower  # noqa: E402
from linopy.testing import synthetic_sources  # noqa: E402

MULTI_COLUMN = "linopy reads no relation of more than one key or value column"
REFUSED: dict[str, tuple[type[Exception], str]] = {
    "operators/at_columns.yaml": (SpecDataError, MULTI_COLUMN),
    "operators/sum_by_columns.yaml": (SpecDataError, MULTI_COLUMN),
    "operators/sum_by_column_lists.yaml": (SpecDataError, MULTI_COLUMN),
    "operators/shift_by_parameter.yaml": (
        SpecDataError,
        "the synthetic data leaves a 0 >= demand row",
    ),
    "pypsa.yaml": (
        SpecDataError,
        "bare and multi-key relations, a quadratic constraint",
    ),
}


def example(path: str) -> Any:
    """*path* as a sweep case, expected to raise where :data:`REFUSED` names it."""
    name = Path(path).relative_to(EXAMPLES_DIR or "").as_posix()
    marks = []
    if name in REFUSED:
        raises, reason = REFUSED[name]
        marks = [pytest.mark.xfail(strict=True, raises=raises, reason=reason)]
    return pytest.param(Path(path), id=name, marks=marks)


EXAMPLES = (
    sorted(
        glob.glob(f"{EXAMPLES_DIR}/*.yaml")
        + glob.glob(f"{EXAMPLES_DIR}/operators/*.yaml")
        + glob.glob(f"{EXAMPLES_DIR}/ports/*.yaml")
    )
    if EXAMPLES_DIR
    else []
)


@pytest.mark.skipif(
    not EXAMPLES, reason="set MATHSPEC_EXAMPLES to a mathspec examples directory"
)
@pytest.mark.parametrize("path", [example(p) for p in EXAMPLES])
def test_every_mathspec_example_builds_and_solves(path: Path) -> None:
    """Every whole mathspec example builds and solves on synthetic data, bar the ones :data:`REFUSED` names, which are refused."""
    program = lower(path)
    m = solved(path, synthetic_sources(path))
    assert m.nvars == sum(int(m.variables[v].labels.count()) for v in program.variables)
    assert m.termination_condition in ("optimal", "infeasible")


def test_the_dispatch_example_solves_and_its_expressions_fold() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    assert m.objective.value == pytest.approx(2500.0)
    xr.testing.assert_allclose(m.solution["p"], DISPATCH_P)
    folded = evaluate(yaml_dict(), m)
    assert list(folded) == ["spend", "usage"]
    xr.testing.assert_allclose(
        folded["spend"], (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )
    xr.testing.assert_allclose(
        folded["usage"], (DISPATCH_P / [100.0, 200.0]).rename("usage")
    )
    assert set(m.parameters.data_vars) == set(lower(yaml_dict()).parameters)
    assert m.parameters.indexes["generator"].equals(GENERATOR)


# ---------------------------------------------------------------------------
# absence: a missing row by position
# ---------------------------------------------------------------------------

T = pd.Index([0, 1, 2], name="t")
SPARSE_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}},
    "parameters": {"c": {"dims": ["t"]}, "w": {"dims": ["t"]}},
    "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}}},
    "constraints": {"cap": {"dims": ["t"], "expression": "w * x <= c"}},
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}
FULL_W = pd.Series([1.0, 1.0, 1.0], index=T)
FULL_C = pd.Series([0.0, 4.0, 5.0], index=T)
HOLE_AT_0 = pd.Series([4.0, 5.0], index=T[1:])
W_HOLE_AT_0 = pd.Series([1.0, 1.0], index=T[1:])


C_HOLE_AT_2 = FULL_C.iloc[:2]
LATER = {"dims": ["t"], "where": "t > 0"}
X_LATER = {**LATER, "bounds": {"lower": 0, "upper": 10}}
CAP_LATER = with_(
    SPARSE_SPEC, constraints={"cap": {**LATER, "expression": "w * x <= c"}}
)
SHIFTED = with_(
    SPARSE_SPEC,
    constraints={
        "cap": {**LATER, "expression": "x <= shift(c, along=t, offset=1, edge=0)"}
    },
)
BOUND_LATER = with_(
    CAP_LATER,
    variables={"x": {**LATER, "bounds": {"lower": 0, "upper": "c"}}},
)
OBJECTIVE_READS_W = with_(
    CAP_LATER, objective={"sense": "maximize", "expression": "sum(w * x, over=t)"}
)
BESIDE_A_VACATED_SHIFT = with_(
    SPARSE_SPEC,
    constraints={
        "cap": {
            "dims": ["t"],
            "expression": "w * x + shift(x, along=t, offset=1) <= c",
        }
    },
)
NAMED_WX = with_(
    SPARSE_SPEC,
    variables={"x": X_LATER},
    constraints={"cap": {**LATER, "expression": "wx <= c"}},
    expressions={"wx": "w * x"},
)


def windowed(where: str) -> dict[str, Any]:
    cap = {
        "dims": ["t"],
        "where": where,
        "expression": "x <= sum_back(c, along=t, window=2)",
    }
    return with_(SPARSE_SPEC, constraints={"cap": cap})


def in_region(when: str) -> dict[str, Any]:
    region = {
        "dims": ["t"],
        "cases": {"on": {"when": when, "expression": "c"}},
        "otherwise": 0,
    }
    return with_(CAP_LATER, expressions={"region": region})


@pytest.mark.parametrize(
    ("spec", "data", "match"),
    [
        pytest.param(
            SPARSE_SPEC,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "constraint 'cap' reads parameter 'w' at t=0",
            id="coefficient",
        ),
        pytest.param(
            SPARSE_SPEC,
            {"w": FULL_W, "c": HOLE_AT_0},
            "constraint 'cap' reads parameter 'c' at t=0",
            id="constant-side",
        ),
        pytest.param(
            SHIFTED,
            {"w": FULL_W, "c": HOLE_AT_0},
            "constraint 'cap' reads parameter 'c' at t=0",
            id="shifted-onto-a-live-row",
        ),
        pytest.param(
            OBJECTIVE_READS_W,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "the objective reads parameter 'w' at t=0",
            id="objective",
        ),
        pytest.param(
            windowed("t > 0"),
            {"w": FULL_W, "c": HOLE_AT_0},
            "constraint 'cap' reads parameter 'c' at t=0",
            id="window",
        ),
        pytest.param(
            in_region("t >= 0"),
            {"w": FULL_W, "c": FULL_C.iloc[1:]},
            "expression 'region' reads parameter 'c' at t=0",
            id="cases-region",
        ),
    ],
)
def test_a_refused_row_a_live_row_reads_is_refused(
    spec: dict[str, Any], data: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=rf"{match}, where it has no row"):
        Model.from_spec(spec, {"t": T, **data})


@pytest.mark.parametrize(
    ("spec", "data", "name", "hole"),
    [
        pytest.param(
            CAP_LATER, {"w": W_HOLE_AT_0, "c": HOLE_AT_0}, "w", 0, id="masked-row"
        ),
        pytest.param(
            SHIFTED, {"w": FULL_W, "c": C_HOLE_AT_2}, "c", 2, id="shifted-past-the-end"
        ),
        pytest.param(
            BOUND_LATER, {"w": FULL_W, "c": HOLE_AT_0}, "c", 0, id="masked-bound"
        ),
        pytest.param(
            BESIDE_A_VACATED_SHIFT,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "w",
            0,
            id="beside-a-vacated-shift",
        ),
        pytest.param(
            with_(OBJECTIVE_READS_W, variables={"x": X_LATER}),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "w",
            0,
            id="objective-over-a-masked-variable",
        ),
        pytest.param(
            NAMED_WX,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "w",
            0,
            id="named-expression-over-a-masked-variable",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                parameters={"p": {"dims": ["t"], "missing": "absent"}},
                constraints={"cap": {"dims": ["t"], "expression": "w * p * x <= c"}},
            ),
            {"w": W_HOLE_AT_0, "p": W_HOLE_AT_0, "c": FULL_C},
            "w",
            0,
            id="beside-an-absent-row",
        ),
        pytest.param(
            windowed("t < 2"), {"w": FULL_W, "c": C_HOLE_AT_2}, "c", 2, id="window"
        ),
        pytest.param(
            in_region("t > 0"),
            {"w": FULL_W, "c": FULL_C.iloc[1:]},
            "c",
            0,
            id="cases-region",
        ),
    ],
)
def test_a_refused_row_no_live_row_reads_builds_and_stays_a_hole(
    spec: dict[str, Any], data: dict[str, Any], name: str, hole: int
) -> None:
    m = Model.from_spec(spec, {"t": T, **data})
    values = parameter(spec, m, name)
    assert values.isnull().values.tolist() == [t == hole for t in T]


def test_a_named_expression_reading_a_refused_row_is_refused_by_build_and_evaluate() -> (
    None
):
    data = {"t": T, "w": W_HOLE_AT_0, "c": FULL_C}
    m = Model.from_spec(CAP_LATER, data)
    spec = with_(CAP_LATER, expressions={"weighted": "w * c"})
    match = "expression 'weighted' reads parameter 'w' at t=0, where it has no row"
    with pytest.raises(SpecDataError, match=match):
        evaluate(spec, m, "weighted")
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, data)
    assert parameter(spec, m, "w").isnull().values.tolist() == [True, False, False]


GROUPS = pd.Index(["p", "q"], name="s")
ABSENT_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}, "s": {"dtype": "str"}},
    "relations": {"grp": {"key": "t", "values": "s"}},
    "parameters": {
        "a": {"dims": ["t"], "missing": "absent"},
        "z": {"dims": ["s"], "missing": "absent"},
    },
    "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}}},
    "objective": {"sense": "maximize", "expression": "sum(x)"},
}
ABSENT_DATA: dict[str, Any] = {
    "t": T,
    "s": GROUPS,
    "grp": pd.Series(["p", "p", "q"], index=T),
    "a": pd.Series([2.0, 3.0], index=T[1:]),
    "z": pd.Series([5.0], index=GROUPS[1:]),
}


def absent_row(expression: str) -> dict[str, Any]:
    return with_(
        ABSENT_SPEC, constraints={"row": {"dims": ["t"], "expression": expression}}
    )


@pytest.mark.parametrize(
    ("expression", "kept"),
    [
        pytest.param("x + a <= 10", [1, 2], id="added"),
        pytest.param("a * x <= 4", [1, 2], id="multiplied"),
        pytest.param("x / a <= 1", [1, 2], id="divisor"),
        pytest.param("x <= a", [1, 2], id="constant-side"),
        pytest.param("x <= shift(a, along=t, offset=1, edge=0)", [0, 2], id="shift"),
        pytest.param("x <= at(z, by=grp[s])", [2], id="at"),
        pytest.param("sum_back(a * x, along=t, window=2) <= 4", [1, 2], id="sum-back"),
    ],
)
def test_an_absent_row_masks_the_row_it_reaches(
    expression: str, kept: list[int]
) -> None:
    row = Model.from_spec(absent_row(expression), ABSENT_DATA).constraints["row"]
    assert T[(row.labels != -1).to_numpy()].tolist() == kept


def test_an_absent_summand_is_dropped() -> None:
    spec = with_(
        ABSENT_SPEC,
        constraints={"row": {"dims": [], "expression": "sum(a * x, over=t) <= 4"}},
    )
    m = solved(spec, ABSENT_DATA)
    assert m.objective.value == pytest.approx(12.0)


def test_an_absent_bound_leaves_its_side_open() -> None:
    spec = with_(
        ABSENT_SPEC,
        variables={"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": "a"}}},
    )
    upper = Model.from_spec(spec, ABSENT_DATA).variables["x"].upper
    assert upper.values.tolist() == [np.inf, 2.0, 3.0]


def test_a_masked_variable_bound_needs_no_row_where_it_is_masked() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={
            "c": {"dims": ["t"], "missing": "absent"},
            "live": {"dims": ["t"], "dtype": "bool", "missing": "absent"},
        },
        variables={
            "x": {
                "dims": ["t"],
                "where": "live",
                "bounds": {"lower": 0, "upper": "c"},
            }
        },
        constraints={
            "cap": {"dims": ["t"], "where": "live", "expression": "w * x <= c"}
        },
    )
    live = pd.Series([True, True], index=T[1:])
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": HOLE_AT_0, "live": live})
    assert int(m.variables["x"].labels.count()) == 3
    assert int((m.variables["x"].labels != -1).sum()) == 2


WHERE_MASKS = with_(
    SPARSE_SPEC,
    parameters={"c": {"dims": ["t"]}, "w": {"dims": ["t"], "missing": "absent"}},
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
                parameters={
                    "c": {"dims": ["t"], "missing": "absent"},
                    "w": {"dims": ["t"]},
                },
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
    "parameters": {
        "gate": {"dims": ["f"], "dtype": "bool", "missing": "absent"},
        "relmax": {"dims": ["f"]},
    },
    "variables": {
        "x": {"dims": ["f"], "bounds": {"lower": 0, "upper": 100}},
        "size": {
            "dims": ["f"],
            "where": "gate",
            "bounds": {"lower": 0, "upper": 50},
        },
    },
    "constraints": {
        "envelope": {"dims": ["f"], "expression": "x - relmax * size <= 0"}
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
            "dims": ["f"],
            "where": "size",
            "expression": "x - relmax * size <= 0",
        },
        "pinned": {"dims": ["f"], "where": "NOT size", "expression": "x <= 0"},
    },
)


def sized(missing: str, expression: str) -> dict[str, Any]:
    """:data:`ENVELOPE_SPEC` with a row over ``size`` alone, which the data empties at ``f='b'``."""
    spec = with_(
        ENVELOPE_SPEC,
        variables={"size": {**ENVELOPE_SPEC["variables"]["size"], "missing": missing}},
    )
    spec["constraints"] = {"sized": {"dims": ["f"], "expression": expression}}
    return spec


def test_a_bare_variable_in_a_where_asks_whether_it_exists() -> None:
    m = solved(DEFINED_SPEC, ENVELOPE_DATA)
    x = m.solution["x"]
    assert float(x.sel(f="a")) == pytest.approx(25.0)
    assert float(x.sel(f="b")) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        pytest.param(ENVELOPE_SPEC, "envelope", id="an-absent-term"),
        pytest.param(
            sized("absent", "size <= relmax"), "sized", id="an-absent-variable"
        ),
    ],
)
def test_absence_takes_the_row_with_it(spec: dict[str, Any], name: str) -> None:
    m = Model.from_spec(spec, ENVELOPE_DATA)
    assert int(m.constraints[name].labels.sel(f="b")) == -1


def test_a_row_the_data_emptied_of_variables_is_refused() -> None:
    """A zero absence leaves ``0 <= relmax`` standing, which linopy cannot carry and a solver would never see."""
    with pytest.raises(
        SpecDataError, match=r"(?s)constraint 'sized'.*Supply the rows of the variables"
    ):
        Model.from_spec(sized("neutral", "size <= relmax"), ENVELOPE_DATA)


def quotient(expression: str, where: str | None = None) -> dict[str, Any]:
    """A row over ``t`` reading the data ``a`` and ``b``, which :data:`ZERO_AT_0` holds at 0 on ``t=0``."""
    row: dict[str, Any] = {"dims": ["t"], "expression": expression}
    return {
        "dimensions": {"t": {"dtype": "int", "ordered": True}},
        "parameters": {"a": {"dims": ["t"]}, "b": {"dims": ["t"]}},
        "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}}},
        "constraints": {"row": row if where is None else {**row, "where": where}},
        "objective": {"sense": "minimize", "expression": "sum(x, over=t)"},
    }


ZERO_AT_0 = {"t": T, "a": pd.Series([0.0, 1.0, 1.0], index=T), "b": FULL_C}
INF_AT_0 = {"t": T, "a": pd.Series([float("inf"), 1.0, 1.0], index=T), "b": FULL_C}


SHIFTED_TERM = "shift(a, along=t, offset=1, edge=0)"


@pytest.mark.parametrize(
    ("expression", "data", "at"),
    [
        pytest.param("x + a / b >= 1", ZERO_AT_0, "t=0", id="added"),
        pytest.param("x * (a / b) >= 1", ZERO_AT_0, "t=0", id="coefficient-right"),
        pytest.param("a / b * x >= 1", ZERO_AT_0, "t=0", id="coefficient-left"),
        pytest.param("x + (a - a) >= 1", INF_AT_0, "t=0", id="inf-minus-inf"),
        pytest.param("x + a - a >= 1", INF_AT_0, "t=0", id="beside-a-term"),
        pytest.param("a - x - a >= 1", INF_AT_0, "t=0", id="across-a-term"),
        pytest.param("x + a >= a", INF_AT_0, "t=0", id="across-sides"),
        pytest.param(
            "x + sum(a, over=t) - sum(a, over=t) >= 1",
            INF_AT_0,
            "t=0",
            id="sums-beside-a-term",
        ),
        pytest.param(
            f"x + {SHIFTED_TERM} - {SHIFTED_TERM} >= 1",
            INF_AT_0,
            "t=1",
            id="shifts-beside-a-term",
        ),
    ],
)
def test_arithmetic_that_makes_data_non_finite_is_refused(
    expression: str, data: dict[str, Any], at: str
) -> None:
    with pytest.raises(SpecDataError, match=rf"constraint 'row'.*not finite at {at}"):
        Model.from_spec(quotient(expression), data)


def test_arithmetic_is_not_refused_on_a_row_masked_out() -> None:
    m = Model.from_spec(quotient("x + a / b >= 1", where="b > 0"), ZERO_AT_0)
    assert int(m.constraints["row"].labels.sel(t=0)) == -1


def test_a_dead_row_of_zero_absences_against_a_zero_side_is_only_warned_about() -> None:
    with pytest.warns(UserWarning, match=r"constraint 'sized'.*trivially true"):
        m = solved(sized("neutral", "size >= 0"), ENVELOPE_DATA)
    assert m.termination_condition == "optimal"
    assert (m.constraints["sized"].vars.sel(f="b") == -1).all()


def test_a_coefficient_needs_no_row_where_its_variable_is_absent() -> None:
    spec = with_(
        DEFINED_SPEC,
        parameters={"relmax": {"dims": ["f"], "missing": "absent"}},
        expressions={"sized": "relmax * size"},
    )
    data = {**ENVELOPE_DATA, "relmax": pd.Series([0.5], index=F[:1])}
    m = solved(spec, data)
    assert float(m.solution["x"].sel(f="a")) == pytest.approx(25.0)
    sized = evaluate(spec, m, "sized")["sized"]
    assert float(sized.sel(f="a")) == pytest.approx(25.0)


SCALAR_SWITCH: dict[str, Any] = {
    "dimensions": {"i": {"dtype": "int"}},
    "parameters": {"on": {"dims": [], "dtype": "bool"}},
    "variables": {
        "x": {"dims": ["i"], "bounds": {"lower": 1, "upper": 5}, "where": "on"},
        "y": {"dims": ["i"], "bounds": {"lower": 2, "upper": 5}},
    },
    "objective": {"sense": "minimize", "expression": "sum(x) + sum(y)"},
}


@pytest.mark.parametrize(("on", "objective"), [(True, 6.0), (False, 4.0)])
def test_a_scalar_where_gates_a_whole_variable(on: bool, objective: float) -> None:
    m = solved(SCALAR_SWITCH, {"i": [1, 2], "on": on})
    assert m.objective.value == pytest.approx(objective)


@pytest.mark.parametrize("expression", ["y <= 5", "5 >= y"], ids=["left", "flipped"])
def test_a_constraint_is_framed_in_its_declared_order(expression: str) -> None:
    spec = {
        "dimensions": {"t": {"dtype": "int"}, "g": {"dtype": "str"}},
        "variables": {"y": {"dims": ["g", "t"], "bounds": {"lower": 0}}},
        "constraints": {"k": {"dims": ["t", "g"], "expression": expression}},
    }
    m = Model.from_spec(spec, {"t": T, "g": ["a", "b"]})
    assert m.constraints["k"].labels.dims == ("t", "g")
    assert m.constraints["k"].lhs.coord_dims == ("t", "g")


@pytest.mark.parametrize("row", ["sum(x, over=t) >= 5", "sum(x, over=t) <= 0"])
def test_a_dimension_with_no_members_empties_a_summed_row(row: str) -> None:
    spec = with_(SPARSE_SPEC, constraints={"budget": {"dims": [], "expression": row}})
    empty = pd.Index([], name="t", dtype=int)
    none = pd.Series([], index=empty, dtype=float)
    with pytest.raises(
        SpecDataError, match="constraint 'budget'.*hold no variable term"
    ):
        Model.from_spec(spec, {"t": empty, "w": none, "c": none})


def test_a_neutral_hole_in_a_where_reads_as_false() -> None:
    spec = {
        "dimensions": {"f": {"dtype": "str"}},
        "parameters": {"count": {"dims": ["f"], "dtype": "int", "missing": "neutral"}},
        "variables": {
            "x": {
                "dims": ["f"],
                "where": "count == 0",
                "bounds": {"lower": 0, "upper": 1},
            }
        },
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    m = Model.from_spec(spec, {"f": ["a", "b"], "count": pd.Series({"a": 0})})
    assert (m.variables["x"].labels != -1).values.tolist() == [True, False]


@pytest.mark.parametrize(
    ("missing", "masked_reads_nan"),
    [("absent", True), ("neutral", False)],
    ids=["undefined-leaves-a-masked-slot-nan", "zero-fills-a-masked-slot"],
)
def test_a_fold_reads_a_masked_slot_the_way_its_absence_says(
    missing: str, masked_reads_nan: bool
) -> None:
    spec = yaml_dict()
    spec["variables"]["p"]["missing"] = missing
    spec["expressions"] = {"spend_by_unit": "p * cost"}
    data = {**DISPATCH_DATA, "p_max": pd.Series([200.0, 0.0], index=GENERATOR)}
    spend = evaluate(spec, solved(spec, data), "spend_by_unit")["spend_by_unit"]
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


# ---------------------------------------------------------------------------
# a constant on the left is the same row, a power hides nothing
# ---------------------------------------------------------------------------


def test_a_constant_on_the_left_is_the_same_row() -> None:
    flipped = with_(
        SPARSE_SPEC, constraints={"cap": {"dims": ["t"], "expression": "c >= w * x"}}
    )
    m = solved(flipped, {"t": T, "w": FULL_W, "c": FULL_C})
    assert m.objective.value == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("expression", "upper"),
    [
        pytest.param("x <= c", 10, id="constant-side"),
        pytest.param("x <= c ** 2", 10, id="constant-side-under-a-power"),
        pytest.param("x <= -(c ** 2)", 10, id="constant-side-under-a-negation"),
        pytest.param("x / c <= 1", 10, id="divisor"),
        pytest.param("x / (c ** 2) <= 1", 10, id="divisor-under-a-power"),
        pytest.param("x <= 10", "c", id="bound"),
    ],
)
def test_a_neutral_hole_where_no_neutral_value_exists_is_refused(
    expression: str, upper: Any
) -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={"w": {"dims": ["t"]}, "c": {"dims": ["t"], "missing": "neutral"}},
        variables={"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": upper}}},
        constraints={"cap": {"dims": ["t"], "expression": expression}},
    )
    with pytest.raises(SpecDataError, match="parameter 'c' is missing: neutral"):
        Model.from_spec(spec, {"t": T, "w": FULL_W, "c": HOLE_AT_0})


def test_an_operator_under_a_power_keeps_the_parameters_it_reads() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={**SPARSE_SPEC["parameters"], "lag": {"dims": [], "dtype": "int"}},
        expressions={"e": "shift(c, along=t, offset=lag, edge=0) ** 1"},
    )
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": FULL_C, "lag": 1})
    assert {"c", "lag"} <= set(m.parameters.data_vars)
    xr.testing.assert_allclose(
        evaluate(spec, m, "e")["e"],
        xr.DataArray([0.0, 0.0, 4.0], coords={"t": T}, name="e"),
    )


# ---------------------------------------------------------------------------
# what linopy cannot build
# ---------------------------------------------------------------------------

QUADRATIC_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
        "y": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
    },
    "constraints": {"cap": {"dims": ["t"], "expression": "x * y <= 5"}},
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}


def test_a_quadratic_constraint_is_refused() -> None:
    with pytest.raises(
        NotImplementedError, match="quadratic term in the objective only"
    ):
        Model.from_spec(QUADRATIC_SPEC, {"t": T})


def test_a_quadratic_objective_builds() -> None:
    spec = with_(
        QUADRATIC_SPEC,
        constraints={"cap": {"dims": ["t"], "expression": "x + y <= 5"}},
        objective={"sense": "maximize", "expression": "sum(x * y, over=t)"},
    )
    m = Model.from_spec(spec, {"t": T})
    assert isinstance(m.objective.expression, linopy.QuadraticExpression)


def test_a_named_expression_reading_a_dual_is_not_built() -> None:
    """A dual needs a solved model, so the body cannot be folded at build time."""
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = Model.from_spec(spec, DISPATCH_DATA)
    assert list(m.expressions) == []
    with pytest.raises(AttributeError, match="no dual yet"):
        evaluate(spec, m, "price")


# ---------------------------------------------------------------------------
# assumptions: the ones a piecewise method implies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "data", "spend"),
    [
        pytest.param(
            CURVE_SPEC, {"bp_x": FULL_X, "bp_y": FULL_Y}, 400.0, id="whole-curves"
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": RAGGED_X, "bp_y": RAGGED_Y},
            275.0,
            id="ragged-curves-under-points",
        ),
    ],
)
def test_a_piecewise_cost_lands_on_the_curve(
    spec: dict[str, Any], data: dict[str, Any], spend: float
) -> None:
    m = solved(spec, {**CURVE_DATA, **data})
    assert evaluate(spec, m, "spend")["spend"].item() == pytest.approx(spend)
    assert m.objective.value == pytest.approx(spend)


def without(series: pd.Series, *keys: tuple[str, int]) -> pd.Series:
    return series.drop(index=list(keys))


@pytest.mark.parametrize(
    ("spec", "data", "match"),
    [
        pytest.param(
            with_(
                CURVE_SPEC,
                parameters={
                    **CURVE_SPEC["parameters"],
                    "bp_x": {"dims": ["generator", "bp"], "missing": "absent"},
                },
            ),
            {"bp_x": without(FULL_X, ("gas", 3)), "bp_y": FULL_Y},
            "assumption 'cost_curve_complete' does not hold.*\n  Not so at generator='gas', bp=3",
            id="a-hole-in-a-whole-curve",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": RAGGED_X, "bp_y": without(RAGGED_Y, ("gas", 3))},
            "assumption 'cost_curve_complete' does not hold.*narrow points: 'bp_x'",
            id="a-hole-inside-the-mask",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": without(FULL_X, ("gas", 1)), "bp_y": FULL_Y},
            "assumption 'cost_curve_contiguous' does not hold.*\n  Not so at generator='gas'",
            id="a-mask-with-a-gap",
        ),
        pytest.param(
            CURVE_SPEC,
            {
                "bp_x": curve(
                    {
                        (g, k): x
                        for g in UNITS
                        for k, x in enumerate([0.0, 20.0, 20.0, 80.0])
                    }
                ),
                "bp_y": FULL_Y,
            },
            "strictly increasing",
            id="breakpoints-that-do-not-increase",
        ),
        pytest.param(
            CURVE_SPEC,
            {
                "bp_x": FULL_X,
                "bp_y": curve(
                    {
                        (g, k): y
                        for g in UNITS
                        for k, y in enumerate([0.0, 300.0, 500.0, 600.0])
                    }
                ),
            },
            "exact only for a convex curve",
            id="a-concave-curve-under-lp",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": without(RAGGED_X, ("hydro", 1)), "bp_y": RAGGED_Y},
            "at least two breakpoints per curve.*\n  Not so at generator='hydro'",
            id="a-one-point-curve-under-lp",
        ),
    ],
)
def test_a_curve_the_method_cannot_build_is_refused(
    spec: dict[str, Any], data: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {**CURVE_DATA, **data})


def test_a_convex_hull_curve_may_bend_either_way_but_not_both() -> None:
    spec = with_(
        CURVE_SPEC,
        piecewise={
            "cost_curve": {
                "over": "bp",
                "links": [["p", "bp_x"], ["op_cost", "bp_y"]],
                "method": "convex",
            }
        },
    )
    concave = curve(
        {(g, k): y for g in UNITS for k, y in enumerate([0.0, 300.0, 500.0, 600.0])}
    )
    mixed = curve(
        {(g, k): y for g in UNITS for k, y in enumerate([0.0, 300.0, 350.0, 600.0])}
    )
    assert (
        "cost_curve_lam"
        in Model.from_spec(
            spec, {**CURVE_DATA, "bp_x": FULL_X, "bp_y": concave}
        ).variables
    )
    with pytest.raises(SpecDataError, match="exact only for a single bend"):
        Model.from_spec(spec, {**CURVE_DATA, "bp_x": FULL_X, "bp_y": mixed})


# ---------------------------------------------------------------------------
# a spec's own assumptions
# ---------------------------------------------------------------------------

CURVES = {"bp_x": FULL_X, "bp_y": FULL_Y}


@pytest.mark.parametrize(
    ("assumption", "match"),
    [
        pytest.param("p_max > 0", None, id="holds"),
        pytest.param(
            {"holds": "p_max >= 50", "where": "p_max > 40"},
            None,
            id="holds-where-checked",
        ),
        pytest.param(
            {
                "holds": "p_max >= 50",
                "description": "a small unit is not worth a curve",
            },
            "assumption 'sized' does not hold for the data attached to 'p_max': a small "
            "unit is not worth a curve\n  Not so at generator='hydro'",
            id="fails-with-the-description",
        ),
        pytest.param(
            "load <= sum(p_max, over=generator)",
            None,
            id="holds-across-an-expression",
        ),
    ],
)
def test_a_declared_assumption_is_checked_against_the_data(
    assumption: Any, match: str | None
) -> None:
    spec = with_(CURVE_SPEC, assumptions={"sized": assumption})
    if match is None:
        assert "p" in Model.from_spec(spec, {**CURVE_DATA, **CURVES}).variables
        return
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {**CURVE_DATA, **CURVES})
