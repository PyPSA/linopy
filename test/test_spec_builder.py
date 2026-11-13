"""
Building declarations from mathspec programs: variables, constraints, the
objective, SOS-constrained curves, coverage refusals, side-swapped
expressions and the assumptions checked against the data.
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")
pytest.importorskip("highspy")
yaml = pytest.importorskip("yaml")

from mathspec.errors import LanguageError  # noqa: E402
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
    needs_examples,
    solved,
    with_,
    yaml_dict,
)

import linopy  # noqa: E402
from linopy import Model  # noqa: E402
from linopy.spec import SpecDataError, evaluate  # noqa: E402
from linopy.spec.api import lower  # noqa: E402
from linopy.testing import (  # noqa: E402
    assert_conequal,
    assert_linequal,
    synthetic_sources,
)

REFUSED: dict[str, tuple[type[Exception], str]] = {
    "operators/shift_by_parameter.yaml": (
        SpecDataError,
        "the synthetic data leaves a 0 >= demand row",
    ),
    "pypsa.yaml": (
        SpecDataError,
        "the synthetic data breaks Link_output_delay_under_uniform_weighting",
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


@needs_examples
@pytest.mark.parametrize("path", [example(p) for p in EXAMPLES])
def test_every_mathspec_example_builds_and_solves(path: Path) -> None:
    """Every whole mathspec example builds and solves on synthetic data, bar the ones :data:`REFUSED` names, which are refused."""
    program = lower(path)
    m = solved(path, synthetic_sources(path), store=False)
    assert set(m.variables) == set(program.variables)
    assert m.termination_condition in ("optimal", "infeasible")


PYPSA = Path(EXAMPLES_DIR or "", "pypsa.yaml")
COMMITTED = ("Generator", "Link", "Process")
PYPSA_SCHEDULES = (
    "Generator_p_set Link_p_set Process_p_set Line_s_set Transformer_s_set "
    "StorageUnit_p_set StorageUnit_p_dispatch_set StorageUnit_p_store_set "
    "StorageUnit_state_of_charge_set Store_e_set Store_p_set"
).split()
PYPSA_DATA: dict[str, Any] = {
    "snapshot_weightings_generators": 1.0,
    "StorageUnit_opens_late": False,
    "Store_opens_late": False,
    **{
        f"{c}_marginal_cost_quadratic": 0.0
        for c in (*COMMITTED, "StorageUnit", "Store")
    },
    "CVaR_alpha": 0.5,
    **dict.fromkeys(PYPSA_SCHEDULES, 0.0),
    "Line_loss_offset": 0.0,
    "Transformer_loss_offset": 0.0,
    "Transformer_phase_shift_min": 0.0,
    "Transformer_phase_shift_weight": 0.0,
    "StorageUnit_p_min_pu": 0.0,
    "Store_e_min_pu": 0.0,
    **{f"{c}_e_sum_min": -np.inf for c in ("Generator", "Link", "Process", "Store")},
    "Carrier_max_growth": 1e3,
    **{f"{c}_maintenance_start_blocked": False for c in COMMITTED},
    **{f"{c}_maintenance_duration": 1 for c in COMMITTED},
    **{f"{c}_{d}_time_brought_in": 0 for c in COMMITTED for d in ("up", "down")},
    **{
        f"{c}_nom_min": 0.0
        for c in (
            "Generator_p",
            "Link_p",
            "Process_p",
            "StorageUnit_p",
            "Store_e",
            "Line_s",
            "Transformer_s",
        )
    },
    "Store_p_min_pu": -1.0,
    "StorageUnit_state_of_charge_initial_per_period": False,
    "Store_e_initial_per_period": False,
    "period_weight_years": 1.0,
    "Generator_marginal_cost_curve_p": np.nan,
}


def pypsa_sources(**overrides: Any) -> dict[str, Any]:
    """Two labels per dimension of synthetic data that pypsa.yaml's assumptions accept and that solves."""
    sources = synthetic_sources(PYPSA, n=2)
    for name, value in {**PYPSA_DATA, **overrides}.items():
        held = sources[name]
        sources[name] = (
            xr.full_like(held, value) if isinstance(held, xr.DataArray) else value
        )
    return sources


@needs_examples
@pytest.mark.slow
def test_the_pypsa_example_solves_on_data_its_assumptions_accept() -> None:
    sources = pypsa_sources()
    m = solved(PYPSA, sources, store=False)
    assert m.termination_condition == "optimal"
    assert int((m.constraints["CVaR_excess"].labels != -1).sum()) == 2
    assert set(evaluate(PYPSA, m, sources=sources)) == set(lower(PYPSA).expressions)


@needs_examples
def test_the_pypsa_example_refuses_a_quadratic_cost_under_a_risk_preference() -> None:
    sources = pypsa_sources(Generator_marginal_cost_quadratic=1.0)
    with pytest.raises(SpecDataError, match="quadratic_without_risk_preference"):
        Model.from_spec(PYPSA, sources, store=False)


def test_the_dispatch_example_solves_and_its_expressions_fold() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    assert m.objective.value == pytest.approx(2500.0)
    xr.testing.assert_allclose(m.solution["p"], DISPATCH_P.T)
    folded = evaluate(yaml_dict(), m)
    assert list(folded) == ["spend", "usage"]
    xr.testing.assert_allclose(
        folded["spend"], (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )
    xr.testing.assert_allclose(
        folded["usage"], (DISPATCH_P / [100.0, 200.0]).T.rename("usage")
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
FULL_W = xr.DataArray([1.0, 1.0, 1.0], coords=[T])
FULL_C = xr.DataArray([0.0, 4.0, 5.0], coords=[T])
HOLE_AT_0 = xr.DataArray([4.0, 5.0], coords=[T[1:]])
W_HOLE_AT_0 = xr.DataArray([1.0, 1.0], coords=[T[1:]])


C_HOLE_AT_2 = FULL_C[:2]
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
            {"w": FULL_W, "c": FULL_C[1:]},
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
            {"w": FULL_W, "c": FULL_C[1:]},
            "c",
            0,
            id="cases-region",
        ),
    ],
)
def test_a_refused_row_no_live_row_reads_builds_and_stays_a_hole(
    spec: dict[str, Any], data: dict[str, Any], name: str, hole: int
) -> None:
    m = Model.from_spec(spec, {"t": T, **data}, store=True)
    assert m.parameters[name].isnull().values.tolist() == [t == hole for t in T]


def test_a_named_expression_reading_a_refused_row_is_refused_by_build_and_evaluate() -> (
    None
):
    data = {"t": T, "w": W_HOLE_AT_0, "c": FULL_C}
    m = Model.from_spec(CAP_LATER, data, store=True)
    spec = with_(CAP_LATER, expressions={"weighted": "w * c"})
    match = "expression 'weighted' reads parameter 'w' at t=0, where it has no row"
    with pytest.raises(SpecDataError, match=match):
        evaluate(spec, m, "weighted")
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, data)
    assert m.parameters["w"].isnull().values.tolist() == [True, False, False]


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
    "grp": pd.DataFrame({"t": T, "s": ["p", "p", "q"]}),
    "a": xr.DataArray([2.0, 3.0], coords=[T[1:]]),
    "z": xr.DataArray([5.0], coords=[GROUPS[1:]]),
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
    live = xr.DataArray([True, True], coords=[T[1:]])
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
    "gate": xr.DataArray([True], coords=[F[:1]]),
    "relmax": xr.DataArray([0.5, 0.5], coords=[F]),
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


def sized(missing: Any, expression: str) -> dict[str, Any]:
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


def ramp(expression: str, where: str | None = None) -> dict[str, Any]:
    """One row per ``s`` reading ``p`` and ``q``, as the holes section of the spec-data docs writes it."""
    row: dict[str, Any] = {"dims": ["s"], "expression": expression}
    return {
        "dimensions": {"s": {"dtype": "int", "ordered": True}},
        "variables": {
            v: {"dims": ["s"], "bounds": {"lower": 0, "upper": 1}} for v in "pq"
        },
        "constraints": {"ramp": row if where is None else {**row, "where": where}},
        "objective": {"sense": "minimize", "expression": "sum(p, over=s)"},
    }


S = {"s": pd.Index([0, 1, 2, 3], name="s")}


@pytest.mark.parametrize(
    ("expression", "where", "dropped"),
    [
        pytest.param("p <= shift(p, along=s, offset=1)", None, [0], id="shift"),
        pytest.param("p <= shift(p, along=s, offset=2)", None, [0, 1], id="offset-2"),
        pytest.param(
            "p + q <= shift(p, along=s, offset=1)", None, [0], id="beside-a-term"
        ),
        pytest.param(
            "p <= shift(p, along=s, offset=1)", "position(s) > 0", [0], id="where"
        ),
        pytest.param("p <= shift(p, along=s, offset=1, edge=0)", None, [], id="edge"),
        pytest.param(
            "p <= shift(p, along=s, offset=1, edge='wrap')", None, [], id="wrap"
        ),
        pytest.param(
            "p <= sum(shift(q, along=s, offset=1), over=s)", None, [], id="in-a-sum"
        ),
        pytest.param("sum_back(p, along=s, window=3) <= q", None, [], id="sum-back"),
    ],
)
def test_the_rows_an_operator_vacates_are_dropped_as_documented(
    expression: str, where: str | None, dropped: list[int]
) -> None:
    """The cases ``doc/spec-data.rst`` lists under holes that operators create."""
    labels = Model.from_spec(ramp(expression, where), S).constraints["ramp"].labels
    assert labels.s[labels == -1].values.tolist() == dropped


def test_an_edge_value_stands_in_for_the_vacated_row() -> None:
    m = Model.from_spec(ramp("p <= shift(p, along=s, offset=1, edge=0)"), S)
    first = m.constraints["ramp"].sel(s=0)
    assert (first.vars == m.variables["p"].labels.sel(s=0)).any()
    assert float(first.rhs) == 0.0


def test_a_row_the_data_emptied_of_variables_is_refused() -> None:
    """A zero absence leaves ``0 <= relmax`` standing, which linopy cannot carry and a solver would never see."""
    with pytest.raises(
        SpecDataError, match=r"(?s)constraint 'sized'.*Supply the rows of the variables"
    ):
        Model.from_spec(sized(0, "size <= relmax"), ENVELOPE_DATA)


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


ZERO_AT_0 = {"t": T, "a": xr.DataArray([0.0, 1.0, 1.0], coords=[T]), "b": FULL_C}
INF_AT_0 = {
    "t": T,
    "a": xr.DataArray([float("inf"), 1.0, 1.0], coords=[T]),
    "b": FULL_C,
}


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
        pytest.param("(x + a) * 2 - 2 * a >= 1", INF_AT_0, "t=0", id="scaled-term"),
        pytest.param("2 * (x + a) - 2 * a >= 1", INF_AT_0, "t=0", id="scaled-left"),
        pytest.param("(x + a) / 2 - a / 2 >= 1", INF_AT_0, "t=0", id="divided-term"),
        pytest.param("(x + a) * 2 >= 2 * a", INF_AT_0, "t=0", id="scaled-across-sides"),
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


GATED_ON = {
    "variables": {
        "on": {
            "dims": ["t"],
            "where": "a < 2",
            "missing": 0,
            "bounds": {"lower": 0, "upper": 1},
        }
    }
}


def test_an_infinity_times_a_term_absent_there_is_not_refused_on_a_row_masked_out() -> (
    None
):
    """``a * on`` read 0 * inf at t=0, a NaN core linopy refused though no row built there."""
    spec = with_(quotient("x <= a * on", where="a < 2"), **GATED_ON)
    m = Model.from_spec(spec, INF_AT_0)
    assert m.constraints["row"].labels.values.tolist()[0] == -1, (
        "the row at t=0 is masked out"
    )


def test_an_infinity_times_a_term_absent_there_is_refused_on_a_built_row() -> None:
    with pytest.raises(ValueError, match="produced NaN"):
        Model.from_spec(with_(quotient("x <= a * on"), **GATED_ON), INF_AT_0)


def test_arithmetic_masked_out_for_one_row_is_refused_for_the_next_reading_it() -> None:
    masked = quotient("x + a / b >= 1", where="b > 0")
    spec = with_(
        masked, constraints={"open": {"dims": ["t"], "expression": "x + a / b >= 1"}}
    )
    with pytest.raises(SpecDataError, match=r"constraint 'open'.*not finite at t=0"):
        Model.from_spec(spec, ZERO_AT_0)


@pytest.mark.parametrize(
    ("missing", "expression"), [(0, "size >= 0"), (5, "size >= 5")]
)
def test_a_dead_row_of_missing_values_whose_sides_are_equal_is_only_warned_about(
    missing: float, expression: str
) -> None:
    with pytest.warns(UserWarning, match=r"constraint 'sized'.*trivially true"):
        m = solved(sized(missing, expression), ENVELOPE_DATA)
    assert m.termination_condition == "optimal"
    assert (m.constraints["sized"].vars.sel(f="b") == -1).all()


def test_a_coefficient_needs_no_row_where_its_variable_is_absent() -> None:
    spec = with_(
        DEFINED_SPEC,
        parameters={"relmax": {"dims": ["f"], "missing": "absent"}},
        expressions={"sized": "relmax * size"},
    )
    data = {**ENVELOPE_DATA, "relmax": xr.DataArray([0.5], coords=[F[:1]])}
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


def test_a_fully_masked_constraint_adds_no_group_and_reads_a_nan_dual() -> None:
    spec = with_(
        SCALAR_SWITCH,
        constraints={
            "cap": {"dims": ["i"], "expression": "y <= 4", "where": "on"},
            "floor": {"dims": ["i"], "expression": "y >= 3"},
        },
        expressions={"price": "dual(cap)"},
    )
    m = solved(spec, {"i": [1, 2], "on": False})
    assert list(m.constraints) == ["floor"]
    assert evaluate(spec, m, "price")["price"].isnull().all()


@pytest.mark.parametrize("expression", ["x <= 4", "x <= 2 * x", "x + x <= x"])
def test_a_constraint_of_masked_variables_alone_adds_no_group(expression: str) -> None:
    spec = with_(
        SCALAR_SWITCH,
        constraints={"cap": {"dims": ["i"], "expression": expression}},
    )
    m = solved(spec, {"i": [1, 2], "on": False})
    assert list(m.constraints) == []
    assert m.objective.value == pytest.approx(4.0)


@pytest.mark.parametrize("expression", ["y <= 5", "5 >= y"], ids=["left", "flipped"])
def test_a_constraint_is_framed_in_the_name_order_of_its_dims(expression: str) -> None:
    spec = {
        "dimensions": {"t": {"dtype": "int"}, "g": {"dtype": "str"}},
        "variables": {"y": {"dims": ["t", "g"], "bounds": {"lower": 0}}},
        "constraints": {"k": {"dims": ["t", "g"], "expression": expression}},
    }
    m = Model.from_spec(spec, {"t": T, "g": ["a", "b"]})
    assert m.constraints["k"].labels.dims == ("g", "t")
    assert m.constraints["k"].lhs.coord_dims == ("g", "t")


@pytest.mark.parametrize("row", ["sum(x, over=t) >= 5", "sum(x, over=t) <= 0"])
def test_a_dimension_with_no_members_empties_a_summed_row(row: str) -> None:
    spec = with_(SPARSE_SPEC, constraints={"budget": {"dims": [], "expression": row}})
    empty = pd.Index([], name="t", dtype=int)
    none = xr.DataArray(np.array([], dtype=float), coords=[empty])
    with pytest.raises(
        SpecDataError, match="constraint 'budget'.*hold no variable term"
    ):
        Model.from_spec(spec, {"t": empty, "w": none, "c": none})


@pytest.mark.parametrize(("missing", "kept"), [("absent", False), (0, True)])
def test_a_missing_row_in_a_where_reads_the_way_missing_says(
    missing: Any, kept: bool
) -> None:
    spec = {
        "dimensions": {"f": {"dtype": "str"}},
        "parameters": {"count": {"dims": ["f"], "dtype": "int", "missing": missing}},
        "variables": {
            "x": {
                "dims": ["f"],
                "where": "count == 0",
                "bounds": {"lower": 0, "upper": 1},
            }
        },
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    m = Model.from_spec(
        spec, {"f": ["a", "b"], "count": pd.DataFrame({"f": ["a"], "count": [0]})}
    )
    assert (m.variables["x"].labels != -1).values.tolist() == [True, kept]


@pytest.mark.parametrize(
    ("missing", "rows", "x"),
    [
        pytest.param(7, {0: 4.0, 2: 5.0}, [4.0, 7.0, 10.0], id="a-value-inside"),
        pytest.param(
            "refused", {0: 4.0, 1: 6.0}, [4.0, 6.0, 10.0], id="refused-inside"
        ),
    ],
)
def test_a_parameter_is_absent_outside_its_where(
    missing: Any, rows: dict[int, float], x: list[float]
) -> None:
    spec = {
        "dimensions": {"t": {"dtype": "int"}},
        "parameters": {
            "on": {"dims": ["t"], "dtype": "bool"},
            "c": {"dims": ["t"], "where": "on", "missing": missing},
        },
        "variables": {
            "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
            "y": {"dims": ["t"], "where": "c != 1", "bounds": {"lower": 0, "upper": 1}},
        },
        "constraints": {"cap": {"dims": ["t"], "expression": "x <= c"}},
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    on = xr.DataArray([True, True, False], coords=[T])
    c = xr.DataArray(list(rows.values()), coords=[pd.Index(list(rows), name="t")])
    m = solved(spec, {"t": T, "on": on, "c": c})
    assert m.solution["x"].values.tolist() == pytest.approx(x)
    assert (m.variables["y"].labels != -1).values.tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("missing", "masked_reads_nan"),
    [("absent", True), (0, False)],
    ids=["undefined-leaves-a-masked-slot-nan", "zero-fills-a-masked-slot"],
)
def test_a_fold_reads_a_masked_slot_the_way_its_absence_says(
    missing: Any, masked_reads_nan: bool
) -> None:
    spec = yaml_dict()
    spec["variables"]["p"]["missing"] = missing
    spec["expressions"] = {"spend_by_unit": "p * cost"}
    data = {**DISPATCH_DATA, "p_max": xr.DataArray([200.0, 0.0], coords=[GENERATOR])}
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
                "along": "bp",
                "dims": ["snapshot", "generator"],
                "links": {"p": ["p", "bp_x"], "op_cost": ["op_cost", "bp_y"]},
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


def zero_filled(expression: str, upper: Any = 10) -> dict[str, Any]:
    """:data:`SPARSE_SPEC` with ``c`` declared ``missing: 0``, read by *expression* and the bound *upper*."""
    return with_(
        SPARSE_SPEC,
        parameters={"w": {"dims": ["t"]}, "c": {"dims": ["t"], "missing": 0}},
        variables={"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": upper}}},
        constraints={"cap": {"dims": ["t"], "expression": expression}},
    )


@pytest.mark.parametrize(
    ("expression", "upper"),
    [
        pytest.param("x <= c", 10, id="constant-side"),
        pytest.param("x <= c ** 2", 10, id="constant-side-under-a-power"),
        pytest.param("x <= 10", "c", id="bound"),
    ],
)
def test_a_missing_value_stands_in_a_bound_and_a_constant_side(
    expression: str, upper: Any
) -> None:
    m = solved(zero_filled(expression, upper), {"t": T, "w": FULL_W, "c": HOLE_AT_0})
    assert float(m.solution["x"].sel(t=0)) == pytest.approx(0.0)


@pytest.mark.parametrize("expression", ["x / c <= 1", "x / (c ** 2) <= 1"])
def test_an_unguarded_divisor_a_missing_value_makes_zero_is_refused_at_load(
    expression: str,
) -> None:
    with pytest.raises(LanguageError, match="the divisor contains 'c'"):
        lower(zero_filled(expression))


def test_an_operator_under_a_power_keeps_the_parameters_it_reads() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={**SPARSE_SPEC["parameters"], "lag": {"dims": [], "dtype": "int"}},
        expressions={"e": "shift(c, along=t, offset=lag, edge=0) ** 1"},
    )
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": FULL_C, "lag": 1}, store=True)
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
    "parameters": {"k": {"dims": ["t"], "missing": "absent"}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
        "y": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
    },
    "constraints": {
        "cap": {"dims": ["t"], "where": "t > 0", "expression": "k * x * y + x <= 5"}
    },
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}


def capped(expression: str) -> dict[str, Any]:
    cap = {"dims": ["t"], "where": "t > 0", "expression": expression}
    return with_(QUADRATIC_SPEC, constraints={"cap": cap})


@pytest.mark.parametrize(
    ("expression", "k"),
    [
        pytest.param("k * x * y + x <= 5", [0.0, 0.0, 0.0], id="zero"),
        pytest.param("k * x * y + x <= 5", [2.0, 0.0, 0.0], id="outside-where"),
        pytest.param("x <= k * x * y + 5", [0.0, 0.0, 0.0], id="right-hand-side"),
        pytest.param(
            "sum(k * x * y, over=t) + x <= 5", [0.0, 0.0], id="absent-summand"
        ),
    ],
)
def test_a_quadratic_constraint_the_data_linearises_builds(
    expression: str, k: list[float]
) -> None:
    data = {"t": T, "k": xr.DataArray(k, coords=[T[: len(k)]])}
    m = Model.from_spec(capped(expression), data)
    linear = Model.from_spec(capped("x <= 5"), data)
    assert_conequal(m.constraints["cap"], linear.constraints["cap"], strict=False)


@pytest.mark.parametrize("expression", ["k * x * y + x <= 5", "x <= k * x * y + 5"])
def test_a_quadratic_constraint_the_data_keeps_is_refused(expression: str) -> None:
    with pytest.raises(
        NotImplementedError,
        match="constraint 'cap': 1 row.*quadratic term in the objective only",
    ):
        Model.from_spec(
            capped(expression), {"t": T, "k": xr.DataArray([0.0, 0.0, 2.0], coords=[T])}
        )


def test_a_quadratic_objective_builds() -> None:
    spec = with_(
        QUADRATIC_SPEC,
        constraints={"cap": {"dims": ["t"], "expression": "x + y <= 5"}},
        objective={"sense": "maximize", "expression": "sum(x * y, over=t)"},
    )
    m = Model.from_spec(spec, {"t": T, "k": xr.zeros_like(FULL_C)})
    assert isinstance(m.objective.expression, linopy.QuadraticExpression)


@pytest.mark.parametrize(
    ("k", "quadratic"),
    [
        pytest.param([0.0, 0.0, 0.0], False, id="zeroed"),
        pytest.param([0.0, 0.0, 2.0], True, id="kept"),
    ],
)
def test_an_objective_is_quadratic_only_where_the_data_keeps_a_product(
    k: list[float], quadratic: bool
) -> None:
    """A quadratic cost whose coefficients are all zero left a QuadraticExpression objective."""
    spec = with_(
        QUADRATIC_SPEC,
        constraints={"cap": {"dims": ["t"], "expression": "x + y <= 5"}},
        objective={"sense": "maximize", "expression": "sum(k * x * y + x, over=t)"},
    )
    m = Model.from_spec(spec, {"t": T, "k": xr.DataArray(k, coords=[T])})
    assert isinstance(m.objective.expression, linopy.QuadraticExpression) is quadratic


@pytest.mark.parametrize("product", ["x * y * k", "k * (x * y)"])
def test_a_product_the_data_zeroes_keeps_a_sparse_objective_sparse(
    backend: Model, product: str
) -> None:
    """A product the data zeroes stayed a quadratic, which has no sparse backing, so every term it met densified."""
    spec = with_(
        QUADRATIC_SPEC,
        constraints={"cap": {"dims": ["t"], "expression": "x + y <= 5"}},
        objective={"sense": "maximize", "expression": f"sum({product} + x, over=t)"},
    )
    data = {"t": T, "k": xr.DataArray([0.0, 0.0, 0.0], coords=[T])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        m = Model.from_spec(spec, data, sparse=backend.sparse)
    objective = m.objective.expression
    assert isinstance(objective, linopy.LinearExpression)
    assert objective.is_sparse == backend.sparse
    assert_linequal(objective, Model.from_spec(spec, data).objective.expression)


def test_a_named_expression_reading_a_dual_is_not_built() -> None:
    """A dual needs a solved model, so the body cannot be folded at build time."""
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = Model.from_spec(spec, DISPATCH_DATA, store=True)
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


def without(arr: xr.DataArray, *keys: tuple[str, int]) -> xr.DataArray:
    """*arr* with the rows at *keys* missing."""
    return xr.DataArray.from_series(arr.to_series().dropna().drop(index=list(keys)))


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
            "assumption 'cost_curve_complete' does not hold.*\n  Not so at bp=3, generator='gas'",
            id="a-hole-in-a-whole-curve",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": RAGGED_X, "bp_y": without(RAGGED_Y, ("gas", 3))},
            "assumption 'cost_curve_complete' does not hold.*narrow where: 'bp_x'",
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
                "bp_y": curve(
                    {
                        (g, k): y
                        for g in UNITS
                        for k, y in enumerate([0.0, 150.0, 150.0, 900.0])
                    }
                ),
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
                "along": "bp",
                "dims": ["snapshot", "generator"],
                "links": {"p": ["p", "bp_x"], "op_cost": ["op_cost", "bp_y"]},
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
