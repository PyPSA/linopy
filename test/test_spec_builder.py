"""
Building declarations from mathspec programs: variables, constraints, the
objective, SOS-constrained curves, coverage refusals and side-swapped
expressions.
"""

from __future__ import annotations

import glob
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from conftest import (  # noqa: E402
    CURVE_DATA,
    CURVE_SPEC,
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    EXAMPLES_DIR,
    FULL_X,
    FULL_Y,
    GENERATOR,
    solved,
    with_,
    yaml_dict,
)
from linopy import Model  # noqa: E402
from linopy.constraints import CSRConstraint  # noqa: E402
from linopy.spec import SpecDataError  # noqa: E402
from linopy.spec.accessor import lower  # noqa: E402
from linopy.spec.testing import synthetic_sources  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

MULTI_COLUMN = "linopy reads no relation of more than one key or value column"
REFUSED = {
    "operators/at_columns.yaml": MULTI_COLUMN,
    "operators/sum_by_columns.yaml": MULTI_COLUMN,
    "operators/sum_by_column_lists.yaml": MULTI_COLUMN,
    "operators/shift_by_parameter.yaml": "the synthetic data leaves a 0 >= demand row",
}


def example(path: str) -> Any:
    """*path* as a sweep case, expected to raise :class:`SpecDataError` where :data:`REFUSED` names it."""
    name = Path(path).relative_to(EXAMPLES_DIR or "").as_posix()
    marks = (
        [pytest.mark.xfail(strict=True, raises=SpecDataError, reason=REFUSED[name])]
        if name in REFUSED
        else []
    )
    return pytest.param(path, id=name, marks=marks)


EXAMPLES = (
    sorted(glob.glob(f"{EXAMPLES_DIR}/*.yaml") + glob.glob(f"{EXAMPLES_DIR}/*/*.yaml"))
    if EXAMPLES_DIR
    else []
)


@pytest.mark.skipif(
    not EXAMPLES, reason="set MATHSPEC_EXAMPLES to a mathspec examples directory"
)
@pytest.mark.parametrize("path", [example(p) for p in EXAMPLES])
def test_every_mathspec_example_builds_and_solves(path: str) -> None:
    """Every mathspec example builds and solves on synthetic data, bar the ones :data:`REFUSED` names, which are refused."""
    if Path(path).parent.name == "symbols":
        pytest.skip("typesetting input, not a spec")
    program = lower(path)
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
    "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}}},
    "constraints": {"cap": {"dims": ["t"], "expression": "w * x <= c"}},
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}
FULL_W = pd.Series([1.0, 1.0, 1.0], index=T)
FULL_C = pd.Series([0.0, 4.0, 5.0], index=T)
HOLE_AT_0 = pd.Series([4.0, 5.0], index=T[1:])
W_HOLE_AT_0 = pd.Series([1.0, 1.0], index=T[1:])

NO_W_CONSTRAINT = {"cap": {"dims": ["t"], "expression": "x <= c"}}


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
                variables={"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": "c"}}},
            ),
            {"w": FULL_W, "c": HOLE_AT_0},
            "variable 'x': 1 rows have NULL bounds",
            id="bound",
        ),
        pytest.param(
            with_(
                SPARSE_SPEC,
                constraints={"cap": {"dims": ["t"], "expression": "x / w <= c"}},
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


def sized(absence: str, expression: str) -> dict[str, Any]:
    """:data:`ENVELOPE_SPEC` with a row over ``size`` alone, which the data empties at ``f='b'``."""
    spec = with_(
        ENVELOPE_SPEC,
        variables={"size": {**ENVELOPE_SPEC["variables"]["size"], "absence": absence}},
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
            sized("undefined", "size <= relmax"), "sized", id="an-absent-variable"
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
        Model.from_spec(sized("zero", "size <= relmax"), ENVELOPE_DATA)


def quotient(expression: str, where: str | None = None) -> dict[str, Any]:
    """A row over ``t`` reading the data ``a`` and ``b``, which :data:`ZERO_AT_0` holds at 0 on ``t=0``."""
    row: dict[str, Any] = {"dims": ["t"], "expression": expression}
    return {
        "dimensions": {"t": {"dtype": "int"}},
        "parameters": {"a": {"dims": ["t"]}, "b": {"dims": ["t"]}},
        "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}}},
        "constraints": {"row": row if where is None else {**row, "where": where}},
        "objective": {"sense": "minimize", "expression": "sum(x, over=t)"},
    }


ZERO_AT_0 = {"t": T, "a": pd.Series([0.0, 1.0, 1.0], index=T), "b": FULL_C}
INF_AT_0 = {"t": T, "a": pd.Series([float("inf"), 1.0, 1.0], index=T), "b": FULL_C}


SHIFTED = "shift(a, along=t, offset=1, edge=0)"


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
            "the only row",
            id="sums-beside-a-term",
        ),
        pytest.param(
            f"x + {SHIFTED} - {SHIFTED} >= 1",
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
        m = solved(sized("zero", "size >= 0"), ENVELOPE_DATA)
    assert m.termination_condition == "optimal"
    assert (m.constraints["sized"].vars.sel(f="b") == -1).all()


def test_a_coefficient_needs_no_row_where_its_variable_is_absent() -> None:
    spec = with_(DEFINED_SPEC, expressions={"sized": "relmax * size"})
    data = {**ENVELOPE_DATA, "relmax": pd.Series([0.5], index=F[:1])}
    m = solved(spec, data)
    assert float(m.solution["x"].sel(f="a")) == pytest.approx(25.0)
    assert float(m.spec.expressions["sized"].solution.sel(f="a")) == pytest.approx(25.0)


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


def test_a_dimension_with_no_members_builds_no_row() -> None:
    spec = with_(
        SPARSE_SPEC,
        constraints={"budget": {"dims": [], "expression": "sum(x, over=t) <= 10"}},
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
    # Expanding the block writes it out as ordinary declarations, so none of it is drift.
    assert not m.spec.unspecified


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
        SPARSE_SPEC, constraints={"cap": {"dims": ["t"], "expression": expression}}
    )
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {"t": T, "w": FULL_W, "c": HOLE_AT_0})


def test_an_operator_under_a_power_keeps_its_parameters_retained() -> None:
    spec = with_(
        SPARSE_SPEC,
        parameters={**SPARSE_SPEC["parameters"], "lag": {"dims": [], "dtype": "int"}},
        expressions={"e": "shift(c, along=t, offset=lag, edge=0) ** 1"},
    )
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": FULL_C, "lag": 1})
    assert {"c", "lag"} <= set(m.spec.parameters.data_vars)
    xr.testing.assert_allclose(
        m.spec.expressions["e"].solution,
        xr.DataArray([0.0, 0.0, 4.0], coords={"t": T}, name="e"),
    )


# ---------------------------------------------------------------------------
# what linopy cannot build
# ---------------------------------------------------------------------------

QUADRATIC_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int"}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
        "y": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
    },
    "constraints": {"cap": {"dims": ["t"], "expression": "x * y <= 5"}},
    "objective": {"sense": "maximize", "expression": "sum(x, over=t)"},
}


def test_a_quadratic_constraint_is_refused_before_anything_is_built() -> None:
    m = Model()
    with pytest.raises(
        NotImplementedError, match="quadratic term in the objective only"
    ):
        m.add_spec(QUADRATIC_SPEC, {"t": T})
    assert list(m.variables) == [] and list(m.constraints) == []


def test_a_quadratic_objective_builds() -> None:
    spec = with_(
        QUADRATIC_SPEC,
        constraints={"cap": {"dims": ["t"], "expression": "x + y <= 5"}},
        objective={"sense": "maximize", "expression": "sum(x * y, over=t)"},
    )
    m = Model.from_spec(spec, {"t": T})
    assert isinstance(m.objective.expression, linopy.QuadraticExpression)


# ---------------------------------------------------------------------------
# the stamp every built declaration carries
# ---------------------------------------------------------------------------


def test_what_the_spec_builds_names_it() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    assert m.variables["p"].spec == "spec"
    assert m.constraints["power_balance"].spec == "spec"
    assert m.expressions["spend"].spec == "spec"


def test_a_spec_read_from_a_file_is_named_after_it(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    m = Model.from_spec(path, DISPATCH_DATA)
    assert m.spec.name == "dispatch"
    assert m.variables["p"].spec == "dispatch"


def test_a_sparse_constraint_carries_the_stamp_through_its_dense_form() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, sparse=True)
    con = m.constraints["power_balance"]
    assert isinstance(con, CSRConstraint)
    assert con.spec == "spec"
    assert con.to_dense().spec == "spec"


@pytest.mark.parametrize(
    "add",
    [
        pytest.param(
            lambda m: m.add_expressions(m.expressions["spend"] * 2, name="twice"),
            id="scaled",
        ),
        pytest.param(
            lambda m: m.add_expressions(
                linopy.merge([m.expressions["spend"]] * 2, dim="copy"), name="twinned"
            ),
            id="merged",
        ),
        pytest.param(
            lambda m: m.add_constraints(
                m.expressions["spend"] >= 0, name="spend_positive"
            ),
            id="constraint",
        ),
    ],
)
def test_what_the_caller_derives_from_a_stamped_expression_is_the_callers_own(
    add: Callable[[Model], Any],
) -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    assert add(m).spec is None
    assert m.expressions["spend"].spec == "spec"


def test_a_named_expression_reading_a_dual_stays_on_the_spec() -> None:
    """A dual needs a solved model, so the body cannot be folded at build time."""
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = Model.from_spec(spec, DISPATCH_DATA)
    assert list(m.expressions) == []
    assert set(m.spec.expressions) == {"price"}
