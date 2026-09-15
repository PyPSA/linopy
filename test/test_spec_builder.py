"""
Building declarations from math-spec programs: variables, constraints, the
objective, SOS-constrained curves, coverage refusals and side-swapped
expressions.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable
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
    SNAPSHOT,
    WHERE_DATA,
    WHERE_SPEC,
    solved,
    with_,
    yaml_dict,
)
from linopy import Model, Variable  # noqa: E402
from linopy.constraints import CSRConstraint  # noqa: E402
from linopy.spec import SpecDataError  # noqa: E402
from linopy.spec.testing import synthetic_sources  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

# ---------------------------------------------------------------------------
# a hand-built base, and a spec that extends it
# ---------------------------------------------------------------------------

EXTRA_SPEC: dict[str, Any] = {
    "dimensions": {"snapshot": {"dtype": "int"}, "generator": {"dtype": "str"}},
    "parameters": {"cap": {"dims": ["generator"]}},
    "variables": {"p": {"dims": ["snapshot", "generator"]}},
    "constraints": {
        "p_cap": {"dims": ["snapshot", "generator"], "expression": "p <= cap"}
    },
    "expressions": {"total": "sum(p)"},
}
EXTRA_DATA: dict[str, Any] = {
    "snapshot": SNAPSHOT,
    "generator": GENERATOR,
    "cap": pd.Series([90.0, 200.0], index=GENERATOR),
}


SECOND_SPEC: dict[str, Any] = {
    **EXTRA_SPEC,
    "parameters": {"floor": {"dims": ["generator"]}},
    "constraints": {
        "p_floor": {
            "dims": ["snapshot", "generator"],
            "where": "p",
            "expression": "p >= floor",
        }
    },
    "expressions": {"peak": "sum(p, over=generator)"},
}
SECOND_DATA: dict[str, Any] = {
    "snapshot": SNAPSHOT,
    "generator": GENERATOR,
    "floor": pd.Series([0.0, 0.0], index=GENERATOR),
}
SOS_SPEC = with_(
    EXTRA_SPEC,
    sos={"one_at_a_time": {"variable": "p", "over": "generator", "type": 1}},
)


def dispatch_p(m: Model, name: str = "p") -> Variable:
    """The dispatch variable ``p`` added to *m* by hand, as the spec would build it."""
    p_max = DISPATCH_DATA["p_max"].to_xarray()
    return m.add_variables(
        lower=0, upper=p_max, coords=[SNAPSHOT, GENERATOR], name=name
    )


def BASE_MODEL() -> Model:
    """The dispatch example built by hand: ``p``, its power balance and its cost."""
    m = Model()
    p = dispatch_p(m)
    load = DISPATCH_DATA["load"].to_xarray()
    m.add_constraints(p.sum("generator") == load, name="power_balance")
    m.add_objective((p * DISPATCH_DATA["cost"].to_xarray()).sum())
    return m


def extended(spec: dict[str, Any] = EXTRA_SPEC, **sources: Any) -> Model:
    """:func:`BASE_MODEL` extended by *spec*, its ``p`` bound to the hand-built one."""
    m = BASE_MODEL()
    data = {**EXTRA_DATA, "p": m.variables["p"], **sources}
    return m.add_spec(spec, data, name="extra")


THREE = pd.Index(["wind", "gas", "solar"], name="generator")


SUBSET_SPEC = with_(
    EXTRA_SPEC,
    constraints={"p_cap": {**EXTRA_SPEC["constraints"]["p_cap"], "where": "p"}},
)


def subset_bound() -> Model:
    """:func:`extended` with the spec over three generators, the hand-built ``p`` spanning two."""
    return extended(
        SUBSET_SPEC, generator=THREE, cap=pd.Series([100.0, 200.0, 50.0], index=THREE)
    )


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


def bound_dispatch() -> Model:
    """The dispatch spec reading a hand-built ``p`` instead of building one."""
    m = Model()
    dispatch_p(m)
    spec = with_(yaml_dict(), variables={"p": {"dims": ["snapshot", "generator"]}})
    m.add_spec(spec, {**DISPATCH_DATA, "p": m.variables["p"]})
    m.solve(solver_name="highs", output_flag=False)
    return m


@pytest.mark.parametrize(
    "build",
    [lambda: solved(yaml_dict(), DISPATCH_DATA), bound_dispatch],
    ids=["built", "bound"],
)
def test_the_dispatch_example_solves_and_its_expressions_fold(
    build: Callable[[], Model],
) -> None:
    m = build()
    assert list(m.variables) == ["p"]
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
    ("spec", "match"),
    [
        pytest.param(
            ENVELOPE_SPEC,
            r"(?s)constraint 'envelope': 1 row\(s\) hold no variable term.*f='b'",
            id="an-absent-term-empties-the-whole-row",
        ),
        pytest.param(
            sized("undefined", "size <= relmax"),
            r"(?s)constraint 'sized'.*absence: zero on the variables",
            id="an-undefined-absence-is-refused-either-way",
        ),
        pytest.param(
            sized("zero", "size <= relmax"),
            r"(?s)constraint 'sized'.*Supply the rows of the variables",
            id="a-zero-absence-is-refused-where-the-other-side-binds",
        ),
    ],
)
def test_a_row_the_data_emptied_of_variables_is_refused(
    spec: dict[str, Any], match: str
) -> None:
    """Such a row would read ``0 sense rhs``, leave the problem and let the solver call it optimal."""
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, ENVELOPE_DATA)


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
    assert not m.spec.unspecified


@pytest.mark.parametrize("model_name", ["p", "Generator-p"])
def test_a_sos_on_a_bound_variable_is_written_onto_the_model_owned_one(
    model_name: str,
) -> None:
    m = Model()
    p = dispatch_p(m, model_name)
    m.add_spec(SOS_SPEC, {**EXTRA_DATA, "p": p}, name="extra")
    assert m.variables[model_name].attrs["sos_type"] == 1
    assert m.variables[model_name].attrs["sos_dim"] == "generator"
    assert m.spec.unspecified.sos == ()


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
        expressions={"e": "shift(c, over=t, offset=lag, edge=0) ** 1"},
    )
    m = Model.from_spec(spec, {"t": T, "w": FULL_W, "c": FULL_C, "lag": 1})
    assert {"c", "lag"} <= set(m.spec.parameters.data_vars)
    xr.testing.assert_allclose(
        m.spec.expressions["e"].solution,
        xr.DataArray([0.0, 0.0, 4.0], coords={"t": T}, name="e"),
    )


# ---------------------------------------------------------------------------
# the layer stamp every built declaration carries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "layer", "constraint", "expression"),
    [
        pytest.param(
            lambda: Model.from_spec(yaml_dict(), DISPATCH_DATA),
            "spec",
            "power_balance",
            "spend",
            id="whole",
        ),
        pytest.param(extended, "extra", "p_cap", "total", id="extending"),
    ],
)
def test_a_built_constraint_and_named_expression_name_their_layer(
    build: Callable[[], Model], layer: str, constraint: str, expression: str
) -> None:
    m = build()
    assert m.constraints[constraint].spec == layer
    assert m.expressions[expression].spec == layer


def test_a_layer_stamps_the_variable_it_builds_and_not_the_one_it_binds() -> None:
    assert Model.from_spec(yaml_dict(), DISPATCH_DATA).variables["p"].spec == "spec"
    assert extended().variables["p"].spec is None


def test_a_frozen_constraint_carries_the_stamp_through_its_dense_form() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, freeze_constraints=True)
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
    assert "price" not in m.expressions
    assert set(m.spec.expressions) == {"price"}


def test_a_named_expression_of_a_name_the_model_holds_is_refused_before_the_build() -> (
    None
):
    m = BASE_MODEL()
    m.add_expressions(m.variables["p"].sum(), name="total")
    with pytest.raises(ValueError, match=r"named expression\(s\) \['total'\]"):
        m.add_spec(EXTRA_SPEC, {**EXTRA_DATA, "p": m.variables["p"]}, name="extra")
    assert list(m.constraints) == ["power_balance"]
    assert list(m.expressions) == ["total"]


def test_a_layer_can_keep_its_named_expressions_lazy() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, build_expressions=False)
    assert list(m.expressions) == []
    assert isinstance(m.spec.expressions["spend"].expression, linopy.LinearExpression)
