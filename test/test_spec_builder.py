"""
Building linopy models from math-spec programs, and reading named expressions back.

``EXAMPLE_DISPATCH`` is math-spec's ``examples/dispatch.yaml`` with two named
expressions added, so the end-to-end check runs on a spec the language ships.
Setting ``MATH_SPEC_EXAMPLES`` to a math-spec ``examples`` directory builds
and solves every example in it with synthetic data.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from linopy import Model  # noqa: E402
from linopy.spec import ModelSpec, SpecDataError  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

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


def solved(spec: Any, sources: Mapping[str, Any], **kwargs: Any) -> Model:
    m = Model.from_spec(spec, sources, **kwargs)
    m.solve(solver_name="highs", output_flag=False, reformulate_sos=True)
    return m


# ---------------------------------------------------------------------------
# inputs and model integration
# ---------------------------------------------------------------------------


SPEC_FORMS: dict[str, Callable[[Path], Any]] = {
    "path": lambda path: path,
    "path-string": str,
    "yaml-text": lambda path: path.read_text(),
    "dict": lambda path: math_spec.to_spec(path).to_dict(),
    "spec": lambda path: math_spec.to_spec(path),
}


@pytest.mark.parametrize("form", SPEC_FORMS.values(), ids=SPEC_FORMS.keys())
def test_spec_forms_build_the_same_model(
    tmp_path: Path, form: Callable[[Path], Any]
) -> None:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    m = Model.from_spec(form(path), DISPATCH_DATA)
    assert list(m.variables) == ["p"]
    assert list(m.constraints) == ["power_balance"]
    reread = math_spec.to_program(yaml.safe_load(m.spec.text))
    assert reread.constraints == m.spec.program.constraints
    assert isinstance(m.spec, ModelSpec)


def yaml_dict() -> dict[str, Any]:
    return math_spec.to_spec(yaml.safe_load(EXAMPLE_DISPATCH)).to_dict()


def test_a_lowered_program_is_refused() -> None:
    program = math_spec.to_program(yaml_dict())
    with pytest.raises(TypeError, match="not a lowered Program"):
        Model().add_spec(program, DISPATCH_DATA)


def test_add_spec_needs_an_empty_model() -> None:
    m = Model()
    m.add_variables(name="x")
    with pytest.raises(ValueError, match="empty model"):
        m.add_spec(yaml_dict(), DISPATCH_DATA)


def test_legacy_semantics_is_refused() -> None:
    with linopy.options as options:
        options["semantics"] = "legacy"
        with pytest.raises(ValueError, match="v1"):
            Model.from_spec(yaml_dict(), DISPATCH_DATA)


def test_a_model_without_a_spec_has_no_accessor() -> None:
    with pytest.raises(AttributeError, match="not built from a spec"):
        _ = Model().spec


def test_from_spec_passes_model_kwargs_and_chains() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, force_dim_names=True)
    assert m.force_dim_names
    assert Model().add_spec(
        yaml_dict(), DISPATCH_DATA
    ).spec.program.variables.keys() == {"p"}


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_the_dispatch_example_solves_and_its_expressions_fold() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    assert m.objective.value == pytest.approx(2500.0)
    xr.testing.assert_allclose(m.solution["p"], DISPATCH_P)
    spend = m.spec.expressions["spend"]
    xr.testing.assert_allclose(
        spend, (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )
    usage = m.spec.expressions["usage"]
    xr.testing.assert_allclose(usage, (DISPATCH_P / [100.0, 200.0]).rename("usage"))
    assert (
        set(m.spec.expressions) == {"spend", "usage"} and len(m.spec.expressions) == 2
    )
    assert set(m.spec.parameters.data_vars) == {"cost", "p_max"}
    assert m.spec.coords["generator"].equals(GENERATOR)


def synthetic_sources(program: Any, n: int = 3) -> dict[str, Any]:
    """Dense data for every declaration: labels per dimension, a linear ramp per parameter, cyclic lookups."""
    sources: dict[str, Any] = {}
    for dim, decl in program.dimensions.items():
        if decl.dtype == "int":
            sources[dim] = pd.Index(range(n), name=dim)
        elif decl.dtype == "datetime":
            sources[dim] = pd.date_range("2030-01-01", periods=n, freq="h", name=dim)
        else:
            sources[dim] = pd.Index([f"{dim}{i}" for i in range(n)], name=dim)
    for over, lk in program.lookups:
        if lk.target is not None:
            values = [sources[lk.target][i % n] for i in range(n)]
        else:
            values = (
                list(range(n))
                if lk.dtype == "int"
                else [f"{lk.name}{i}" for i in range(n)]
            )
        sources[lk.name] = pd.Series(values, index=sources[over])
    ramp = 1.0 + np.arange(n)
    for name, p in program.parameters.items():
        if p.derivation is not None:
            continue
        shape = [n] * len(p.dims)
        if p.dtype == "float":
            data = np.broadcast_to(ramp, shape).copy() if p.dims else np.array(1.0)
        elif p.dtype == "int":
            data = np.ones(shape, dtype=int)
        elif p.dtype == "bool":
            data = np.ones(shape, dtype=bool)
        else:
            data = np.full(shape, "a", dtype=object)
        if not p.dims:
            sources[name] = data.item()
        else:
            sources[name] = xr.DataArray(
                data, coords={d: sources[d] for d in p.dims}, dims=p.dims
            )
    return sources


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


# ---------------------------------------------------------------------------
# retain and evaluate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retain", "kept"),
    [
        ("report", {"cost", "p_max"}),
        ("all", {"cost", "load", "p_max"}),
        ("none", set()),
    ],
)
def test_retain_decides_what_the_fold_can_read(retain: str, kept: set[str]) -> None:
    m = solved(yaml_dict(), DISPATCH_DATA, retain=retain)
    assert set(m.parameters.data_vars) == kept
    want = (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    xr.testing.assert_allclose(m.spec.evaluate("spend", DISPATCH_DATA), want)
    if "cost" in kept:
        xr.testing.assert_allclose(m.spec.expressions["spend"], want)
    else:
        with pytest.raises(SpecDataError, match="not retained"):
            m.spec.expressions["spend"]


def test_an_unknown_expression_is_a_key_error_with_a_hint() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    with pytest.raises(KeyError, match="unknown named expression 'spent'.*spend"):
        m.spec.expressions["spent"]


def test_a_fold_over_variables_needs_a_solution_and_one_over_data_does_not() -> None:
    spec = {
        **yaml_dict(),
        "parameters": {
            **yaml_dict()["parameters"],
            "rate": {"dims": []},
            "years": {"dims": []},
        },
        "expressions": {
            "spend": "sum(p * cost, over=generator)",
            "growth": "rate ** years",
        },
    }
    m = Model.from_spec(spec, {**DISPATCH_DATA, "rate": 1.05, "years": 3.0})
    assert float(m.spec.expressions["growth"]) == pytest.approx(1.05**3)
    with pytest.raises(RuntimeError, match="no solution yet"):
        m.spec.expressions["spend"]


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


def with_(spec: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    for section, entries in sections.items():
        out[section] = {**spec.get(section, {}), **entries}
    return out


@pytest.mark.parametrize(
    ("spec", "data", "objective"),
    [
        pytest.param(
            SPARSE_SPEC,
            {"w": W_HOLE_AT_0, "c": FULL_C},
            19.0,
            id="coefficient-reads-as-zero",
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
            id="constant-side-behind-a-where-is-no-row",
        ),
    ],
)
def test_a_missing_row_is_a_zero_coefficient_or_no_row(
    spec: dict[str, Any], data: dict[str, Any], objective: float
) -> None:
    m = solved(spec, {"t": T, **data})
    assert m.objective.value == pytest.approx(objective)


@pytest.mark.parametrize(
    ("spec", "data", "match"),
    [
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
                objective={"sense": "maximize", "expression": "sum(x / w, over=t)"},
            ),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "the objective.*divisor",
            id="divisor-in-the-objective",
        ),
        pytest.param(
            with_(SPARSE_SPEC, expressions={"ratio": "x / w"}),
            {"w": W_HOLE_AT_0, "c": FULL_C},
            "expression 'ratio'.*divisor",
            id="divisor-in-a-named-expression",
        ),
    ],
)
def test_a_missing_row_is_refused_as_bound_constant_side_or_divisor(
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


GROUPED_SPEC: dict[str, Any] = {
    "dimensions": {"generator": {}, "bus": {"dtype": "str"}},
    "lookups": {"gen_bus": {"over": "generator", "into": "bus"}},
    "parameters": {"capacity": {"dims": ["generator"]}},
    "variables": {
        "imports": {"foreach": ["bus"], "bounds": {"lower": 0, "upper": 100}}
    },
    "constraints": {
        "import_limit": {
            "foreach": ["bus"],
            "expression": "imports <= sum(capacity, by=gen_bus)",
        }
    },
    "objective": {"sense": "maximize", "expression": "sum(imports, over=bus)"},
}
GENS = pd.Index(["g1", "g2"], name="generator")


def grouped_sources(capacity: pd.Series) -> dict[str, Any]:
    return {
        "bus": ["north", "south"],
        "generator": GENS,
        "gen_bus": pd.Series(["north", "north"], index=GENS),
        "capacity": capacity,
    }


def test_an_empty_group_on_the_constant_side_is_a_zero_and_not_a_gap() -> None:
    m = solved(GROUPED_SPEC, grouped_sources(pd.Series([3.0, 4.0], index=GENS)))
    assert m.objective.value == pytest.approx(7.0)
    assert float(m.solution["imports"].sel(bus="south")) == pytest.approx(0.0)


def test_a_member_with_no_value_is_still_refused_through_a_group() -> None:
    with pytest.raises(SpecDataError, match="parameter 'capacity' covers 1 fewer"):
        Model.from_spec(GROUPED_SPEC, grouped_sources(pd.Series([3.0], index=GENS[:1])))


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
    spend = solved(spec, data).spec.expressions["spend_by_unit"]
    masked = spend.sel(generator="gas")
    assert bool(masked.isnull().all()) is masked_reads_nan
    if not masked_reads_nan:
        assert float(masked.max()) == pytest.approx(0.0)
    assert not bool(spend.sel(generator="wind").isnull().any())


# ---------------------------------------------------------------------------
# operators, built as a constraint and folded as a named expression
# ---------------------------------------------------------------------------

TT = pd.Index([0, 1, 2, 3], name="t")
S = pd.Index(["a", "b"], name="s")
V = np.array([1.0, 2.0, 4.0, 8.0])
OPERATORS: dict[str, tuple[str, list[str], list[float]]] = {
    "shift-edge-0": ("shift(x, over=t, offset=1, edge=0)", ["t"], [0, 1, 2, 4]),
    "shift-ahead-edge-0": ("shift(x, over=t, offset=-1, edge=0)", ["t"], [2, 4, 8, 0]),
    "shift-wrap": ("shift(x, over=t, offset=1, edge='wrap')", ["t"], [8, 1, 2, 4]),
    "shift-wrap-in-groups": (
        "shift(x, over=t, offset=1, edge='wrap', by=season_of)",
        ["t"],
        [2, 1, 8, 4],
    ),
    "shift-by-group-offset": (
        "shift(x, over=t, offset=lag, edge=0, by=season_of)",
        ["t"],
        [0, 1, 0, 0],
    ),
    "sum-back": ("sum_back(x, over=t, within=2)", ["t"], [1, 3, 6, 12]),
    "sum-back-wrap": (
        "sum_back(x, over=t, within=2, edge='wrap')",
        ["t"],
        [9, 3, 6, 12],
    ),
    "sum-back-in-groups": (
        "sum_back(x, over=t, within=2, by=season_of)",
        ["t"],
        [1, 3, 4, 12],
    ),
    "sum-back-group-width": (
        "sum_back(x, over=t, within=width, by=season_of)",
        ["t"],
        [1, 2, 4, 12],
    ),
    "sum-by": ("sum(x, by=season_of)", ["s"], [3, 12]),
    "at": ("x * at(z, by=season_of)", ["t"], [10, 20, 80, 160]),
    "cases": ("x_state", ["t"], [100, 1, 2, 4]),
}


def operator_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int"}, "s": {"dtype": "str"}},
        "lookups": {"season_of": {"over": "t", "into": "s"}},
        "parameters": {
            "v": {"dims": ["t"]},
            "z": {"dims": ["s"]},
            "lag": {"dims": ["s"], "dtype": "int"},
            "width": {"dims": ["s"], "dtype": "int"},
        },
        "variables": {"x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": 100}}},
        "constraints": {"fix": {"foreach": ["t"], "expression": "x == v"}},
        "expressions": {
            "x_state": {
                "foreach": ["t"],
                "cases": {"first": {"when": "position(t) == 0", "expression": 100}},
                "otherwise": "shift(x, over=t, offset=1)",
            }
        },
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }
    for key, (expression, dims, _) in OPERATORS.items():
        name = key.replace("-", "_")
        spec["variables"][f"y_{name}"] = {
            "foreach": dims,
            "bounds": {"lower": -1000, "upper": 1000},
        }
        spec["constraints"][f"link_{name}"] = {
            "foreach": dims,
            "expression": f"y_{name} == {expression}",
        }
        spec["expressions"][f"probe_{name}"] = expression
    return spec


OPERATOR_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "season_of": pd.Series(["a", "a", "b", "b"], index=TT),
    "v": pd.Series(V, index=TT),
    "z": pd.Series([10.0, 20.0], index=S),
    "lag": pd.Series([1, 2], index=S),
    "width": pd.Series([1, 2], index=S),
}


@pytest.fixture(scope="module")
def operators_model() -> Model:
    with linopy.options as options:
        options["semantics"] = "v1"
        return solved(operator_spec(), OPERATOR_DATA, retain="all")


@pytest.mark.parametrize("key", OPERATORS)
def test_an_operator_builds_and_folds_alike(operators_model: Model, key: str) -> None:
    _, dims, expected = OPERATORS[key]
    name = key.replace("-", "_")
    want = xr.DataArray(expected, coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims)
    built = operators_model.solution[f"y_{name}"]
    folded = operators_model.spec.expressions[f"probe_{name}"]
    xr.testing.assert_allclose(built, want.rename(f"y_{name}"))
    xr.testing.assert_allclose(folded, want.rename(f"probe_{name}"))


# ---------------------------------------------------------------------------
# piecewise curves
# ---------------------------------------------------------------------------

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
    m = solved(spec, {**CURVE_DATA, **data}, retain="all")
    assert m.spec.expressions["spend"].item() == pytest.approx(spend)
    assert m.objective.value == pytest.approx(spend)


def without(series: pd.Series, *keys: tuple[str, int]) -> pd.Series:
    return series.drop(index=list(keys))


@pytest.mark.parametrize(
    ("spec", "data", "match"),
    [
        pytest.param(
            CURVE_SPEC,
            {"bp_x": without(FULL_X, ("gas", 3)), "bp_y": FULL_Y},
            "parameter 'bp_x' has no value at \\(generator='gas', bp=3\\)",
            id="a-hole-in-a-whole-curve",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": RAGGED_X, "bp_y": without(RAGGED_Y, ("gas", 3))},
            "Shorten it    'bp_x' claims this breakpoint",
            id="a-hole-inside-the-mask",
        ),
        pytest.param(
            MASKED_CURVE_SPEC,
            {"bp_x": without(FULL_X, ("gas", 1)), "bp_y": FULL_Y},
            "Not so at generator='gas'",
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
            "This curve carries 1",
            id="a-one-point-curve-under-lp",
        ),
    ],
)
def test_a_curve_the_method_cannot_build_is_refused(
    spec: dict[str, Any], data: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, {**CURVE_DATA, **data})


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
# where predicates
# ---------------------------------------------------------------------------

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
DAYS = pd.date_range("2030-01-01", periods=4, freq="D", name="d")
WHERE_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "d": DAYS,
    "season_of": pd.Series(["a", "a", "b"], index=TT[:3]),
    "other_of": pd.Series(["a", "b", "b", "a"], index=TT),
    "tag": pd.Series(["p", "q"], index=TT[:2]),
    "flag": pd.Series([True, False], index=TT[:2]),
    "cost": pd.Series([1.0, np.inf, 3.0], index=TT[:3]),
    "label": pd.Series(["u", "v"], index=TT[1:3]),
    "day_cost": pd.Series([1.0, 2.0, 3.0, 4.0], index=DAYS),
}
WHERE_CASES: dict[str, tuple[str, str, list[Any]]] = {
    "dimension-comparison": ("x", "t > 1", [2, 3]),
    "lookup-comparison": ("x", "season_of == 'a'", [0, 1]),
    "lookup-not-equal-skips-unmapped": ("x", "season_of != 'a'", [2]),
    "lookup-pair": ("x", "season_of != other_of", [1]),
    "lookup-defined": ("x", "season_of", [0, 1, 2]),
    "label-space-lookup": ("x", "tag == 'q'", [1]),
    "not": ("x", "NOT (t > 1)", [0, 1]),
    "and": ("x", "t > 0 AND t < 3", [1, 2]),
    "or": ("x", "t == 0 OR t == 3", [0, 3]),
    "position": ("x", "position(t) == -1", [3]),
    "position-in-groups": ("x", "position(t, by=season_of) == 0", [0, 2]),
    "bool-parameter": ("x", "flag", [0]),
    "float-parameter-must-be-finite": ("x", "cost", [0, 2]),
    "str-parameter": ("x", "label", [1, 2]),
    "parameter-comparison": ("x", "cost > 2", [2, 1]),
    "datetime-axis": ("y", "d >= '2030-01-03'", list(DAYS[2:])),
}


@pytest.mark.parametrize("case", WHERE_CASES)
def test_a_where_picks_the_rows_it_names(case: str) -> None:
    variable, predicate, labels = WHERE_CASES[case]
    spec = with_(
        WHERE_SPEC,
        variables={variable: {**WHERE_SPEC["variables"][variable], "where": predicate}},
    )
    built = Model.from_spec(spec, WHERE_DATA).variables[variable]
    dim = built.dims[0]
    present = built.labels[dim][(built.labels != -1).to_numpy()]
    assert sorted(present.to_numpy().tolist()) == sorted(labels)


@pytest.mark.parametrize(
    ("predicate", "match"),
    [
        ("position(t) == 7", "names position 7 of 't', which has 4"),
        ("position(t, by=season_of) == 1", "shorter than that: \\['b'\\]"),
    ],
)
def test_a_position_no_coordinate_holds_is_refused(predicate: str, match: str) -> None:
    spec = with_(
        WHERE_SPEC,
        variables={"x": {**WHERE_SPEC["variables"]["x"], "where": predicate}},
    )
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, WHERE_DATA)


# ---------------------------------------------------------------------------
# edges: partial lookups, swapped sides, constants, empty dimensions
# ---------------------------------------------------------------------------

PARTIAL_CASES: dict[str, list[float]] = {
    "sum-by": [3.0, 4.0],
    "at": [10.0, 20.0, 80.0, np.nan],
    "shift-wrap-in-groups": [2.0, 1.0, 4.0, np.nan],
    "sum-back-in-groups": [1.0, 3.0, 4.0, np.nan],
}


@pytest.mark.parametrize("key", PARTIAL_CASES)
def test_a_member_a_lookup_sends_nowhere_reaches_nothing(key: str) -> None:
    data = {**OPERATOR_DATA, "season_of": pd.Series(["a", "a", "b"], index=TT[:3])}
    m = solved(operator_spec(), data, retain="all")
    _, dims, _ = OPERATORS[key]
    name = key.replace("-", "_")
    folded = m.spec.expressions[f"probe_{name}"]
    want = xr.DataArray(
        PARTIAL_CASES[key], coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims
    )
    xr.testing.assert_allclose(folded, want.rename(folded.name))
    if dims == ["t"]:
        assert int(m.constraints[f"link_{name}"].labels.sel(t=3)) == -1


def test_a_constant_on_the_left_is_the_same_row() -> None:
    flipped = with_(
        SPARSE_SPEC, constraints={"cap": {"foreach": ["t"], "expression": "c >= w * x"}}
    )
    m = solved(flipped, {"t": T, "w": FULL_W, "c": FULL_C})
    assert m.objective.value == pytest.approx(9.0)


def test_a_constant_expression_folds_to_a_scalar() -> None:
    spec = {**yaml_dict(), "expressions": {"answer": "6 * 7"}}
    got = Model.from_spec(spec, DISPATCH_DATA).spec.expressions["answer"]
    assert got.ndim == 0 and float(got) == 42.0


def test_a_sum_beside_an_empty_dimension_is_the_empty_sum() -> None:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int"}, "s": {"dtype": "str"}},
        "variables": {"x": {"foreach": ["t", "s"], "bounds": {"lower": 0, "upper": 1}}},
        "constraints": {"cap": {"foreach": ["s"], "expression": "sum(x, over=t) <= 1"}},
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    m = Model.from_spec(spec, {"t": [0, 1], "s": pd.Index([], name="s", dtype=object)})
    assert "cap" not in m.constraints


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
# a power hides nothing
# ---------------------------------------------------------------------------


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
    assert {"c", "lag"} <= set(m.parameters.data_vars)
    xr.testing.assert_allclose(
        m.spec.expressions["e"],
        xr.DataArray([0.0, 0.0, 4.0], coords={"t": T}, name="e"),
    )


OTHER = pd.Index(["x", "y"], name="generator")


@pytest.mark.parametrize(
    ("generator", "match"),
    [
        pytest.param(GENERATOR[::-1], "as \\['gas', 'wind'\\]", id="reordered"),
        pytest.param(OTHER, "as \\['x', 'y'\\]", id="relabelled"),
    ],
)
def test_evaluate_refuses_sources_on_other_labels_than_the_model(
    generator: pd.Index, match: str
) -> None:
    m = solved({**yaml_dict(), "expressions": {"twice": "cost * 2"}}, DISPATCH_DATA)
    sources = {
        **DISPATCH_DATA,
        "generator": generator,
        "p_max": pd.Series([100.0, 200.0], index=generator),
        "cost": pd.Series([0.0, 50.0], index=generator),
    }
    with pytest.raises(SpecDataError, match=f"dimension 'generator' {match}"):
        m.spec.evaluate("twice", sources)


def test_a_window_width_no_member_carries_is_a_window_of_nothing() -> None:
    data = {
        **OPERATOR_DATA,
        "season_of": pd.Series(
            [], index=pd.Index([], name="t", dtype=int), dtype=object
        ),
    }
    m = solved(operator_spec(), data, retain="all")
    folded = m.spec.expressions["probe_sum_back_group_width"]
    assert bool(folded.isnull().all())
