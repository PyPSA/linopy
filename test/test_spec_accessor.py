"""
``model.spec``, ``ModelSpec``, ``NamedExpression``, ``evaluate``, typesetting,
and the ``add_spec``/``from_spec`` argument handling that builds them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

from test_spec_builder import (  # noqa: E402
    BASE_MODEL,
    EXTRA_DATA,
    EXTRA_SPEC,
    SECOND_SPEC,
    SOS_SPEC,
    THREE,
    dispatch_p,
    extended,
    subset_bound,
)

import linopy  # noqa: E402
from conftest import (  # noqa: E402
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    GENERATOR,
    SNAPSHOT,
    solved,
    with_,
    yaml_dict,
)
from linopy import Model, breakpoints  # noqa: E402
from linopy.spec import (  # noqa: E402
    ModelSpec,
    NamedExpression,
    SpecDataError,
    Unspecified,
)
from linopy.testing import assert_linequal  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

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


def test_a_lowered_program_is_refused() -> None:
    program = math_spec.to_program(yaml_dict())
    with pytest.raises(TypeError, match="not a lowered Program"):
        Model().add_spec(program, DISPATCH_DATA)


def test_a_second_spec_must_bind_or_not_collide() -> None:
    m = Model()
    m.add_variables(name="x")
    m.add_spec(yaml_dict(), DISPATCH_DATA)
    assert list(m.variables) == ["x", "p"]
    with pytest.raises(ValueError, match="bind it or rename it"):
        m.add_spec(yaml_dict(), DISPATCH_DATA)


def test_a_bound_variable_is_read_not_built() -> None:
    m = extended()
    assert list(m.variables) == ["p"]
    assert m.spec.names == {"p": "p"}
    total = m.spec.expressions["total"]
    assert_linequal(total.expression, m.variables["p"].sum())
    m.solve(solver_name="highs", output_flag=False)
    assert float(total.solution) == pytest.approx(float(DISPATCH_P.sum()))


def test_a_binding_must_be_a_variable() -> None:
    with pytest.raises(SpecDataError, match="must be a linopy Variable to bind"):
        extended(p=3.0)


P_OVER_GENERATOR = with_(
    EXTRA_SPEC,
    variables={"p": {"foreach": ["generator"]}},
    constraints={"p_cap": {"foreach": ["generator"], "expression": "p <= cap"}},
)


def p_declared(**more: Any) -> dict[str, Any]:
    return with_(EXTRA_SPEC, variables={"p": {**EXTRA_SPEC["variables"]["p"], **more}})


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        pytest.param(P_OVER_GENERATOR, "Dimensions match by name", id="dims"),
        pytest.param(
            p_declared(bounds={"lower": 0}),
            "owns this variable's bounds and mask",
            id="bounds",
        ),
        pytest.param(
            p_declared(where="cap > 0"),
            "owns this variable's bounds and mask",
            id="where",
        ),
        pytest.param(
            p_declared(domain="binary"),
            "declared binary and the bound variable 'p' is continuous",
            id="domain",
        ),
    ],
)
def test_a_bound_variable_keeps_its_declared_shape(
    spec: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        extended(spec)


def test_a_bound_subset_is_reindexed_onto_the_master() -> None:
    """The spec spans three generators, the bound ``p`` two: its third column is absent, not a stranger."""
    m = subset_bound()
    assert m.spec.coords["generator"].equals(THREE)
    assert m.variables["p"].indexes["generator"].equals(GENERATOR)
    labels = m.constraints["p_cap"].labels
    assert labels.shape == (3, 3)
    assert (labels.sel(generator="solar") == -1).all()
    assert (labels.sel(generator=GENERATOR) != -1).all()
    total = m.spec.expressions["total"]
    read = total.expression.vars.values
    assert set(read[read != -1]) == set(m.variables["p"].labels.values.ravel())
    m.solve(solver_name="highs", output_flag=False)
    assert float(total.solution) == pytest.approx(float(DISPATCH_P.sum()))

    wider = Model()
    wider.add_variables(coords=[SNAPSHOT, THREE], name="p")
    with pytest.raises(SpecDataError, match="variable 'p' has label.*'solar'"):
        wider.add_spec(EXTRA_SPEC, {**EXTRA_DATA, "p": wider.variables["p"]})


def test_a_bound_variable_can_supply_a_dimension() -> None:
    m = BASE_MODEL()
    data = {"cap": EXTRA_DATA["cap"], "p": m.variables["p"]}
    m.add_spec(EXTRA_SPEC, data)
    assert m.spec.coords["generator"].equals(GENERATOR)
    assert m.spec.coords["snapshot"].equals(SNAPSHOT)
    with pytest.raises(SpecDataError, match="or bind a variable that spans it"):
        Model().add_spec(EXTRA_SPEC, {"cap": EXTRA_DATA["cap"]})


def two_bound_variables(q_generator: pd.Index, first: str) -> None:
    """``p`` and ``q`` bound with no generator source, *first* declared before the other."""
    m = BASE_MODEL()
    m.add_variables(coords=[SNAPSHOT, q_generator], name="q")
    declared = {**EXTRA_SPEC["variables"], "q": {"foreach": ["snapshot", "generator"]}}
    ordered = {first: declared[first], **declared}
    spec = {**EXTRA_SPEC, "variables": ordered}
    data = {"cap": EXTRA_DATA["cap"], "p": m.variables["p"], "q": m.variables["q"]}
    m.add_spec(spec, data)


def second_layer_disagrees() -> None:
    m = extended()
    again = {
        k: v for k, v in EXTRA_SPEC.items() if k not in ("constraints", "expressions")
    }
    data = {**EXTRA_DATA, "generator": GENERATOR[::-1], "p": m.variables["p"]}
    m.add_spec(again, data, name="again")


@pytest.mark.parametrize(
    "build",
    [
        lambda: extended(generator=GENERATOR[::-1]),
        lambda: two_bound_variables(GENERATOR[::-1], "p"),
        lambda: two_bound_variables(THREE, "p"),
        lambda: two_bound_variables(THREE, "q"),
        second_layer_disagrees,
    ],
    ids=[
        "sources-vs-bound",
        "bound-vs-bound",
        "narrower-bound-first",
        "wider-bound-first",
        "layer-vs-layer",
    ],
)
def test_dimension_labels_must_agree(build: Callable[[], None]) -> None:
    with pytest.raises(SpecDataError, match="same dimension name means the same axis"):
        build()


def test_a_later_layer_inherits_the_dimensions_it_does_not_key() -> None:
    """No generator source and a bound ``p`` over two: the master is the first layer's three."""
    m = subset_bound()
    floor = pd.Series([0.0, 0.0, 0.0], index=THREE)
    m.add_spec(SECOND_SPEC, {"floor": floor, "p": m.variables["p"]}, name="second")
    assert m.spec["second"].coords["generator"].equals(THREE)
    assert m.spec["second"].coords["snapshot"].equals(SNAPSHOT)
    labels = m.constraints["p_floor"].labels
    assert labels.indexes["generator"].equals(THREE)
    assert labels.indexes["snapshot"].equals(SNAPSHOT)
    assert (labels.sel(generator="solar") == -1).all()
    assert (labels.sel(generator=GENERATOR) != -1).all()


def with_sos(m: Model) -> None:
    m.add_sos_constraints(m.variables["p"], sos_type=1, sos_dim="generator")


@pytest.mark.parametrize(
    ("spec", "sources", "prepare", "match"),
    [
        pytest.param(
            EXTRA_SPEC, {"p": None}, None, "bind it or rename it", id="variable"
        ),
        pytest.param(
            with_(
                EXTRA_SPEC,
                constraints={"power_balance": EXTRA_SPEC["constraints"]["p_cap"]},
            ),
            {},
            None,
            r"constraint\(s\) \['power_balance'\]",
            id="constraint",
        ),
        pytest.param(
            SOS_SPEC,
            {},
            with_sos,
            r"special-ordered set on variable\(s\) \['p'\]",
            id="sos",
        ),
    ],
)
def test_collisions_are_refused(
    spec: dict[str, Any],
    sources: dict[str, Any],
    prepare: Callable[[Model], None] | None,
    match: str,
) -> None:
    m = BASE_MODEL()
    if prepare is not None:
        prepare(m)
    data = {**EXTRA_DATA, "p": m.variables["p"], **sources}
    data = {k: v for k, v in data.items() if v is not None}
    with pytest.raises(ValueError, match=match):
        m.add_spec(spec, data)


def test_an_expression_name_is_taken_once_across_specs() -> None:
    m = extended()
    again = {k: v for k, v in EXTRA_SPEC.items() if k != "constraints"}
    with pytest.raises(ValueError, match=r"named expression\(s\) \['total'\]"):
        m.add_spec(again, {**EXTRA_DATA, "p": m.variables["p"]})


def test_an_objective_on_a_non_empty_model_is_refused() -> None:
    spec = with_(EXTRA_SPEC, objective={"sense": "minimize", "expression": "sum(p)"})
    with pytest.raises(ValueError, match="already has one"):
        extended(spec)

    m = Model()
    dispatch_p(m)
    m.add_spec(spec, {**EXTRA_DATA, "p": m.variables["p"]}, name="extra")
    assert m.objective.sense == "min"
    assert m.spec.name == "extra"
    assert m.spec.objective_owner == "extra"
    assert m.spec.unspecified.objective is False


def test_a_binding_needs_a_mapping_source() -> None:
    ds = xr.Dataset(
        {"cap": EXTRA_DATA["cap"].to_xarray()}, coords={"snapshot": SNAPSHOT}
    )
    with pytest.raises(ValueError, match="bind it or rename it"):
        BASE_MODEL().add_spec(EXTRA_SPEC, ds)
    m = Model().add_spec(EXTRA_SPEC, ds)
    assert "p" in m.variables and m.spec.names == {}


def test_legacy_semantics_is_refused() -> None:
    with linopy.options as options:
        options["semantics"] = "legacy"
        with pytest.raises(ValueError, match="v1"):
            Model.from_spec(yaml_dict(), DISPATCH_DATA)


def test_a_model_without_a_spec_has_no_accessor() -> None:
    with pytest.raises(AttributeError, match="holds no spec"):
        _ = Model().spec


def test_from_spec_passes_model_kwargs_and_chains() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, force_dim_names=True)
    assert m.force_dim_names
    assert Model().add_spec(
        yaml_dict(), DISPATCH_DATA
    ).spec.program.variables.keys() == {"p"}


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
def test_retain_decides_what_is_kept_and_not_what_can_be_read(
    retain: str, kept: set[str]
) -> None:
    """A parameter retain dropped is read from the sources the model still holds."""
    m = solved(yaml_dict(), DISPATCH_DATA, retain=retain)
    assert set(m.spec.parameters.data_vars) == kept
    assert not m.parameters.data_vars
    want = (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    xr.testing.assert_allclose(m.spec.expressions["spend"].solution, want)
    xr.testing.assert_allclose(m.spec.evaluate("spend", DISPATCH_DATA).solution, want)


def test_the_spec_keeps_its_parameters_off_the_model() -> None:
    """``model.parameters`` is the caller's: a build neither reads nor writes it."""
    own = xr.DataArray(np.array(["a", "b", "c"], dtype=object), dims=["own"])
    m = Model()
    m.parameters["cost"] = own
    m.add_spec(yaml_dict(), DISPATCH_DATA, retain="all")

    assert m.parameters["cost"].equals(own)
    assert m.spec.parameters["cost"].dims == ("generator",)


def test_a_build_that_cannot_retain_leaves_the_model_buildable() -> None:
    """retain='all' reaches parameters no declaration does, and must not half-build on one."""
    spec = with_(yaml_dict(), parameters={"spare": {"dims": ["generator"]}})
    m = Model()
    with pytest.raises(SpecDataError, match="no data provided for parameter 'spare'"):
        m.add_spec(spec, DISPATCH_DATA, retain="all")
    assert not len(m.variables) and not len(m.constraints)

    spare = pd.Series([1.0, 2.0], index=GENERATOR)
    m.add_spec(spec, {**DISPATCH_DATA, "spare": spare}, retain="all")
    assert "spare" in m.spec.parameters


def test_a_declared_dimension_with_no_source_still_reprs() -> None:
    """A dimension nothing reaches needs no source, so the repr must do without its labels."""
    spec = with_(
        yaml_dict(), dimensions={"spare": {"dtype": "int", "description": "unreached"}}
    )
    m = Model.from_spec(spec, DISPATCH_DATA)

    assert "spare (unreached)" in repr(m.spec)
    assert "snapshot (3)" in repr(m.spec)


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
    assert float(m.spec.expressions["growth"].solution) == pytest.approx(1.05**3)
    with pytest.raises(RuntimeError, match="no solution yet"):
        m.spec.expressions["spend"].solution


# ---------------------------------------------------------------------------
# three views: math, the linopy expression and the solution
# ---------------------------------------------------------------------------

VIEWS_SPEC: dict[str, Any] = {
    **math_spec.to_spec(yaml.safe_load(EXAMPLE_DISPATCH)).to_dict(),
    "expressions": {
        "spend": "sum(p * cost, over=generator)",
        "bare": "p",
        "levels": "cost * 2",
        "answer": "6 * 7",
    },
}


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("spend", linopy.LinearExpression),
        ("bare", linopy.Variable),
        ("levels", xr.DataArray),
        ("answer", float),
    ],
)
def test_expression_is_the_unsolved_linopy_term(name: str, kind: type) -> None:
    m = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA)
    assert isinstance(m.spec.expressions[name].expression, kind)


def test_expression_reads_unsolved_but_solution_waits_for_a_solve() -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    assert isinstance(e.expression, linopy.LinearExpression)
    with pytest.raises(RuntimeError, match="no solution yet"):
        e.solution


def test_the_named_expression_bundles_the_three_views() -> None:
    m = solved(VIEWS_SPEC, DISPATCH_DATA)
    e = m.spec.expressions["spend"]
    assert e.node is m.spec.program.named_expressions["spend"].expression
    assert isinstance(e.expression, linopy.LinearExpression)
    xr.testing.assert_allclose(
        e.solution, (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )


def test_evaluate_returns_a_named_expression() -> None:
    m = solved(VIEWS_SPEC, DISPATCH_DATA, retain="none")
    e = m.spec.evaluate("spend", DISPATCH_DATA)
    assert isinstance(e, NamedExpression)
    assert isinstance(e.expression, linopy.LinearExpression)
    xr.testing.assert_allclose(
        e.solution, (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend")
    )


def test_repr_summarises_every_section() -> None:
    text = repr(Model.from_spec(yaml_dict(), DISPATCH_DATA).spec)
    assert text.startswith("ModelSpec: Least-cost dispatch")
    assert "Dimensions:  snapshot (3), generator (2)" in text
    assert "Variables:   p" in text
    assert "Constraints: power_balance" in text
    assert "Objective:   minimize" in text
    assert "Expressions: spend, usage" in text


def test_repr_of_several_layers_names_each() -> None:
    spec = two_layers().spec
    text = repr(spec)
    assert text.startswith("ModelSpec: layers spec, extra")
    assert "Layer 'spec': Least-cost dispatch" in text
    assert "Layer 'extra'\n" in text
    assert repr(spec["extra"]).startswith("Layer 'extra'\n  Dimensions:")


def test_repr_caps_long_sections() -> None:
    spec = with_(yaml_dict(), expressions={f"e{i}": "p / p_max" for i in range(12)})
    text = repr(Model.from_spec(spec, DISPATCH_DATA).spec)
    assert "(+6 more)" in text
    assert "e11" not in text


def test_model_repr_shows_the_spec_and_tags_only_expressions() -> None:
    text = repr(Model.from_spec(yaml_dict(), DISPATCH_DATA))
    assert "Linopy LP model, built from a math-spec" in text
    assert "Least-cost dispatch of a generator fleet against an hourly load." in text
    assert " * spend (snapshot) [spec]" in text
    assert " * usage (snapshot, generator) [spec]" in text
    assert " * p (snapshot, generator)\n" in text
    assert " * power_balance (snapshot)\n" in text
    assert "<empty>" not in text


def test_model_repr_of_an_extended_model_names_its_layers() -> None:
    m = extended()
    m.add_variables(lower=0, coords=[GENERATOR], name="reserve")
    text = repr(m)
    assert "Linopy LP model, extended by math-spec layer(s) extra" in text
    assert " * p (snapshot, generator) [extra]" in text
    assert " * reserve (generator)\n" in text
    assert " * p_cap (snapshot, generator) [extra]" in text
    assert " * power_balance (snapshot)\n" in text
    assert " * total () [extra]" in text


def test_model_repr_of_a_spec_without_a_description() -> None:
    spec = {k: v for k, v in yaml_dict().items() if k != "description"}
    m = Model.from_spec(spec, DISPATCH_DATA)
    assert m.spec.description == ""
    assert repr(m).startswith("Linopy LP model, built from a math-spec\n=")


def test_hybrid_model_tags_spec_variables_constraints_and_expressions() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    v = m.add_variables(lower=0, coords=[GENERATOR], name="reserve")
    m.add_expressions(v * 2.0, name="reserve_cost")
    m.add_constraints(v <= 10.0, name="reserve_cap")
    text = repr(m)
    assert " * p (snapshot, generator) [spec]" in text
    assert " * reserve (generator)\n" in text
    assert " * power_balance (snapshot) [spec]" in text
    assert " * reserve_cap (generator)\n" in text
    assert " * reserve_cost (generator)\n" in text
    assert " * spend (snapshot) [spec]" in text
    assert "<empty>" not in text


def two_layers() -> Model:
    """The dispatch example built from its spec, then extended by a second layer."""
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    return m.add_spec(EXTRA_SPEC, {**EXTRA_DATA, "p": m.variables["p"]}, name="extra")


def test_the_spec_typesets_in_every_format() -> None:
    spec = Model.from_spec(yaml_dict(), DISPATCH_DATA).spec
    assert "align" in spec.to_latex()
    assert "$$" in spec.to_markdown()
    assert spec.to_typst()
    assert spec._repr_markdown_() == spec.to_markdown()


@pytest.mark.parametrize("fmt", ["latex", "markdown", "typst"])
def test_two_layers_typeset_one_after_the_other(fmt: str) -> None:
    spec = two_layers().spec
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rendered = spec.typeset(fmt)
    assert rendered == f"{spec['spec'].typeset(fmt)}\n\n{spec['extra'].typeset(fmt)}"
    with pytest.raises(ValueError, match=r"model\.spec\[name\]\.typeset"):
        spec.typeset(fmt, standalone=True)
    assert spec["extra"].typeset(fmt, standalone=True)


def test_model_spec_layers() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    assert list(m.spec.layers) == ["spec"]
    assert m.spec.program is m.spec["spec"].program
    assert m.spec.whole and m.spec.objective_owner == "spec"

    m = two_layers()
    assert list(m.spec.layers) == ["spec", "extra"]
    assert m.spec["extra"].program.constraints.keys() == {"p_cap"}
    assert m.spec["extra"].names == {"p": "p"}
    with pytest.raises(ValueError, match=r"\['spec', 'extra'\]"):
        m.spec.program
    assert set(m.spec.expressions) == {"spend", "usage", "total"}
    assert m.spec.declaration("p_cap").to_latex()
    with pytest.raises(KeyError, match="unknown spec layer 'extr'.*extra"):
        m.spec["extr"]
    assert m.spec.whole and not extended().spec.whole


def test_layer_names(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    assert list(Model.from_spec(path, DISPATCH_DATA).spec.layers) == ["dispatch"]
    assert list(Model.from_spec(yaml_dict(), DISPATCH_DATA).spec.layers) == ["spec"]
    m = extended()
    assert list(m.spec.layers) == ["extra"]
    again = {
        k: v for k, v in EXTRA_SPEC.items() if k not in ("constraints", "expressions")
    }
    with pytest.raises(ValueError, match="layer named 'extra' is already"):
        m.add_spec(again, {**EXTRA_DATA, "p": m.variables["p"]}, name="extra")


@pytest.mark.parametrize("fmt", ["latex", "markdown", "typst"])
def test_typeset_and_its_named_aliases_agree(fmt: str) -> None:
    """The format is a parameter; the named methods only spell a common one."""
    spec = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec
    declaration = spec.declaration("p")
    assert spec.typeset(fmt) == getattr(spec, f"to_{fmt}")()
    assert declaration.typeset(fmt) == getattr(declaration, f"to_{fmt}")()


def hybrid() -> Model:
    """A spec-built model grown past its spec by hand."""
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    m.add_variables(lower=0, coords=[GENERATOR], name="reserve")
    m.add_constraints(m.variables["reserve"] <= 10.0, name="reserve_cap")
    return m


def test_unspecified_names_what_the_spec_does_not_declare() -> None:
    assert not Model.from_spec(yaml_dict(), DISPATCH_DATA).spec.unspecified
    assert hybrid().spec.unspecified == Unspecified(
        variables=("reserve",),
        constraints=("reserve_cap",),
        expressions=(),
        sos=(),
        piecewise=(),
        objective=False,
    )
    found = extended().spec.unspecified
    assert found.variables == ()
    assert found.constraints == ("power_balance",)


def test_unspecified_sees_what_carries_no_name_of_its_own() -> None:
    """An SOS is attributes on a variable, and a replaced objective is no name at all."""
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    m.add_expressions(m.variables["p"].sum("generator"), name="hand_expr")
    m.add_sos_constraints(m.variables["p"], sos_type=2, sos_dim="generator")
    m.add_objective(m.variables["p"].sum() * 3.0, overwrite=True)

    found = m.spec.unspecified
    assert found.expressions == ("hand_expr",)
    assert found.sos == ("p",)
    assert found.objective
    assert found.variables == () and found.constraints == ()


def test_a_piecewise_formulation_is_named_as_one_and_not_as_its_parts() -> None:
    """Its own variables and constraints are the formulation's business, not the tally's."""
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    k = pd.Index([0, 1], name="k")
    pts = {"k": k, "_breakpoint": [0, 1, 2]}
    x = m.add_variables(lower=0, upper=10, coords=[k], name="pw_x")
    y = m.add_variables(lower=0, upper=10, coords=[k], name="pw_y")
    m.add_piecewise_formulation(
        (x, breakpoints(xr.DataArray([[0.0, 5.0, 10.0]] * 2, coords=pts))),
        (y, breakpoints(xr.DataArray([[0.0, 1.0, 4.0]] * 2, coords=pts))),
        name="curve",
    )

    found = m.spec.unspecified
    assert found.piecewise == ("curve",)
    assert found.variables == ("pw_x", "pw_y")
    assert found.constraints == ()


@pytest.mark.parametrize(
    ("build", "match", "tallied"),
    [
        pytest.param(
            hybrid,
            "drifted from the spec",
            ["1 variable (reserve)", "1 constraint (reserve_cap)"],
            id="whole",
        ),
        pytest.param(
            extended,
            "extends a model it does not describe",
            ["1 constraint (power_balance)"],
            id="extended",
        ),
    ],
)
@pytest.mark.parametrize(
    ("fmt", "opener"), [("latex", "%"), ("markdown", "<!--"), ("typst", "//")]
)
def test_typesetting_a_hybrid_model_warns_and_says_so_in_the_source(
    build: Callable[[], Model], match: str, tallied: list[str], fmt: str, opener: str
) -> None:
    """The tally is a comment of the format's own: gone once compiled, there in the source."""
    with pytest.warns(UserWarning, match=match):
        rendered = build().spec.typeset(fmt)

    first = rendered.splitlines()[0]
    assert first.startswith(opener)
    for part in tallied:
        assert part in first


def test_a_spec_that_is_the_whole_model_typesets_without_a_word() -> None:
    spec = Model.from_spec(yaml_dict(), DISPATCH_DATA).spec
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert not spec.to_latex().startswith("%")


def test_a_notebook_sees_a_note_the_warning_would_not_reach() -> None:
    """A notebook swallows warnings, so the rendered Markdown carries the tally visibly."""
    with pytest.warns(UserWarning):
        rendered = hybrid().spec._repr_markdown_()

    assert rendered.splitlines()[-1] == (
        "*This model has drifted from the spec typeset here: "
        "1 variable (reserve) and 1 constraint (reserve_cap).*"
    )


@pytest.mark.parametrize("fmt", ["to_latex", "to_markdown", "to_typst"])
def test_a_named_expression_typesets_to_one_line(fmt: str) -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    line = getattr(e, fmt)()
    assert "spend" in line
    assert "\n" not in line
    assert "align" not in line and "$$" not in line


def test_a_named_expression_repr_markdown_wraps_only_itself() -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    assert e._repr_markdown_() == f"$$\n{e.to_markdown()}\n$$"


def test_a_named_expression_typeset_passes_options() -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    assert e.to_latex(
        symbols={"notation": "latex", "names": {"spend": "S"}}
    ).startswith("S")


@pytest.mark.parametrize("name", ["power_balance", "p"])
@pytest.mark.parametrize("fmt", ["to_latex", "to_markdown", "to_typst"])
def test_a_constraint_or_variable_typesets_to_one_line(name: str, fmt: str) -> None:
    d = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.declaration(name)
    line = getattr(d, fmt)()
    assert line
    assert "\n" not in line
    assert "align" not in line and "$$" not in line


def test_declaration_reaches_every_kind_and_an_unknown_name_is_a_key_error() -> None:
    spec = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec
    assert spec.declaration("spend").to_latex() == spec.expressions["spend"].to_latex()
    with pytest.raises(KeyError, match="unknown declaration 'spent'.*spend"):
        spec.declaration("spent")


def test_a_constant_expression_folds_to_a_scalar() -> None:
    spec = {**yaml_dict(), "expressions": {"answer": "6 * 7"}}
    got = Model.from_spec(spec, DISPATCH_DATA).spec.expressions["answer"].solution
    assert got.ndim == 0 and float(got) == 42.0


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


def test_a_reported_dual_folds_to_the_constraint_dual() -> None:
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = solved(spec, DISPATCH_DATA)
    xr.testing.assert_allclose(
        m.spec.expressions["price"].solution,
        m.constraints["power_balance"].dual.rename("price"),
    )


def test_a_dual_needs_a_solution() -> None:
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = Model.from_spec(spec, DISPATCH_DATA)
    with pytest.raises(RuntimeError, match="no dual yet"):
        m.spec.expressions["price"].expression


def test_spec_api_warns_once_per_session() -> None:
    from linopy import EvolvingAPIWarning
    from linopy.constants import _emitted_evolving_warnings

    _emitted_evolving_warnings.discard("spec")
    with pytest.warns(EvolvingAPIWarning, match="spec: Model.add_spec"):
        Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EvolvingAPIWarning)
        Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)


@pytest.mark.parametrize(
    ("build", "name", "dims"),
    [
        (lambda: Model.from_spec(yaml_dict(), DISPATCH_DATA), "spend", ("snapshot",)),
        (
            lambda: Model.from_spec(yaml_dict(), DISPATCH_DATA),
            "usage",
            ("snapshot", "generator"),
        ),
        (extended, "total", ()),
    ],
    ids=["spend", "usage", "bound"],
)
def test_named_expression_dims_are_static(
    build: Callable[[], Model], name: str, dims: tuple[str, ...]
) -> None:
    expr = build().spec.expressions[name]
    assert expr.dims == dims
    assert set(expr.expression.coord_dims) == set(dims)
