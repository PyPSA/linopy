"""
``model.spec``, ``ModelSpec``, ``NamedExpression``, typesetting,
and the ``add_spec``/``from_spec`` argument handling that builds them.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

mathspec = pytest.importorskip("mathspec")
yaml = pytest.importorskip("yaml")
pytest.importorskip("highspy")

from mathspec import FormatName  # noqa: E402
from spec_helpers import (  # noqa: E402
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    GENERATOR,
    solved,
    with_,
    yaml_dict,
)

import linopy  # noqa: E402
from linopy import LinearExpression, Model  # noqa: E402
from linopy.spec import (  # noqa: E402
    Declaration,
    ModelSpec,
    NamedExpression,
    SpecDataError,
    lower,
)

# ---------------------------------------------------------------------------
# inputs and model integration
# ---------------------------------------------------------------------------


SPEC_FORMS: dict[str, Callable[[Path], Any]] = {
    "path": lambda path: path,
    "path-string": str,
    "yaml-text": lambda path: path.read_text(),
    "flow-yaml": lambda path: yaml.safe_dump(
        mathspec.to_spec(path).to_dict(), default_flow_style=True, width=10**6
    ).strip(),
    "dict": lambda path: mathspec.to_spec(path).to_dict(),
    "spec": lambda path: mathspec.to_spec(path),
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
    reread = lower(yaml.safe_load(m.spec.text))
    assert reread.constraints == m.spec.program.constraints
    assert isinstance(m.spec, ModelSpec)


def test_a_lowered_program_is_refused() -> None:
    program = lower(yaml_dict())
    with pytest.raises(TypeError, match="mathspec.Spec, not a Program"):
        Model().add_spec(program, DISPATCH_DATA)  # type: ignore[arg-type]


def _opened(tmp_path: Path) -> Any:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    with path.open() as handle:
        return handle


UNREADABLE_SPECS: list[Any] = [
    pytest.param(_opened, TypeError, "mathspec.Spec, not a", id="file-object"),
    pytest.param(
        lambda tmp_path: io.StringIO(EXAMPLE_DISPATCH),
        TypeError,
        "mathspec.Spec, not a StringIO",
        id="string-io",
    ),
    pytest.param(
        lambda tmp_path: tmp_path / "absent.yaml",
        FileNotFoundError,
        "No such file",
        id="missing-path",
    ),
    pytest.param(
        lambda tmp_path: str(tmp_path / "absent.yaml"),
        mathspec.errors.SchemaError,
        "holds a str, not a mapping of sections",
        id="missing-path-string-is-yaml-text",
    ),
    pytest.param(
        lambda tmp_path: {"description": "a spec that declares nothing"},
        SpecDataError,
        "the spec declares nothing",
        id="empty-spec",
    ),
    pytest.param(
        lambda tmp_path: "- p_max\n- load\n",
        mathspec.errors.SchemaError,
        "not a mapping of sections",
        id="sequence",
    ),
]


@pytest.mark.parametrize(("form", "error", "match"), UNREADABLE_SPECS)
def test_an_unreadable_spec_is_refused(
    tmp_path: Path, form: Callable[[Path], Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Model().add_spec(form(tmp_path), DISPATCH_DATA)


def test_add_spec_needs_an_empty_model() -> None:
    m = Model()
    m.add_variables(name="x")
    with pytest.raises(ValueError, match="empty model"):
        m.add_spec(yaml_dict(), DISPATCH_DATA)


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
# kept parameters
# ---------------------------------------------------------------------------


def test_the_spec_keeps_its_parameters_off_the_model() -> None:
    """``model.parameters`` is the caller's: a build neither reads nor writes it."""
    own = xr.DataArray(np.array(["a", "b", "c"], dtype=object), dims=["own"])
    m = Model()
    m.parameters["cost"] = own
    m.add_spec(with_(yaml_dict(), expressions={"twice": "cost * 2"}), DISPATCH_DATA)

    assert m.parameters["cost"].equals(own)
    assert m.spec.parameters["cost"].dims == ("generator",)


def test_a_build_missing_a_kept_parameter_leaves_the_model_buildable() -> None:
    """A dual-reading expression is skipped by the build, so what it reads is read before it."""
    spec = with_(
        yaml_dict(),
        parameters={"spare": {"dims": ["generator"]}},
        expressions={"price": "dual(power_balance) * sum(spare, over=generator)"},
    )
    m = Model()
    with pytest.raises(SpecDataError, match="no data provided for parameter 'spare'"):
        m.add_spec(spec, DISPATCH_DATA)
    assert not len(m.variables) and not len(m.constraints)

    spare = pd.Series([1.0, 2.0], index=GENERATOR)
    m.add_spec(spec, {**DISPATCH_DATA, "spare": spare})
    assert "spare" in m.spec.parameters


def test_a_removed_built_expression_names_what_is_not_kept() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    m.remove_expressions("spend")
    with pytest.raises(SpecDataError, match="parameter 'cost' is not kept"):
        m.spec.expressions["spend"].solution


def test_a_declared_dimension_with_no_source_still_reprs() -> None:
    """A dimension nothing reaches needs no source, so the repr must do without its labels."""
    spec = with_(
        yaml_dict(), dimensions={"spare": {"dtype": "int", "description": "unreached"}}
    )
    m = Model.from_spec(spec, DISPATCH_DATA)

    assert "spare (unreached)" in repr(m.spec)
    assert "snapshot (3)" in repr(m.spec)


def test_a_fold_over_data_needs_no_solve() -> None:
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


@pytest.mark.parametrize(
    ("name", "match"),
    [
        pytest.param("spend", "not optimized", id="built"),
        pytest.param("price", "no dual yet", id="unbuilt"),
    ],
)
def test_an_unsolved_model_has_no_solution_to_fold(name: str, match: str) -> None:
    spec = with_(yaml_dict(), expressions={"price": "dual(power_balance)"})
    m = Model.from_spec(spec, DISPATCH_DATA)
    with pytest.raises(AttributeError, match=match):
        m.spec.expressions[name].solution


# ---------------------------------------------------------------------------
# three views: math, the linopy expression and the solution
# ---------------------------------------------------------------------------

VIEWS_SPEC: dict[str, Any] = {
    **mathspec.to_spec(yaml.safe_load(EXAMPLE_DISPATCH)).to_dict(),
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
        ("bare", linopy.LinearExpression),
        ("levels", xr.DataArray),
        ("answer", float),
    ],
)
def test_expression_is_the_unsolved_linopy_term(name: str, kind: type) -> None:
    m = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA)
    assert isinstance(m.spec.expressions[name].expression, kind)


@pytest.mark.parametrize("name", ["spend", "bare"])
def test_a_variable_bearing_named_expression_is_the_model_expression(name: str) -> None:
    """The container holds one object per name; the data-only bodies stay on the spec."""
    m = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA)
    assert set(m.expressions) == {"spend", "bare"}
    assert m.spec.expressions[name].expression is m.expressions[name]
    assert m.expressions[name].spec == "spec"


def test_the_spec_keeps_what_its_unbuilt_expressions_read() -> None:
    m = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA)
    assert set(m.spec.parameters.data_vars) == {"cost"}
    assert not m.parameters.data_vars


def test_the_named_expression_bundles_the_three_views() -> None:
    m = solved(VIEWS_SPEC, DISPATCH_DATA)
    e = m.spec.expressions["spend"]
    assert e.node is m.spec.program.expressions["spend"].expression
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


def test_repr_caps_long_sections() -> None:
    spec = with_(yaml_dict(), expressions={f"e{i}": "p / p_max" for i in range(12)})
    text = repr(Model.from_spec(spec, DISPATCH_DATA).spec)
    assert "(+6 more)" in text
    assert "e11" not in text


def test_model_repr_of_a_spec_model_carries_no_tags() -> None:
    text = repr(Model.from_spec(yaml_dict(), DISPATCH_DATA))
    assert " * spend (snapshot)\n" in text
    assert " * usage (snapshot, generator)\n" in text
    assert " * p (snapshot, generator)\n" in text
    assert " * power_balance (snapshot)\n" in text
    assert "[spec]" not in text


def test_a_spec_without_a_description_reprs_bare() -> None:
    spec = {k: v for k, v in yaml_dict().items() if k != "description"}
    m = Model.from_spec(spec, DISPATCH_DATA)
    assert repr(m.spec).startswith("ModelSpec\n")


def test_the_spec_typesets_in_every_format() -> None:
    spec = Model.from_spec(yaml_dict(), DISPATCH_DATA).spec
    assert "align" in spec.to_latex()
    assert "```math" in spec.to_markdown()
    assert spec.to_typst()
    shown = spec._repr_markdown_()
    assert "```math" not in shown and "$`" not in shown
    assert "$$\n" in shown and " $t$ " in shown


TYPESETTERS: dict[FormatName, tuple[Callable[[Any], str], Callable[[Any], str]]] = {
    "latex": (ModelSpec.to_latex, NamedExpression.to_latex),
    "markdown": (ModelSpec.to_markdown, NamedExpression.to_markdown),
    "typst": (ModelSpec.to_typst, NamedExpression.to_typst),
}
LINE_TYPESETTERS = [line for _, line in TYPESETTERS.values()]


@pytest.mark.parametrize("fmt", TYPESETTERS)
def test_typeset_and_its_named_aliases_agree(fmt: FormatName) -> None:
    """The format is a parameter; the named methods only spell a common one."""
    spec = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec
    declaration = spec.declaration("p")
    document, line = TYPESETTERS[fmt]
    assert spec.typeset(fmt) == document(spec)
    assert declaration.typeset(fmt) == line(declaration)


HAND_BUILT: dict[str, Callable[[Model], Any]] = {
    "variable": lambda m: m.add_variables(lower=0, coords=[GENERATOR], name="r"),
    "constraint": lambda m: m.add_constraints(m.variables["p"] <= 10.0, name="cap"),
    "expression": lambda m: m.add_expressions(m.variables["p"] * 2.0, name="twice"),
    "objective": lambda m: m.add_objective(2 * m.objective.expression, overwrite=True),
    "objective-scaled": lambda m: setattr(m, "objective", m.objective * 2),
    "objective-extended": lambda m: setattr(
        m, "objective", m.objective + m.variables["p"].sum()
    ),
    "objective-expression": lambda m: setattr(
        m.objective, "expression", 2 * m.objective.expression
    ),
    "sos": lambda m: m.add_sos_constraints(
        m.variables["p"], sos_type=1, sos_dim="generator"
    ),
}


@pytest.mark.parametrize("add", HAND_BUILT.values(), ids=HAND_BUILT.keys())
def test_typesetting_a_model_grown_past_its_spec_warns(
    add: Callable[[Model], Any],
) -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    add(m)
    with pytest.warns(UserWarning, match="the spec does not declare") as record:
        m.spec.to_latex()
    assert record[0].filename == __file__


def test_an_objective_from_a_named_expression_keeps_its_stamp() -> None:
    spec = with_(yaml_dict(), expressions={"total": "sum(p * cost)"})
    m = Model.from_spec(spec, DISPATCH_DATA)
    total = m.expressions["total"]
    stamp = total.spec
    m.add_objective(total, overwrite=True)
    assert stamp is not None
    assert total.spec == stamp
    assert m.objective.expression.spec is None


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(yaml_dict(), id="plain"),
        pytest.param(
            with_(
                yaml_dict(),
                sos={"one": {"variable": "p", "along": "generator", "type": 1}},
            ),
            id="sos",
        ),
    ],
)
def test_a_spec_that_is_the_whole_model_typesets_without_a_word(
    declared: dict[str, Any],
) -> None:
    spec = Model.from_spec(declared, DISPATCH_DATA).spec
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert not spec.to_latex().startswith("%")


def test_a_named_expression_repr_markdown_wraps_only_itself() -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    assert e._repr_markdown_() == f"$$\n{e.to_markdown()}\n$$"


def test_a_named_expression_typeset_passes_options() -> None:
    e = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.expressions["spend"]
    assert e.to_latex(
        symbols={"notation": "latex", "names": {"spend": "S"}}
    ).startswith("S")


@pytest.mark.parametrize("name", ["spend", "power_balance", "p"])
@pytest.mark.parametrize("typeset", LINE_TYPESETTERS)
def test_a_declaration_typesets_to_one_line(
    name: str, typeset: Callable[[Declaration], str]
) -> None:
    d = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.declaration(name)
    line = typeset(d)
    assert line
    assert "\n" not in line
    assert "align" not in line and "$$" not in line


def test_declaration_reaches_every_kind_and_an_unknown_name_is_a_key_error() -> None:
    spec = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec
    assert spec.declaration("spend").to_latex() == spec.expressions["spend"].to_latex()
    with pytest.raises(KeyError, match="unknown declaration 'spent'.*spend"):
        spec.declaration("spent")


VIEWS: dict[str, Callable[[Any], Any]] = {
    "node": lambda d: d.node,
    "dims": lambda d: d.dims,
    "expression": lambda d: d.expression,
    "solution": lambda d: d.solution,
}


@pytest.mark.parametrize("name", ["power_balance", "p"])
@pytest.mark.parametrize("view", VIEWS.values(), ids=VIEWS.keys())
def test_a_constraint_or_variable_has_only_its_typesetting(
    name: str, view: Callable[[Any], Any]
) -> None:
    d = Model.from_spec(VIEWS_SPEC, DISPATCH_DATA).spec.declaration(name)
    assert type(d) is Declaration
    with pytest.raises(AttributeError):
        view(d)


def test_a_constant_expression_folds_to_a_scalar() -> None:
    spec = {**yaml_dict(), "expressions": {"answer": "6 * 7"}}
    got = Model.from_spec(spec, DISPATCH_DATA).spec.expressions["answer"].solution
    assert got.ndim == 0 and float(got) == 42.0


DUAL_SPEC: dict[str, Any] = {
    **yaml_dict(),
    "expressions": {"price": "dual(power_balance)"},
}


def test_a_reported_dual_folds_to_the_constraint_dual() -> None:
    m = solved(DUAL_SPEC, DISPATCH_DATA)
    xr.testing.assert_allclose(
        m.spec.expressions["price"].solution,
        m.constraints["power_balance"].dual.rename("price"),
    )


@pytest.mark.parametrize("build", [Model.from_spec, solved], ids=["unsolved", "solved"])
def test_a_dual_has_no_symbolic_form(build: Callable[..., Model]) -> None:
    m = build(DUAL_SPEC, DISPATCH_DATA)
    with pytest.raises(TypeError, match="no symbolic form"):
        m.spec.expressions["price"].expression


@pytest.mark.parametrize(
    ("sense", "rows", "duals"),
    [
        pytest.param(
            "minimize",
            {"left": "x >= 2", "right": "2 <= y", "fixed": "z == 4"},
            {"left": 1.0, "right": -1.0, "fixed": 1.0},
            id="minimize",
        ),
        pytest.param(
            "maximize",
            {"left": "x <= 2", "right": "2 >= y", "fixed": "4 == z"},
            {"left": 1.0, "right": -1.0, "fixed": -1.0},
            id="maximize",
        ),
    ],
)
def test_a_dual_is_the_rate_the_objective_rises_with_the_right_side(
    sense: str, rows: dict[str, str], duals: dict[str, float]
) -> None:
    bounds = {"bounds": {"lower": 0, "upper": 10}}
    spec = {
        "variables": {v: {"dims": [], **bounds} for v in "xyz"},
        "constraints": {k: {"dims": [], "expression": e} for k, e in rows.items()},
        "objective": {"sense": sense, "expression": "x + y + z"},
        "expressions": {f"dual_{k}": f"dual({k})" for k in rows},
    }
    m = solved(spec, {})
    got = {k: float(m.spec.expressions[f"dual_{k}"].solution) for k in rows}
    assert got == pytest.approx(duals)


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
    ("name", "dims"), [("spend", ("snapshot",)), ("usage", ("snapshot", "generator"))]
)
def test_named_expression_dims_are_static(name: str, dims: tuple[str, ...]) -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    expr = m.spec.expressions[name]
    assert expr.dims == dims
    assert isinstance(expr.expression, LinearExpression)
    assert set(expr.expression.coord_dims) == set(dims)


FRAMED: dict[str, dict[str, Any]] = {
    "built": {"dims": ["generator", "snapshot"], "expression": "p"},
    "unbuilt": {
        "dims": ["generator", "snapshot"],
        "cases": {"free": {"when": "cost == 0", "expression": "load"}},
        "otherwise": 0,
    },
}


@pytest.mark.parametrize("body", FRAMED.values(), ids=FRAMED.keys())
def test_a_named_expression_is_framed_in_its_declared_order(
    body: dict[str, Any],
) -> None:
    m = solved(with_(yaml_dict(), expressions={"framed": body}), DISPATCH_DATA)
    framed = m.spec.expressions["framed"]
    assert framed.dims == ("generator", "snapshot")
    assert framed.solution.dims == ("generator", "snapshot")


def test_aliasing_a_named_expression_leaves_the_original_stamped() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    alias = m.add_expressions(m.expressions["spend"], name="alias")
    spend = m.expressions["spend"]
    assert (spend.attrs["name"], spend.spec) == ("spend", "spec")
    assert (alias.attrs["name"], alias.spec) == ("alias", None)
