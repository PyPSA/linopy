"""
``model.spec``, ``ModelSpec``, ``NamedExpression``, ``evaluate``, typesetting,
and the ``add_spec``/``from_spec`` argument handling that builds them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from conftest import (  # noqa: E402
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    GENERATOR,
    solved,
    yaml_dict,
)
from linopy import Model  # noqa: E402
from linopy.spec import ModelSpec, NamedExpression, SpecDataError  # noqa: E402

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
    xr.testing.assert_allclose(m.spec.evaluate("spend", DISPATCH_DATA).solution, want)
    if "cost" in kept:
        xr.testing.assert_allclose(m.spec.expressions["spend"].solution, want)
    else:
        with pytest.raises(SpecDataError, match="not retained"):
            m.spec.expressions["spend"].solution


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
    assert e.node is m.spec.program.named_expressions["spend"]
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


def test_the_whole_model_typesets() -> None:
    spec = Model.from_spec(yaml_dict(), DISPATCH_DATA).spec
    assert "align" in spec.to_latex()
    assert "$$" in spec.to_markdown()
    assert spec.to_typst()
    assert spec._repr_markdown_() == spec.to_markdown()


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
