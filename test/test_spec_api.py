"""
``Model.from_spec`` argument handling, the data it stores in ``model.parameters``,
and ``linopy.spec.evaluate`` reading a spec-built model.
"""

from __future__ import annotations

import io
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

mathspec = pytest.importorskip("mathspec")
yaml = pytest.importorskip("yaml")
pytest.importorskip("highspy")

from spec_helpers import (  # noqa: E402
    DISPATCH_DATA,
    DISPATCH_P,
    EXAMPLE_DISPATCH,
    GENERATOR,
    TT,
    WHERE_DATA,
    WHERE_SPEC,
    solved,
    with_,
    yaml_dict,
)

import linopy  # noqa: E402
from linopy import LinearExpression, Model  # noqa: E402
from linopy.spec import SpecDataError, evaluate  # noqa: E402
from linopy.spec.api import lower  # noqa: E402
from linopy.testing import synthetic_sources  # noqa: E402

# ---------------------------------------------------------------------------
# inputs and model integration
# ---------------------------------------------------------------------------


SPEC_FORMS: dict[str, Callable[[Path], Any]] = {
    "path": lambda path: path,
    "yaml-text": lambda path: path.read_text(),
    "flow-yaml": lambda path: yaml.safe_dump(
        mathspec.to_spec(path).to_dict(), default_flow_style=True, width=10**6
    ).strip(),
    "dict": lambda path: mathspec.to_spec(path).to_dict(),
    "spec": lambda path: mathspec.to_spec(path),
}


def _written(tmp_path: Path) -> Path:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    return path


@pytest.mark.parametrize("form", SPEC_FORMS.values(), ids=SPEC_FORMS.keys())
def test_spec_forms_build_the_same_model(
    tmp_path: Path, form: Callable[[Path], Any]
) -> None:
    m = Model.from_spec(form(_written(tmp_path)), DISPATCH_DATA)
    assert list(m.variables) == ["p"]
    assert list(m.constraints) == ["power_balance"]


@pytest.mark.parametrize("form", SPEC_FORMS.values(), ids=SPEC_FORMS.keys())
def test_synthetic_sources_take_every_spec_form(
    tmp_path: Path, form: Callable[[Path], Any]
) -> None:
    path = _written(tmp_path)
    sources = synthetic_sources(form(path))
    assert sources.keys() == synthetic_sources(path).keys()
    assert list(Model.from_spec(path, sources).variables) == ["p"]


@pytest.mark.parametrize(
    "use",
    [
        pytest.param(lambda s: Model.from_spec(s, DISPATCH_DATA), id="from_spec"),
        pytest.param(synthetic_sources, id="synthetic_sources"),
    ],
)
def test_a_lowered_program_is_refused(use: Callable[[Any], Any]) -> None:
    with pytest.raises(TypeError, match="mathspec.Spec, not a Program"):
        use(lower(yaml_dict()))


def _opened(tmp_path: Path) -> Any:
    with _written(tmp_path).open() as handle:
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
        ValueError,
        r"pathlib\.Path",
        id="missing-path-string-is-yaml-text",
    ),
    pytest.param(
        lambda tmp_path: str(_written(tmp_path)),
        ValueError,
        r"pathlib\.Path",
        id="existing-path-string-is-yaml-text",
    ),
    pytest.param(
        lambda tmp_path: " Dispatch.YML ", ValueError, r"pathlib\.Path", id="yml-name"
    ),
    pytest.param(
        lambda tmp_path: "description: see a.yaml",
        SpecDataError,
        "the spec declares nothing",
        id="one-line-yaml-mentioning-a-file",
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
        Model.from_spec(form(tmp_path), DISPATCH_DATA)


def test_from_spec_passes_model_kwargs() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA, force_dim_names=True)
    assert m.force_dim_names


def test_the_spec_package_exports() -> None:
    assert isinstance(linopy.spec.attach, types.ModuleType)
    assert set(linopy.spec.__all__) == {"SpecDataError", "evaluate"}
    assert callable(linopy.spec.evaluate)


def test_spec_api_warns_once_per_session() -> None:
    from linopy import EvolvingAPIWarning
    from linopy.constants import _emitted_evolving_warnings

    _emitted_evolving_warnings.discard("spec")
    with pytest.warns(EvolvingAPIWarning, match="spec: Model.from_spec"):
        Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("error", EvolvingAPIWarning)
        Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)


def test_model_repr_of_a_spec_model_carries_no_tags() -> None:
    text = repr(Model.from_spec(yaml_dict(), DISPATCH_DATA))
    assert " * spend (snapshot)\n" in text
    assert " * usage (snapshot, generator)\n" in text
    assert " * p (snapshot, generator)\n" in text
    assert " * power_balance (snapshot)\n" in text
    assert "[spec]" not in text


def test_aliasing_a_named_expression_leaves_the_original_named() -> None:
    m = Model.from_spec(yaml_dict(), DISPATCH_DATA)
    alias = m.add_expressions(m.expressions["spend"], name="alias")
    assert m.expressions["spend"].attrs["name"] == "spend"
    assert alias.attrs["name"] == "alias"


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("store", [False, True])
def test_from_spec_stores_parameters_only_when_asked(store: bool) -> None:
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, store=store)
    program = lower(WHERE_SPEC)
    names = {*program.parameters, *program.relations} if store else set()
    assert set(m.parameters.data_vars) == names


def test_from_spec_stores_raw_parameters_and_relations() -> None:
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, store=True)
    for name in ("cost", "season_of"):
        assert m.parameters[name].isnull().values.tolist() == [False] * 3 + [True]


def test_a_missing_parameter_is_refused() -> None:
    """Every parameter is read, even one only an unbuilt expression reads."""
    spec = with_(
        yaml_dict(),
        parameters={"spare": {"dims": ["generator"]}},
        expressions={"price": "dual(power_balance) * sum(spare, over=generator)"},
    )
    with pytest.raises(SpecDataError, match="no data provided for parameter 'spare'"):
        Model.from_spec(spec, DISPATCH_DATA)

    spare = pd.Series([1.0, 2.0], index=GENERATOR)
    m = Model.from_spec(spec, {**DISPATCH_DATA, "spare": spare}, store=True)
    assert "spare" in m.parameters


def test_extra_stored_parameters_are_ignored(recwarn: Any) -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    m.parameters["costs"] = m.parameters["cost"] * 2
    m.parameters["note"] = xr.DataArray(1.0)
    rebuilt = Model.from_spec(yaml_dict(), m.parameters, store=True)
    got = evaluate(yaml_dict(), m, "spend")["spend"]
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []
    xr.testing.assert_allclose(got, m.expressions["spend"].solution)
    assert set(rebuilt.parameters.data_vars) == {"p_max", "load", "cost"}


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "want"),
    [((), ["spend", "usage"]), (("usage",), ["usage"])],
    ids=["all", "one"],
)
def test_evaluate_folds_the_named_expressions(
    names: tuple[str, ...], want: list[str]
) -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    got = evaluate(yaml_dict(), m, *names)
    assert list(got) == want
    for name in want:
        xr.testing.assert_allclose(got[name], m.expressions[name].solution)


def test_evaluate_matches_the_dispatch() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    xr.testing.assert_allclose(
        evaluate(yaml_dict(), m, "spend")["spend"],
        (DISPATCH_P * [0.0, 50.0]).sum("generator").rename("spend"),
    )


INDEXED_ONLY: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int"}},
    "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 1}}},
    "objective": {"sense": "maximize", "expression": "sum(x)"},
    "expressions": {"tot": "sum(x)"},
}


@pytest.mark.parametrize(
    ("spec", "model", "name", "match"),
    [
        pytest.param(
            yaml_dict(),
            lambda: Model.from_spec(yaml_dict(), DISPATCH_DATA),
            "spent",
            "unknown expression 'spent'",
            id="unknown-name",
        ),
        pytest.param(
            yaml_dict(), Model, "spend", "model.parameters holds no", id="not-built"
        ),
        pytest.param(
            yaml_dict(),
            lambda: Model.from_spec(yaml_dict(), DISPATCH_DATA),
            "spend",
            r"sources=\.\.\.\), or build the model with .*store=True",
            id="not-stored",
        ),
        pytest.param(
            INDEXED_ONLY,
            lambda: Model.from_spec(INDEXED_ONLY, {"t": TT}),
            "tot",
            r"holds no 't'.*store=True",
            id="index-not-stored",
        ),
    ],
)
def test_evaluate_refuses_with_a_key_error(
    spec: dict[str, Any], model: Callable[[], Model], name: str, match: str
) -> None:
    with pytest.raises(KeyError, match=match):
        evaluate(spec, model(), name)


def test_evaluate_reads_given_sources_over_stored_parameters() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    unstored = m.copy(include_solution=True)
    unstored.parameters = xr.Dataset()
    cheap = {**DISPATCH_DATA, "cost": DISPATCH_DATA["cost"] / 2}
    stored = evaluate(yaml_dict(), m, "spend")["spend"]
    given = evaluate(yaml_dict(), unstored, "spend", sources=DISPATCH_DATA)["spend"]
    xr.testing.assert_allclose(given, stored)
    halved = evaluate(yaml_dict(), m, "spend", sources=cheap)["spend"]
    xr.testing.assert_allclose(halved, stored / 2)


def test_evaluate_after_removing_the_built_expression() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    original = m.expressions["spend"].solution
    m.remove_expressions("spend")
    xr.testing.assert_allclose(evaluate(yaml_dict(), m, "spend")["spend"], original)


def test_evaluate_follows_assign_coords() -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    before = evaluate(yaml_dict(), m, "spend")["spend"]
    m.assign_coords(generator=["w", "g"])
    xr.testing.assert_allclose(evaluate(yaml_dict(), m, "spend")["spend"], before)


def test_evaluate_applies_missing_in_a_condition() -> None:
    """A ``neutral`` hole fills to 0 as a value, but a condition still reads it as absent."""
    spec = {
        "dimensions": {"t": {"dtype": "int"}},
        "parameters": {"v": {"dims": ["t"], "missing": "neutral"}},
        "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 1}}},
        "objective": {"sense": "maximize", "expression": "sum(x)"},
        "expressions": {
            "big": {
                "dims": ["t"],
                "cases": {"on": {"when": "v < 5", "expression": "x"}},
                "otherwise": 0,
            }
        },
    }
    t = TT[:3]
    m = solved(spec, {"t": t, "v": pd.Series([1.0, 9.0], index=t[:2])})
    assert evaluate(spec, m, "big")["big"].values.tolist() == [1.0, 0.0, 0.0]


def test_a_fold_over_data_needs_no_solve() -> None:
    spec = with_(
        yaml_dict(),
        parameters={"rate": {"dims": []}, "years": {"dims": []}},
        expressions={"growth": "rate ** years"},
    )
    sources = {**DISPATCH_DATA, "rate": 1.05, "years": 3.0}
    m = Model.from_spec(spec, sources)
    growth = evaluate(spec, m, "growth", sources=sources)["growth"]
    assert float(growth) == pytest.approx(1.05**3)


@pytest.mark.parametrize(
    ("name", "match"),
    [
        pytest.param("spend", "has no solution yet", id="built"),
        pytest.param("price", "no dual yet", id="unbuilt"),
    ],
)
def test_an_unsolved_model_has_no_solution_to_fold(name: str, match: str) -> None:
    spec = with_(yaml_dict(), expressions={"price": "dual(power_balance)"})
    m = Model.from_spec(spec, DISPATCH_DATA, store=True)
    with pytest.raises(AttributeError, match=match):
        evaluate(spec, m, name)


def test_a_constant_expression_folds_to_a_scalar() -> None:
    spec = {**yaml_dict(), "expressions": {"answer": "6 * 7"}}
    m = Model.from_spec(spec, DISPATCH_DATA)
    got = evaluate(spec, m, "answer", sources=DISPATCH_DATA)["answer"]
    assert got.ndim == 0 and float(got) == 42.0


def test_a_reported_dual_folds_to_the_constraint_dual() -> None:
    spec = {**yaml_dict(), "expressions": {"price": "dual(power_balance)"}}
    m = solved(spec, DISPATCH_DATA)
    xr.testing.assert_allclose(
        evaluate(spec, m, "price")["price"],
        m.constraints["power_balance"].dual.rename("price"),
    )


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
    folded = evaluate(spec, solved(spec, {}))
    assert {k: float(folded[f"dual_{k}"]) for k in rows} == pytest.approx(duals)


@pytest.mark.parametrize(
    ("name", "dims"), [("spend", ("snapshot",)), ("usage", ("snapshot", "generator"))]
)
def test_named_expression_dims_are_static(name: str, dims: tuple[str, ...]) -> None:
    m = solved(yaml_dict(), DISPATCH_DATA)
    assert isinstance(m.expressions[name], LinearExpression)
    assert set(m.expressions[name].coord_dims) == set(dims)
    assert evaluate(yaml_dict(), m, name)[name].dims == dims


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
    spec = with_(yaml_dict(), expressions={"framed": body})
    folded = evaluate(spec, solved(spec, DISPATCH_DATA), "framed")["framed"]
    assert folded.dims == ("generator", "snapshot")
