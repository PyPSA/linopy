"""
The term path a sparse model builds its declarations on: it builds the model
the eager evaluator builds, which takes over wherever the term path does not
cover a node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")

from spec_helpers import (  # noqa: E402
    CURVE_DATA,
    CURVE_SPEC,
    DISPATCH_DATA,
    FULL_X,
    FULL_Y,
    MASKED_CURVE_SPEC,
    RAGGED_X,
    RAGGED_Y,
    TT,
    needs_examples,
    with_,
    yaml_dict,
)
from test_spec_builder import (  # noqa: E402
    ABSENT_DATA,
    ABSENT_SPEC,
    BESIDE_A_VACATED_SHIFT,
    EXAMPLES,
    FULL_C,
    FULL_W,
    NAMED_WX,
    SPARSE_SPEC,
    T,
    example,
)
from test_spec_operators import (  # noqa: E402
    OPERATOR_DATA,
    RELATION_DATA,
    RELATION_SUMS,
    operator_spec,
    relations_spec,
)

from linopy import LinearExpression, Model, QuadraticExpression  # noqa: E402
from linopy.spec.terms import eager_only  # noqa: E402
from linopy.testing import (  # noqa: E402
    assert_conequal,
    assert_equivalent,
    assert_quadequal,
    assert_varequal,
    synthetic_sources,
)


def assert_alike(
    a: LinearExpression | QuadraticExpression,
    b: LinearExpression | QuadraticExpression,
) -> None:
    if isinstance(b, QuadraticExpression):
        assert_quadequal(a, b)
    else:
        assert_equivalent(a, b)


def assert_built_alike(spec: Any, data: dict[str, Any]) -> None:
    """
    *spec* built on the term path and on the eager path, equal as the solver
    reads it: the term path spreads a term onto the nonzero entries of a factor
    only, so it holds no term of coefficient zero where the eager path does.
    """
    with eager_only():
        eager = Model.from_spec(spec, data, sparse=True)
    built = Model.from_spec(spec, data, sparse=True)
    assert list(built.variables) == list(eager.variables)
    for name in eager.variables:
        assert_varequal(built.variables[name], eager.variables[name])
    assert list(built.constraints) == list(eager.constraints)
    for name in eager.constraints:
        con = eager.constraints[name]
        assert_conequal(built.constraints[name], con, strict=False)
        xr.testing.assert_equal(built.constraints[name].labels, con.labels)
    assert list(built.expressions) == list(eager.expressions)
    for name, expression in eager.expressions.items():
        assert_alike(built.expressions[name], expression)
    assert_alike(built.objective.expression, eager.objective.expression)
    assert built.objective.sense == eager.objective.sense


SPARSE_DATA = {"t": T, "w": FULL_W, "c": FULL_C}
PROGRAMS: dict[str, tuple[Any, dict[str, Any]]] = {
    "dispatch": (yaml_dict(), DISPATCH_DATA),
    "curve": (CURVE_SPEC, {**CURVE_DATA, "bp_x": FULL_X, "bp_y": FULL_Y}),
    "ragged-curve": (
        MASKED_CURVE_SPEC,
        {**CURVE_DATA, "bp_x": RAGGED_X, "bp_y": RAGGED_Y},
    ),
    "operators": (operator_spec(), OPERATOR_DATA),
    "relations": (relations_spec(), RELATION_DATA),
    **{f"relation-sum-{k}": v for k, v in RELATION_SUMS.items()},
    "sparse": (SPARSE_SPEC, SPARSE_DATA),
    "named-expression": (NAMED_WX, SPARSE_DATA),
    "beside-a-vacated-shift": (BESIDE_A_VACATED_SHIFT, SPARSE_DATA),
    "absent-rows": (
        with_(
            ABSENT_SPEC,
            constraints={"cap": {"dims": ["t"], "expression": "a * x <= 5"}},
        ),
        ABSENT_DATA,
    ),
}


@pytest.mark.parametrize(("spec", "data"), PROGRAMS.values(), ids=list(PROGRAMS))
def test_the_term_path_builds_the_model_the_eager_path_builds(
    spec: Any, data: dict[str, Any]
) -> None:
    assert_built_alike(spec, data)


@needs_examples
@pytest.mark.parametrize("path", [example(p) for p in EXAMPLES])
def test_every_mathspec_example_builds_alike_on_both_paths(path: Path) -> None:
    assert_built_alike(path, synthetic_sources(path))


U = pd.Index(["a", "b"], name="u")
G = pd.Index(["x", "z"], name="g")
TWO_DIMS: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}, "g": {"dtype": "str"}},
    "parameters": {"w": {"dims": ["t", "g"]}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0}},
        "y": {"dims": ["t", "g"], "bounds": {"lower": 0}},
    },
    "objective": {"sense": "minimize", "expression": "sum(x)"},
}
TWO_DIMS_DATA = {"t": TT, "g": G, "w": xr.DataArray(1.0, coords=[TT, G])}
FALLBACKS: dict[str, tuple[Any, dict[str, Any]]] = {
    "a-row-of-zero-weights": (
        {
            "dimensions": {"u": {"dtype": "str"}, "g": {"dtype": "str"}},
            "variables": {"b": {"dims": ["u"], "bounds": {"lower": 0}}},
            "parameters": {"w": {"dims": ["g", "u"]}},
            "constraints": {
                "cap": {"dims": ["g"], "expression": "sum(b * w, over=u) <= 4"}
            },
        },
        {"u": U, "g": G, "w": xr.DataArray([[1.0, 2.0], [0.0, 0.0]], coords=[G, U])},
    ),
    "a-product-of-two-terms": (
        with_(TWO_DIMS, objective={"sense": "minimize", "expression": "sum(x * x)"}),
        TWO_DIMS_DATA,
    ),
    "a-window-sum": (
        with_(
            TWO_DIMS,
            constraints={
                "cap": {
                    "dims": ["t"],
                    "expression": "sum_back(x, along=t, window=2) <= 3",
                }
            },
        ),
        TWO_DIMS_DATA,
    ),
    "a-right-side-over-a-dimension-the-left-lacks": (
        with_(
            TWO_DIMS,
            constraints={"cap": {"dims": ["t", "g"], "expression": "x <= w * y"}},
        ),
        TWO_DIMS_DATA,
    ),
}


@pytest.mark.parametrize(("spec", "data"), FALLBACKS.values(), ids=list(FALLBACKS))
def test_a_node_the_term_path_does_not_cover_builds_on_the_eager_path(
    spec: Any, data: dict[str, Any]
) -> None:
    assert_built_alike(spec, data)
