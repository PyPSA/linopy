"""
Piecewise curve derivation and validation: whole and ragged breakpoint
tables, the checks that refuse a curve a method cannot build, and the
convex-hull method's single-bend requirement.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from conftest import (  # noqa: E402
    CURVE_DATA,
    CURVE_SPEC,
    FULL_X,
    FULL_Y,
    MASKED_CURVE_SPEC,
    RAGGED_X,
    RAGGED_Y,
    UNITS,
    curve,
    solved,
    with_,
)
from linopy import Model  # noqa: E402
from linopy.spec import SpecDataError  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

# ---------------------------------------------------------------------------
# piecewise curves
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
    m = solved(spec, {**CURVE_DATA, **data}, retain="all")
    assert m.spec.expressions["spend"].solution.item() == pytest.approx(spend)
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
