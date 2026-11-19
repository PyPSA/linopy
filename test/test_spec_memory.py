"""
The memory a spec build takes for a table parameter over a large grid.

A parameter is laid out on the full grid of its dimensions (#1072), so these
tests pin the build's peak as a multiple of that grid: an unread parameter
takes no grid, and a read one is laid out once. The peak is traced
with ``tracemalloc``, which sees numpy's allocations, so the tests run without
pytest-memray's ``--memray``.
"""

from __future__ import annotations

import gc
import tracemalloc
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mathspec")

from linopy import EvolvingAPIWarning, Model  # noqa: E402

T = pd.Index(range(4000), name="t")
G = pd.Index([f"g{i}" for i in range(1000)], name="g")
GRID = len(T) * len(G) * 8
DECLARATIONS = """
dimensions: {t: {dtype: int}, g: {}}
parameters: {c: {dims: [t, g], missing: absent}}
variables: {x: {dims: [g], bounds: {lower: 0, upper: 1}}}
objective: {sense: minimize, expression: "sum(x)"}
"""
SUMMED = (
    DECLARATIONS + 'constraints: {k: {dims: [g], expression: "x <= sum(c, over=t)"}}\n'
)


@pytest.fixture(scope="module")
def sparse() -> pd.DataFrame:
    """20,000 rows over the 4 million cells of the (t, g) grid."""
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {
            "t": T[rng.integers(0, len(T), 20_000)],
            "g": G[rng.integers(0, len(G), 20_000)],
        }
    )
    return rows.drop_duplicates().assign(c=1.0)


def peak(build: Callable[[], Any]) -> int:
    """The peak traced memory of *build*, after a warm-up call has imported and lowered what it needs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EvolvingAPIWarning)
        build()
        gc.collect()
        tracemalloc.start()
        try:
            build()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()


def test_an_unread_table_parameter_takes_no_grid(sparse: pd.DataFrame) -> None:
    used = peak(lambda: Model.from_spec(DECLARATIONS, {"t": T, "g": G, "c": sparse}))
    assert used < 0.25 * GRID


def test_a_summed_table_parameter_is_laid_out_once(sparse: pd.DataFrame) -> None:
    used = peak(lambda: Model.from_spec(SUMMED, {"t": T, "g": G, "c": sparse}))
    assert used < 2.75 * GRID


def test_a_build_frees_the_grids_it_read_when_it_returns(
    sparse: pd.DataFrame,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EvolvingAPIWarning)
        Model.from_spec(SUMMED, {"t": T, "g": G, "c": sparse})
        gc.collect()
        gc.disable()
        tracemalloc.start()
        try:
            # mathspec caches what it parses, and only a fresh parse leaves
            # the cycles that kept the build's frame and its grids alive
            fresh = SUMMED.replace("over=t)", "over=t) * 1")
            model = Model.from_spec(fresh, {"t": T, "g": G, "c": sparse})
            held = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
            gc.enable()
    assert model.constraints["k"].shape == (len(G),)
    assert held < 0.25 * GRID


BUSES = pd.Index([f"b{i}" for i in range(400)], name="b")
JOINED = """
dimensions: {t: {dtype: int}, g: {}, b: {}}
relations: {gen_bus: {key: g, values: b}}
parameters: {c: {dims: [g]}, d: {dims: [b]}}
variables: {x: {dims: [g, t], bounds: {lower: 0, upper: 1}}}
expressions:
  rated: {dims: [g, t], cases: {on: {when: "c > 0", expression: "c * x"}}, otherwise: 0}
  rating: {dims: [b], cases: {on: {when: "d > 0", expression: "d"}}, otherwise: 0}
objective: {sense: minimize, expression: "sum(x)"}
"""


@pytest.mark.parametrize(
    "constraint",
    [
        pytest.param(
            "{dims: [b, t], where: \"b != 'b0' and t >= 0\", "
            'expression: "sum(rated, over=g, by=gen_bus[b]) <= 1"}',
            id="grouped-sum",
        ),
        pytest.param(
            "{dims: [g, t], where: \"g != 'g0' and t >= 0\", "
            'expression: "x <= at(rating, by=gen_bus[b])"}',
            id="lookup",
        ),
    ],
)
def test_a_join_checks_its_operand_rows_without_the_dims_it_adds(
    constraint: str,
) -> None:
    """The rows a join checks reach its operand without the dims it adds, not as one mask over bus x generator x t (#1190)."""
    t = T[:50]
    sources = {
        "t": t,
        "g": G,
        "b": BUSES,
        "gen_bus": pd.DataFrame({"g": G, "b": np.resize(BUSES, len(G))}),
        "c": pd.DataFrame({"g": G, "c": 1.0}),
        "d": pd.DataFrame({"b": BUSES, "d": 1.0}),
    }
    spec = JOINED + f"constraints: {{k: {constraint}}}\n"
    used = peak(lambda: Model.from_spec(spec, sources))
    assert used < 0.5 * len(BUSES) * len(t) * len(G)
