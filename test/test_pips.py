from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_io import _pips_time_plant_model

import linopy.pips as pips
from linopy import Model


def _time_model(n_time: int) -> Model:
    m = Model()
    time = pd.RangeIndex(n_time, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x <= 1, name="c")
    m.add_objective(x.sum())
    return m


def _synthetic_storage_model() -> Model:
    m = Model()
    time = pd.RangeIndex(6, name="time")
    plant = pd.Index([0, 1], name="plant")
    x = m.add_variables(lower=0, coords=[time, plant], name="x")
    soc = m.add_variables(coords=[time], name="soc")
    cap = m.add_variables(lower=0, coords=[plant], name="cap")
    m.add_constraints(x - cap <= 0, name="capacity")
    continuity = (soc - soc.shift(time=1) - x.sum("plant")).isel(time=slice(1, None))
    m.add_constraints(continuity == 0, name="storage")
    m.add_constraints(x.sum() <= 100, name="budget")
    m.add_objective(x.sum() + soc.sum() + cap.sum())
    return m


@pytest.mark.parametrize(
    "n_time, n_blocks, sizes",
    [(6, 2, [3, 3]), (6, 4, [2, 2, 1, 1]), (10, 3, [4, 3, 3])],
)
def test_assign_blocks_contiguous(n_time: int, n_blocks: int, sizes: list[int]) -> None:
    m = _time_model(n_time)
    out = pips.assign_blocks(m, "time", n_blocks)
    expected = np.repeat(np.arange(1, n_blocks + 1), sizes)
    assert out is m
    assert m.blocks.dims == ("time",)
    assert list(m.blocks.coords["time"].values) == list(range(n_time))
    np.testing.assert_array_equal(m.blocks.values, expected)


def test_assign_blocks_dtype_is_best_int() -> None:
    m = _time_model(6)
    m.assign_blocks("time", 2)
    assert m.blocks.dtype == np.dtype("int8")


def test_assign_blocks_method_matches_function() -> None:
    a = _time_model(6).assign_blocks("time", 3)
    b = pips.assign_blocks(_time_model(6), "time", 3)
    np.testing.assert_array_equal(a.blocks.values, b.blocks.values)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"dim": "nope", "n_blocks": 2}, ValueError),
        ({"dim": "time", "n_blocks": 0}, ValueError),
        ({"dim": "time", "n_blocks": 7}, ValueError),
        ({"dim": "time", "n_blocks": 2, "boundary": "custom"}, NotImplementedError),
    ],
)
def test_assign_blocks_fail_fast(kwargs: dict, exc: type[Exception]) -> None:
    m = _time_model(6)
    with pytest.raises(exc):
        pips.assign_blocks(m, **kwargs)


def test_diagnose_synthetic_exact_counts() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    r = pips.diagnose(m)

    assert r.n_blocks == 3
    assert r.n_vars == 20
    assert r.n_cons == 18
    assert r.nnz == 56

    assert r.n_global_cols == 2
    assert r.block_cols == {1: 6, 2: 6, 3: 6}
    assert r.n_global_cols + sum(r.block_cols.values()) == r.n_vars

    assert r.block_nnz == {1: 12, 2: 12, 3: 12}
    assert r.balance_ratio == 1.0

    assert (r.n_local_rows, r.n_global_rows, r.n_linking_rows) == (15, 0, 3)
    assert r.n_local_rows + r.n_global_rows + r.n_linking_rows == r.n_cons
    assert (r.n_adjacent_rows, r.n_border_rows) == (2, 1)
    assert r.n_adjacent_rows + r.n_border_rows == r.n_linking_rows

    assert r.border_nnz == 32
    assert r.border_fraction == pytest.approx(32 / 56)


def test_diagnose_border_nnz_bruteforce() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    r = pips.diagnose(m)

    N = int(m.blocks.max())
    block_map = m.variables.get_blockmap(m.blocks.dtype.type)
    col_blocks = block_map[m.matrices.vlabels]
    row_blocks = np.concatenate(
        [
            c.data["blocks"].values.ravel()[c.active_row_mask()]
            for _, c in m.constraints.items()
            if not c.is_indicator
        ]
    )
    coo = m.matrices.A.tocoo()
    is_border = (row_blocks[coo.row] == N + 1) | (col_blocks[coo.col] == 0)
    assert r.border_nnz == int(is_border.sum())
    assert 0.0 <= r.border_fraction <= 1.0


def test_diagnose_preconditions() -> None:
    m = _synthetic_storage_model()
    with pytest.raises(ValueError, match="no blocks assigned"):
        pips.diagnose(m)


def test_diagnose_no_regular_constraints() -> None:
    m = Model()
    time = pd.RangeIndex(4, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_objective(x.sum())
    m.assign_blocks("time", 2)
    with pytest.raises(ValueError, match="no regular constraints"):
        pips.diagnose(m)


@pytest.mark.parametrize(
    "target_cores, rec_ranks, rec_threads, capped",
    [(None, 50, 1, False), (50, 50, 1, False), (200, 50, 4, True), (30, 30, 1, False)],
)
def test_diagnose_recommendation(
    target_cores: int | None, rec_ranks: int, rec_threads: int, capped: bool
) -> None:
    m = _time_model(50)
    m.assign_blocks("time", 50)
    r = pips.diagnose(m, target_cores=target_cores)
    assert r.max_ranks == 50
    assert r.rec_ranks == rec_ranks
    assert r.rec_threads == rec_threads
    assert any("capped" in w for w in r.warnings) == capped
    assert r.rec_ranks <= r.max_ranks
    assert r.rec_threads >= 1


def _imbalanced_model() -> Model:
    m = Model()
    time = pd.RangeIndex(9, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x <= 1, name="c")
    for i in range(3):
        m.add_constraints(x.isel(time=slice(0, 3)) >= -5, name=f"extra{i}")
    m.add_objective(x.sum())
    m.assign_blocks("time", 3)
    return m


def _empty_block_model() -> Model:
    m = Model()
    time = pd.RangeIndex(9, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x.isel(time=slice(0, 6)) <= 1, name="c")
    m.add_objective(x.sum())
    m.assign_blocks("time", 3)
    return m


@pytest.mark.parametrize(
    "builder, substring",
    [
        (_synthetic_storage_model, "high border fraction"),
        (_imbalanced_model, "block imbalance"),
        (_empty_block_model, "empty local blocks"),
        (lambda: _time_model(6).assign_blocks("time", 1), "not decomposed"),
    ],
)
def test_diagnose_warnings(builder, substring: str) -> None:
    m = builder()
    if m.blocks is None:
        m.assign_blocks("time", 3)
    r = pips.diagnose(m)
    assert any(substring in w for w in r.warnings)


def test_diagnose_str_renders_all_groups() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    text = str(pips.diagnose(m, target_cores=8))
    for token in [
        "BlockReport",
        "columns",
        "block nnz",
        "rows",
        "border",
        "parallel",
        "warnings",
    ]:
        assert token in text


@pytest.mark.parametrize("masked", [False, True])
def test_diagnose_realistic_consistency(masked: bool) -> None:
    m = _pips_time_plant_model(masked=masked)
    m.assign_blocks("time", 2)
    r = pips.diagnose(m, target_cores=8)
    assert r.n_global_cols + sum(r.block_cols.values()) == r.n_vars
    assert r.n_local_rows + r.n_global_rows + r.n_linking_rows == r.n_cons
    assert r.n_adjacent_rows + r.n_border_rows == r.n_linking_rows
    assert r.border_nnz == pytest.approx(r.border_fraction * r.nnz)
    assert r.rec_ranks <= r.max_ranks == r.n_blocks
