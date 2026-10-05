"""
Coordinate concatenation contracts: input backing, row order, joins, absence,
explicit zeros, ragged terms, metadata, empty grids, and unsupported fallbacks.
Term merging already owns addition coverage; these tests stack disjoint rows.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy.csr import CSRLinearExpression
from linopy.expressions import LinearExpression


def parts(model: linopy.Model, buses: list[list[str]]) -> list[LinearExpression]:
    result = []
    for i, labels in enumerate(buses):
        coords = [pd.Index([2 - i], name="snapshot"), pd.Index(labels, name="bus")]
        x = model.add_variables(coords=coords)
        dense = x * (i + 1) + (i + 3)
        dense = dense.assign_coords(season=("snapshot", [f"s{i}"]))
        result.append(
            LinearExpression._from_csr(
                CSRLinearExpression.from_dense(dense.data, model), model
            )
        )
    return result


@pytest.mark.v1
@pytest.mark.parametrize(
    "join", ["outer", "inner", "left", "right", "exact", "override"]
)
def test_coordinate_concat_preserves_sparse_inputs_and_rows(join: str) -> None:
    model = linopy.Model(sparse=True)
    labels = [["b", "a"], ["b", "a"]] if join == "exact" else [["b", "a"], ["c", "b"]]
    sparse = parts(model, labels)
    dense = [e._csr.to_dense() for e in sparse]
    expected = linopy.merge(dense, dim="snapshot", join=join)
    result = linopy.merge(sparse, dim="snapshot", join=join)
    assert result.is_sparse and all(e.is_sparse for e in sparse)
    actual = result._csr.to_dense()
    xr.testing.assert_equal(actual.const, expected.const)
    xr.testing.assert_equal(actual.coords.to_dataset(), expected.coords.to_dataset())
    np.testing.assert_array_equal(
        result._csr.const.reshape(result._csr.grid.shape)[0], expected.const.values[0]
    )
    for row, snapshot in enumerate([2, 1]):
        for bus in result._csr.grid.indexes["bus"]:
            cell = result._csr.cell((row, result._csr.grid.indexes["bus"].get_loc(bus)))
            if join == "override" or bus in labels[row]:
                np.testing.assert_array_equal(cell[0], [row + 1])
                source_position = (
                    result._csr.grid.indexes["bus"].get_loc(bus)
                    if join == "override"
                    else labels[row].index(bus)
                )
                np.testing.assert_array_equal(cell[1], [row * 2 + source_position])
                assert cell[2] == row + 3
            else:
                assert np.isnan(cell[2]) and len(cell[0]) == 0


@pytest.mark.v1
@pytest.mark.parametrize("empty", ["snapshot", "bus", None])
def test_ragged_concat_empty_absence_and_frozen_matrix(empty: str | None) -> None:
    model = linopy.Model(sparse=True)
    blocks = parts(model, [["a", "b"], ["a", "b"]])
    # A real zero term remains stored, while a masked cell remains absent.
    first = blocks[0]._csr
    from dataclasses import replace

    matrix = first.csr.copy()
    matrix.data[0] = 0.0
    blocks[0] = LinearExpression._from_csr(replace(first, csr=matrix), model)
    y = model.add_variables(
        coords=[pd.Index([1], name="snapshot"), pd.Index(["a", "b"], name="bus")]
    )
    second_dense = blocks[1]._csr.to_dense() + 5 * y
    second = CSRLinearExpression.from_dense(second_dense.data, model)
    blocks[1] = LinearExpression._from_csr(
        second.with_const(np.array([np.nan, 9.0])), model
    )
    # Reverse dimension order without expanding term rectangles.
    blocks[1] = LinearExpression._from_csr(
        blocks[1]._csr.reindexed(blocks[1]._csr.grid.reordered(("bus", "snapshot"))),
        model,
    )
    if empty is not None:
        blocks[1] = blocks[1].isel({empty: slice(0, 0)})
    dense = [b._csr.to_dense() for b in blocks]
    expected = linopy.merge(dense, dim="snapshot", join="outer")
    actual = linopy.merge(blocks, dim="snapshot", join="outer")
    assert actual.is_sparse and all(b.is_sparse for b in blocks)
    xr.testing.assert_equal(actual._csr.to_dense().const, expected.const)
    assert np.count_nonzero(actual._csr.csr.data == 0) == 1
    dense_csr = CSRLinearExpression.from_dense(expected.data, model)
    np.testing.assert_array_equal(actual._csr.csr.toarray(), dense_csr.csr.toarray())
    np.testing.assert_array_equal(actual._csr.const, dense_csr.const)
    # Frozen constraints exercise the solver-facing export, including masked rows.
    left = model.add_constraints(actual == 0, name="sparse", freeze=True)
    right = model.add_constraints(expected == 0, name="dense", freeze=True)
    np.testing.assert_array_equal(left.rhs.values, right.rhs.values)
    from linopy.testing import assert_conequal

    assert_conequal(left, right, strict=False)


@pytest.mark.v1
@pytest.mark.parametrize("unsupported", ["overlap", "repeated", "mixed", "kwargs"])
def test_unsupported_concat_keeps_observable_dense_fallback(unsupported: str) -> None:
    from linopy.constants import PerformanceWarning

    model = linopy.Model(sparse=True)
    blocks = parts(model, [["a"], ["a"]])
    kwargs = {}
    if unsupported == "overlap":
        blocks[1] = blocks[1].reindex(snapshot=[2])
    elif unsupported == "repeated":
        blocks[0] = blocks[0].isel(snapshot=[0, 0])
    elif unsupported == "mixed":
        blocks[1] = blocks[1]._csr.to_dense()
    else:
        kwargs = {"compat": "override", "coords": "minimal"}
    dense = [b._csr.to_dense() if b.is_sparse else b for b in blocks]
    expected = linopy.merge(dense, dim="snapshot", join="outer", **kwargs)
    with linopy.options as options:
        options.set_value(warn_on_densify=True)
        with pytest.warns(PerformanceWarning, match="merge along a coordinate"):
            actual = linopy.merge(blocks, dim="snapshot", join="outer", **kwargs)
    assert not actual.is_sparse
    xr.testing.assert_equal(actual.data, expected.data)


@pytest.mark.v1
@pytest.mark.parametrize("size", [0, 1])
def test_single_block_preserves_empty_grid_and_scalar_metadata(size: int) -> None:
    model = linopy.Model(sparse=True)
    block = parts(model, [["a", "b"]])[0].isel(snapshot=slice(0, size))
    from dataclasses import replace

    csr = block._csr
    grid = replace(csr.grid, aux=csr.grid.aux | {"country": ((), np.array("NL"))})
    block = LinearExpression._from_csr(replace(csr, grid=grid), model)
    expected = block._csr.to_dense()
    actual = linopy.merge([block], dim="snapshot", join="outer")
    assert actual.is_sparse and block.is_sparse
    xr.testing.assert_equal(actual._csr.to_dense().data, expected.data)


@pytest.mark.v1
@pytest.mark.parametrize("contract", ["exact", "override", "auxiliary"])
def test_coordinate_alignment_errors_preserve_inputs(contract: str) -> None:
    model = linopy.Model(sparse=True)
    blocks = parts(model, [["a", "b"], ["b", "c"]])
    join = "outer"
    if contract == "exact":
        join = "exact"
    elif contract == "override":
        join = "override"
        blocks[1] = blocks[1].isel(bus=[0])
    else:
        for i, country in enumerate(["NL", "DE"]):
            dense = blocks[i]._csr.to_dense().assign_coords(country=country)
            blocks[i] = LinearExpression._from_csr(
                CSRLinearExpression.from_dense(dense.data, model), model
            )
    expected = [b._csr.to_dense() for b in blocks]
    with pytest.raises(ValueError):
        linopy.merge(expected, dim="snapshot", join=join)
    with pytest.raises(ValueError):
        linopy.merge(blocks, dim="snapshot", join=join)
    assert all(b.is_sparse for b in blocks)
