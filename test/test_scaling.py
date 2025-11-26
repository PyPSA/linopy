import numpy as np
import pandas as pd
import pytest

from linopy.constants import Result, Solution, Status
from linopy.model import Model
from linopy.scaling import ScaleOptions, ScalingContext, resolve_options


def _build_simple_model() -> Model:
    m = Model()
    x = m.add_variables(lower=0, name="x", coords=[range(2)])
    # Row 0: max coeff 4 -> scale factor 0.25
    m.add_constraints(2 * x.isel(dim_0=0) + 4 * x.isel(dim_0=1) == 8, name="row0")
    # Row 1: max coeff 0.5 -> scale factor 2.0
    m.add_constraints(0.5 * x.isel(dim_0=0) + 0.25 * x.isel(dim_0=1) >= 1, name="row1")
    return m


def test_row_scaling_and_dual_unscale() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True, variable_scaling=False, method="row-max")
    ctx = ScalingContext.from_model(m.matrices, opts)
    M = ctx.matrices

    assert np.isclose(ctx.row_scale[0], 0.25)
    assert np.isclose(ctx.row_scale[1], 2.0)
    assert M.A is not None
    A = np.asarray(M.A.todense())
    # Row 0 scaled
    assert np.isclose(A[0, 0], 0.5)
    assert np.isclose(A[0, 1], 1.0)
    # RHS scaled with the same factor
    assert np.isclose(M.b[0], 2.0)

    dual = pd.Series([1.0, 1.5], index=m.matrices.clabels, dtype=float)
    unscaled_dual = ctx.unscale_dual(dual)
    assert np.isclose(unscaled_dual.loc[m.matrices.clabels[0]], 0.25)
    assert np.isclose(unscaled_dual.loc[m.matrices.clabels[1]], 3.0)


def test_column_scaling_continuous_only() -> None:
    m = Model()
    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y")
    b = m.add_variables(name="b", binary=True)

    m.add_constraints(1000 * x + 2 * y + b >= 10, name="row0")

    opts = ScaleOptions(
        enabled=True,
        variable_scaling=True,
        scale_integer_variables=False,
        method="row-max",
    )
    ctx = ScalingContext.from_model(m.matrices, opts)

    # After row scaling: row factor = 1/1000; col norms become [1, 0.002, 0.001]
    # Only continuous vars should be scaled.
    assert np.isclose(ctx.col_scale[0], 1.0)
    assert np.isclose(ctx.col_scale[1], 500.0)
    # binary/int columns remain unscaled
    assert np.isclose(ctx.col_scale[2], 1.0)

    scaled = pd.Series([1.0, 5.0, 1.0], index=m.matrices.vlabels, dtype=float)
    unscaled = ctx.unscale_primal(scaled)
    assert np.isclose(unscaled.loc[m.matrices.vlabels[0]], 1.0)
    assert np.isclose(unscaled.loc[m.matrices.vlabels[1]], 0.01)
    assert np.isclose(unscaled.loc[m.matrices.vlabels[2]], 1.0)


def test_row_l2_scaling() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True, variable_scaling=False, method="row-l2")
    ctx = ScalingContext.from_model(m.matrices, opts)

    # RMS of row 0: sqrt(mean([2^2, 4^2])) = sqrt(10) ≈ 3.162
    # scale factor = 1 / sqrt(10)
    assert np.isclose(ctx.row_scale[0], 1 / np.sqrt(10), rtol=1e-5)


def test_column_scaling_with_l2_norm() -> None:
    m = Model()
    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y")
    m.add_constraints(100 * x + 10 * y >= 10, name="row0")

    opts = ScaleOptions(enabled=True, variable_scaling=True, method="row-l2")
    ctx = ScalingContext.from_model(m.matrices, opts)

    # After row scaling, column norms are computed with L2
    assert ctx.col_scale[0] != 1.0
    assert ctx.col_scale[1] != 1.0


def test_resolve_options_variations() -> None:
    # None -> disabled
    opts = resolve_options(None)
    assert not opts.enabled

    # False -> disabled
    opts = resolve_options(False)
    assert not opts.enabled

    # True -> enabled with defaults
    opts = resolve_options(True)
    assert opts.enabled
    assert opts.method == "row-max"

    # String -> enabled with method
    opts = resolve_options("row-l2")
    assert opts.enabled
    assert opts.method == "row-l2"

    # ScaleOptions passthrough
    custom = ScaleOptions(enabled=True, variable_scaling=True)
    opts = resolve_options(custom)
    assert opts is custom


def test_resolve_options_invalid_string() -> None:
    with pytest.raises(ValueError, match="Invalid scale method"):
        resolve_options("invalid-method")


def test_resolve_options_invalid_type() -> None:
    with pytest.raises(ValueError, match="scale must be one of"):
        resolve_options(123)  # type: ignore[arg-type]


def test_scaling_disabled_error() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=False)
    with pytest.raises(ValueError, match="scaling disabled"):
        ScalingContext.from_model(m.matrices, opts)


def test_variable_scaling_without_bounds_error() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True, variable_scaling=True, scale_bounds=False)
    with pytest.raises(ValueError, match="scaling bounds"):
        ScalingContext.from_model(m.matrices, opts)


def test_unscale_solution_none_result() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True)
    ctx = ScalingContext.from_model(m.matrices, opts)

    # None result returns None
    assert ctx.unscale_solution(None) is None


def test_unscale_solution_none_solution() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True)
    ctx = ScalingContext.from_model(m.matrices, opts)

    # Result with None solution returns unchanged
    result = Result(Status("warning", "infeasible"), solution=None)
    unscaled = ctx.unscale_solution(result)
    assert unscaled is result
    assert unscaled.solution is None


def test_unscale_solution_empty_primal_dual() -> None:
    m = _build_simple_model()
    opts = ScaleOptions(enabled=True)
    ctx = ScalingContext.from_model(m.matrices, opts)

    # Empty primal/dual should not fail
    sol = Solution(primal=pd.Series(dtype=float), dual=pd.Series(dtype=float))
    result = Result(Status("ok", "optimal"), solution=sol)
    unscaled = ctx.unscale_solution(result)
    assert unscaled.solution is not None
    assert unscaled.solution.primal.empty
    assert unscaled.solution.dual.empty
