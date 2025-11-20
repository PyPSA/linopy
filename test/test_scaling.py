import numpy as np
import pandas as pd

from linopy.model import Model
from linopy.scaling import ScaleOptions, ScalingContext


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
