"""
Benchmark: Memory usage of linopy expressions under varying sparsity.

Tests:
  A) Baseline (float64 coeffs, int64 vars)
  B) int32 vars only
  C) float32 coeffs only
  D) int32 vars + float32 coeffs
  E) Deferred chunked constraint building (batch of CHUNK_SIZE)
  F) No-mask baseline (skip mask entirely, for comparison)

Each test builds a model with N_EXPR sparse expressions and sums them into
constraints, measuring peak memory and wall-clock time.
"""

import gc
import time
import tracemalloc

import numpy as np
import pandas as pd
import xarray as xr

import linopy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Problem dimensions – adjust to fit your machine
DIM_SIZES = {
    "small": (50, 50, 10),
    "medium": (100, 100, 20),
    "large": (200, 200, 50),
}

SPARSITY_LEVELS = [0.01, 0.05, 0.10, 0.50]  # fraction of non-zero elements
N_EXPR = 5  # number of sparse expressions to sum

SCENARIO = "medium"  # change to "large" if you have enough RAM

CHUNK_SIZE = 2000  # batch size for test E


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_coords(shape):
    return [pd.RangeIndex(s, name=f"d{i}") for i, s in enumerate(shape)]


def make_mask_da(mask_np, coords):
    """Convert numpy bool mask to xarray DataArray with correct dims."""
    dims = [c.name for c in coords]
    return xr.DataArray(mask_np, dims=dims, coords={c.name: c for c in coords})


def make_sparse_coeffs(shape, sparsity, rng, dtype=np.float64):
    """Return a dense array with (1 - sparsity) fraction set to zero."""
    mask = rng.random(shape) < sparsity
    vals = rng.standard_normal(shape).astype(dtype)
    vals[~mask] = 0.0
    return vals, mask


def measure(func, label=""):
    """Run *func*, return (peak_memory_MB, elapsed_seconds, result)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2, elapsed, result


def bytes_of_dataset(ds):
    """Approximate nbytes of an xarray Dataset."""
    total = 0
    for v in ds.data_vars.values():
        total += v.values.nbytes
    return total


def cast_expression(expr, coeffs_dtype=None, vars_dtype=None):
    """
    Monkey-patch expression dtypes by mutating the backing arrays directly.
    This avoids linopy's internal float/int coercion.
    """
    data = expr.data
    if coeffs_dtype is not None:
        data["coeffs"].values = data["coeffs"].values.astype(coeffs_dtype)
    if vars_dtype is not None:
        data["vars"].values = data["vars"].values.astype(vars_dtype)
    return expr


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def build_model_and_variables(shape, sparsity, rng):
    """Shared setup: create model, variables, sparse coefficient arrays."""
    coords = make_coords(shape)
    m = linopy.Model()
    variables = []
    coeff_arrays = []
    masks = []
    for i in range(N_EXPR):
        v = m.add_variables(lower=0, coords=coords, name=f"x{i}")
        variables.append(v)
        c, mask = make_sparse_coeffs(shape, sparsity, rng)
        coeff_arrays.append(c)
        masks.append(mask)
    # Combined mask: True where ANY expression has a non-zero coefficient
    combined_mask_np = np.zeros(shape, dtype=bool)
    for mask in masks:
        combined_mask_np |= mask
    combined_mask_da = make_mask_da(combined_mask_np, coords)
    return m, variables, coeff_arrays, masks, combined_mask_np, combined_mask_da, coords


def test_A_baseline(shape, sparsity, rng):
    """Baseline: dense float64/int64, vectorized sum."""
    m, variables, coeff_arrays, _, _, combined_mask_da, _ = build_model_and_variables(
        shape, sparsity, rng
    )

    def run():
        exprs = [coeff_arrays[i] * variables[i] for i in range(N_EXPR)]
        total = sum(exprs)
        m.add_constraints(total <= 1, name="con", mask=combined_mask_da)
        return bytes_of_dataset(m.constraints["con"].data)

    peak_mb, elapsed, con_bytes = measure(run, "A_baseline")
    return {
        "test": "A_baseline (f64/i64)",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "constraint_bytes": con_bytes,
    }


def test_B_int32_vars(shape, sparsity, rng):
    """int32 for vars array only."""
    m, variables, coeff_arrays, _, _, combined_mask_da, _ = build_model_and_variables(
        shape, sparsity, rng
    )

    def run():
        exprs = []
        for i in range(N_EXPR):
            e = coeff_arrays[i] * variables[i]
            cast_expression(e, vars_dtype=np.int32)
            exprs.append(e)
        total = sum(exprs)
        cast_expression(total, vars_dtype=np.int32)
        m.add_constraints(total <= 1, name="con", mask=combined_mask_da)
        return bytes_of_dataset(m.constraints["con"].data)

    peak_mb, elapsed, con_bytes = measure(run, "B_int32_vars")
    return {
        "test": "B_int32_vars",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "constraint_bytes": con_bytes,
    }


def test_C_float32_coeffs(shape, sparsity, rng):
    """float32 for coeffs only."""
    m, variables, coeff_arrays, _, _, combined_mask_da, _ = build_model_and_variables(
        shape, sparsity, rng
    )

    def run():
        exprs = []
        for i in range(N_EXPR):
            c32 = coeff_arrays[i].astype(np.float32)
            e = c32 * variables[i]
            cast_expression(e, coeffs_dtype=np.float32)
            exprs.append(e)
        total = sum(exprs)
        cast_expression(total, coeffs_dtype=np.float32)
        m.add_constraints(total <= 1, name="con", mask=combined_mask_da)
        return bytes_of_dataset(m.constraints["con"].data)

    peak_mb, elapsed, con_bytes = measure(run, "C_float32_coeffs")
    return {
        "test": "C_float32_coeffs",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "constraint_bytes": con_bytes,
    }


def test_D_both(shape, sparsity, rng):
    """int32 vars + float32 coeffs."""
    m, variables, coeff_arrays, _, _, combined_mask_da, _ = build_model_and_variables(
        shape, sparsity, rng
    )

    def run():
        exprs = []
        for i in range(N_EXPR):
            c32 = coeff_arrays[i].astype(np.float32)
            e = c32 * variables[i]
            cast_expression(e, coeffs_dtype=np.float32, vars_dtype=np.int32)
            exprs.append(e)
        total = sum(exprs)
        cast_expression(total, coeffs_dtype=np.float32, vars_dtype=np.int32)
        m.add_constraints(total <= 1, name="con", mask=combined_mask_da)
        return bytes_of_dataset(m.constraints["con"].data)

    peak_mb, elapsed, con_bytes = measure(run, "D_both")
    return {
        "test": "D_int32+float32",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "constraint_bytes": con_bytes,
    }


def test_E_deferred_chunked(shape, sparsity, rng):
    """
    Deferred: process mask elements in chunks of CHUNK_SIZE.

    Instead of materializing the full dense sum, we iterate through
    non-zero mask positions in batches, indexing into each expression
    and summing only the relevant elements.
    """
    m, variables, coeff_arrays, _, combined_mask_np, _, coords = (
        build_model_and_variables(shape, sparsity, rng)
    )

    def run():
        true_indices = np.argwhere(combined_mask_np)
        n_constraints = len(true_indices)
        con_count = 0
        ndim = len(shape)

        for chunk_start in range(0, n_constraints, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_constraints)
            chunk_idx = true_indices[chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            # Build a flat "chunk" coordinate
            chunk_coord = pd.RangeIndex(chunk_len, name="__chunk")

            # Sum expressions for this chunk only
            chunk_expr = None
            for i in range(N_EXPR):
                # Extract coefficient values for this chunk
                idx_tuple = tuple(chunk_idx[:, d] for d in range(ndim))
                c_chunk = coeff_arrays[i][idx_tuple]

                # Extract variable labels for this chunk via numpy indexing
                var_labels = m.variables["x" + str(i)].data["labels"].values[idx_tuple]

                # Build a tiny LinearExpression for this chunk
                coeffs_da = xr.DataArray(
                    c_chunk.reshape(chunk_len, 1),
                    dims=("__chunk", "_term"),
                    coords={"__chunk": chunk_coord},
                )
                vars_da = xr.DataArray(
                    var_labels.reshape(chunk_len, 1),
                    dims=("__chunk", "_term"),
                    coords={"__chunk": chunk_coord},
                )
                ds = xr.Dataset({"coeffs": coeffs_da, "vars": vars_da})
                e = linopy.LinearExpression(ds, m)

                if chunk_expr is None:
                    chunk_expr = e
                else:
                    chunk_expr = chunk_expr + e

            if chunk_expr is not None:
                m.add_constraints(
                    chunk_expr <= 1,
                    name=f"con_chunk_{con_count}",
                )
                con_count += 1

        return con_count

    peak_mb, elapsed, n_chunks = measure(run, "E_chunked")
    return {
        "test": f"E_chunked({CHUNK_SIZE})",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "n_chunks": n_chunks,
    }


def test_F_no_mask(shape, sparsity, rng):
    """No mask: dense vectorized sum, no filtering. Shows mask overhead."""
    m, variables, coeff_arrays, _, _, _, _ = build_model_and_variables(
        shape, sparsity, rng
    )

    def run():
        exprs = [coeff_arrays[i] * variables[i] for i in range(N_EXPR)]
        total = sum(exprs)
        m.add_constraints(total <= 1, name="con")
        return bytes_of_dataset(m.constraints["con"].data)

    peak_mb, elapsed, con_bytes = measure(run, "F_no_mask")
    return {
        "test": "F_no_mask (f64/i64)",
        "peak_mb": peak_mb,
        "elapsed_s": elapsed,
        "constraint_bytes": con_bytes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    shape = DIM_SIZES[SCENARIO]
    total_elements = int(np.prod(shape))
    print(f"Scenario: {SCENARIO}  shape={shape}  total_elements={total_elements:,}")
    print(f"N_EXPR={N_EXPR}  CHUNK_SIZE={CHUNK_SIZE}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print("=" * 80)

    all_results = []

    for sparsity in SPARSITY_LEVELS:
        n_nonzero_approx = int(total_elements * (1 - (1 - sparsity) ** N_EXPR))
        print(
            f"\n--- Sparsity: {sparsity:.0%} non-zero  (~{n_nonzero_approx:,} constraints) ---"
        )

        tests = [
            test_A_baseline,
            test_B_int32_vars,
            test_C_float32_coeffs,
            test_D_both,
            test_E_deferred_chunked,
            test_F_no_mask,
        ]

        for test_fn in tests:
            gc.collect()
            rng_copy = np.random.default_rng(42)  # same seed each time
            try:
                result = test_fn(shape, sparsity, rng_copy)
                result["sparsity"] = sparsity
                result["shape"] = str(shape)
                all_results.append(result)
                peak = result["peak_mb"]
                elapsed = result["elapsed_s"]
                print(
                    f"  {result['test']:40s}  peak={peak:8.1f} MB  time={elapsed:7.2f}s"
                )
            except Exception as ex:
                print(f"  {test_fn.__name__:40s}  FAILED: {ex}")
                all_results.append(
                    {
                        "test": test_fn.__name__,
                        "sparsity": sparsity,
                        "shape": str(shape),
                        "error": str(ex),
                    },
                )

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(all_results)
    cols = ["sparsity", "test", "peak_mb", "elapsed_s"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    # Save to CSV
    out_path = "dev-scripts/benchmark_sparsity_memory_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
