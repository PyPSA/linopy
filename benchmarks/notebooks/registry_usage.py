# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Benchmark model registry — usage guide
#
# This file is the canonical walkthrough for the benchmark **model registry**.
# It's authored in [jupytext](https://jupytext.readthedocs.io/) percent format,
# which means:
#
# - **Run as a script:** `python benchmarks/notebooks/registry_usage.py` — every
#   pattern below executes end-to-end. CI runs it this way on every PR, so the
#   examples can't silently rot.
# - **Open as a notebook:** in JupyterLab or VSCode with the jupytext extension,
#   this file appears as a notebook with markdown + code cells.
# - **Lint:** `ruff check` works because it's plain Python.
#
# The registry lives in `benchmarks/registry.py`. Each model file under
# `benchmarks/models/` self-registers a `ModelSpec` on import, so just touching
# the `benchmarks` package populates `REGISTRY`.

# %% [markdown]
# ## 1. Import the registry
#
# Single entry point: `from benchmarks import REGISTRY` plus the feature / phase
# constants you need for filtering.

# %%
# Put the repo root on sys.path so the file runs from anywhere
# (e.g. ``python benchmarks/notebooks/registry_usage.py``).
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import (  # noqa: E402
    INTEGER,
    QUADRATIC,
    REGISTRY,
    TO_GUROBIPY,
    filter_by,
    get,
)

print(f"{len(REGISTRY)} models registered: {sorted(REGISTRY)}")

# %% [markdown]
# ## 2. Look up one model by name
#
# `REGISTRY[name]` returns a `ModelSpec` (frozen dataclass). `.build(size)`
# constructs and returns a `linopy.Model`.

# %%
spec = REGISTRY["basic"]
print(f"name:            {spec.name}")
print(f"sizes:           {spec.sizes}")
print(f"features:        {sorted(spec.features)}")
print(f"quick_threshold: {spec.quick_threshold}")
print(f"long_threshold:  {spec.long_threshold}")

m = spec.build(50)
print(
    f"\nbuilt at n=50: {len(m.variables)} variable arrays, "
    f"{len(m.constraints)} constraint arrays"
)

# %% [markdown]
# `get("name")` is an equivalent functional accessor — handy when you don't
# want to import `REGISTRY` directly.

# %%
assert get("basic") is REGISTRY["basic"]

# %% [markdown]
# ## 3. Iterate the whole registry
#
# Useful when you want to sweep your own test or profiling logic across every
# model — e.g. checking that a refactor didn't break any spec.

# %%
print(f"{'name':<25} {'features':<35} {'sizes':<20}")
print("-" * 80)
for name, spec in REGISTRY.items():
    feats = ",".join(sorted(spec.features))
    sizes = f"{spec.sizes[0]}..{spec.sizes[-1]}"
    print(f"{name:<25} {feats:<35} {sizes:<20}")

# %% [markdown]
# ## 4. Filter by feature
#
# `filter_by(has_feature=...)` returns specs that advertise that feature. The
# feature tag constants (`CONTINUOUS`, `BINARY`, `INTEGER`, `QUADRATIC`, `SOS`,
# `PIECEWISE`, `MASKED`) are exported from `benchmarks`.

# %%
qp_specs = filter_by(has_feature=QUADRATIC)
print("Quadratic models:", [s.name for s in qp_specs])

mip_specs = filter_by(has_feature=INTEGER)
print("Integer models:  ", [s.name for s in mip_specs])

# %% [markdown]
# ## 5. Filter by phase
#
# Each spec declares which **phases** apply — `BUILD`, `MATRICES`, `LP_WRITE`,
# `NETCDF`, `SOLVER_BUILD`, plus per-solver `TO_HIGHSPY` / `TO_GUROBIPY` /
# `TO_MOSEK` / `TO_XPRESS`. Use `has_phase=` to narrow to solver-compatible
# models, e.g. when writing a Gurobi-specific regression test.

# %%
gurobi_specs = filter_by(has_phase=TO_GUROBIPY)
print(f"{len(gurobi_specs)} models declare TO_GUROBIPY:")
for s in gurobi_specs:
    print(f"  - {s.name}")

# %% [markdown]
# ## 6. Reuse pattern — parametrize your own pytest
#
# The pattern the suite itself uses (see `benchmarks/test_build.py` etc.) —
# `iter_params(phase)` returns `(spec, size)` pairs for the given phase, and
# `param_ids(...)` builds stable test IDs for `pytest.mark.parametrize`:
#
# ```python
# import pytest
# from benchmarks import BUILD, iter_params, param_ids
#
# _PARAMS = iter_params(BUILD)
#
# @pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
# def test_my_invariant(spec, size):
#     m = spec.build(size)
#     # ... assertion that should hold for every model
# ```

# %% [markdown]
# ## 7. Reuse pattern — one-off profiling
#
# Grab a single model at a chosen size, measure something, throw it away.
# `tracemalloc` works well for in-process peak-RSS spot checks (use
# `benchmarks/memory.py` + pytest-memray for the real metric).

# %%
import tracemalloc  # noqa: E402

tracemalloc.start()
m = REGISTRY["sparse_network"].build(100)
_current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"sparse_network n=100: built, peak allocation ≈ {peak / 1e6:.1f} MB")
print(
    f"  {m.variables.nvars} scalar variables, {m.constraints.ncons} scalar constraints"
)

# %% [markdown]
# ## 8. Running the benchmark suite
#
# Three size tiers, configured per-spec via `quick_threshold` and
# `long_threshold`:
#
# | Flag        | Sizes included            | Use case                              |
# | ----------- | ------------------------- | ------------------------------------- |
# | `--quick`   | `size <= quick_threshold` | CI smoke (~18s, one size per model)   |
# | _(none)_    | `size <= long_threshold`  | Local regression run (~45s)           |
# | `--long`    | all sizes                 | Full sweep (~2 min, slow stuff)       |
#
# ```bash
# # Quickest smoke
# pytest benchmarks/ --quick --benchmark-disable
#
# # Default timing
# pytest benchmarks/ --benchmark-only
#
# # Full sweep with the slow sizes
# pytest benchmarks/ --benchmark-only --long
#
# # Pick a single (phase, model) pair
# pytest benchmarks/test_lp_write.py -k "knapsack and n=1000"
# ```

# %% [markdown]
# ## 9. Adding a new model
#
# 1. Drop `benchmarks/models/<name>.py` with a `build_<name>(size) -> Model`.
# 2. Build a `ModelSpec` and call `register(...)` at module scope. Declare
#    realistic `quick_threshold` / `long_threshold` so the smoke stays fast.
# 3. Add an import in `benchmarks/models/__init__.py` so registration fires.
#
# That's it — every phase test picks the spec up automatically through
# `iter_params(phase)`.
