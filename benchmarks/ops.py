"""
Registry of arithmetic *operation* micro-benchmarks.

Where :mod:`benchmarks.registry` benchmarks whole model builds, this benchmarks
single operations — ``var * array``, ``expr + expr``, ``expr <= c`` — with the
operands built *outside* the measured region, so a run isolates one op rather
than a whole build. That granularity attributes regressions to a specific path
(a whole-build benchmark says "kvl got heavier"; an op benchmark says "expr+expr
broadcast got heavier").

One 3-D size profile (``3×4×1000``, ~12 K elements): multi-dim so it exercises
broadcast/alignment across dims; ~MB-scale ops sit above the memory-measurement
noise floor; the asymmetric shape catches dim-order/transpose bugs. CodSpeed
records time *and* memory on every benchmark, so a second size isn't needed to
separate the two signals.

The one axis beyond the op itself is **alignment** — for binary labelled ops,
``match`` (identical coords, the fast path) vs ``broadcast`` (an extra dim → §9
cross-product). That's where the alignment-path regressions live, so it's
first-class, not incidental.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import linopy

# --- size profiles ----------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """A benchmark size: named dimensions and their lengths."""

    key: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


GRID = Profile("grid", ("d0", "d1", "d2"), (3, 4, 1000))

# a broadcast operand always adds this one extra dim (kept small so the
# cross-product stays cheap while still exercising the broadcast path)
EXTRA_DIM = "b"
EXTRA_LEN = 5


# --- operand builders (run in setup, never measured) ------------------------


def _coords(dims: tuple[str, ...], shape: tuple[int, ...]) -> dict[str, pd.Index]:
    return {d: pd.RangeIndex(n, name=d) for d, n in zip(dims, shape)}


def var(profile: Profile, name: str = "x") -> linopy.Variable:
    """A variable spanning the profile's dimensions."""
    m = linopy.Model()
    return m.add_variables(
        coords=list(_coords(profile.dims, profile.shape).values()),
        dims=list(profile.dims),
        name=name,
    )


def array(profile: Profile) -> xr.DataArray:
    """A coefficient array matching the profile's dims (the ``match`` case)."""
    return xr.DataArray(
        np.linspace(-1.0, 1.0, profile.size).reshape(profile.shape),
        dims=list(profile.dims),
        coords=_coords(profile.dims, profile.shape),
    )


def extra_array(_: Profile) -> xr.DataArray:
    """An array on a *new* dim — broadcasting it introduces that dim (§9)."""
    return xr.DataArray(
        np.linspace(1.0, 2.0, EXTRA_LEN),
        dims=[EXTRA_DIM],
        coords={EXTRA_DIM: pd.RangeIndex(EXTRA_LEN, name=EXTRA_DIM)},
    )


def extra_var(profile: Profile, name: str = "z") -> linopy.Variable:
    """A variable on a *new* dim — for var+var broadcast."""
    m = linopy.Model()
    return m.add_variables(
        coords=[pd.RangeIndex(EXTRA_LEN, name=EXTRA_DIM)], dims=[EXTRA_DIM], name=name
    )


def expr(profile: Profile) -> linopy.LinearExpression:
    """A linear expression spanning the profile's dims (coeffs vary)."""
    return array(profile) * var(profile)


def cond(profile: Profile) -> xr.DataArray:
    """A boolean mask over the profile's dims (~half the slots)."""
    return array(profile) > 0.0


def masked_expr(profile: Profile) -> linopy.LinearExpression:
    """An expression carrying absence (§4) — masked in place."""
    return expr(profile).where(cond(profile))


def grouped_expr(profile: Profile) -> linopy.LinearExpression:
    """An expression with a coarse ``g`` group coord on the last dim (8 groups)."""
    last, n = profile.dims[-1], profile.shape[-1]
    g = xr.DataArray(
        np.arange(n) * 8 // n,
        dims=[last],
        coords={last: pd.RangeIndex(n, name=last)},
    )
    return expr(profile).assign_coords(g=g)


# --- op registry ------------------------------------------------------------


@dataclass(frozen=True)
class OpSpec:
    """One operation benchmark: build operands, then measure ``op(*operands)``."""

    name: str
    group: str
    setup: Callable[[Profile], tuple]
    op: Callable[..., object]


OP_REGISTRY: dict[str, OpSpec] = {}


def register_op(
    name: str,
    group: str,
    setup: Callable[[Profile], tuple],
    op: Callable[..., object],
) -> None:
    if name in OP_REGISTRY:
        raise ValueError(f"op {name!r} already registered")
    OP_REGISTRY[name] = OpSpec(name, group, setup, op)


def iter_ops() -> list[OpSpec]:
    """Every registered op — the pytest parametrize source."""
    return list(OP_REGISTRY.values())


# --- the operations ---------------------------------------------------------
# Binary labelled ops register a `match` and a `broadcast` variant; the
# alignment case is baked into the operands the setup builds.

# scaling / construction
register_op("var_mul_scalar", "scale", lambda p: (var(p),), lambda x: 2.0 * x)
register_op("var_div_scalar", "scale", lambda p: (var(p),), lambda x: x / 2.0)
register_op("var_neg", "scale", lambda p: (var(p),), lambda x: -x)
register_op("var_to_linexpr", "scale", lambda p: (var(p),), lambda x: 1 * x)
register_op(
    "var_mul_array_match", "scale", lambda p: (var(p), array(p)), lambda x, a: a * x
)
register_op(
    "var_mul_array_bcast",
    "scale",
    lambda p: (var(p), extra_array(p)),
    lambda x, a: a * x,
)

# variable arithmetic
register_op("var_add_scalar", "var_arith", lambda p: (var(p),), lambda x: x + 2.0)
register_op(
    "var_add_array_match", "var_arith", lambda p: (var(p), array(p)), lambda x, a: x + a
)
register_op(
    "var_add_array_bcast",
    "var_arith",
    lambda p: (var(p), extra_array(p)),
    lambda x, a: x + a,
)
register_op(
    "var_add_var_match",
    "var_arith",
    lambda p: (var(p, "x"), var(p, "y")),
    lambda x, y: x + y,
)
register_op(
    "var_add_var_bcast",
    "var_arith",
    lambda p: (var(p, "x"), extra_var(p)),
    lambda x, z: x + z,
)
register_op(
    "var_sub_var_match",
    "var_arith",
    lambda p: (var(p, "x"), var(p, "y")),
    lambda x, y: x - y,
)

# quadratic
register_op(
    "var_mul_var", "quad", lambda p: (var(p, "x"), var(p, "y")), lambda x, y: x * y
)
register_op(
    "expr_mul_var", "quad", lambda p: (expr(p), var(p, "y")), lambda e, y: e * y
)

# expression arithmetic
register_op("expr_add_scalar", "expr_arith", lambda p: (expr(p),), lambda e: e + 2.0)
register_op(
    "expr_add_array_match",
    "expr_arith",
    lambda p: (expr(p), array(p)),
    lambda e, a: e + a,
)
register_op(
    "expr_add_array_bcast",
    "expr_arith",
    lambda p: (expr(p), extra_array(p)),
    lambda e, a: e + a,
)
register_op(
    "expr_add_var", "expr_arith", lambda p: (expr(p), var(p, "y")), lambda e, y: e + y
)
register_op(
    "expr_add_expr_match",
    "expr_arith",
    lambda p: (expr(p), expr(p)),
    lambda a, b: a + b,
)
register_op(
    "expr_add_expr_bcast",
    "expr_arith",
    lambda p: (expr(p), extra_array(p) * var(p)),
    lambda a, b: a + b,
)
register_op(
    "expr_sub_expr_match",
    "expr_arith",
    lambda p: (expr(p), expr(p)),
    lambda a, b: a - b,
)
register_op("expr_mul_scalar", "expr_arith", lambda p: (expr(p),), lambda e: 2.0 * e)
register_op(
    "expr_mul_array_match",
    "expr_arith",
    lambda p: (expr(p), array(p)),
    lambda e, a: a * e,
)
register_op(
    "expr_mul_array_bcast",
    "expr_arith",
    lambda p: (expr(p), extra_array(p)),
    lambda e, a: a * e,
)

# reductions
register_op("var_sum_dim", "reduce", lambda p: (var(p),), lambda x: x.sum("d0"))
register_op("expr_sum_dim", "reduce", lambda p: (expr(p),), lambda e: e.sum("d0"))
register_op("expr_sum_all", "reduce", lambda p: (expr(p),), lambda e: e.sum())

# constraint construction
register_op("con_le_scalar", "constraint", lambda p: (expr(p),), lambda e: e <= 2.0)
register_op(
    "con_le_array", "constraint", lambda p: (expr(p), array(p)), lambda e, a: e <= a
)
register_op(
    "con_eq_expr", "constraint", lambda p: (expr(p), expr(p)), lambda a, b: a == b
)

# absence / masking (§4–§7)
register_op("expr_where", "mask", lambda p: (expr(p), cond(p)), lambda e, c: e.where(c))
register_op("expr_fillna", "mask", lambda p: (masked_expr(p),), lambda e: e.fillna(0.0))
register_op(
    "expr_add_masked",
    "mask",
    lambda p: (expr(p), masked_expr(p)),
    lambda a, b: a + b,
)

# groupby
register_op(
    "expr_groupby_sum",
    "groupby",
    lambda p: (grouped_expr(p),),
    lambda e: e.groupby("g").sum(),
)

# N-way assembly (constraint building sums many terms)
register_op(
    "merge_sum",
    "merge",
    lambda p: tuple(expr(p) for _ in range(8)),
    lambda *es: sum(es[1:], es[0]),
)
