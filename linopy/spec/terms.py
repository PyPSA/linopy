"""
Deferred long-format terms: a prototype evaluator for linear declarations.

A node carrying a variable evaluates to a :class:`Table` instead of a linopy
expression. A table holds blocks of terms in long format, one entry per term:
integer position columns, the variable label, and the factors its coefficient
is still to be multiplied by. Beside the blocks sits one dense constant, NaN
where the table is absent. ``+`` concatenates blocks, a sum drops the column
of a dimension, a grouped sum gathers the group of each term through its
relation, and a product with data appends a factor. A table becomes a CSR
store once, when its declaration is added to the model.

A node the prototype does not handle is evaluated by
:func:`linopy.spec.evaluator.evaluate` and its result converted to a table.
The switch is the environment variable ``LINOPY_SPEC_TERMS=1``; the
declarations it covers are named in ``LINOPY_SPEC_TERMS_SCOPE``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.csr import CSRStore, Grid, _grouped, coo_to_csr
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec import operators
from linopy.spec.context import Context, widened
from linopy.spec.evaluator import (
    _narrowed,
    _nothing_summed,
    _off_rows,
    _regions,
    _sums_nothing,
    _zeros,
    evaluate,
    refuse_made,
)
from linopy.spec.where import looked_up
from linopy.store import as_sparse
from linopy.variables import Variable

DEFAULT_SCOPE = "Bus_nodal_balance,objective"
DENSE_KEYS = 50_000_000
FALLBACKS: list[str] = []


def enabled(name: str) -> bool:
    if os.environ.get("LINOPY_SPEC_TERMS") != "1":
        return False
    scope = os.environ.get("LINOPY_SPEC_TERMS_SCOPE", DEFAULT_SCOPE)
    return scope == "*" or name in scope.split(",")


class Unsupported(Exception):
    def __init__(self, reason: str, fallback: bool = True) -> None:
        super().__init__(reason)
        self.fallback = fallback


Factor = float | tuple[np.ndarray, tuple[int, ...]]
Data = xr.DataArray | float


@dataclass(frozen=True)
class Block:
    cols: tuple[np.ndarray, ...]
    dims: dict[str, int]
    var: np.ndarray
    factors: tuple[Factor, ...] = ()
    coeff: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.var)

    def positions(self, dims: Sequence[str]) -> tuple[np.ndarray, ...]:
        return tuple(self.cols[self.dims[d]] for d in dims)

    def times(self, factor: Factor) -> Block:
        return replace(self, factors=(*self.factors, factor))

    def coefficients(self) -> np.ndarray:
        c = np.ones(len(self)) if self.coeff is None else self.coeff
        for f in self.factors:
            if isinstance(f, tuple):
                values, at = f
                c = c * values[tuple(self.cols[i] for i in at)]
            else:
                c = c * f
        return c

    def take(self, idx: np.ndarray) -> Block:
        coeff = None if self.coeff is None else self.coeff[idx]
        cols = tuple(c[idx] for c in self.cols)
        return Block(cols, self.dims, self.var[idx], self.factors, coeff)

    def folded(self) -> Block:
        if not self.factors and len(self.cols) == len(self.dims):
            return self
        live = sorted(set(self.dims.values()))
        at = {old: new for new, old in enumerate(live)}
        dims = {d: at[i] for d, i in self.dims.items()}
        cols = tuple(self.cols[i] for i in live)
        return Block(cols, dims, self.var, (), self.coefficients())

    def regrouped(
        self, drop: Sequence[str], add: dict[str, np.ndarray]
    ) -> Block:
        dims = {d: i for d, i in self.dims.items() if d not in drop}
        dims |= {d: len(self.cols) + k for k, d in enumerate(add)}
        return replace(self, cols=(*self.cols, *add.values()), dims=dims)

    def broadcast(self, sizes: dict[str, int]) -> Block:
        new = [d for d in sizes if d not in self.dims]
        if not new:
            return self
        b = self.folded()
        shape = [sizes[d] for d in new]
        k = math.prod(shape)
        n = len(b)
        cells = np.unravel_index(np.tile(np.arange(k), n), shape)
        cols = tuple(np.repeat(c, k) for c in b.cols) + tuple(cells)
        dims = b.dims | {d: len(b.cols) + i for i, d in enumerate(new)}
        coeff = np.repeat(b.coefficients(), k)
        return Block(cols, dims, np.repeat(b.var, k), (), coeff)


@dataclass(frozen=True)
class Table:
    dims: tuple[str, ...]
    blocks: tuple[Block, ...]
    const: xr.DataArray | None
    blank: bool = False

    @property
    def flat(self) -> bool:
        """Whether the constant is zero wherever it is present: nothing to scale, sum or gather."""
        return self.const is None or self.blank


def _union(*dims: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(d for ds in dims for d in ds))


def _const(value: Table | Data) -> xr.DataArray | None:
    if isinstance(value, Table):
        return value.const
    return xr.DataArray(value)


def _flat(value: Table | Data) -> bool:
    """Whether the constant of *value* is zero wherever it is present."""
    if isinstance(value, Table):
        return value.flat
    values = np.asarray(getattr(value, "values", value))
    if values.dtype.kind != "f":
        return not values.any()
    return not (values[~np.isnan(values)] != 0).any()


def _dims(value: Table | Data) -> tuple[str, ...]:
    if isinstance(value, Table):
        return value.dims
    return tuple(str(d) for d in getattr(value, "dims", ()))


def _matches(
    tkey: np.ndarray, rkey: np.ndarray, size: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """For each term key the relation rows with that key: term and row indices, terms None where each matches once."""
    if size <= DENSE_KEYS:
        counts = np.bincount(rkey, minlength=size)
        if counts.max(initial=0) <= 1:
            at = np.full(size, -1, dtype=np.int64)
            at[rkey] = np.arange(len(rkey))
            ri = at[tkey]
            hit = ri >= 0
            if hit.all():
                return None, ri
            ti = np.flatnonzero(hit)
            return ti, ri[ti]
    if size <= DENSE_KEYS:
        order, bounds = _grouped(rkey, size)
        lo, cnt = bounds[tkey], counts[tkey]
    else:
        order = np.argsort(rkey, kind="stable")
        skey = rkey[order]
        lo = np.searchsorted(skey, tkey, "left")
        cnt = np.searchsorted(skey, tkey, "right") - lo
    ti = np.repeat(np.arange(len(tkey)), cnt)
    off = np.arange(len(ti)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return ti, order[np.repeat(lo, cnt) + off]


class Evaluator:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.carry: dict[int, bool] = {}

    def sizes(self, dims: Sequence[str]) -> dict[str, int]:
        return {d: len(self.ctx.coords[d]) for d in dims}

    def carries(self, node: ms.Expression) -> bool:
        key = id(node)
        if key not in self.carry:
            self.carry[key] = isinstance(node, ms.VariableReference) or any(
                self.carries(c) for c in ms.children(node)
            )
        return self.carry[key]

    def value(self, node: ms.Expression, ctx: Context) -> Table | Data:
        if not self.carries(node):
            return evaluate(node, ctx)
        return self.table(node, ctx)

    def table(self, node: ms.Expression, ctx: Context) -> Table:
        try:
            made = self._own(node, ctx)
        except Unsupported as e:
            FALLBACKS.append(f"{type(node).__name__} {getattr(node, 'name', '')}: {e}")
            if not e.fallback:
                raise Unsupported(str(e)) from e
            return self.converted(evaluate(node, ctx))
        ctx.memo.used(node)
        return made

    def _own(self, node: ms.Expression, ctx: Context) -> Table:
        if isinstance(node, ms.ExpressionReference):
            return self.named(node.body, ctx)
        if isinstance(node, ms.VariableReference):
            return self.variable(node.name)
        if _sums_nothing(node, ctx):
            raise Unsupported("over an empty dimension")
        if isinstance(node, ms.Negate):
            return self.scaled(self.table(node.operand, ctx), -1.0)
        if isinstance(node, ms.Add):
            return self.added(node, ctx)
        if isinstance(node, ms.Multiply):
            return self.multiplied(node, ctx)
        if isinstance(node, ms.Sum):
            return self.summed(node, ctx)
        if isinstance(node, ms.Join):
            return self.looked_up(node, ctx)
        if isinstance(node, ms.Cases):
            return self.cases(node, ctx)
        raise Unsupported(type(node).__name__)

    def named(self, body: ms.Expression, ctx: Context) -> Table:
        """The table of a named expression's *body*, kept for the build unless a refusal was masked making it, as the eager path keeps its term."""
        memo = ctx.memo
        key = ("terms.table", body)
        if key in memo.values:
            if body not in memo.inner:
                memo.inner[body] = tuple(ms.walk(body))
            for inner in memo.inner[body]:
                memo.used(inner)
            return cast(Table, memo.values[key])
        masked = memo.masked
        memo.uses.pop(body, None)
        made = self.table(body, ctx)
        if memo.masked == masked:
            memo.values[key] = made
        return made

    def variable(self, name: str) -> Table:
        return self.ctx.memo.kept(("terms.variable", name), lambda: self._variable(name))

    def _variable(self, name: str) -> Table:
        ctx = self.ctx
        labels = ctx.model.variables[ctx.rename(name)].labels
        dims = tuple(str(d) for d in labels.dims)
        for d in dims:
            if not labels.indexes[d].equals(ctx.coords[d]):
                raise Unsupported(f"variable '{name}' is off the master coordinates")
        flat = labels.values.reshape(-1)
        live = flat != -1
        everywhere = bool(live.all())
        idx = np.arange(flat.size) if everywhere else np.flatnonzero(live)
        cols = np.unravel_index(idx, labels.shape) if dims else ()
        block = Block(tuple(cols), {d: i for i, d in enumerate(dims)}, flat[idx])
        missing = ctx.program.variables[name].missing
        const = None
        if not everywhere and missing != 0:
            fill = np.nan if missing == "absent" else float(missing)
            const = labels.copy(data=np.where(live, 0.0, fill).reshape(labels.shape))
        return Table(dims, (block,), const, missing == "absent")

    def scaled(self, table: Table, factor: float) -> Table:
        const = table.const if table.flat else table.const * factor
        blocks = tuple(b.times(factor) for b in table.blocks)
        return Table(table.dims, blocks, const, table.blank)

    def added(self, node: ms.Add, ctx: Context) -> Table:
        first, second = node.left, node.right
        sub = isinstance(second, ms.Negate)
        if sub:
            second = second.operand
        left = _off_rows(self.value(first, ctx), ctx)
        right = _off_rows(self.value(second, ctx), ctx)
        a, b = _const(left), _const(right)
        op = (lambda x, y: x - y) if sub else (lambda x, y: x + y)
        if a is not None and b is not None:
            const = refuse_made("add", (node,), op, a, b, ctx)
        elif b is not None:
            const = -b if sub else b
        else:
            const = a
        blocks = list(left.blocks) if isinstance(left, Table) else []
        if isinstance(right, Table):
            blocks += [b.times(-1.0) if sub else b for b in right.blocks]
        blank = _flat(left) and _flat(right)
        return Table(_union(_dims(left), _dims(right)), tuple(blocks), const, blank)

    def multiplied(self, node: ms.Multiply, ctx: Context) -> Table:
        if self.carries(node.left) and self.carries(node.right):
            raise Unsupported("a product of two terms", fallback=False)
        dead = self.dead(node, ctx)
        if dead is not None:
            return dead
        left, right = self.value(node.left, ctx), self.value(node.right, ctx)
        table, data = (left, right) if isinstance(left, Table) else (right, left)
        return self.times(table, _off_rows(data, ctx))

    def dead(self, node: ms.Multiply, ctx: Context) -> Table | None:
        """A product of two terms times data zero wherever present: the constant the eager path's dead-term rule leaves."""
        product, other = node.left, node.right
        if not isinstance(product, ms.Multiply):
            product, other = other, product
        if not (
            isinstance(product, ms.Multiply)
            and self.carries(product.left)
            and self.carries(product.right)
            and not self.carries(other)
        ):
            return None
        data = _off_rows(evaluate(other, ctx), ctx)
        values = np.asarray(getattr(data, "values", data), dtype=float)
        if np.isinf(values).any() or (values[~np.isnan(values)] != 0).any():
            raise Unsupported("a live product of two terms", fallback=False)
        a = self.table(product.left, ctx)
        b = self.table(product.right, ctx)
        ctx.memo.used(product)
        const = xr.DataArray(0.0)
        for c in (a.const, b.const):
            if c is not None:
                const = const.where(c.notnull())
        data = xr.DataArray(data)
        if data.ndim:
            data = self.aligned(data)
        const = const.where(data.notnull()) * data.fillna(1.0)
        dims = _union(a.dims, b.dims, _dims(data))
        return Table(dims, (), self.full(const, dims), True)

    def aligned(self, data: xr.DataArray) -> xr.DataArray:
        stray = [c for c in data.coords if c not in data.dims]
        if stray:
            data = data.drop_vars(stray)
        for d in data.dims:
            index = data.indexes[d]
            if index is not self.ctx.coords[d] and not index.equals(self.ctx.coords[d]):
                raise Unsupported(f"data off the master coordinates on '{d}'")
        return data

    def times(self, table: Table, data: Data) -> Table:
        if not isinstance(data, xr.DataArray) or data.ndim == 0:
            return self.scaled(table, float(data))
        data = self.aligned(data)
        values = data.values
        if values.dtype.kind != "f":
            values = values.astype(float)
        if np.isinf(values).any():
            raise Unsupported("infinite data beside a term")
        holes = np.isnan(values)
        if holes.any():
            values = np.where(holes, 1.0, values)
        dims = tuple(str(d) for d in data.dims)
        const = table.const
        if holes.any():
            present = data.copy(data=~holes)
            const = (xr.DataArray(0.0) if const is None else const).where(present)
        if not table.flat:
            const = const * data.copy(data=values)
        sizes = self.sizes(dims)
        support = None
        blocks = []
        for b in table.blocks:
            if any(d not in b.dims for d in dims):
                if support is None:
                    support = np.nonzero((values != 0) & ~holes)
                b = self.spread(b, dims, support)
            blocks.append(b.times((values, tuple(b.dims[d] for d in dims))))
        return Table(_union(table.dims, dims), tuple(blocks), const, table.flat)

    def spread(
        self, b: Block, dims: tuple[str, ...], support: tuple[np.ndarray, ...]
    ) -> Block:
        """*b* spread onto the dims of a factor it lacks, only where the factor is nonzero."""
        at = dict(zip(dims, support))
        shared = [d for d in dims if d in b.dims]
        new = [d for d in dims if d not in b.dims]
        n, m = len(b), len(support[0]) if support else 0
        if shared:
            shape = list(self.sizes(shared).values())
            rkey = np.ravel_multi_index([at[d] for d in shared], shape)
            tkey = np.ravel_multi_index(b.positions(shared), shape)
            ti, ri = _matches(tkey, rkey, math.prod(shape))
        else:
            ti, ri = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
        if ti is not None:
            b = b.take(ti)
        return b.regrouped([], {d: at[d][ri] for d in new})

    def absorbed(self, table: Table) -> Table:
        if table.const is None:
            return table
        absent = table.const.isnull()
        if not bool(absent.any()):
            return table
        dims = tuple(str(d) for d in absent.dims)
        mask = absent.values
        sizes = self.sizes(dims)
        blocks = []
        for b in table.blocks:
            b = b.broadcast(sizes)
            keep = ~mask[b.positions(dims)]
            blocks.append(b if keep.all() else b.take(keep))
        return replace(table, blocks=tuple(blocks))

    def full(self, const: xr.DataArray, dims: Sequence[str]) -> xr.DataArray:
        lacking = {d: self.ctx.coords[d] for d in dims if d not in const.dims}
        return const.expand_dims(lacking) if lacking else const

    def summed(self, node: ms.Sum, ctx: Context) -> Table:
        operand = node.operand
        join = operand if isinstance(operand, ms.Join) and operand.columns.axes else None
        inner = widened(node, ctx)
        if join is not None:
            inner = widened(join, inner)
            table = self.through(self.table(join.operand, inner), join.columns)
        else:
            table = self.table(operand, inner)
        for axis in node.over:
            if axis.column is None:
                table = self.sum_over(table, axis.dimension)
        return table

    def sum_over(self, table: Table, dim: str) -> Table:
        if dim not in table.dims:
            raise Unsupported(f"a sum over '{dim}', which the operand lacks")
        table = self.absorbed(table)
        size = len(self.ctx.coords[dim])
        blocks = []
        for b in table.blocks:
            if dim in b.dims:
                blocks.append(replace(b, dims={d: i for d, i in b.dims.items() if d != dim}))
            else:
                blocks.append(b.times(float(size)))
        const = None
        if not table.flat:
            const = table.const.fillna(0)
            const = const.sum(dim) if dim in const.dims else const * size
        dims = tuple(d for d in table.dims if d != dim)
        return Table(dims, tuple(blocks), const, const is not None and _flat(const))

    def through(self, table: Table, columns: ms.JoinColumns) -> Table:
        ctx = self.ctx
        rows = ctx.relations[columns.name].rows
        joined = {columns.dim(r): rows[r].to_numpy() for r in columns.joined}
        grouped = {columns.dim(r): rows[r].to_numpy() for r in columns.grouped}
        onto = {d: ctx.coords[d] for d in grouped}
        if any(d not in table.dims for d in joined):
            raise Unsupported("a grouped sum along a dimension the operand lacks")
        if any(d in table.dims and d not in joined for d in grouped):
            raise Unsupported("a grouped sum onto a dimension the operand keeps")
        table = self.absorbed(table)
        jdims = list(joined)
        jpos = [ctx.coords[d].get_indexer(v) for d, v in joined.items()]
        gpos = {d: onto[d].get_indexer(v) for d, v in grouped.items()}
        if any((p < 0).any() for p in (*jpos, *gpos.values())):
            raise Unsupported("a relation row off the coordinates")
        jsizes = self.sizes(jdims)
        jshape = list(jsizes.values())
        rkey = np.ravel_multi_index(jpos, jshape) if jdims else np.zeros(len(rows), int)
        size = math.prod(jshape)
        blocks = []
        for b in table.blocks:
            b = b.broadcast(jsizes)
            tkey = np.ravel_multi_index(b.positions(jdims), jshape)
            ti, ri = _matches(tkey, rkey, size)
            if ti is not None:
                b = b.take(ti)
            blocks.append(b.regrouped(jdims, {d: p[ri] for d, p in gpos.items()}))
        const = None
        if not table.flat:
            const = self.full(table.const.fillna(0), table.dims)
            const = operators.grouped_sum(const, joined, grouped, onto)
        dims = _union([d for d in table.dims if d not in joined], grouped)
        return Table(dims, tuple(blocks), const, const is not None and _flat(const))

    def looked_up(self, node: ms.Join, ctx: Context) -> Table:
        table = self.table(node.operand, widened(node, ctx))
        columns = node.columns
        relation = ctx.relations[columns.name]
        rows = relation.rows
        if not len(rows):
            raise Unsupported("a lookup through an empty relation")
        key_dims = relation.key_dims
        read = {columns.dim(r): ctx.coords[columns.dim(r)].get_indexer(rows[r].to_numpy()) for r in columns.joined}
        keys = {d: ctx.coords[d].get_indexer(rows[r].to_numpy()) for r, d in zip(relation.declared.key, key_dims)}
        if any((p < 0).any() for p in (*read.values(), *keys.values())):
            raise Unsupported("a relation row off the coordinates")
        if any(d in table.dims and d not in read for d in key_dims):
            raise Unsupported("a lookup onto a dimension the operand keeps")
        if any(d not in table.dims for d in read):
            raise Unsupported("a lookup along a dimension the operand lacks")
        rdims = list(read)
        rsizes = self.sizes(rdims)
        rshape = list(rsizes.values())
        rkey = np.ravel_multi_index(list(read.values()), rshape)
        size = math.prod(rshape)
        blocks = []
        for b in table.blocks:
            b = b.broadcast(rsizes)
            ti, ri = _matches(np.ravel_multi_index(b.positions(rdims), rshape), rkey, size)
            if ti is not None:
                b = b.take(ti)
            blocks.append(b.regrouped(rdims, {d: p[ri] for d, p in keys.items()}))
        present = relation.defined
        const = table.const
        if const is not None:
            const = self.aligned(looked_up(self.full(const, table.dims), columns, ctx))
        elif not bool(present.all()):
            const = xr.DataArray(0.0).where(present)
        dims = _union([d for d in table.dims if d not in read], key_dims)
        return Table(dims, tuple(blocks), const, table.flat)

    def cases(self, node: ms.Cases, ctx: Context) -> Table:
        parts: list[Table | Data] = []
        for region, rows in _regions(node, ctx):
            value = self.value(region.value, _narrowed(ctx, rows))
            parts.append(self.in_region(value, rows))
        if len(parts) < len(node.regions):
            parts.append(_zeros(node, ctx))
        total = parts[0]
        for part in parts[1:]:
            total = self.plus(total, part)
        if isinstance(total, Table):
            return total
        return Table(_dims(total), (), xr.DataArray(total), _flat(total))

    def in_region(self, value: Table | Data, rows: xr.DataArray) -> Table | Data:
        if isinstance(value, float | int):
            return rows * value
        if isinstance(value, xr.DataArray):
            return value.where(rows, 0)
        if rows.ndim == 0:
            return value if bool(rows) else Table(value.dims, (), xr.DataArray(0.0))
        dims = tuple(str(d) for d in rows.dims)
        mask = self.aligned(rows).values
        sizes = self.sizes(dims)
        blocks = []
        for b in value.blocks:
            b = b.broadcast(sizes)
            keep = mask[b.positions(dims)]
            blocks.append(b if keep.all() else b.take(keep))
        const = None if value.const is None else value.const.where(rows, 0)
        return Table(_union(value.dims, dims), tuple(blocks), const, value.blank)

    def plus(self, left: Table | Data, right: Table | Data) -> Table | Data:
        if not isinstance(left, Table) and not isinstance(right, Table):
            return left + right
        a, b = _const(left), _const(right)
        if a is not None and b is not None:
            const = a + b
        else:
            const = a if b is None else b
        blocks = [*getattr(left, "blocks", ()), *getattr(right, "blocks", ())]
        blank = _flat(left) and _flat(right)
        return Table(_union(_dims(left), _dims(right)), tuple(blocks), const, blank)

    def converted(self, value: object) -> Table:
        if isinstance(value, float | int | xr.DataArray):
            data = xr.DataArray(value)
            return Table(_dims(data), (), data)
        if isinstance(value, Variable):
            value = value.to_linexpr()
        if isinstance(value, QuadraticExpression) or not isinstance(value, LinearExpression):
            raise Unsupported(f"a {type(value).__name__} from the eager path")
        store = as_sparse(value._store)
        grid = store.grid
        dims = tuple(grid.dims)
        for d in dims:
            if not grid.indexes[d].equals(self.ctx.coords[d]):
                raise Unsupported(f"an eager result off the master coordinates on '{d}'")
        csr = store.csr
        cells = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
        cols = np.unravel_index(cells, grid.shape) if dims else ()
        block = Block(
            tuple(cols),
            {d: i for i, d in enumerate(dims)},
            csr.indices.astype(np.int64),
            coeff=np.asarray(csr.data, dtype=float),
        )
        values = store.const.reshape(grid.shape)
        const = None
        if not (values == 0).all():
            const = xr.DataArray(values, coords={d: grid.indexes[d] for d in dims}, dims=dims)
        return Table(dims, (block,), const, _flat(values))


def linear(node: ms.Expression, ctx: Context, dims: tuple[str, ...]) -> LinearExpression:
    """*node* as a sparse linear expression framed the way the builder frames a declaration over *dims*."""
    model = ctx.model
    table = Evaluator(ctx).table(node, ctx)
    order = [d for d in ctx.frame(dims) if d in table.dims or d in ctx.dim_order]
    if any(d not in order for d in table.dims):
        raise Unsupported("a term over a dimension the declaration lacks")
    sizes = {d: len(ctx.coords[d]) for d in order}
    shape = list(sizes.values())
    n_cells = math.prod(shape)
    if table.const is None:
        const = np.zeros(n_cells)
    else:
        own = [d for d in order if d in table.const.dims]
        values = np.asarray(table.const.transpose(*own).values, dtype=float)
        values = values.reshape([sizes[d] if d in own else 1 for d in order])
        const = np.empty(shape)
        const[...] = values
        const = const.reshape(-1)
    absent = np.isnan(const)
    data, rows, cols = [], [], []
    for b in table.blocks:
        b = b.broadcast(sizes)
        coeff = b.coefficients()
        if order:
            row = np.ravel_multi_index(b.positions(order), shape)
        else:
            row = np.zeros(len(b), dtype=np.int64)
        var = b.var
        if absent.any():
            keep = ~absent[row]
            row, var, coeff = row[keep], var[keep], coeff[keep]
        data.append(coeff)
        rows.append(row)
        cols.append(var)
    cat = (lambda a, t: np.concatenate(a) if a else np.empty(0, dtype=t))
    csr = coo_to_csr(
        cat(data, float),
        cat(rows, np.int64),
        cat(cols, np.int64),
        (n_cells, model._xCounter),
        model,
    )
    grid = Grid({d: pd.Index(ctx.coords[d]).rename(d) for d in order})
    FALLBACKS.append(f"materialised {getattr(node, 'name', type(node).__name__)} nnz={csr.nnz}")
    return LinearExpression._from_store(CSRStore(csr, const, grid, model), model)
