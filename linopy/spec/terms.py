"""
Term tables: the evaluator of linear declarations.

A node carrying a variable evaluates to a :class:`Table` instead of a linopy
expression. A table holds blocks of terms in long format, one entry per term:
integer position columns, the variable label, and the factors its coefficient
is still to be multiplied by. Beside the blocks sits one dense constant, NaN
where the table is absent. ``+`` concatenates blocks, a sum drops the column
of a dimension, a grouped sum gathers the group of each term through its
relation, and a product with data appends a factor. A table becomes a CSR
store once, when its declaration is added to the model.

:class:`Unsupported` names what a table does not hold: a window sum or a
power of a term, a product of two terms the data leaves standing, a division
by a term or by zero, infinite data beside a term, a lookup of a term through
an empty relation, and a variable, data or a relation row off the master
coordinates. The builder takes every declaration from this path and builds
one that raises it with :func:`linopy.spec.evaluator.evaluate` instead;
:func:`eager_only` turns the path off.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.csr import CSRStore, Grid, _grouped, coo_to_csr
from linopy.expressions import LinearExpression
from linopy.spec import operators
from linopy.spec.context import Context, widened
from linopy.spec.evaluator import (
    _data_under,
    _moved,
    _narrowed,
    _off_rows,
    _regions,
    _sums_nothing,
    evaluate,
    refuse_made,
)
from linopy.spec.where import looked_up

DENSE_KEYS = 50_000_000
_EAGER_ONLY = False


@contextmanager
def eager_only() -> Iterator[None]:
    """Build every declaration with the eager evaluator inside the block: for tests and debugging."""
    global _EAGER_ONLY
    before, _EAGER_ONLY = _EAGER_ONLY, True
    try:
        yield
    finally:
        _EAGER_ONLY = before


def active() -> bool:
    """Whether the builder tries this path: outside :func:`eager_only`."""
    return not _EAGER_ONLY


class Unsupported(Exception):
    """What a declaration holds that a term table does not."""


Data = xr.DataArray | float
Positions = dict[str, np.ndarray]


class Factor(NamedTuple):
    """Data a coefficient is still to be multiplied with or divided by: ``values`` read at the block columns ``cols``, a number where there is none."""

    values: np.ndarray
    cols: tuple[int, ...] = ()
    divide: bool = False


NEGATIVE = Factor(np.array(-1.0))


@dataclass(frozen=True)
class Block:
    """
    Terms in long format: the variable label ``var`` of each, its position
    along each dim in the column ``dims`` names, and the ``factors`` its
    coefficient is still to be multiplied with.

    A column outlives its dim, as a factor reads it by its place in ``cols``.
    """

    cols: tuple[np.ndarray, ...]
    dims: dict[str, int]
    var: np.ndarray
    factors: tuple[Factor, ...] = ()

    def __len__(self) -> int:
        return len(self.var)

    def key(self, sizes: dict[str, int]) -> np.ndarray:
        """The row-major position of each term on the grid of *sizes*."""
        cols = [self.cols[self.dims[d]] for d in sizes]
        return _key(cols, list(sizes.values()), len(self))

    def times(self, factor: Factor) -> Block:
        return replace(self, factors=(*self.factors, factor))

    def coefficients(self) -> np.ndarray:
        """The coefficient of each term: the factors applied in their order, a sign last."""
        c: np.ndarray | None = None
        negative = False
        for values, cols, divide in self.factors:
            if cols:
                at = _key([self.cols[i] for i in cols], values.shape, len(self))
                by = np.take(values.reshape(-1), at)
            elif abs(values) == 1:
                negative ^= bool(values < 0)
                continue
            else:
                by = np.full(len(self), float(values))
            if c is None:
                c = np.divide(1.0, by, out=by) if divide else by
            else:
                (np.divide if divide else np.multiply)(c, by, out=c)
        if c is None:
            return np.full(len(self), -1.0 if negative else 1.0)
        return np.negative(c, out=c) if negative else c

    def take(self, idx: np.ndarray) -> Block:
        """The terms *idx* picks, by position or by mask."""
        cols = tuple(c[idx] for c in self.cols)
        return Block(cols, self.dims, self.var[idx], self.factors)

    def without(self, dims: Iterable[str]) -> Block:
        drop = set(dims)
        return replace(self, dims={d: i for d, i in self.dims.items() if d not in drop})

    def with_cols(self, cols: Positions) -> Block:
        dims = self.dims | {d: len(self.cols) + i for i, d in enumerate(cols)}
        return replace(self, cols=(*self.cols, *cols.values()), dims=dims)

    def broadcast(self, sizes: dict[str, int]) -> Block:
        """Each term repeated onto every cell of the dims of *sizes* it lacks."""
        new = {d: size for d, size in sizes.items() if d not in self.dims}
        if not new:
            return self
        k = math.prod(new.values())
        cells = np.unravel_index(np.tile(np.arange(k), len(self)), list(new.values()))
        repeated = self.take(np.repeat(np.arange(len(self)), k))
        return repeated.with_cols(dict(zip(new, cells)))


@dataclass(frozen=True)
class Table:
    """
    A linear expression over ``dims``: the ``blocks`` of its terms and its
    constant ``const``, zero where None and NaN where the expression is
    absent. ``zero`` says the constant is known to be zero wherever present.
    ``pending`` names the lookup whose joined dims the table is still over.
    """

    dims: tuple[str, ...]
    blocks: tuple[Block, ...] = ()
    const: xr.DataArray | None = None
    zero: bool = False
    pending: ms.JoinColumns | None = None

    @property
    def homogeneous(self) -> bool:
        """Whether the constant is zero wherever present: nothing to scale, sum or gather."""
        return self.const is None or self.zero

    @property
    def offset(self) -> xr.DataArray | None:
        """The constant with a zero where the table is absent, None where it is zero throughout."""
        return None if self.const is None or self.zero else self.const.fillna(0)


def _constant(data: Data) -> Table:
    """*data* as a table without terms."""
    const = data if isinstance(data, xr.DataArray) else xr.DataArray(data)
    return Table(_names(const.dims), (), const, _is_zero(const.values))


def _plus(
    left: Table,
    right: Table,
    minus: bool = False,
    combine: Callable[..., xr.DataArray] = lambda op, a, b: op(a, b),
) -> Table:
    """*left* plus *right*, or *minus* it: the terms of both, the constants put together by *combine*."""
    a, b = left.const, right.const
    if a is None or b is None:
        const = a if b is None else -b if minus else b
    else:
        const = combine(operator.sub if minus else operator.add, a, b)
    theirs = right.blocks
    if minus:
        theirs = tuple(block.times(NEGATIVE) for block in theirs)
    dims = _union(left.dims, right.dims)
    zero = left.homogeneous and right.homogeneous
    return Table(dims, left.blocks + theirs, const, zero)


def _names(dims: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(d) for d in dims)


def _union(*dims: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(d for ds in dims for d in ds))


def _data(node: ms.Expression, ctx: Context) -> Data:
    """The data of *node*, which carries no variable, an infinity no built row reads set to zero."""
    return cast(Data, _off_rows(evaluate(node, ctx), ctx))


def _is_zero(values: np.ndarray) -> bool:
    """Whether *values* are zero wherever they are not NaN."""
    return not np.nan_to_num(values).any()


def _key(cols: Sequence[np.ndarray], shape: Sequence[int], n: int) -> np.ndarray:
    """The row-major position of each of *n* entries of *cols*, positions inside *shape*."""
    if not cols:
        return np.zeros(n, dtype=np.int64)
    key = cols[0]
    for col, size in zip(cols[1:], shape[1:]):
        key = key * size
        key += col
    return key


def _pairs(
    term_keys: np.ndarray, row_keys: np.ndarray, size: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Every pair of a term and a row with the same key, the keys inside ``range(size)``.

    The index of the term and of the row of each pair, the pairs of a term
    together and in the order of its rows. The term index is None where
    every term has one row, so the pairs are the terms.
    """
    if size <= DENSE_KEYS:
        counts = np.bincount(row_keys, minlength=size)
        if counts.max(initial=0) <= 1:
            row_at = np.full(size, -1, dtype=np.int64)
            row_at[row_keys] = np.arange(len(row_keys))
            row = row_at[term_keys]
            found = row >= 0
            if found.all():
                return None, row
            term = np.flatnonzero(found)
            return term, row[term]
        order, start = _grouped(row_keys, size)
        first, count = start[term_keys], counts[term_keys]
    else:
        order = np.argsort(row_keys, kind="stable")
        ordered = row_keys[order]
        first = np.searchsorted(ordered, term_keys, "left")
        count = np.searchsorted(ordered, term_keys, "right") - first
    term = np.repeat(np.arange(len(term_keys)), count)
    nth = np.arange(len(term)) - np.repeat(np.cumsum(count) - count, count)
    return term, order[np.repeat(first, count) + nth]


class Evaluator:
    """The term tables of the nodes of one declaration."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.carry: dict[int, bool] = {}

    def sizes(self, dims: Iterable[str]) -> dict[str, int]:
        return {d: len(self.ctx.coords[d]) for d in dims}

    def carries(self, node: ms.Expression) -> bool:
        """Whether a variable stands under *node*."""
        key = id(node)
        if key not in self.carry:
            self.carry[key] = isinstance(node, ms.VariableReference) or any(
                self.carries(c) for c in ms.children(node)
            )
        return self.carry[key]

    def table(self, node: ms.Expression, ctx: Context, lifted: bool = False) -> Table:
        """The table of *node*, which carries a variable, one use of it consumed; *lifted* leaves a pending lookup for a sum to settle."""
        made = self._table(node, ctx)
        ctx.memo.used(node)
        return made if lifted else self.settled(made)

    def _table(self, node: ms.Expression, ctx: Context) -> Table:
        if isinstance(node, ms.ExpressionReference):
            return self.named(node.body, ctx)
        if isinstance(node, ms.VariableReference):
            return self.variable(node.name)
        if _sums_nothing(node, ctx):
            _data_under(node, ctx)
            return Table(node.dims)
        if isinstance(node, ms.Negate):
            return self.times(self.table(node.operand, ctx), -1.0)
        if isinstance(node, ms.Add):
            return self.added(node, ctx)
        if isinstance(node, ms.Multiply):
            return self.multiplied(node, ctx)
        if isinstance(node, ms.Divide):
            return self.divided(node, ctx)
        if isinstance(node, ms.Sum):
            return self.summed(node, ctx)
        if isinstance(node, ms.Join):
            return self.looked_up(node, ctx)
        if isinstance(node, ms.Translate):
            return self.moved(node, ctx)
        if isinstance(node, ms.Cases):
            return self.cases(node, ctx)
        raise Unsupported(f"a {type(node).__name__} of a term")

    def named(self, body: ms.Expression, ctx: Context) -> Table:
        """The table of a named expression's *body*, kept for the build unless a refusal was masked making it, as the eager path keeps its term."""
        memo = ctx.memo
        key = ("terms.table", body)
        if key in memo.values:
            if body not in memo.inner:
                memo.inner[body] = tuple(ms.walk(body))
            for inner in memo.inner[body]:
                memo.used(inner)
        else:
            memo.uses.pop(body, None)
        return memo.kept(key, lambda: self.table(body, ctx))

    def variable(self, name: str) -> Table:
        """A term per label of the variable *name*, its ``missing`` value the constant where it has none."""
        return self.ctx.memo.kept(
            ("terms.variable", name), lambda: self._variable(name)
        )

    def _variable(self, name: str) -> Table:
        ctx = self.ctx
        labels = self.aligned(ctx.model.variables[ctx.rename(name)].labels)
        dims = _names(labels.dims)
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

    def added(self, node: ms.Add, ctx: Context) -> Table:
        first, second, minus = node.left, node.right, False
        if isinstance(second, ms.Negate):
            second, minus = second.operand, True
        left, right = self.addend(first, ctx), self.addend(second, ctx)

        def refused(
            op: Callable[..., xr.DataArray], a: xr.DataArray, b: xr.DataArray
        ) -> xr.DataArray:
            return refuse_made("add", (node,), op, a, b, ctx)

        return _plus(left, right, minus, refused)

    def addend(self, node: ms.Expression, ctx: Context) -> Table:
        return (
            self.table(node, ctx) if self.carries(node) else _constant(_data(node, ctx))
        )

    def multiplied(self, node: ms.Multiply, ctx: Context) -> Table:
        if self.carries(node.left) and self.carries(node.right):
            raise Unsupported("a product of two terms")
        dead = self.dead(node, ctx)
        if dead is not None:
            return dead
        term, other = node.left, node.right
        if not self.carries(term):
            term, other = other, term
        table = self.table(term, ctx)
        if isinstance(other, ms.Join) and self.defers(other, ctx):
            return self.joined(table, other, ctx)
        return self.times(table, _data(other, ctx))

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
        data = self.aligned(_data(other, ctx))
        if not _is_zero(data.values):
            raise Unsupported("a live product of two terms")
        a = self.table(product.left, ctx)
        b = self.table(product.right, ctx)
        ctx.memo.used(product)
        const = xr.DataArray(0.0).where(data.notnull())
        for c in (a.const, b.const):
            if c is not None:
                const = const.where(c.notnull())
        dims = _union(a.dims, b.dims, _names(data.dims))
        return Table(dims, (), self.full(const, dims), True)

    def divided(self, node: ms.Divide, ctx: Context) -> Table:
        if self.carries(node.divisor):
            raise Unsupported("a division by a term")
        table = self.table(node.numerator, ctx)
        data = _data(node.divisor, ctx)
        if (np.asarray(data) == 0).any():
            raise Unsupported("a division by zero")
        return self.times(table, data, divide=True)

    def times(self, table: Table, data: Data, divide: bool = False) -> Table:
        """*table* times *data*, or with *divide* over it: a factor on each term, spread onto the dims it lacks; a hole in *data* is an absence."""
        data = self.aligned(data)
        values = np.asarray(data.values, dtype=float, order="C")
        if np.isinf(values).any():
            raise Unsupported("infinite data beside a term")
        holes = np.isnan(values)
        if holes.any():
            values = np.where(holes, 1.0, values)
        const = table.const
        if const is not None and not table.zero:
            filled = data.copy(data=values)
            const = const / filled if divide else const * filled
        if holes.any():
            const = xr.DataArray(0.0) if const is None else const
            const = const.where(data.copy(data=~holes))
        dims = _names(data.dims)
        support: Positions | None = None
        blocks = []
        for b in table.blocks:
            if any(d not in b.dims for d in dims):
                if support is None:
                    support = dict(zip(dims, np.nonzero((values != 0) & ~holes)))
                on = {d: at for d, at in support.items() if d in b.dims}
                add = {d: at for d, at in support.items() if d not in b.dims}
                b = self.matched(b, on, add)
            blocks.append(
                b.times(Factor(values, tuple(b.dims[d] for d in dims), divide))
            )
        zero = table.homogeneous
        return Table(_union(table.dims, dims), tuple(blocks), const, zero)

    def defers(self, join: ms.Join, ctx: Context) -> bool:
        """Whether a lookup of data can be read per term: not kept already, through a relation with a row at every key, along value columns only."""
        columns = join.columns
        relation = ctx.relations[columns.name]
        return (
            join not in ctx.memo.values
            and len(relation.rows) > 0
            and not any(r in relation.declared.key for r in columns.joined)
            and not set(join.operand.dims) & set(relation.key_dims)
            and bool(relation.defined.all())
        )

    def joined(self, table: Table, join: ms.Join, ctx: Context) -> Table:
        """*table* times the lookup *join* of data, the dense grid over the key built only where the gather cannot stand in."""
        data = cast(Data, evaluate(join.operand, widened(join, ctx)))
        ctx.memo.used(join)
        made = self.gathered(table, data, join.columns)
        if made is not None:
            return made
        read = looked_up(cast(xr.DataArray, data), join.columns, ctx)
        return self.times(table, cast(Data, _off_rows(read, ctx)))

    def gathered(
        self, table: Table, data: Data, columns: ms.JoinColumns
    ) -> Table | None:
        """*table* times *data* read through a relation: each term gathers the row its key reads, where the data is finite and present throughout."""
        relation = self.ctx.relations[columns.name]
        rows = relation.rows
        keys = dict(zip(relation.key_dims, relation.declared.key))
        read = {columns.dim(r): r for r in columns.joined}
        const_dims = () if table.const is None else table.const.dims
        if (
            not isinstance(data, xr.DataArray)
            or data.dtype.kind != "f"
            or np.isinf(data.values).any()
            or any(d not in table.dims or d in const_dims for d in keys)
            or any(d in table.dims for d in read)
        ):
            return None
        try:
            on = self.positions({d: rows[r] for d, r in keys.items()})
            add = self.positions({d: rows[r] for d, r in read.items()})
        except Unsupported:
            return None
        blocks = tuple(self.matched(b, on, add) for b in table.blocks)
        lifted = replace(table, dims=_union(table.dims, read), blocks=blocks)
        return replace(self.times(lifted, data), pending=columns)

    def settled(self, table: Table) -> Table:
        """*table* off the dims a pending lookup reads along: its constant read through the relation, each term at the key it was gathered for."""
        columns = table.pending
        if columns is None:
            return table
        read = columns.joined_dims
        const = table.const
        if const is not None:
            const = self.read(self.full(const, read), columns)
        dims = tuple(d for d in table.dims if d not in read)
        blocks = tuple(b.without(read) for b in table.blocks)
        return Table(dims, blocks, const, table.zero)

    def summed(self, node: ms.Sum, ctx: Context) -> Table:
        operand = node.operand
        join = (
            operand if isinstance(operand, ms.Join) and operand.columns.axes else None
        )
        inner = widened(node, ctx)
        if join is not None:
            inner = widened(join, inner)
            table = self.through(self.table(join.operand, inner), join.columns)
        else:
            table = self.table(operand, inner, lifted=True)
        for axis in node.over:
            if axis.column is None:
                table = self.sum_over(table, axis.dimension)
        return self.settled(table)

    def sum_over(self, table: Table, dim: str) -> Table:
        pending = table.pending
        if pending is not None and dim in (*pending.joined_dims, *pending.grouped_dims):
            table, pending = self.settled(table), None
        table = self.absorbed(table)
        size = len(self.ctx.coords[dim])
        blocks = tuple(
            b.without([dim])
            if dim in b.dims
            else b.times(Factor(np.array(float(size))))
            for b in table.blocks
        )
        const = table.offset
        if const is not None:
            const = const.sum(dim) if dim in const.dims else const * size
        dims = tuple(d for d in table.dims if d != dim)
        zero = const is not None and _is_zero(const.values)
        return Table(dims, blocks, const, zero, pending)

    def through(self, table: Table, columns: ms.JoinColumns) -> Table:
        """*table* summed through the rows of a relation: each term goes to the group of every row it joins."""
        ctx = self.ctx
        rows = ctx.relations[columns.name].rows
        joined = {columns.dim(r): rows[r].to_numpy() for r in columns.joined}
        grouped = {columns.dim(r): rows[r].to_numpy() for r in columns.grouped}
        table = self.absorbed(table)
        on, add = self.positions(joined), self.positions(grouped)
        blocks = tuple(self.matched(b, on, add, drop=on) for b in table.blocks)
        const = table.offset
        if const is not None:
            onto = {d: ctx.coords[d] for d in grouped}
            const = self.full(const, table.dims)
            const = cast(
                xr.DataArray, operators.grouped_sum(const, joined, grouped, onto)
            )
        dims = _union([d for d in table.dims if d not in joined], grouped)
        return Table(dims, blocks, const, const is not None and _is_zero(const.values))

    def looked_up(self, node: ms.Join, ctx: Context) -> Table:
        """A lookup: each term goes to the key of every relation row that reads its cell."""
        table = self.table(node.operand, widened(node, ctx))
        columns = node.columns
        relation = ctx.relations[columns.name]
        rows = relation.rows
        if not len(rows):
            raise Unsupported("a lookup through an empty relation")
        on = self.positions({columns.dim(r): rows[r] for r in columns.joined})
        keys = zip(relation.key_dims, relation.declared.key)
        add = self.positions({d: rows[r] for d, r in keys})
        blocks = tuple(self.matched(b, on, add, drop=on) for b in table.blocks)
        const = table.const
        if const is not None:
            const = self.read(self.full(const, table.dims), columns)
        elif not bool(relation.defined.all()):
            const = xr.DataArray(0.0).where(relation.defined)
        dims = _union([d for d in table.dims if d not in on], add)
        return Table(dims, blocks, const, table.homogeneous)

    def moved(self, node: ms.Translate, ctx: Context) -> Table:
        """A shift: each term goes to the rows that read its row, found by shifting the positions along the axis as the eager path shifts data."""
        table = self.table(node.operand, widened(node, ctx))
        along = node.along
        labels = ctx.coords[along]
        ordinal = xr.DataArray(np.arange(len(labels), dtype=float), {along: labels})
        source = self.aligned(cast(Data, _moved(node, ordinal, ctx, -1.0)))
        dims = _names(source.dims)
        reached = source.values
        valid = reached >= 0
        to = dict(zip(dims, np.nonzero(valid)))
        read = {**to, along: reached[valid].astype(np.int64)}
        blocks = []
        for b in table.blocks:
            b = b.broadcast(self.sizes([along]))
            on = {d: read[d] for d in dims if d in b.dims}
            add = {d: to[d] for d in dims if d == along or d not in b.dims}
            blocks.append(self.matched(b, on, add, drop=[along]))
        fill = np.nan if node.fill is None else float(node.fill)
        if table.const is not None:
            full = self.full(table.const, table.dims)
            const = self.aligned(cast(Data, _moved(node, full, ctx, node.fill)))
        elif valid.all():
            const = None
        else:
            vacated = np.where(np.isnan(reached), np.nan, fill)
            const = source.copy(data=np.where(valid, 0.0, vacated))
        zero = table.homogeneous and not fill > 0 and not fill < 0
        return Table(_union(table.dims, dims), tuple(blocks), const, zero)

    def cases(self, node: ms.Cases, ctx: Context) -> Table:
        """The cases added up, each its value where it applies and a hard zero elsewhere."""
        total = Table(node.dims)
        for region, rows in _regions(node, ctx):
            inner = _narrowed(ctx, rows)
            if self.carries(region.value):
                part = self.table(region.value, inner)
            else:
                part = _constant(cast(Data, evaluate(region.value, inner)))
            const = None if part.const is None else part.const.where(rows, 0)
            blocks = self.where(part.blocks, self.aligned(rows))
            dims = _union(part.dims, _names(rows.dims))
            total = _plus(total, Table(dims, blocks, const, part.zero))
        return total

    def matched(
        self, b: Block, on: Positions, add: Positions, drop: Iterable[str] = ()
    ) -> Block:
        """*b* joined to the rows *on* and *add* hold the positions of: each term meets the rows at its own positions along the dims of *on*, and takes their positions along the dims of *add* in place of its dims *drop*."""
        sizes = self.sizes(on)
        b = b.broadcast(sizes)
        n_rows = len(next(iter((on | add).values())))
        row_keys = _key(list(on.values()), list(sizes.values()), n_rows)
        term, row = _pairs(b.key(sizes), row_keys, math.prod(sizes.values()))
        if term is not None:
            b = b.take(term)
        return b.without(drop).with_cols({d: at[row] for d, at in add.items()})

    def where(self, blocks: Iterable[Block], keep: xr.DataArray) -> tuple[Block, ...]:
        """*blocks* with the terms on the cells *keep* holds."""
        sizes = self.sizes(_names(keep.dims))
        mask = keep.values.reshape(-1)
        kept = []
        for b in blocks:
            b = b.broadcast(sizes)
            on = mask[b.key(sizes)]
            kept.append(b if on.all() else b.take(on))
        return tuple(kept)

    def absorbed(self, table: Table) -> Table:
        """*table* without the terms on its absent cells, which a sum skips."""
        if table.const is None:
            return table
        present = table.const.notnull()
        if bool(present.all()):
            return table
        return replace(table, blocks=self.where(table.blocks, present))

    def positions(self, labels: dict[str, np.ndarray]) -> Positions:
        """The position of each of *labels* along its dimension."""
        at = {
            d: self.ctx.coords[d].get_indexer(np.asarray(v)) for d, v in labels.items()
        }
        if any((p < 0).any() for p in at.values()):
            raise Unsupported("a relation row off the coordinates")
        return at

    def aligned(self, data: Data) -> xr.DataArray:
        """*data* as an array without stray coordinates, which has to lie on the master coordinates."""
        if not isinstance(data, xr.DataArray):
            return xr.DataArray(data)
        stray = [c for c in data.coords if c not in data.dims]
        if stray:
            data = data.drop_vars(stray)
        for d in data.dims:
            index, master = data.indexes[d], self.ctx.coords[str(d)]
            if index is not master and not index.equals(master):
                raise Unsupported(f"data off the master coordinates on '{d}'")
        return data

    def read(self, const: xr.DataArray, columns: ms.JoinColumns) -> xr.DataArray:
        """*const* read through the relation *columns* name."""
        return self.aligned(cast(xr.DataArray, looked_up(const, columns, self.ctx)))

    def full(self, const: xr.DataArray, dims: Iterable[str]) -> xr.DataArray:
        lacking = {d: self.ctx.coords[d] for d in dims if d not in const.dims}
        return const.expand_dims(lacking) if lacking else const


def linear(
    node: ms.Expression, ctx: Context, dims: tuple[str, ...]
) -> LinearExpression:
    """*node* as a sparse linear expression framed the way the builder frames a declaration over *dims*."""
    return materialised(Evaluator(ctx).table(node, ctx), ctx, dims)


def difference(
    lhs: ms.Expression, rhs: ms.Expression, ctx: Context, dims: tuple[str, ...]
) -> LinearExpression:
    """*lhs* minus *rhs* as one sparse linear expression, framed on the dims of *lhs*: linopy's sum of the two expressions, the constants subtracted unchecked."""
    ev = Evaluator(ctx)
    left, right = ev.table(lhs, ctx), ev.table(rhs, ctx)
    if any(d not in left.dims and d not in ctx.dim_order for d in right.dims):
        raise Unsupported("a right side over a dimension the left side lacks")
    return materialised(_plus(left, right, minus=True), ctx, dims)


def materialised(table: Table, ctx: Context, dims: tuple[str, ...]) -> LinearExpression:
    """*table* as a sparse linear expression framed the way the builder frames a declaration over *dims*."""
    model = ctx.model
    order = [d for d in ctx.frame(dims) if d in table.dims or d in ctx.dim_order]
    if any(d not in order for d in table.dims):
        raise Unsupported("a term over a dimension the declaration lacks")
    sizes = {d: len(ctx.coords[d]) for d in order}
    shape = list(sizes.values())
    const = np.zeros(shape)
    if table.const is not None:
        own = [d for d in order if d in table.const.dims]
        values = table.const.transpose(*own).values
        const[...] = values.reshape([sizes[d] if d in own else 1 for d in order])
    const = const.reshape(-1)
    absent = np.isnan(const)
    masked = bool(absent.any())
    data, rows, cols = [], [], []
    for b in table.blocks:
        b = b.broadcast(sizes)
        coeff, row, var = b.coefficients(), b.key(sizes), b.var
        if masked:
            keep = ~absent[row]
            coeff, row, var = coeff[keep], row[keep], var[keep]
        data.append(coeff)
        rows.append(row)
        cols.append(var)
    csr = coo_to_csr(
        _concatenated(data, float),
        _concatenated(rows, np.int64),
        _concatenated(cols, np.int64),
        (const.size, model._xCounter),
        model,
    )
    grid = Grid({d: pd.Index(ctx.coords[d]).rename(d) for d in order})
    return LinearExpression._from_store(CSRStore(csr, const, grid, model), model)


def _concatenated(arrays: list[np.ndarray], dtype: type) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)
