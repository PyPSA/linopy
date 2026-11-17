"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TypeAlias, TypeVar, assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.constants import FACTOR_DIM
from linopy.csr import CSRStore
from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.spec import operators
from linopy.spec.context import (
    Array,
    Context,
    Memo,
    Term,
    Value,
    as_expression,
    is_empty,
)
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.where import admitted, evaluate_where, groups_of, looked_up
from linopy.variables import Variable

_Binary: TypeAlias = ms.Add | ms.Multiply | ms.Divide | ms.Power
_OPERATORS: dict[type, Callable[[Value, Value], Value]] = {
    ms.Add: operator.add,
    ms.Multiply: operator.mul,
    ms.Divide: operator.truediv,
    ms.Power: operator.pow,
}
_Framed = TypeVar(
    "_Framed", bound=xr.DataArray | LinearExpression | QuadraticExpression
)
Shadow: TypeAlias = float | xr.DataArray
_Sign: TypeAlias = Callable[[Value, Value], Value]


class _Unmirrored(Exception):
    """A term whose constant :func:`_shadow` does not mirror, so it is built in full."""


def fold(name: str, ctx: Context) -> xr.DataArray:
    """The named expression *name* as data, folded over the solution and the parameters *ctx* holds, framed as declared."""
    declared = ctx.program.expressions[name]
    refuse_holes(f"expression '{name}'", [declared.expression], ctx)
    value = evaluate(declared.expression, ctx)
    if isinstance(value, xr.DataArray):
        return without_strays(value).rename(name).transpose(*declared.dims, ...)
    return xr.DataArray(float(cast(float, value)), name=name)


def without_strays(value: _Framed) -> _Framed:
    """*value* without the non-dimension coordinates a lookup leaves beside its dimensions, which no file keeps."""
    stray = [c for c in value.coords if c not in value.dims]
    return cast(_Framed, value.drop_vars(stray)) if stray else value


def evaluate(node: ms.Expression, ctx: Context) -> Value:
    """One node as a linopy term, an array or a number, each node evaluated once per context."""
    if isinstance(node, ms.Constant):
        return node.value
    pinned = isinstance(
        node, ms.ExpressionReference | ms.VariableReference | ms.ParameterReference
    )
    if isinstance(node, ms.ExpressionReference):
        node = node.body
    return ctx.memo.evaluated(node, lambda: _evaluated(node, ctx), pinned=pinned)


def _evaluated(node: ms.Expression, ctx: Context) -> Value:
    if isinstance(node, ms.ExpressionReference):
        return evaluate(node.body, ctx)
    if isinstance(node, ms.Constant):
        return node.value
    if isinstance(node, ms.VariableReference):
        return _variable(node.name, ctx)
    if isinstance(node, ms.Dual):
        return _dual(node.constraint, ctx)
    if isinstance(node, ms.ParameterReference):
        return ctx.parameter(node.name).values
    if not ctx.solved and _sums_nothing(node, ctx) and _carries(node, ctx):
        return _nothing_summed(node, ctx)
    if _prunes(ctx) and _carries(node, ctx) and not _live(node, ctx):
        dead = _dead(node, ctx)
        if dead is not None:
            return dead
    if isinstance(node, ms.Negate):
        return -evaluate(node.operand, ctx)
    if isinstance(node, _Binary):
        return _combine(node, ctx)
    if isinstance(node, ms.Sum):
        return _sum(node, ctx)
    if isinstance(node, ms.Join):
        return looked_up(_operand(node.operand, ctx), node.columns, ctx)
    if isinstance(node, ms.Translate):
        return _moved(node, _operand(node.operand, ctx), ctx, node.fill)
    if isinstance(node, ms.WindowSum):
        return _moved(node, _operand(node.operand, ctx), ctx, None)
    if isinstance(node, ms.Cases):
        return _cases(node, ctx)
    assert_never(node)


def _cases(node: ms.Cases, ctx: Context) -> Value:
    """
    The regions of *node* added up, each its value where it holds and a hard zero elsewhere.

    A region that holds nowhere adds a zero to every row: its term is left
    unbuilt, its data read and refused as if it were built.
    """
    total: Value | None = None
    unbuilt: dict[str, None] = {}
    for region in node.regions:
        if (
            _prunes(ctx)
            and _carries(region.value, ctx)
            and not _admits(region.when, ctx)
            and _mirrored(region.value, ctx)
        ):
            rows = evaluate_where(region.when, ctx)
            unbuilt.update(dict.fromkeys([*region.value.dims, *map(str, rows.dims)]))
            continue
        value = evaluate(region.value, ctx)
        whole = ctx.prune and _everywhere(region.when, ctx)
        part = _in_region(value, evaluate_where(region.when, ctx), whole)
        total = part if total is None else operator.add(*_paired(total, part))
    if not unbuilt:
        return cast(Value, total)
    ctx.memo.pruned += 1
    if isinstance(total, Term) and set(unbuilt) <= set(map(str, total.coord_dims)):
        return total
    zero = operators.constant_term(ctx.model, _full(0.0, tuple(unbuilt), ctx))
    return zero if total is None else operator.add(*_paired(total, zero))


def unpruned(ctx: Context) -> Context:
    """
    *ctx* building every term in full, on a memo of its own that keeps the data and drops the terms.

    A term the data zeroes still counts as a term of its row and of its
    expression, so whatever asks whether one holds a term at all is decided
    on the terms built in full.
    """
    kept = {
        key: value
        for key, value in ctx.memo.values.items()
        if not isinstance(value, LinearExpression | QuadraticExpression)
    }
    return replace(ctx, prune=False, memo=Memo(values=kept))


def _prunes(ctx: Context) -> bool:
    """Whether a term without a live term is left unbuilt: in a build, where a variable is its term."""
    return ctx.prune and not ctx.solved


def _carries(node: ms.Expression, ctx: Context) -> bool:
    """Whether a variable stands under *node*, asked once per node."""
    facts, key = ctx.memo.facts, ("carries", id(node))
    if key not in facts:
        facts[key] = isinstance(node, ms.VariableReference) or any(
            _carries(child, ctx) for child in ms.children(node)
        )
    return cast(bool, facts[key])


def _admits(mask: ms.Mask, ctx: Context) -> bool:
    """Whether *mask* holds at a row, asked once per mask."""
    facts, key = ctx.memo.facts, ("admits", id(mask))
    if key not in facts:
        facts[key] = bool(evaluate_where(mask, ctx).any())
    return cast(bool, facts[key])


def _everywhere(mask: ms.Mask, ctx: Context) -> bool:
    """Whether *mask* holds at every row, asked once per mask."""
    facts, key = ctx.memo.facts, ("everywhere", id(mask))
    if key not in facts:
        facts[key] = bool(evaluate_where(mask, ctx).all())
    return cast(bool, facts[key])


def _presence(name: str, ctx: Context) -> bool | xr.DataArray:
    """Where variable *name* has a label: ``True`` everywhere, ``False`` nowhere, else the rows."""

    def read() -> bool | xr.DataArray:
        present = ctx.model.variables[ctx.rename(name)].labels != -1
        if not present.size or not bool(present.any()):
            return False
        return True if bool(present.all()) else present

    return ctx.memo.kept(("presence", name), read)


def _live(node: ms.Expression, ctx: Context) -> bool:
    """
    Whether *node* may hold a live term: a variable with a label stands
    under it, outside every case that holds nowhere. Asked once per node.
    """
    facts, key = ctx.memo.facts, ("live", id(node))
    if key in facts:
        return cast(bool, facts[key])
    if isinstance(node, ms.VariableReference):
        live = _presence(node.name, ctx) is not False
    elif isinstance(node, ms.Cases):
        live = any(
            _carries(region.value, ctx)
            and _admits(region.when, ctx)
            and _live(region.value, ctx)
            for region in node.regions
        )
    else:
        live = any(
            _live(child, ctx) for child in ms.children(node) if _carries(child, ctx)
        )
    facts[key] = live
    return live


def _dead(node: ms.Expression, ctx: Context) -> LinearExpression | None:
    """The term of *node*, which holds no live term, as its constant alone; ``None`` where :func:`_shadow` does not mirror it."""
    try:
        const = _full(_shadow(node, ctx), node.dims, ctx)
    except _Unmirrored:
        return None
    ctx.memo.pruned += 1
    return operators.constant_term(ctx.model, const)


def _mirrored(node: ms.Expression, ctx: Context) -> bool:
    """Whether the term of *node* is read and refused by its shadow as it is when built."""
    try:
        _shadow(node, ctx)
    except _Unmirrored:
        return False
    return True


def _full(shadow: Shadow, dims: tuple[str, ...], ctx: Context) -> xr.DataArray:
    """*shadow* over every coordinate of *dims*."""
    coords = {d: ctx.coords[d] for d in dims}
    shape = [len(index) for index in coords.values()]
    if not isinstance(shadow, xr.DataArray):
        return xr.DataArray(np.full(shape, float(shadow)), coords=coords, dims=dims)
    held = [d for d in dims if d in shadow.dims]
    if len(held) != shadow.ndim or any(shadow.sizes[d] != len(coords[d]) for d in held):
        raise _Unmirrored
    values = shadow.transpose(*held).values.astype(float, copy=False)
    values = values.reshape([len(coords[d]) if d in held else 1 for d in dims])
    return xr.DataArray(np.broadcast_to(values, shape).copy(), coords=coords, dims=dims)


def _shadow(node: ms.Expression, ctx: Context) -> Shadow:
    """
    The constant of the term *node* evaluates to, NaN where the term is absent, its variables unbuilt.

    A number where it is the same at every row, else an array over dims of
    the node. The data beside the terms is evaluated, so it is read, checked
    and refused as :func:`evaluate` does. Raises :class:`_Unmirrored` where
    the constant of the built term is not known to be this one: data with
    an infinity or a zero divisor beside a term.
    """
    if isinstance(node, ms.VariableReference):
        return _variable_shadow(node.name, ctx)
    if isinstance(node, ms.ExpressionReference):
        if node.body in ctx.memo.values:
            return _constant(evaluate(node, ctx), ctx)
        return _shadow(node.body, ctx)
    if isinstance(node, ms.Negate):
        return -_shadow(node.operand, ctx)
    if isinstance(node, ms.Add):
        return _shadow_added(node, ctx)
    if isinstance(node, ms.Multiply | ms.Divide):
        first, second = _operands(node)
        left, right = _part(first, ctx), _part(second, ctx)
        if _carries(first, ctx) and _carries(second, ctx):
            return left * right
        term, data = (left, right) if _carries(first, ctx) else (right, left)
        return _scaled(term, _off_rows(data, ctx), _OPERATORS[type(node)])
    if isinstance(node, ms.Sum):
        operand = node.operand
        if isinstance(operand, ms.Join) and operand.columns.axes:
            summed = _shadow_through(operand, ctx)
        else:
            summed = _shadow(operand, ctx)
        for axis in node.over:
            if axis.column is None:
                summed = _shadow_summed(summed, axis.dimension, ctx)
        return summed
    if isinstance(node, ms.Join):
        return _shadow_read(node, ctx)
    if isinstance(node, ms.Translate | ms.WindowSum):
        return _shadow_moved(node, ctx)
    if isinstance(node, ms.Cases):
        return _shadow_cases(node, ctx)
    raise _Unmirrored


def _part(node: ms.Expression, ctx: Context) -> Shadow:
    """An operand beside a shadow: the shadow of a term, the value of data."""
    if _carries(node, ctx):
        return _shadow(node, ctx)
    return cast(Shadow, evaluate(node, ctx))


def _is_number(value: object) -> bool:
    return isinstance(value, float | int)


def _variable_shadow(name: str, ctx: Context) -> Shadow:
    """Zero where variable *name* has a label; elsewhere its ``missing:`` number, or NaN for ``absent``."""
    present = _presence(name, ctx)
    missing = ctx.program.variables[name].missing
    fill = np.nan if missing == "absent" else float(missing)
    if isinstance(present, bool):
        return 0.0 if present else fill
    return xr.where(present, 0.0, fill)


def _constant(value: Value, ctx: Context) -> Shadow:
    """The constant of a built *value*, NaN where it is absent."""
    if isinstance(value, Variable):
        return xr.where(value.labels != -1, 0.0, np.nan)
    if isinstance(value, Term):
        return value.const
    raise _Unmirrored


def _shadow_added(node: ms.Add, ctx: Context) -> Shadow:
    """The shadow of a sum or a difference, refused as :func:`_combine` refuses the constants it adds."""
    op, second = operator.add, node.right
    if isinstance(second, ms.Negate):
        op, second = operator.sub, second.operand
    left, right = _part(node.left, ctx), _part(second, ctx)
    if not _carries(node.left, ctx):
        left = cast(Shadow, _off_rows(left, ctx))
    if not _carries(second, ctx):
        right = cast(Shadow, _off_rows(right, ctx))
    if _is_number(left) and _is_number(right):
        made = op(left, right)
        if math.isfinite(made) or math.isnan(left) or math.isnan(right):
            return float(made)
    return refuse_made("add", (node,), op, xr.DataArray(left), xr.DataArray(right), ctx)


def _scaled(term: Shadow, data: Shadow, op: Callable[[Value, Value], Value]) -> Shadow:
    """The shadow *term* multiplied or divided by *data*, a hole in the data an absence."""
    if _is_number(term) and math.isnan(term):
        return np.nan
    factor = np.asarray(data, dtype=float)
    if np.isinf(factor).any() or np.isinf(np.asarray(term, dtype=float)).any():
        raise _Unmirrored
    if op is operator.truediv and (factor == 0).any():
        raise _Unmirrored
    return cast(Shadow, op(term, data))


def _shadow_through(join: ms.Join, ctx: Context) -> Shadow:
    """The shadow of a term summed through the relation of *join*: a group no present row reaches holds zero."""
    inner = _shadow(join.operand, ctx)
    if _is_number(inner) and (math.isnan(inner) or inner == 0):
        return 0.0
    return cast(
        Shadow, _through(_full(inner, join.operand.dims, ctx), join.columns, ctx)
    )


def _shadow_summed(shadow: Shadow, dimension: str, ctx: Context) -> Shadow:
    """*shadow* summed over *dimension*, an absence adding nothing."""
    size = len(ctx.coords[dimension])
    if not isinstance(shadow, xr.DataArray):
        return 0.0 if math.isnan(shadow) else float(shadow) * size
    if dimension in shadow.dims:
        return cast(xr.DataArray, operators.sum_over(shadow, dimension))
    return shadow.fillna(0.0) * size


def _shadow_read(node: ms.Join, ctx: Context) -> Shadow:
    """The shadow of a term read through the relation of *node*: absent where the relation has no row."""
    inner = _shadow(node.operand, ctx)
    if _is_number(inner) and math.isnan(inner):
        return np.nan
    columns = node.columns
    table = ctx.relations[columns.name]
    if _is_number(inner) and not is_empty(columns.joined_dims, ctx):
        if not len(table.rows):
            return np.nan
        present = table.defined
        return float(inner) if bool(present.all()) else xr.where(present, inner, np.nan)
    array = _full(inner, node.operand.dims, ctx)
    return cast(xr.DataArray, looked_up(array, columns, ctx))


def _shadow_moved(node: ms.Translate | ms.WindowSum, ctx: Context) -> Shadow:
    """The shadow of a term shifted or summed back, moved the way the term is."""
    inner = _shadow(node.operand, ctx)
    by = _partition(node.partition, ctx)
    amount = _amount(_amount_of(node), ctx)
    if isinstance(node, ms.Translate) and _is_number(inner) and math.isnan(inner):
        return np.nan
    array = _full(inner, node.operand.dims, ctx)
    if isinstance(node, ms.WindowSum):
        moved = operators.sum_back(
            array, over=node.along, within=amount, wrap=node.wrap, by=by
        )
    else:
        moved = operators.shift(
            array,
            over=node.along,
            offset=amount,
            wrap=node.wrap,
            fill=node.fill,
            by=by,
            as_term=True,
        )
    return cast(xr.DataArray, moved)


def _shadow_cases(node: ms.Cases, ctx: Context) -> Shadow:
    """The shadow of the regions added up, a region that holds nowhere read and left at zero."""
    total: Shadow | None = None
    for region in node.regions:
        rows = evaluate_where(region.when, ctx)
        part: Shadow
        if not _carries(region.value, ctx):
            part = cast(Shadow, _in_region(evaluate(region.value, ctx), rows))
        else:
            value = _shadow(region.value, ctx)
            if not _admits(region.when, ctx):
                part = 0.0
            elif _everywhere(region.when, ctx):
                part = value
            else:
                part = xr.where(rows, value, 0.0)
        total = part if total is None else total + part
    return cast(Shadow, total)


def _sums_nothing(node: ms.Expression, ctx: Context) -> bool:
    """Whether *node* is over a dimension without a coordinate, or sums an operand that is."""
    return is_empty((node.operand if isinstance(node, ms.Sum) else node).dims, ctx)


def _nothing_summed(node: ms.Expression, ctx: Context) -> LinearExpression:
    """
    A term over a dimension without a coordinate, or a sum of one: the constant zero.

    No term under it is built, but what it reads beside its terms is evaluated
    all the same, so its data is read, checked and refused as anywhere else.
    """
    _data_under(node, ctx)
    return operators.constant_term(ctx.model, _zeros(node, ctx))


def _data_under(node: ms.Expression, ctx: Context) -> None:
    """
    Evaluate what *node* reads beside its terms: the operands that hold no
    variable, the conditions of its cases, the amount and partition of a
    shift or window. A use of every other operand is consumed unbuilt.
    """
    if isinstance(node, ms.Cases):
        for region in node.regions:
            evaluate_where(region.when, ctx)
    if isinstance(node, ms.Translate | ms.WindowSum):
        _partition(node.partition, ctx)
        _amount(_amount_of(node), ctx)
    for operand in ms.children(node):
        if not ms.carries_variable(operand):
            evaluate(operand, ctx)
            continue
        ctx.memo.used(operand)
        _data_under(operand, ctx)


def _sum(
    node: ms.Sum, ctx: Context, of: Callable[[ms.Expression], Array] | None = None
) -> Array:
    """
    A sum over its axes: the join axes a relation opens are closed by grouping
    the operand through the relation, the plain ones by summing them away.
    *of* evaluates an operand, :func:`evaluate` when not given.
    """
    operand, columns = _summed(node)
    if of is not None:
        return _closed(node, of(operand), columns, ctx)
    if _prunes(ctx) and _carries(operand, ctx) and not is_empty(operand.dims, ctx):
        steps, term = _scalings(operand, ctx)
        if steps:
            return _weighted_sum(node, steps, term, ctx)
    return _closed(node, _operand(operand, ctx), columns, ctx)


def _summed(node: ms.Sum) -> tuple[ms.Expression, ms.JoinColumns | None]:
    """What *node* sums, and the relation it is summed through, if any."""
    operand = node.operand
    if isinstance(operand, ms.Join) and operand.columns.axes:
        return operand.operand, operand.columns
    return operand, None


def _closed(
    node: ms.Sum, array: Array, columns: ms.JoinColumns | None, ctx: Context
) -> Array:
    """*array*, the evaluated operand of *node*, summed through the relation *columns* and over the plain axes."""
    summed = array if columns is None else _through(array, columns, ctx)
    for axis in node.over:
        if axis.column is None:
            summed = operators.sum_over(summed, axis.dimension)
    return summed


_Scaling: TypeAlias = tuple[ms.Multiply | ms.Divide, ms.Expression, ms.Expression]
_LIVE_SHARE = 0.5
"""A sum of a term scaled by data is built from the live cells alone where the data is live on at most this share of its cells."""


def _scalings(
    node: ms.Expression, ctx: Context
) -> tuple[list[_Scaling], ms.Expression]:
    """
    The scalings of a term by data that end in *node*, the outermost first: per
    scaling its node, the operand carrying the term and the data. Beside them
    the term the innermost one scales.
    """
    steps: list[_Scaling] = []
    while isinstance(node, ms.Multiply | ms.Divide):
        first, second = _operands(node)
        if _carries(first, ctx) == _carries(second, ctx):
            break
        if isinstance(node, ms.Divide) and not _carries(first, ctx):
            break
        term, data = (first, second) if _carries(first, ctx) else (second, first)
        steps.append((node, term, data))
        node = term
    return steps, node


def _weighted_sum(
    node: ms.Sum, steps: Sequence[_Scaling], term: ms.Expression, ctx: Context
) -> Array:
    """
    The sum *node* of *term* scaled as *steps* say, the data read before the term as :func:`_combine` reads it.

    Where the data is zero or a hole at most cells, the sum is built from
    the other cells alone, in one step. Else, and wherever that build is not
    known to give what the operators give one by one, they are applied one
    by one to what is evaluated by then.
    """
    columns = _summed(node)[1]
    factors: list[Value] = []
    value: Value | None = None
    for _, _, data in steps:
        factors.append(evaluate(data, ctx))
        if _annihilates(factors[-1]):
            break
    else:
        value = evaluate(term, ctx)
        fused = _fused(node, columns, steps, factors, value, ctx)
        if fused is not None:
            ctx.memo.pruned += 1
            for product, _, _ in steps:
                ctx.memo.used(product)
            return fused
    reached = list(zip(steps, factors))
    for (product, inner, _), factor in reversed(reached):
        value = _product(product, inner, factor, ctx, value)
        ctx.memo.used(product)
    return _closed(node, cast(Array, value), columns, ctx)


def _fused(
    node: ms.Sum,
    columns: ms.JoinColumns | None,
    steps: Sequence[_Scaling],
    factors: Sequence[Value],
    value: Value,
    ctx: Context,
) -> LinearExpression | None:
    """
    The sum *node* of the term *value* scaled by *factors*, built from the cells where they are live; ``None`` where it is built operator by operator.

    That is where the data is live at more than :data:`_LIVE_SHARE` of its
    cells, where it holds an infinity or a zero divisor, where the term is
    quadratic or its constant infinite, and where a relation reads a key
    twice or names a label the coordinates lack. A zero drops terms the
    operators keep with a zero coefficient, which ``zeroed`` counts.

    A constant of the term is scaled and summed by the operators themselves,
    on a term without a variable, so it is added up in their order.
    """
    if not isinstance(value, Variable) and type(value) is not LinearExpression:
        return None
    operand = steps[0][0]
    scales: list[operators.Scale] = []
    masks: list[tuple[tuple[str, ...], np.ndarray, np.ndarray]] = []
    dead = False
    for (product, _, _), factor in reversed(list(zip(steps, factors))):
        factor = _off_rows(factor, ctx)
        if isinstance(factor, xr.DataArray) and factor.ndim:
            held = factor.values
            if factor.dtype.kind not in "fiu" or np.isinf(held).any():
                return None
            nothing, absent = held == 0, held != held
            if isinstance(product, ms.Divide) and nothing.any():
                return None
            dead = dead or bool(nothing.any()) or bool(absent.any())
            masks.append((tuple(map(str, factor.dims)), nothing, absent))
        else:
            factor = float(cast(float, factor))
            if not math.isfinite(factor) or factor == 0:
                return None
        scales.append((_OPERATORS[type(product)], factor))
    if not dead:
        return None
    zero = functools.reduce(
        operator.or_, (xr.DataArray(nothing, dims=d) for d, nothing, _ in masks)
    )
    hole = functools.reduce(
        operator.or_, (xr.DataArray(absent, dims=d) for d, _, absent in masks)
    )
    live = ~(zero | hole)
    if np.count_nonzero(live.values) > _LIVE_SHARE * live.size:
        return None
    dims = {*map(str, live.dims), *map(str, _coord_dims(value))}
    if dims != set(operand.dims) or not _on_master(value, scales, ctx):
        return None
    const = _constant(value, ctx)
    consts = np.asarray(const, dtype=float)
    if np.isinf(consts).any():
        return None
    joined: dict[str, np.ndarray] | None = None
    grouped: dict[str, np.ndarray] | None = None
    if columns is not None:
        rows = ctx.relations[columns.name].rows
        joined = {
            columns.dim(r): ctx.coords[columns.dim(r)].get_indexer(rows[r].to_numpy())
            for r in columns.joined
        }
        grouped = {
            columns.dim(r): ctx.coords[columns.dim(r)].get_indexer(rows[r].to_numpy())
            for r in columns.grouped
        }
        if not _reads_once(joined, grouped, dims, ctx):
            return None
        dims = (dims - set(joined)) | set(grouped)
    plain = {axis.dimension for axis in node.over if axis.column is None}
    if dims - plain != set(node.dims):
        return None
    own = _coord_dims(value)
    frame = _frame(own, scales, joined, grouped, plain)
    onto = {d: value.indexes[d] if d in own else ctx.coords[d] for d in frame}
    made = operators.weighted_sum(value, scales, live, onto, joined, grouped)
    if bool((zero & ~hole).any()) or _stores_zero(value):
        ctx.memo.zeroed += 1
    if not ((consts != 0) & ~np.isnan(consts)).any():
        return made
    bare: Value = operators.constant_term(ctx.model, xr.DataArray(const))
    for (product, inner, _), factor in reversed(list(zip(steps, factors))):
        bare = _product(product, inner, factor, ctx, bare)
    summed = as_expression(cast(Term, _closed(node, cast(Array, bare), columns, ctx)))
    store = made._store
    flat = _full(summed.const, tuple(store.dims), ctx).values.reshape(-1)
    return LinearExpression._from_store(store.assign_const(flat), made.model)


def _frame(
    own: Sequence[str],
    scales: Sequence[operators.Scale],
    joined: Mapping[str, np.ndarray] | None,
    grouped: Mapping[str, np.ndarray] | None,
    plain: set[str],
) -> list[str]:
    """
    The dimensions of a sum of a scaled term in the order the operators leave them in.

    A scaling puts the dimensions its data adds behind the term's, a
    relation its grouped dimensions where the first joined one stood, and
    a sum keeps the rest in place. A constant summed later is added up in
    the order its cells lie in, so the order is kept to the last bit.
    """
    frame = list(own)
    for _, data in scales:
        if isinstance(data, xr.DataArray):
            frame += [str(d) for d in data.dims if d not in frame]
    if joined is not None and grouped is not None:
        slot = min(frame.index(d) for d in joined)
        frame[slot : slot + 1] = [*grouped, frame[slot]]
        frame = [d for d in frame if d not in joined]
    return [d for d in frame if d not in plain]


def _coord_dims(term: Variable | LinearExpression) -> tuple[str, ...]:
    dims = term.dims if isinstance(term, Variable) else term.coord_dims
    return tuple(str(d) for d in dims)


def _on_master(
    term: Variable | LinearExpression,
    scales: Sequence[operators.Scale],
    ctx: Context,
) -> bool:
    """Whether the term and the data of *scales* lie on the master coordinates of their dimensions."""
    arrays = [data for _, data in scales if isinstance(data, xr.DataArray)]
    held = [(str(d), term.indexes[d]) for d in _coord_dims(term)]
    held += [(str(d), a.indexes[d]) for a in arrays for d in a.dims]
    return all(d in ctx.coords and index.equals(ctx.coords[d]) for d, index in held)


def _reads_once(
    joined: Mapping[str, np.ndarray],
    grouped: Mapping[str, np.ndarray],
    dims: set[str],
    ctx: Context,
) -> bool:
    """Whether a relation has a row, names labels of the coordinates alone, reads no key twice and groups onto dimensions the operand lacks."""
    positions = [*joined.values(), *grouped.values()]
    if not len(positions[0]) or any((p < 0).any() for p in positions):
        return False
    if not set(joined) <= dims or set(grouped) & dims:
        return False
    shape = [len(ctx.coords[d]) for d in joined]
    keys = np.ravel_multi_index(list(joined.values()), shape)
    return len(np.unique(keys)) == len(keys)


def _stores_zero(term: Variable | LinearExpression) -> bool:
    """Whether *term* stores a term with a zero coefficient, which its live terms leave out."""
    if isinstance(term, Variable):
        return False
    store = term._store
    if isinstance(store, CSRStore):
        return bool((store.csr.data == 0).any())
    data = store.ds
    return bool(((data.coeffs.values == 0) & (data.vars.values != -1)).any())


def _through(array: Array, columns: ms.JoinColumns, ctx: Context) -> Array:
    """*array* summed through the rows of the relation *columns* name, joined and grouped as the call says."""
    rows = ctx.relations[columns.name].rows
    joined = {columns.dim(r): rows[r].to_numpy() for r in columns.joined}
    grouped = {columns.dim(r): rows[r].to_numpy() for r in columns.grouped}
    onto = {d: ctx.coords[d] for d in grouped}
    return operators.grouped_sum(array, joined, grouped, onto)


def _variable(name: str, ctx: Context) -> Value:
    variable = ctx.model.variables[ctx.rename(name)]
    missing = ctx.program.variables[name].missing
    if not ctx.solved:
        return variable if missing == "absent" else variable.fillna(missing)
    if "solution" not in variable.data:
        raise AttributeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    solution = variable.solution
    return solution if missing == "absent" else solution.fillna(missing)


def _dual(constraint: str, ctx: Context) -> xr.DataArray:
    """
    The dual of *constraint* as the spec reads it: the rate the objective rises
    with the right side of the written comparison. The builder gathers the
    variables onto the left, flipping a row written with them on the right,
    and such a row's dual is negated back.
    """
    if not ctx.solved:
        raise TypeError(
            f"the dual of constraint '{constraint}' has no symbolic form, read `.solution`"
        )
    built = ctx.model.constraints.data
    renamed = ctx.rename(constraint)
    reported: xr.DataArray | None = None
    if renamed in built:
        reported = built[renamed]._dual
    elif any(c._dual is not None for c in built.values()):
        dims = ctx.program.constraints[constraint].dims
        shape = [len(ctx.coords[d]) for d in dims]
        coords = {d: ctx.coords[d] for d in dims}
        reported = xr.DataArray(np.full(shape, np.nan), coords, dims=dims)
    if reported is None:
        raise AttributeError(
            f"constraint '{constraint}' has no dual yet: solve the model, with a solver "
            f"and a problem that report duals, before reading one"
        )
    return -reported if flipped(ctx.program.constraints[constraint]) else reported


def flipped(constraint: ms.Constraint) -> bool:
    """Whether the builder gathers the constraint's variables from its right side onto the left."""
    return not ms.carries_variable(constraint.lhs)


def _operands(node: _Binary) -> tuple[ms.Expression, ms.Expression]:
    if isinstance(node, ms.Divide):
        return node.numerator, node.divisor
    if isinstance(node, ms.Power):
        return node.base, node.exponent
    return node.left, node.right


def _combine(node: _Binary, ctx: Context) -> Value:
    """
    The operands of *node* combined by its operator.

    Two arrays have to agree on their shared coordinates, and the data they
    make has to be finite wherever both are: a NaN made of present data would
    read as an absence and silently drop the row, an infinity would stand in
    the problem. A hole beside a term becomes the term's absence.
    """
    if isinstance(node, ms.Add) and _prunes(ctx) and _carries(node, ctx):
        return _added(node, ctx)
    first, second = _operands(node)
    if isinstance(node, ms.Multiply | ms.Divide) and _prunes(ctx):
        if _carries(first, ctx) != _carries(second, ctx):
            term, data = (first, second) if _carries(first, ctx) else (second, first)
            return _product(node, term, evaluate(data, ctx), ctx)
    if isinstance(node, ms.Add) and isinstance(second, ms.Negate):
        second = second.operand
    return _made(node, evaluate(first, ctx), evaluate(second, ctx), ctx)


def _product(
    node: ms.Multiply | ms.Divide,
    term: ms.Expression,
    factor: Value,
    ctx: Context,
    scaled: Value | None = None,
) -> Value:
    """
    The term of *term* scaled by the data *factor* as *node* says, *scaled* being the term where it is evaluated already.

    A term scaled by zeros and holes alone is left unbuilt, as its constant.
    """
    if scaled is None:
        if _annihilates(factor):
            zeroed = _zeroed(node, term, factor, ctx)
            if zeroed is not None:
                return zeroed
        scaled = evaluate(term, ctx)
    if term is _operands(node)[0]:
        return _made(node, scaled, factor, ctx)
    return _made(node, factor, scaled, ctx)


def _made(node: _Binary, left: Value, right: Value, ctx: Context) -> Value:
    """The evaluated operands *left* and *right* of *node* combined by its operator, refused as :func:`_combine` says."""
    op = _OPERATORS[type(node)]
    if isinstance(node, ms.Add) and isinstance(node.right, ms.Negate):
        op = operator.sub
    kind = type(node).__name__.lower()
    if _zeroes_decide(node, left, right, ctx):
        return _combine(node, unpruned(ctx))
    if isinstance(left, Term) or isinstance(right, Term):
        left, right = _off_rows(left, ctx), _off_rows(right, ctx)
        if isinstance(node, ms.Add):
            a, b = constant_of(left), constant_of(right)
            refuse_made(kind, (node,), op, a, b, ctx)
        made = op(*_paired(left, right))
        return _without_dead_terms(made) if isinstance(node, ms.Multiply) else made
    if not isinstance(left, xr.DataArray) and not isinstance(right, xr.DataArray):
        return op(left, right)
    a, b = xr.DataArray(left), xr.DataArray(right)
    for dim in set(a.dims) & set(b.dims):
        if not a.indexes[dim].equals(b.indexes[dim]):
            raise SpecDataError(
                f"operands are not aligned on '{dim}': {a.indexes[dim].tolist()[:5]} against "
                f"{b.indexes[dim].tolist()[:5]}. Every operand is read on the master "
                f"coordinates, so the data was attached against other labels than the model was built on."
            )
    return refuse_made(kind, (node,), op, a, b, ctx)


def _chain(
    node: ms.Expression, ctx: Context
) -> tuple[ms.Expression, list[tuple[ms.Add, _Sign, ms.Expression]]]:
    """
    The chain of sums and differences ending in *node*: its first operand,
    then per step the node, its sign and the operand it adds. The chain runs
    down the left operands for as long as they are sums carrying a variable.
    """
    steps: list[tuple[ms.Add, _Sign, ms.Expression]] = []
    while isinstance(node, ms.Add) and _carries(node, ctx):
        sign, operand = operator.add, node.right
        if isinstance(operand, ms.Negate):
            sign, operand = operator.sub, operand.operand
        steps.append((node, sign, operand))
        node = node.left
    return node, steps[::-1]


def _added(node: ms.Add, ctx: Context) -> Value:
    """
    A chain of sums and differences that carries a variable, its terms merged in one step.

    Where an operand holds an infinity or data, the constants are added
    operand by operand in the order of the chain and refused as each step
    of it is, and the result holds exactly that constant: no step refuses
    operands that are finite throughout. Data adds to the constant alone,
    in one step; a hole in it is the absence of the row.
    """
    first, steps = _chain(node, ctx)
    ctx.memo.pruned += 1
    operands = [(None, operator.add, first), *steps]
    values = [_off_rows(evaluate(operand, ctx), ctx) for _, _, operand in operands]
    const: xr.DataArray | None = None
    if any(not isinstance(v, Term) or _holds_infinity(v) for v in values):
        try:
            const = _chain_constant(operands, values, ctx, exact=False)
        except SpecDataError:
            const = _chain_constant(operands, values, ctx, exact=True)
    terms: list[LinearExpression | QuadraticExpression] = []
    negate: list[bool] = []
    data: Value | None = None
    for (_, sign, _), value in zip(operands, values):
        if isinstance(value, Term):
            terms.append(as_expression(value))
            negate.append(sign is operator.sub)
        else:
            signed = -value if sign is operator.sub else value
            data = signed if data is None else data + signed
    made: LinearExpression | QuadraticExpression
    if all(type(term) is LinearExpression for term in terms):
        if len(terms) == 1 and not negate[0]:
            made = terms[0]
        else:
            made = merge(cast(list[LinearExpression], terms), negate=negate)
    else:
        signed_terms = (-t if flip else t for t, flip in zip(terms, negate))
        made = functools.reduce(operator.add, signed_terms)
    if data is None or const is None:
        return made
    return _with_constant(operator.add(*_paired(made, data)), const, ctx)


def _holds_infinity(term: Term) -> bool:
    """Whether the constant of *term* is an infinity at a row."""
    if isinstance(term, Variable):
        return False
    if isinstance(term, LinearExpression):
        return bool(np.isinf(term._store.const).any())
    return bool(np.isinf(term.const.values).any())


def _chain_constant(
    operands: Sequence[tuple[ms.Add | None, _Sign, ms.Expression]],
    values: Sequence[Value],
    ctx: Context,
    exact: bool,
) -> xr.DataArray:
    """
    The constant a chain of sums adds up to, refused step by step as :func:`_combine` refuses a sum.

    A bare variable adds a zero, also where it is masked, which refuses
    whatever the chain built step by step refuses and at most more. *exact*
    gives each partial sum the absences and the dims of the term built by
    then, so it refuses exactly the same and names the same coordinates.
    """
    const = xr.DataArray(0.0)
    over: dict[str, None] = {}
    for (step, sign, operand), value in zip(operands, values):
        part = xr.DataArray(
            _constant_part(operand, value, ctx) if exact else constant_of(value)
        )
        if step is None:
            const = part
        else:
            if exact and over:
                dims = (*over, *(str(d) for d in const.dims if d not in over))
                const = _full(const, dims, ctx)
            const = refuse_made("add", (step,), sign, const, part, ctx)
        if isinstance(value, Term):
            over.update(dict.fromkeys(map(str, value.coord_dims)))
    return const


def _constant_part(operand: ms.Expression, value: Value, ctx: Context) -> Shadow:
    """The constant *value* adds to a sum, NaN where it is absent."""
    if isinstance(operand, ms.VariableReference) and isinstance(value, Variable):
        return _variable_shadow(operand.name, ctx)
    if isinstance(value, Term):
        return _constant(value, ctx)
    return cast(Shadow, value)


def _with_constant(made: Value, const: xr.DataArray, ctx: Context) -> Value:
    """The linear expression *made* holding exactly *const* wherever it is present."""
    if not isinstance(made, LinearExpression):
        return made
    store = made._store
    held = np.asarray(store.const).reshape(-1)
    values = _full(const, tuple(store.dims), ctx).values.reshape(-1)
    values[np.isnan(held)] = np.nan
    if np.array_equal(values, held, equal_nan=True):
        return made
    return LinearExpression._from_store(store.assign_const(values), made.model)


def _zeroes_decide(node: _Binary, left: Value, right: Value, ctx: Context) -> bool:
    """Whether a product of two terms has a factor without a term, where a term the data zeroed is left unbuilt: linopy multiplies a constant factor another way."""
    if not (ctx.prune and ctx.memo.zeroed and isinstance(node, ms.Multiply)):
        return False
    if not (isinstance(left, Term) and isinstance(right, Term)):
        return False
    return any(as_expression(side).is_constant for side in (left, right))


def _annihilates(factor: Value) -> bool:
    """Whether the data *factor* is a zero or a hole at every row, so no term it scales stays live."""
    values = np.asarray(factor)
    if values.dtype.kind not in "fiub" or not values.size:
        return False
    head = values.flat[0]
    if head != 0 and head == head:
        return False
    return bool(((values == 0) | np.isnan(values)).all())


def _zeroed(
    node: ms.Multiply | ms.Divide, term: ms.Expression, factor: Value, ctx: Context
) -> LinearExpression | None:
    """
    The product of the term of *term* and a *factor* of zeros and holes as its constant alone.

    ``None`` where :func:`_shadow` does not mirror the term. A zero drops
    terms a full build keeps with a zero coefficient, which ``zeroed`` counts.
    """
    try:
        shadow = _scaled(
            _shadow(term, ctx), cast(Shadow, factor), _OPERATORS[type(node)]
        )
        const = _full(shadow, node.dims, ctx)
    except _Unmirrored:
        return None
    ctx.memo.pruned += 1
    if bool((np.asarray(factor) == 0).any()):
        ctx.memo.zeroed += 1
    return operators.constant_term(ctx.model, const)


def _without_dead_terms(made: Value) -> Value:
    """
    A quadratic *made* whose terms the data zeroes throughout, as its constant.

    A quadratic has no sparse (CSR) backing, so one left standing densifies
    every sparse term it meets; its constant is built on the model's store.
    """
    if not isinstance(made, QuadraticExpression):
        return made
    data = made.data
    live = (data.coeffs != 0) & (data.vars != -1).any(FACTOR_DIM)
    if bool(live.any()):
        return made
    return LinearExpression.from_constant(made.model, data.const)


def _off_rows(value: Value, ctx: Context) -> Value:
    """*value* with an infinity no built row reads set to zero, so a term's zero there does not make a NaN that core linopy refuses."""
    if (
        ctx.rows is None
        or not isinstance(value, xr.DataArray)
        or value.dtype.kind not in "fc"
    ):
        return value
    infinite = np.isinf(value)
    if not bool(infinite.any()):
        return value
    other = [d for d in ctx.rows.dims if d not in value.dims]
    read = ctx.rows.any(other) if other else ctx.rows
    off = infinite & ~read
    if not bool(off.any()):
        return value
    ctx.memo.masked += 1
    return value.where(~off, 0.0)


def _paired(left: Value, right: Value) -> tuple[Value, Value]:
    """*left* and *right* to combine, a hole in data beside a term becoming the term's absence."""
    if isinstance(left, xr.DataArray) and isinstance(right, Term):
        right, left = carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(left, Term):
        left, right = carried(left, right)
    return left, right


def constant_of(value: Value) -> xr.DataArray:
    """The constant part of *value*: a term's constant, or the data itself."""
    if isinstance(value, Variable):
        return xr.DataArray(0.0)
    if isinstance(value, Term):
        return value.const
    return xr.DataArray(value)


def refuse_made(
    kind: str,
    nodes: Sequence[ms.Expression],
    op: Callable[[Value, Value], Value],
    left: xr.DataArray,
    right: xr.DataArray,
    ctx: Context,
) -> xr.DataArray:
    """*op* of *left* and *right*, refused wherever it is not finite though both are present, the parameters *nodes* read named."""
    made = cast(xr.DataArray, op(left, right))
    if made.dtype.kind not in "fc" or bool(np.isfinite(made.values).all()):
        return made
    nan_made = made.isnull() & left.notnull() & right.notnull()
    bad = nan_made | (np.isinf(made) & _finite(left) & _finite(right))
    if ctx.rows is not None and bool(bad.any()):
        ctx.memo.masked += 1
        bad = bad & ctx.rows
    if not bool(bad.any()):
        return made
    named = ", ".join(f"'{p}'" for p in sorted(ms.parameters_of(*nodes)))
    raise SpecDataError(
        f"{ctx.subject}: the {kind} of {named} is not finite at "
        f"{first_coordinates(bad, 3)}, though its operands are present. A NaN made there "
        f"would read as an absence and silently drop the row; an infinity would "
        f"stand in the problem.\n"
        f"  Mask the coordinates out with a where, or supply data the arithmetic "
        f"is defined on."
    )


def _finite(arr: xr.DataArray) -> xr.DataArray:
    if arr.dtype.kind in "fc":
        return arr.notnull() & np.isfinite(arr.fillna(0.0))
    return xr.ones_like(arr, dtype=bool)


def carried(term: Term, data: xr.DataArray) -> tuple[Term, xr.DataArray]:
    """A hole an operator left in *data* is an absence the term takes: the slot leaves the row, and the hole reads as a harmless one."""
    if not bool(data.isnull().any()):
        return term, data
    return term.where(data.notnull()), data.fillna(1.0)


def _operand(node: ms.Expression, ctx: Context) -> Array:
    """An operand the language gives dims, so never a bare number."""
    return cast(Array, evaluate(node, ctx))


def _in_region(value: Value, rows: xr.DataArray, whole: bool = False) -> Value:
    """
    *value* where the region holds and a hard zero everywhere else: a fill, so absence inside the region stands.

    A region that is *whole*, holding at every row, leaves a value over its dims as it is.
    """
    if isinstance(value, float | int):
        return rows * value
    expression = as_expression(value)
    dims = value.dims if isinstance(value, xr.DataArray) else value.coord_dims
    if whole and set(rows.dims) <= set(dims):
        return expression
    return expression.where(rows, 0)


def _amount(amount: int | str, ctx: Context) -> operators.Amount:
    """An amount, a ``refused`` hole moving nothing: no live row reads it, as :func:`refuse_holes` checks."""
    if not isinstance(amount, str):
        return amount
    values = ctx.parameter(amount).values
    if ctx.program.parameters[amount].missing == "refused":
        return values.fillna(0)
    return values


def _moved(
    node: ms.Translate | ms.WindowSum, operand: Array, ctx: Context, fill: float | None
) -> Array:
    """*operand* shifted or summed back the way *node* says, a vacated row of a shift *fill*."""
    by = _partition(node.partition, ctx)
    amount = _amount(_amount_of(node), ctx)
    if isinstance(node, ms.Translate):
        return operators.shift(
            operand, over=node.along, offset=amount, wrap=node.wrap, fill=fill, by=by
        )
    return operators.sum_back(
        operand, over=node.along, within=amount, wrap=node.wrap, by=by
    )


def _amount_of(node: ms.Translate | ms.WindowSum) -> int | str:
    return node.offset if isinstance(node, ms.Translate) else node.width


def _partition(partition: ms.Partition | None, ctx: Context) -> xr.DataArray | None:
    """The relation a windowed operator stays inside, named for the dimension its group column is over."""
    return None if partition is None else groups_of(partition, ctx)


def refuse_holes(
    subject: str,
    nodes: Sequence[ms.Expression],
    ctx: Context,
    rows: xr.DataArray | None = None,
) -> None:
    """
    Refuse a missing row of a ``missing: refused`` parameter that a row of *nodes* reads.

    *nodes* are read together, as the sides of a comparison: a row exists
    where *rows* keeps it, every row when not given, and none of *nodes* is
    absent there. A hole is read where it reaches such a row through the
    operators that move and gather data on the way, unless a term it meets
    is absent there: a masked variable or an ``absent`` parameter row takes
    the hole's term with it, as in :func:`evaluate`. A condition never reads
    a row: a missing row compares as false there, whatever its ``missing:``.
    Outside a parameter's ``where:`` it is absent, so no row there is missing.
    """
    live = xr.DataArray(True) if rows is None else rows
    for name in sorted(_read_parameters(nodes)):
        present = ctx.parameter(name).present
        if ctx.program.parameters[name].missing != "refused" or bool(present.all()):
            continue
        holes = ~present & admitted(name, ctx)
        found = _read_holes(holes, functools.partial(_reads, nodes, name, live, ctx))
        if bool(found.any()):
            raise SpecDataError(
                f"{subject} reads parameter '{name}' at {first_coordinates(found, 3)}, "
                f"where it has no row. A missing: refused parameter needs a row wherever "
                f"a declaration reads it.\n"
                f"  Supply the row, mask the coordinate out with a where:, or say what a "
                f"missing row means with missing:."
            )


def _reads(
    nodes: Sequence[ms.Expression],
    name: str,
    live: xr.DataArray,
    ctx: Context,
    holes: xr.DataArray,
) -> bool:
    """Whether a row of *nodes* that *live* keeps reads one of *holes* of parameter *name*."""
    marks = functools.reduce(operator.add, (_marks(n, name, holes, ctx) for n in nodes))
    return bool(((marks > 0) & live).any())


def _read_parameters(nodes: Sequence[ms.Expression]) -> set[str]:
    """Every parameter *nodes* read, the ones naming a shift or window amount included."""
    amounts = (
        _amount_of(node)
        for node in ms.walk(*nodes)
        if isinstance(node, ms.Translate | ms.WindowSum)
    )
    return set(ms.parameters_of(*nodes)) | {a for a in amounts if isinstance(a, str)}


def _read_holes(
    holes: xr.DataArray, reads: Callable[[xr.DataArray], bool], n: int = 3
) -> xr.DataArray:
    """The first *n* of *holes* that *reads* finds read, found by halving the candidates."""
    found: list[int] = []
    pending = [np.flatnonzero(holes.values)]
    while pending and len(found) < n:
        positions = pending.pop()
        if not reads(_only(holes, positions)):
            continue
        if len(positions) == 1:
            found.append(int(positions[0]))
            continue
        half = len(positions) // 2
        pending += [positions[half:], positions[:half]]
    return _only(holes, found)


def _only(holes: xr.DataArray, positions: Sequence[int] | np.ndarray) -> xr.DataArray:
    picked = np.zeros(holes.size, dtype=bool)
    picked[positions] = True
    return holes.copy(data=picked.reshape(holes.shape))


def _marks(
    node: ms.Expression, name: str, holes: xr.DataArray, ctx: Context
) -> xr.DataArray:
    """
    Where *node* reads one of *holes*, the missing rows of parameter *name*:
    positive there, zero elsewhere, NaN where *node* is absent.

    Absence spreads as it does in :func:`evaluate`: through every arithmetic
    operator, a shift and a lookup, but not out of a sum or a window. A case
    reads its value inside its region only. A parameter amount reads its own
    row at every row the operator gives.
    """

    def of(operand: ms.Expression) -> xr.DataArray:
        return _marks(operand, name, holes, ctx)

    if isinstance(node, ms.ParameterReference):
        return _parameter_marks(node, name, holes, ctx)
    if isinstance(node, ms.VariableReference):
        labels = ctx.model.variables[ctx.rename(node.name)].labels
        zeros = xr.zeros_like(labels, dtype=float)
        if ctx.program.variables[node.name].missing != "absent":
            return zeros
        return zeros.where(labels != -1)
    if isinstance(node, ms.Constant | ms.Dual):
        return _zeros(node, ctx)
    if isinstance(node, ms.ExpressionReference):
        return of(node.body)
    if isinstance(node, ms.Negate):
        return of(node.operand)
    if isinstance(node, _Binary):
        left, right = (of(operand) for operand in _operands(node))
        return left + right
    if isinstance(node, ms.Sum):
        return cast(xr.DataArray, _sum(node, ctx, lambda o: of(o).fillna(0.0)))
    if isinstance(node, ms.Join):
        return cast(xr.DataArray, looked_up(of(node.operand), node.columns, ctx))
    if isinstance(node, ms.Translate | ms.WindowSum):
        operand = of(node.operand)
        if isinstance(node, ms.WindowSum):
            operand = operand.fillna(0.0)
        fill = None if isinstance(node, ms.Translate) and node.fill is None else 0.0
        moved = cast(xr.DataArray, _moved(node, operand, ctx, fill))
        amount = _amount_marks(node, name, holes, ctx)
        return (moved.fillna(0.0) + amount).where(moved.notnull() | (amount > 0))
    if isinstance(node, ms.Cases):
        regions = (
            of(region.value).where(evaluate_where(region.when, ctx), 0.0)
            for region in node.regions
        )
        return functools.reduce(operator.add, regions)
    assert_never(node)


def _parameter_marks(
    node: ms.ParameterReference, name: str, holes: xr.DataArray, ctx: Context
) -> xr.DataArray:
    """*holes* on parameter *name*, NaN where a parameter is absent: a missing ``absent`` row, or outside its ``where:``."""
    declared = ctx.program.parameters[node.name]
    if node.name == name:
        return holes.astype(float).where(admitted(name, ctx))
    if declared.missing != "absent" and declared.where is None:
        return _zeros(node, ctx)
    present = ctx.parameter(node.name).present
    exists = present if declared.missing == "absent" else admitted(node.name, ctx)
    return xr.zeros_like(present, dtype=float).where(exists)


def _zeros(node: ms.Expression, ctx: Context) -> xr.DataArray:
    return xr.DataArray(
        np.zeros([len(ctx.coords[d]) for d in node.dims]),
        coords={d: ctx.coords[d] for d in node.dims},
        dims=node.dims,
    )


def _amount_marks(
    node: ms.Translate | ms.WindowSum, name: str, holes: xr.DataArray, ctx: Context
) -> xr.DataArray:
    """The marks a parameter amount leaves on the rows it gives, read per group under its partition."""
    if _amount_of(node) != name:
        return xr.DataArray(0.0)
    marks = holes.astype(float)
    by = _partition(node.partition, ctx)
    if by is None:
        return marks
    return cast(xr.DataArray, operators.per_group(marks, by)).fillna(0.0)
