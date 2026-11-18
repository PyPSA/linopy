"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import TypeAlias, TypeVar, assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.constants import FACTOR_DIM
from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.spec import operators
from linopy.spec.context import (
    Array,
    Context,
    Term,
    Value,
    as_expression,
    is_empty,
)
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.where import admitted, evaluate_where, groups_of, looked_up
from linopy.store import DenseStore
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
_Step: TypeAlias = tuple[ms.Add, ms.Expression]
_Factor: TypeAlias = tuple[ms.Multiply | ms.Divide, ms.Expression, ms.Expression]
"""A product or quotient that scales a term by data: its node, the operand carrying the term and the data."""
_LIVE_SHARE = 0.5
"""A sum of a term scaled by data is built from the live cells alone where the data is live on at most this share of its cells."""


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
    if not ctx.solved and _sums_nothing(node, ctx) and ms.carries_variable(node):
        return _nothing_summed(node, ctx)
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
    The regions of *node* added up, each its value where it applies and a hard zero elsewhere.

    A region's value is evaluated and checked only where the region
    applies, and a region that applies nowhere is not evaluated: its value
    is never read.
    """
    total: Value | None = None
    for region, rows in _regions(node, ctx):
        part = _in_region(evaluate(region.value, _narrowed(ctx, rows)), rows)
        total = part if total is None else operator.add(*_paired(total, part))
    held = set() if total is None else set(_dims_of(total))
    if total is not None and held >= set(node.dims):
        return total
    zeros = _zeros(node, ctx)
    if ms.carries_variable(node):
        return (
            operators.constant_term(ctx.model, zeros)
            if total is None
            else total + zeros
        )
    return zeros if total is None else total + zeros


def _regions(node: ms.Cases, ctx: Context) -> Iterator[tuple[ms.Region, xr.DataArray]]:
    """The regions of *node* that apply somewhere, each with where it applies; a use of every other is consumed unread."""
    for region in node.regions:
        rows = evaluate_where(region.when, ctx)
        if bool(rows.any()):
            yield region, rows
        else:
            ctx.memo.skipped(region.value)


def _dims_of(value: Value) -> tuple[str, ...]:
    if isinstance(value, xr.DataArray):
        return tuple(map(str, value.dims))
    if isinstance(value, Variable | LinearExpression | QuadraticExpression):
        return tuple(map(str, as_expression(value).coord_dims))
    return ()


def _narrowed(ctx: Context, rows: xr.DataArray) -> Context:
    """*ctx* checking only the rows *rows* keeps of the ones it checks."""
    return replace(ctx, rows=rows if ctx.rows is None else ctx.rows & rows)


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
        for region, rows in _regions(node, ctx):
            _data_of(region.value, _narrowed(ctx, rows))
        return
    if isinstance(node, ms.Translate | ms.WindowSum):
        _partition(node.partition, ctx)
        _amount(_amount_of(node), ctx)
    for operand in ms.children(node):
        _data_of(operand, ctx)


def _data_of(operand: ms.Expression, ctx: Context) -> None:
    """Evaluate *operand* if it holds no variable, else what it reads beside its terms."""
    if not ms.carries_variable(operand):
        evaluate(operand, ctx)
        return
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
    if ctx.fuse and not ctx.solved:
        factors, term = _factors_of(operand, ctx)
        if factors:
            return _weighted_sum(node, factors, term, ctx)
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


def _factors_of(
    node: ms.Expression, ctx: Context
) -> tuple[list[_Factor], ms.Expression]:
    """
    The factors that scale a term by data down from *node*, the outermost
    first, and the term the innermost one scales. A product kept evaluated
    is read back as the term.
    """
    factors: list[_Factor] = []
    while isinstance(node, ms.Multiply | ms.Divide) and node not in ctx.memo.values:
        first, second = _operands(node)
        carries = ms.carries_variable(first)
        if carries == ms.carries_variable(second):
            break
        if isinstance(node, ms.Divide) and not carries:
            break
        term, data = (first, second) if carries else (second, first)
        factors.append((node, term, data))
        node = term
    return factors, node


def _weighted_sum(
    node: ms.Sum, factors: Sequence[_Factor], term: ms.Expression, ctx: Context
) -> Array:
    """
    The sum *node* of *term* scaled by *factors*, built by :func:`_fused`
    where it can, else operator by operator. The operands are evaluated in
    the order :func:`_combine` evaluates them.
    """
    columns = _summed(node)[1]
    read: dict[int, Value] = {
        i: evaluate(data, ctx)
        for i, (product, _, data) in enumerate(factors)
        if data is _operands(product)[0]
    }
    value = evaluate(term, ctx)
    for i in reversed(range(len(factors))):
        if i not in read:
            read[i] = evaluate(factors[i][2], ctx)
    data = [read[i] for i in range(len(factors))]
    for product, _, _ in factors:
        ctx.memo.used(product)
    fused = _fused(node, columns, factors, data, value, ctx)
    if fused is not None:
        return fused
    for factor, datum in reversed(list(zip(factors, data))):
        value = _scaled(factor, value, datum, ctx)
    return _closed(node, cast(Array, value), columns, ctx)


def _scaled(factor: _Factor, value: Value, datum: Value, ctx: Context) -> Value:
    """The term *value* scaled by the data *datum* as *factor* says, its operands in their written order."""
    product, term, _ = factor
    if term is _operands(product)[0]:
        return _made(product, value, datum, ctx)
    return _made(product, datum, value, ctx)


def _fused(
    node: ms.Sum,
    columns: ms.JoinColumns | None,
    factors: Sequence[_Factor],
    data: Sequence[Value],
    value: Value,
    ctx: Context,
) -> LinearExpression | None:
    """
    The sum *node* of the term *value* scaled by *data*, built from the cells where the data is live; ``None`` where it is built operator by operator.

    That is where the data is live at more than :data:`_LIVE_SHARE` of its
    cells, where :func:`_weights` refuses it, where the term is quadratic or
    its constant infinite, where the operands leave the master coordinates,
    where a relation reads a key twice or names a label the coordinates
    lack, and where a coefficient overflows. The sum holds the live terms
    alone, where the operators keep a term with a zero coefficient and an
    empty slot; ``fused`` counts it.

    A constant of the term is scaled and summed by the operators themselves,
    on a term without a variable, so it is added up in their order.
    """
    weighted = _weights(factors, data, ctx)
    if weighted is None or not _mergeable(value):
        return None
    weights, zero, hole = weighted
    term = cast(Variable | LinearExpression, value)
    live = ~(zero | hole)
    if np.count_nonzero(live.values) > _LIVE_SHARE * live.size:
        return None
    own = _term_dims(term)
    dims = {*map(str, live.dims), *own}
    if dims != set(factors[0][0].dims) or not _on_master(term, weights, ctx):
        return None
    const = term.const if isinstance(term, LinearExpression) else _absent(term)
    if bool(np.isinf(const.values).any()):
        return None
    joined = grouped = None
    if columns is not None:
        joined = _positions(columns, columns.joined, ctx)
        grouped = _positions(columns, columns.grouped, ctx)
        if not _reads_once(joined, grouped, dims, ctx):
            return None
        dims = (dims - set(joined)) | set(grouped)
    plain = {axis.dimension for axis in node.over if axis.column is None}
    if dims - plain != set(node.dims):
        return None
    frame = _frame(own, weights, joined, grouped, plain)
    onto = {d: term.indexes[d] if d in own else ctx.coords[d] for d in frame}
    made = operators.weighted_sum(term, weights, live, onto, joined, grouped)
    if made is None:
        return None
    ctx.memo.fused += 1
    if not bool((const.fillna(0.0) != 0).any()):
        return made
    with np.errstate(over="ignore"):
        bare: Value = operators.constant_term(ctx.model, const)
    for factor, datum in reversed(list(zip(factors, data))):
        bare = _scaled(factor, bare, datum, ctx)
    summed = as_expression(cast(Term, _closed(node, cast(Array, bare), columns, ctx)))
    store = made._store
    flat = summed.const.transpose(*store.dims).values.reshape(-1)
    return LinearExpression._from_store(store.assign_const(flat), made.model)


def _weights(
    factors: Sequence[_Factor], data: Sequence[Value], ctx: Context
) -> tuple[list[operators.Weight], xr.DataArray, xr.DataArray] | None:
    """
    The operator and the data of each of *factors*, the innermost first, and
    where the data is zero and where it holds a hole; ``None`` where no data
    is an array, where it holds an infinity a built row reads, where it
    divides by zero and where a number scales by zero.
    """
    weights: list[operators.Weight] = []
    zero: xr.DataArray | bool = False
    hole: xr.DataArray | bool = False
    for (product, _, _), datum in reversed(list(zip(factors, data))):
        datum = _off_rows(datum, ctx)
        if isinstance(datum, xr.DataArray) and datum.ndim:
            if datum.dtype.kind not in "fiu" or bool(np.isinf(datum.values).any()):
                return None
            nothing = datum == 0
            if isinstance(product, ms.Divide) and bool(nothing.any()):
                return None
            zero, hole = zero | nothing, hole | datum.isnull()
        else:
            datum = float(cast(float, datum))
            if not math.isfinite(datum) or datum == 0:
                return None
        weights.append((_OPERATORS[type(product)], datum))
    if not isinstance(zero, xr.DataArray) or not isinstance(hole, xr.DataArray):
        return None
    return weights, zero, hole


def _positions(
    columns: ms.JoinColumns, roles: Sequence[str], ctx: Context
) -> dict[str, np.ndarray]:
    """The position of each row of the relation *columns* name on the dimension of each of *roles*, -1 where the coordinates lack its label."""
    rows = ctx.relations[columns.name].rows
    return {
        columns.dim(r): ctx.coords[columns.dim(r)].get_indexer(rows[r].to_numpy())
        for r in roles
    }


def _absent(variable: Variable) -> xr.DataArray:
    """The constant of *variable*'s term: zero where it has a label, an absence elsewhere."""
    return xr.where(variable.labels != -1, 0.0, np.nan)


def _term_dims(term: Variable | LinearExpression) -> tuple[str, ...]:
    dims = term.dims if isinstance(term, Variable) else term.coord_dims
    return tuple(str(d) for d in dims)


def _frame(
    own: Sequence[str],
    weights: Sequence[operators.Weight],
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
    for _, data in weights:
        if isinstance(data, xr.DataArray):
            frame += [str(d) for d in data.dims if d not in frame]
    if joined is not None and grouped is not None:
        slot = min(frame.index(d) for d in joined)
        frame[slot : slot + 1] = [*grouped, frame[slot]]
        frame = [d for d in frame if d not in joined]
    return [d for d in frame if d not in plain]


def _on_master(
    term: Variable | LinearExpression, weights: Sequence[operators.Weight], ctx: Context
) -> bool:
    """Whether the term and the data of *weights* lie on the master coordinates of their dimensions."""
    arrays = [data for _, data in weights if isinstance(data, xr.DataArray)]
    held = [(d, term.indexes[d]) for d in _term_dims(term)]
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


def _through(array: Array, columns: ms.JoinColumns, ctx: Context) -> Array:
    """*array* summed through the rows of the relation *columns* name, joined and grouped as the call says."""
    rows = ctx.relations[columns.name].rows
    joined = {columns.dim(r): rows[r].to_numpy() for r in columns.joined}
    grouped = {columns.dim(r): rows[r].to_numpy() for r in columns.grouped}
    onto = {d: ctx.coords[d] for d in grouped}
    return operators.grouped_sum(array, joined, grouped, onto)


def _variable(name: str, ctx: Context) -> Value:
    """
    Variable *name* as its term, or in a fold as its solved values.

    Where it has no label it is absent, or its ``missing:`` number; one
    without a label at all is that constant, a term without a variable.
    """
    variable = ctx.model.variables[ctx.rename(name)]
    missing = ctx.program.variables[name].missing
    if not ctx.solved:
        labels = variable.labels
        fill = np.nan if missing == "absent" else float(missing)
        if labels.size and not bool((labels.values != -1).any()):
            const = xr.full_like(labels, fill, dtype=float)
            return operators.constant_term(ctx.model, const)
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
    if isinstance(node, ms.Add) and not ctx.solved and ms.carries_variable(node):
        return _added(node, ctx)
    first, second = _operands(node)
    if isinstance(node, ms.Add) and isinstance(second, ms.Negate):
        second = second.operand
    return _made(node, evaluate(first, ctx), evaluate(second, ctx), ctx)


def _made(node: _Binary, left: Value, right: Value, ctx: Context) -> Value:
    """The evaluated operands *left* and *right* of *node* combined by its operator, refused as :func:`_combine` says."""
    if ctx.fuse and ctx.memo.fused and _of_terms(node, left, right):
        return _combine(node, unfused(ctx))
    op = _OPERATORS[type(node)]
    if _subtracts(node):
        op = operator.sub
    kind = type(node).__name__.lower()
    if isinstance(node, ms.Multiply) and _of_terms(node, left, right):
        left, right = _as_data_if_constant(left), _as_data_if_constant(right)
    if isinstance(left, Term) or isinstance(right, Term):
        left, right = _off_rows(left, ctx), _off_rows(right, ctx)
        if isinstance(node, ms.Add):
            a, b = constant_of(left), constant_of(right)
            refuse_made(kind, (node,), op, a, b, ctx)
        made = op(*_paired(left, right))
        if not isinstance(node, ms.Multiply):
            return made
        if ctx.zeroes:
            made = _without_zeroed_terms(made, left, right)
        return _without_dead_terms(made)
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


def _of_terms(node: _Binary, left: Value, right: Value) -> bool:
    """Whether *node* multiplies two terms or raises one to a power, which linopy builds on their slots."""
    if isinstance(node, ms.Power):
        return isinstance(left, Term)
    return isinstance(node, ms.Multiply) and all(
        isinstance(v, Term) for v in (left, right)
    )


def unfused(ctx: Context) -> Context:
    """
    *ctx* summing every term operator by operator, the terms its memo keeps dropped.

    A sum built from its live cells holds fewer slots than one built
    operator by operator, so whatever asks whether a value holds a term, or
    how many, is decided on terms built operator by operator. The memo is
    shared, so a named expression built again replaces the one kept, and a
    refusal masked on the way counts where it is kept.
    """
    values = ctx.memo.values
    for key in [
        k
        for k, v in values.items()
        if isinstance(v, LinearExpression | QuadraticExpression)
    ]:
        del values[key]
    return replace(ctx, fuse=False)


def _subtracts(node: _Binary) -> bool:
    """Whether *node* is a difference: a sum of a negated right operand."""
    return isinstance(node, ms.Add) and isinstance(node.right, ms.Negate)


def _added(node: ms.Add, ctx: Context) -> Value:
    """
    A chain of sums and differences that carries a variable, each run of linear terms in it merged in one step.

    The operands are evaluated in the order the chain reads them. Data and a
    quadratic term are added as :func:`_made` adds them, once the run before
    them is merged; so is every operand once the magnitudes of the constants
    added so far sum to an infinity. Below that bound no partial sum of the
    constants overflows, so :func:`refuse_made` refuses no step of a merged
    run, and a step it refuses is refused with the same error as before.
    """
    first, steps = _chain(node, ctx)
    total = _off_rows(evaluate(first, ctx), ctx)
    bound = _reach(total)
    run: list[tuple[Variable | LinearExpression, bool]] = []
    for step, operand in steps:
        value = _off_rows(evaluate(operand, ctx), ctx)
        bound += _reach(value)
        if math.isfinite(bound) and _mergeable(total) and _mergeable(value):
            run.append((cast(Variable | LinearExpression, value), _subtracts(step)))
            continue
        total = _made(step, _merged(total, run), value, ctx)
        run = []
    return _merged(total, run)


def _chain(node: ms.Add, ctx: Context) -> tuple[ms.Expression, list[_Step]]:
    """
    The first operand of the chain of sums and differences that ends in *node*,
    and per step its node and the operand it adds or subtracts.

    The chain runs down the left operands for as long as they are sums that
    carry a variable and are not kept evaluated; a use of each one it passes
    is consumed.
    """
    steps: list[_Step] = []
    left: ms.Expression = node
    while isinstance(left, ms.Add):
        right = left.right
        steps.append((left, right.operand if isinstance(right, ms.Negate) else right))
        left = left.left
        if left in ctx.memo.values or not ms.carries_variable(left):
            break
        if isinstance(left, ms.Add):
            ctx.memo.used(left)
    return left, steps[::-1]


def _reach(value: Value) -> float:
    """The largest magnitude of the constant *value* adds to a sum, a hole aside."""
    if isinstance(value, Variable):
        return 0.0
    held = value.const if isinstance(value, Term) else value
    magnitudes = np.abs(np.asarray(held, dtype=float))
    return float(np.fmax.reduce(magnitudes, axis=None, initial=0.0))


def _mergeable(value: Value) -> bool:
    return isinstance(value, Variable) or type(value) is LinearExpression


def _merged(
    total: Value, run: Sequence[tuple[Variable | LinearExpression, bool]]
) -> Value:
    """*total* plus the terms of *run*, a flagged one subtracted, stacked in one step."""
    if not run:
        return total
    linear = [cast(Variable | LinearExpression, total), *(value for value, _ in run)]
    terms = [cast(LinearExpression, as_expression(term)) for term in linear]
    negate = [False, *(flip for _, flip in run)]
    stores = [term._store for term in terms]
    if all(isinstance(store, DenseStore) for store in stores):
        dense = cast(list[DenseStore], stores)
        summed = DenseStore.signed_sum(dense, negate)
        if summed is not None:
            return LinearExpression._from_store(summed, terms[0].model)
    signed = [-term if flip else term for term, flip in zip(terms, negate)]
    return merge(signed, cls=LinearExpression)


def _as_data_if_constant(value: Value) -> Value:
    """
    A term without a variable term as the data it is, so a product of it
    and another term scales that term rather than multiplying two terms, its
    absence a hole.
    """
    if isinstance(value, LinearExpression) and value.is_constant:
        return value.const
    return value


def _without_zeroed_terms(made: Value, left: Value, right: Value) -> Value:
    """
    A linear *made*, the product of a term and data, without the terms the data zeroes.

    Where the data is zero the product is the term's constant times zero:
    a constant, and a row left with nothing but such terms holds no
    variable term.
    """
    if not isinstance(made, LinearExpression) or type(made) is not LinearExpression:
        return made
    data = right if isinstance(left, Term) else left
    if isinstance(data, Term) or not bool(np.any(np.asarray(data) == 0)):
        return made
    return LinearExpression._from_store(made._store.prune(), made.model)


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


def _in_region(value: Value, rows: xr.DataArray) -> Value:
    """*value* where the region holds and a hard zero everywhere else: a fill, so absence inside the region stands."""
    if isinstance(value, float | int):
        return rows * value
    return as_expression(value).where(rows, 0)


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
