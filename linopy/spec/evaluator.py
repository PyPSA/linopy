"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Sequence
from typing import TypeVar, assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec import operators
from linopy.spec.context import Array, Context, Term, Value, as_expression
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.where import evaluate_where, groups_of, looked_up
from linopy.variables import Variable

_Binary = ms.Add | ms.Multiply | ms.Divide | ms.Power
_OPERATORS: dict[type, Callable[[Value, Value], Value]] = {
    ms.Add: operator.add,
    ms.Multiply: operator.mul,
    ms.Divide: operator.truediv,
    ms.Power: operator.pow,
}
_Framed = TypeVar(
    "_Framed", bound=xr.DataArray | LinearExpression | QuadraticExpression
)


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
    """One node as a linopy term, an array or a number, a reference evaluated once per context."""
    if isinstance(node, ms.ExpressionReference):
        node = node.body
    elif not isinstance(node, ms.VariableReference | ms.ParameterReference):
        return _evaluated(node, ctx)
    return ctx.memo.kept((node, ctx.filled), lambda: _evaluated(node, ctx))


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
        return _parameter(node.name, ctx)
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
        regions = (
            _in_region(evaluate(region.value, ctx), evaluate_where(region.when, ctx))
            for region in node.regions
        )
        return functools.reduce(lambda a, b: operator.add(*_paired(a, b)), regions)
    assert_never(node)


def _sum(
    node: ms.Sum, ctx: Context, of: Callable[[ms.Expression], Array] | None = None
) -> Array:
    """
    A sum over its axes: the join axes a relation opens are closed by grouping
    the operand through the relation, the plain ones by summing them away.
    *of* evaluates an operand, :func:`evaluate` when not given.
    """
    value = functools.partial(_operand, ctx=ctx) if of is None else of
    operand = node.operand
    plain = [axis.dimension for axis in node.over if axis.column is None]
    if isinstance(operand, ms.Join) and operand.columns.axes:
        summed = _through(value(operand.operand), operand.columns, ctx)
    else:
        summed = value(operand)
    for dimension in plain:
        summed = operators.sum_over(summed, dimension)
    return summed


def _through(array: Array, columns: ms.JoinColumns, ctx: Context) -> Array:
    """*array* summed through the rows of the relation *columns* name, joined and grouped as the call says."""
    rows = ctx.relations[columns.name].rows
    joined = {columns.dim(r): rows[r].to_numpy() for r in columns.joined}
    grouped = {columns.dim(r): rows[r].to_numpy() for r in columns.grouped}
    onto = {d: ctx.coords[d] for d in grouped}
    return operators.grouped_sum(array, joined, grouped, onto)


def _parameter(name: str, ctx: Context) -> xr.DataArray:
    """The parameter filled as its ``missing:`` says; unfilled, a missing row is a hole that a comparison reads as false."""
    filled, present = ctx.parameter(name)
    if ctx.filled or not isinstance(ctx.program.parameters[name].missing, str):
        return filled
    return filled.where(present)


def _variable(name: str, ctx: Context) -> Value:
    variable = ctx.model.variables[name]
    neutral = ctx.program.variables[name].missing == "neutral"
    if not ctx.solved:
        return variable.fillna(0) if neutral else variable
    if "solution" not in variable.data:
        raise AttributeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    return variable.solution.fillna(0) if neutral else variable.solution


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
    built = ctx.model.constraints[constraint]
    if built._dual is None:
        raise AttributeError(
            f"constraint '{constraint}' has no dual yet: solve the model, with a solver "
            f"and a problem that report duals, before reading one"
        )
    dual = built.dual
    return -dual if flipped(ctx.program.constraints[constraint]) else dual


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
    op = _OPERATORS[type(node)]
    left, right = (evaluate(operand, ctx) for operand in _operands(node))
    kind = type(node).__name__.lower()
    if isinstance(left, Term) or isinstance(right, Term):
        if isinstance(node, ms.Add):
            a, b = constant_of(left), constant_of(right)
            refuse_made(kind, ms.parameters_of(node), operator.add, a, b, ctx)
        return op(*_paired(left, right))
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
    return refuse_made(kind, ms.parameters_of(node), op, a, b, ctx)


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
    params: frozenset[str],
    op: Callable[[Value, Value], Value],
    left: xr.DataArray,
    right: xr.DataArray,
    ctx: Context,
) -> xr.DataArray:
    """*op* of *left* and *right*, refused wherever it is not finite though both are present."""
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
    named = ", ".join(f"'{p}'" for p in sorted(params))
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
    """
    live = xr.DataArray(True) if rows is None else rows
    for name in sorted(_read_parameters(nodes)):
        present = ctx.parameter(name).present
        if ctx.program.parameters[name].missing != "refused" or bool(present.all()):
            continue
        found = _read_holes(~present, functools.partial(_reads, nodes, name, live, ctx))
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
        labels = ctx.model.variables[node.name].labels
        zeros = xr.zeros_like(labels, dtype=float)
        if ctx.program.variables[node.name].missing == "neutral":
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
    """*holes* on parameter *name*, NaN on a missing ``absent`` row of another."""
    if node.name == name:
        return holes.astype(float)
    if ctx.program.parameters[node.name].missing != "absent":
        return _zeros(node, ctx)
    present = ctx.parameter(node.name).present
    return xr.zeros_like(present, dtype=float).where(present)


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
