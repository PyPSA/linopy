"""
What evaluating a node reads beyond the node, and what it evaluates to.

A :class:`Context` holds the model, the program, the coordinates, the
relations and ``parameter``, which gives a parameter by name as an
:class:`~linopy.spec.attach.Parameter`: its values filled the way its
``missing:`` says beside where the data has a row. What a node evaluates
to is a :data:`Value`: a number, an array or a linopy term.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast, overload

import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec.attach import Parameter, Table
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array
Expression = xr.DataArray | LinearExpression | QuadraticExpression
_Kept = TypeVar("_Kept")


@overload
def as_expression(value: Term) -> LinearExpression | QuadraticExpression: ...
@overload
def as_expression(value: Array) -> Expression: ...
def as_expression(value: Array) -> Expression:
    """*value* with a bare variable as its linear expression."""
    return value.to_linexpr() if isinstance(value, Variable) else value


@dataclass
class Memo:
    """
    What a build or a fold has evaluated: a condition by its node, an
    expression node by its node and ``filled``. A reference and a named
    expression's body are kept for the build; another node is kept while
    ``uses`` counts a use of it ahead, and only when it is data: a term is
    made again where it is used, so none is held for a later declaration.

    ``masked`` counts the refusals a ``rows`` narrowing let pass: a value
    made under one is not kept, so a value read back is refused wherever it
    would be when evaluated afresh.
    """

    values: dict[Hashable, Any] = field(default_factory=dict)
    masked: int = 0
    uses: dict[ms.Expression, int] = field(default_factory=dict)
    inner: dict[ms.Expression, tuple[ms.Expression, ...]] = field(default_factory=dict)

    def kept(self, key: Hashable, make: Callable[[], _Kept]) -> _Kept:
        """The value kept at *key*, else made by *make* and kept unless a refusal was masked making it."""
        if key in self.values:
            return cast(_Kept, self.values[key])
        masked = self.masked
        value = make()
        if self.masked == masked:
            self.values[key] = value
        return value

    def evaluated(
        self,
        node: ms.Expression,
        filled: bool,
        make: Callable[[], _Kept],
        pinned: bool = False,
    ) -> _Kept:
        """The value of *node* under *filled*, one use of it consumed: read back if kept, else made by *make* and kept for the build if *pinned*, else while a use is ahead and it is data."""
        key = (node, filled)
        if key in self.values:
            value = cast(_Kept, self.values[key])
            if node not in self.inner:
                self.inner[node] = tuple(ms.walk(node))
            for inner in self.inner[node]:
                self._used(inner)
            return value
        if pinned:
            value = self.kept(key, make)
            self.uses.pop(node, None)
            return value
        if self.uses.get(node, 0) < 2:
            self._used(node)
            return make()
        masked = self.masked
        value = make()
        if self.masked == masked and not isinstance(value, Term):
            self.values[key] = value
        self._used(node)
        return value

    def skipped(self, *roots: ms.Expression) -> None:
        """Consume the uses of every node under *roots*, a declaration left unbuilt."""
        for node in ms.walk(*roots):
            self._used(node)

    def _used(self, node: ms.Expression) -> None:
        left = self.uses.get(node)
        if left is None:
            return
        if left > 1:
            self.uses[node] = left - 1
            return
        del self.uses[node]
        for filled in (True, False):
            self.values.pop((node, filled), None)


def uses_of(program: ms.Program) -> dict[ms.Expression, int]:
    """How often each node stands in what a build evaluates: the bounds, the sides of the constraints, the objective and the named expressions."""
    bounds = (b for v in program.variables.values() for b in (v.lower, v.upper))
    sides = (s for c in program.constraints.values() for s in (c.lhs, c.rhs))
    bodies = (e.expression for e in program.expressions.values())
    objective = () if program.objective is None else (program.objective.expression,)
    roots = [*(b for b in bounds if b is not None), *sides, *bodies, *objective]
    return Counter(ms.walk(*roots))


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

    ``parameter`` reads a parameter by name, filled the way its
    ``missing:`` says, each resolved once and cached.
    ``solved`` is the fold's switch: a build leaves it false
    and a variable enters an expression as its linopy term; a fold sets it
    true and a variable enters as its solved values, so a named expression
    reads off the primal. ``filled`` is the comparison's switch: on, a
    parameter enters as its filled values; a condition turns it off so a
    missing row reaches the comparison as a hole, read as false there. ``rows``
    narrows the finiteness check to the rows a constraint builds, and
    ``subject`` names what is being built in its refusals. ``memo`` is
    shared by every context replaced from one.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, Table]
    parameter: Callable[[str], Parameter]
    solved: bool = False
    filled: bool = True
    rows: xr.DataArray | None = None
    subject: str = "the spec"
    memo: Memo = field(default_factory=Memo, compare=False, repr=False)
