"""
``model.spec``: the program a model was built from, and its named expressions as data.

The model owns the data. The spec text, the retained parameters, the lookups
and the master coordinates all sit on the model, so this accessor holds
nothing a round trip through a file could lose: it re-lowers the text and
reads ``model.parameters``.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr
import yaml
from math_spec import (
    Spec,
    did_you_mean,
    to_latex,
    to_markdown,
    to_program,
    to_spec,
    to_typst,
)
from math_spec import program as ms

from linopy.model import Model
from linopy.semantics import is_v1
from linopy.spec import terms
from linopy.spec.binder import Bound, Retain, bind
from linopy.spec.builder import build, evaluate_named, fold
from linopy.spec.context import Context, Parameters, Resolve
from linopy.spec.errors import SpecDataError

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec


def attach(
    model: Model,
    spec: SpecLike,
    sources: Mapping[str, Any] | xr.Dataset,
    retain: Retain,
) -> ModelSpec:
    """
    Build *spec* with *sources* into the empty *model* and return its accessor.

    Raises:
        ValueError: The model already holds variables or constraints, or runs
            under legacy semantics.
        TypeError: *spec* is a lowered ``Program``, which has no YAML form to
            keep on the model.
    """
    if not is_v1():
        raise ValueError(
            "a spec-built model uses linopy's v1 semantics, and the current setting is "
            "'legacy'. Set linopy.options['semantics'] = 'v1' before building from a spec."
        )
    if len(model.variables) or len(model.constraints):
        raise ValueError(
            "add_spec builds into an empty model, and this one already holds "
            f"{len(model.variables)} variable(s) and {len(model.constraints)} constraint(s)."
        )
    text, program = _source(spec)
    bound: Bound = bind(program, sources, retain=retain)
    build(model, bound)
    model.parameters = bound.retained().assign_coords(dict(bound.coords))
    return ModelSpec(model, program, text)


def restore(model: Model, text: str) -> ModelSpec:
    """The accessor for *model*, with the program lowered afresh from *text*."""
    return ModelSpec(model, to_program(yaml.safe_load(text)), text)


def _source(spec: SpecLike) -> tuple[str, ms.Program]:
    """The spec as the YAML text kept on the model, and lowered."""
    if isinstance(spec, ms.Program):
        raise TypeError(
            "add_spec takes the spec as a path, YAML text, a mapping or a math_spec.Spec, "
            "not a lowered Program: a Program has no YAML form to keep on the model."
        )
    if isinstance(spec, str) and "\n" not in spec:
        spec = Path(spec)
    if isinstance(spec, Path):
        return spec.read_text(), to_program(spec)
    if isinstance(spec, str):
        return spec, to_program(yaml.safe_load(spec))
    loaded = to_spec(dict(spec)) if isinstance(spec, Mapping) else spec
    return loaded.to_yaml(), to_program(loaded)


class ModelSpec:
    """
    The spec a model was built from.

    Attributes:
        program: The lowered spec.
        text: The spec as YAML, verbatim where a file or text was passed.
    """

    def __init__(self, model: Model, program: ms.Program, text: str) -> None:
        self._model = model
        self.program = program
        self.text = text

    def __repr__(self) -> str:
        names = list(self.program.named_expressions)
        return f"ModelSpec(expressions={names})"

    def _rebound(self, model: Model) -> ModelSpec:
        """The same spec, read off *model*."""
        return ModelSpec(model, self.program, self.text)

    @property
    def parameters(self) -> xr.Dataset:
        """The parameters and lookups retained on the model, on the master coordinates."""
        return self._model.parameters

    @property
    def coords(self) -> dict[str, pd.Index]:
        """Master coordinates by dimension, as the model was built on them."""
        return {str(d): index for d, index in self.parameters.indexes.items()}

    @property
    def lookups(self) -> dict[str, dict[str, xr.DataArray]]:
        """By dimension, by name, each lookup as an array over its dimension."""
        out: dict[str, dict[str, xr.DataArray]] = {}
        for over, lk in self.program.lookups:
            out.setdefault(over, {})[lk.name] = self.parameters[lk.name]
        return out

    @property
    def expressions(self) -> NamedExpressions:
        """Each named expression as a :class:`NamedExpression`: its math, its linopy fold and its solution."""
        return NamedExpressions(self)

    def to_latex(self, **options: Any) -> str:
        """The whole model typeset as a LaTeX document."""
        return to_latex(self._schema, **options)

    def to_markdown(self, **options: Any) -> str:
        """The whole model typeset as Markdown, its equations in ``$$`` blocks."""
        return to_markdown(self._schema, **options)

    def to_typst(self, **options: Any) -> str:
        """The whole model typeset as Typst."""
        return to_typst(self._schema, **options)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    @property
    def _schema(self) -> dict[str, Any]:
        """The spec as the mapping the typesetter reads (a bare string it reads as a path)."""
        return yaml.safe_load(self.text)

    def evaluate(
        self, name: str, sources: Mapping[str, Any] | xr.Dataset
    ) -> NamedExpression:
        """
        The named expression *name*, with its parameters bound afresh from *sources*.

        For a model built with ``retain="none"``, or an expression reading a
        parameter ``retain="report"`` did not keep. *sources* is read the way
        ``add_spec`` read it, and must describe the coordinates the model was
        built on.

        Raises:
            SpecDataError: *sources* label a dimension differently than the
                model was built on.
        """
        bound = bind(self.program, sources, retain="none")
        coords = self.coords
        for dim, index in bound.coords.items():
            if dim in coords and not index.equals(coords[dim]):
                raise SpecDataError(
                    f"sources describe dimension '{dim}' as {index.tolist()[:5]}, and the model "
                    f"was built on {coords[dim].tolist()[:5]}. evaluate() reads the solution the "
                    f"model holds, so the data must be bound on the same labels in the same order."
                )
        return NamedExpression(self, name, self._context(bound.parameter))

    def _retained(self, name: str) -> xr.DataArray:
        if name not in self.parameters:
            raise SpecDataError(
                f"parameter '{name}' is not retained on the model: retain='report' keeps only what "
                f"the named expressions read, and retain='none' keeps nothing. Build with "
                f"retain='all', or read the expression with evaluate(name, sources)."
            )
        return self.parameters[name]

    def _context(self, resolve: Resolve) -> Context:
        return Context(
            self._model,
            self.program,
            self.coords,
            self.lookups,
            Parameters(self.program, resolve),
            solved=True,
        )


class NamedExpressions(Mapping[str, "NamedExpression"]):
    """The named expressions of a spec, each a :class:`NamedExpression` on read."""

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec

    def __getitem__(self, name: str) -> NamedExpression:
        if name not in self._spec.program.named_expressions:
            raise KeyError(
                f"unknown named expression '{name}'. "
                + did_you_mean(name, self._spec.program.named_expressions)
            )
        return NamedExpression(
            self._spec, name, self._spec._context(self._spec._retained)
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._spec.program.named_expressions)

    def __len__(self) -> int:
        return len(self._spec.program.named_expressions)

    def __repr__(self) -> str:
        return f"NamedExpressions({list(self)})"


class NamedExpression:
    """
    One named expression, in three views: its math, its linopy fold and its solution.

    The object pins the data sources it was made with for its lifetime, so the
    three views agree. ``expressions[name]`` reads the retained parameters and
    the solution the model holds; ``evaluate(name, sources)`` binds fresh data.

    Attributes:
        node: The lowered expression body, math-spec's own AST handle.
    """

    def __init__(self, spec: ModelSpec, name: str, ctx: Context) -> None:
        self._spec = spec
        self._name = name
        self._ctx = ctx

    @property
    def node(self) -> ms.ExpressionNode:
        """The expression body as lowered, math-spec's own AST handle."""
        return self._spec.program.named_expressions[self._name]

    @functools.cached_property
    def expression(self) -> terms.Value:
        """
        The linopy symbolic expression, its variables unsolved.

        A named expression is read affinely, so this is a ``LinearExpression``
        where the body carries variables, a bare ``Variable``, a ``DataArray``
        for a data-only body or a ``float`` for a constant. Not wrapped: a
        degree-0 array can hold holes that ``from_constant`` would refuse.
        """
        return evaluate_named(self._name, self._ctx.unsolved)

    @functools.cached_property
    def solution(self) -> xr.DataArray:
        """
        The expression folded over the model's solution, as data.

        Raises:
            RuntimeError: The model reads a variable but holds no solution yet.
            SpecDataError: A parameter the body reads was not retained.
        """
        return fold(self._name, self._ctx)

    def __repr__(self) -> str:
        value = self.__dict__.get("solution", self.__dict__.get("expression"))
        if isinstance(value, xr.DataArray):
            return f"NamedExpression('{self._name}', dims={tuple(value.dims)})"
        return f"NamedExpression('{self._name}')"

    def _repr_markdown_(self) -> str:
        return self._spec.to_markdown()
