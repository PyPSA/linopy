"""
``model.spec``: the program a model was built from, and its named expressions as data.

The model owns the data. The spec text, the retained parameters, the lookups
and the master coordinates all sit on the model, so this accessor holds
nothing a round trip through a file could lose: it re-lowers the text and
reads ``model.parameters``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr
import yaml
from math_spec import Spec, to_program, to_spec
from math_spec import program as ms

from linopy.model import Model
from linopy.semantics import is_v1
from linopy.spec.binder import Bound, Retain, bind
from linopy.spec.builder import build, fold
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
        """Each named expression folded over the solution and the retained parameters."""
        return NamedExpressions(self)

    def evaluate(
        self, name: str, sources: Mapping[str, Any] | xr.Dataset
    ) -> xr.DataArray:
        """
        The named expression *name*, with its parameters bound afresh from *sources*.

        For a model built with ``retain="none"``, or an expression reading a
        parameter ``retain="report"`` did not keep. *sources* is read the way
        ``add_spec`` read it, and must describe the coordinates the model was
        built on.
        """
        bound = bind(self.program, sources, retain="none")
        return fold(name, self._context(bound.parameter))

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


class NamedExpressions(Mapping[str, xr.DataArray]):
    """The named expressions of a spec, each folded to data on read."""

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec

    def __getitem__(self, name: str) -> xr.DataArray:
        return fold(name, self._spec._context(self._spec._retained))

    def __iter__(self) -> Iterator[str]:
        return iter(self._spec.program.named_expressions)

    def __len__(self) -> int:
        return len(self._spec.program.named_expressions)

    def __repr__(self) -> str:
        return f"NamedExpressions({list(self)})"
