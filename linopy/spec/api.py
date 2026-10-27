"""
Build a model from a spec, and read a spec-built model's results.

A spec-built model is a plain linopy model. The build reads its data from
the sources it is given and, asked to ``store`` it, writes the spec's
parameters, unfilled with a missing row a NaN hole, and its relations into
``model.parameters``, the data a model is rebuilt from. What a spec adds
beyond the model is read as a function of the spec, the model and the data:
:func:`evaluate` folds named expressions over the solution.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import xarray as xr
from mathspec import program as ms
from mathspec import to_spec
from mathspec.spec import Spec

from linopy.constants import _warn_evolving_api
from linopy.model import Model
from linopy.spec.attach import (
    Attached,
    Parameter,
    attach,
    fill,
    reached_dimensions,
    strict_parameters,
)
from linopy.spec.builder import build
from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError, shown, unknown
from linopy.spec.evaluator import fold

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec
Sources: TypeAlias = Mapping[str, Any] | xr.Dataset

EVOLVING_MESSAGE = (
    "spec: Model.from_spec and linopy.spec.evaluate are newly added "
    "and their details may change in minor releases. Silence with "
    '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.'
)


def build_into(model: Model, spec: SpecLike, sources: Sources, store: bool) -> None:
    """
    Build *spec* with *sources* into the fresh *model*, with *store* its data into ``model.parameters``.

    Raises
    ------
    TypeError, ValueError, FileNotFoundError, mathspec.errors.LanguageError
        What :func:`normalize_spec` raises for a spec it cannot lower.
    SpecDataError
        The spec declares nothing, or the data does not fit it: a missing
        or malformed source, a label no dimension holds, an assumption the
        numbers break, or a constraint row the data empties. With *store*,
        a relation ``model.parameters`` cannot hold, refused before anything
        is stored.
    NotImplementedError
        A constraint multiplies two variable-carrying operands.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE, stacklevel=4)
    program = normalize_spec(spec)
    attached = attach(program, sources)
    relations = _storable(attached) if store else {}
    raw = functools.cache(attached.read)
    unfilled = {name: raw(name).values for name in program.parameters}
    if store:
        model.parameters = xr.Dataset(unfilled | relations, coords=attached.coords)
    build(_context(attached, model, raw, solved=False))


def _storable(attached: Attached) -> dict[str, xr.DataArray]:
    """
    Each relation as ``model.parameters`` stores it: its value column over its key.

    Raises
    ------
    SpecDataError
        A relation of more than one key or value column, or a bare one,
        which no array over a key holds.
    """
    stored = {}
    for name, table in attached.relations.items():
        rel = table.declared
        if len(rel.key) != 1 or len(rel.values) != 1:
            raise SpecDataError(
                f"relation '{name}' has the key column(s) {list(rel.key)} and the value "
                f"column(s) {list(rel.values)}, and model.parameters stores a relation as "
                f"one array over its one key column, holding its one value column. Build "
                f"with store=False and keep the sources to rebuild from."
            )
        stored[name] = table.column(rel.values[0]).rename(name)
    return stored


def evaluate(
    spec: SpecLike, model: Model, *names: str, sources: Sources | None = None
) -> dict[str, xr.DataArray]:
    """
    The named expressions *names* of *spec*, all in declared order when none is named, folded over *model*.

    Each folds over ``model.solution`` and its duals, with the data read from
    *sources*, or from ``model.parameters`` when none is given, as
    ``Model.from_spec(..., store=True)`` writes it: a body over data alone
    needs no solve.

    Raises
    ------
    KeyError
        *spec* declares no expression of one of *names*, or no *sources* are
        given and ``model.parameters`` lacks a parameter, relation or dimension
        index *spec* needs.
    AttributeError
        The model is not solved: a body reads a variable the model holds no
        solution for or a constraint it holds no dual for.
    SpecDataError
        The data does not fit the spec, or a body reads a missing row of a
        ``missing: refused`` parameter.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE)
    program = normalize_spec(spec)
    for name in names:
        if name not in program.expressions:
            raise unknown("expression", name, program.expressions)
    attached = attach(program, _stored(program, model) if sources is None else sources)
    ctx = _context(attached, model, attached.read, solved=True)
    return {name: fold(name, ctx) for name in names or program.expressions}


def _stored(program: ms.Program, model: Model) -> xr.Dataset:
    lacking = [
        name
        for name in (*program.parameters, *program.relations)
        if name not in model.parameters
    ] + sorted(reached_dimensions(program) - set(model.parameters.indexes))
    if lacking:
        raise KeyError(
            f"model.parameters holds no {shown(lacking)} that the spec declares: pass the "
            f"data as evaluate(spec, model, sources=...), or build the model with "
            f"Model.from_spec(spec, sources, store=True)"
        )
    return model.parameters


def _context(
    attached: Attached,
    model: Model,
    raw: Callable[[str], Parameter],
    *,
    solved: bool,
) -> Context:
    program = attached.program
    strict = strict_parameters(program)

    def read(name: str) -> Parameter:
        parameter = raw(name)
        return Parameter(fill(program, name, parameter, strict), parameter.present)

    return Context(
        model,
        program,
        attached.coords,
        attached.relations,
        functools.cache(read),
        solved=solved,
    )


def lower(spec: SpecLike) -> ms.Program:
    """
    *spec* lowered to the program linopy builds, its piecewise curves written out.

    A ``str`` is YAML text, never a file name, so a one-line one gets the
    newline that keeps ``to_spec`` from reading it as a path. Only
    ``piecewise:`` is expanded: a special-ordered set stays a set, which
    linopy builds natively rather than as binaries.
    """
    if isinstance(spec, str):
        spec = f"{spec}\n"
    elif isinstance(spec, Mapping):
        spec = dict(spec)
    return to_spec(spec).expand("piecewise").program


def normalize_spec(spec: SpecLike) -> ms.Program:
    """
    *spec* lowered: a ``Path`` is read, a ``str`` is YAML text, a mapping is a spec as a dict, a ``Spec`` is used as it is.

    Raises
    ------
    TypeError
        *spec* is none of these, such as a lowered ``Program`` or an open file.
    ValueError
        *spec* is a one-line ``str`` ending in ``.yaml`` or ``.yml``, a file
        name rather than YAML text.
    FileNotFoundError
        *spec* is a ``Path`` and there is no file there.
    mathspec.errors.LanguageError
        The spec is not valid under the installed mathspec.
    SpecDataError
        The spec declares no dimension, parameter or variable.
    """
    if isinstance(spec, str):
        one_line = "\n" not in spec and ": " not in spec
        if one_line and spec.strip().lower().endswith((".yaml", ".yml")):
            raise ValueError(
                f"from_spec reads a str as YAML text, and {spec!r} reads like a file name. "
                f"Pass it as pathlib.Path({spec!r})."
            )
    elif not isinstance(spec, Path | Mapping | Spec):
        raise TypeError(
            "from_spec takes the spec as a path, YAML text, a mapping or a mathspec.Spec, "
            f"not a {type(spec).__name__}."
        )
    program = lower(spec)
    if not (program.dimensions or program.parameters or program.variables):
        raise SpecDataError(
            "the spec declares nothing: no dimension, no parameter and no variable. "
            "There is nothing to attach data to and nothing to build."
        )
    return program
