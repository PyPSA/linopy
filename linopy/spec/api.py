"""
Build a model from a spec, and read a spec-built model's data and results.

A spec-built model is a plain linopy model: the build writes the spec's
parameters, unfilled with a missing row a NaN hole, and its relations into
``model.parameters``. What a spec adds beyond the model is read as a function
of the spec and the model: :func:`evaluate` folds named expressions over the
solution and :func:`parameter` fills a parameter the way its ``missing:`` says.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

import xarray as xr
from mathspec import program as ms
from mathspec import to_spec
from mathspec.spec import Spec

from linopy.constants import _warn_evolving_api
from linopy.model import Model
from linopy.spec.attach import Parameter, attach, fill, strict_parameters
from linopy.spec.builder import build
from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError, unknown
from linopy.spec.evaluator import fold

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec

EVOLVING_MESSAGE = (
    "spec: Model.from_spec, linopy.spec.evaluate and linopy.spec.parameter are newly added "
    "and their details may change in minor releases. Silence with "
    '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.'
)


def build_into(
    model: Model, spec: SpecLike, sources: Mapping[str, Any] | xr.Dataset
) -> None:
    """
    Build *spec* with *sources* into the fresh *model*, its data into ``model.parameters``.

    Raises
    ------
    TypeError, ValueError, FileNotFoundError, mathspec.errors.LanguageError
        What :func:`normalize_spec` raises for a spec it cannot lower.
    SpecDataError
        The spec declares nothing, or the data does not fit it: a missing
        or malformed source, a label no dimension holds, an assumption the
        numbers break, or a constraint row the data empties.
    NotImplementedError
        A constraint multiplies two variable-carrying operands.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE, stacklevel=4)
    program = normalize_spec(spec)
    attached = attach(program, sources)
    model.parameters = xr.Dataset(
        {p: attached.read(p).values for p in program.parameters}
        | dict(attached.relations),
        coords=attached.coords,
    )
    build(_context(program, model, solved=False))


def evaluate(spec: SpecLike, model: Model, *names: str) -> dict[str, xr.DataArray]:
    """
    The named expressions *names* of *spec*, all in declared order when none is named, folded over *model*.

    Each folds over ``model.solution``, its duals and ``model.parameters``,
    read afresh on every call: a body over data alone needs no solve.

    Raises
    ------
    KeyError
        *spec* declares no expression of one of *names*, or
        ``model.parameters`` lacks a parameter or relation *spec* declares.
    AttributeError
        The model is not solved: a body reads a variable the model holds no
        solution for or a constraint it holds no dual for.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE)
    program = normalize_spec(spec)
    for name in names:
        if name not in program.expressions:
            raise unknown("expression", name, program.expressions)
    ctx = _context(program, model, solved=True)
    return {name: fold(name, ctx) for name in names or program.expressions}


def parameter(spec: SpecLike, model: Model, name: str) -> xr.DataArray:
    """
    The parameter *name* of *spec* in *model*, filled the way its ``missing:`` says.

    An ``int`` or ``bool`` parameter is at its declared dtype unless an
    ``absent`` row stays a NaN hole.

    Raises
    ------
    KeyError
        *spec* declares no parameter *name*, or ``model.parameters`` lacks it.
    SpecDataError
        A missing row its ``missing:`` refuses.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE)
    program = normalize_spec(spec)
    if name not in program.parameters:
        raise unknown("parameter", name, program.parameters)
    return _filled(program, model, strict_parameters(program), name).values


def _held(model: Model, name: str) -> xr.DataArray:
    if name not in model.parameters:
        raise KeyError(
            f"model.parameters holds no '{name}' that the spec declares: build the model "
            f"with Model.from_spec(spec, ...) or restore its parameters"
        )
    return model.parameters[name]


def _filled(
    program: ms.Program, model: Model, strict: frozenset[str], name: str
) -> Parameter:
    raw = _held(model, name)
    present = raw.notnull()
    return Parameter(fill(program, name, Parameter(raw, present), strict), present)


def _context(program: ms.Program, model: Model, *, solved: bool) -> Context:
    for name in program.parameters:
        _held(model, name)
    read = functools.partial(_filled, program, model, strict_parameters(program))
    return Context(
        model,
        program,
        {str(d): index for d, index in model.parameters.indexes.items()},
        {name: _held(model, name) for name in program.relations},
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
