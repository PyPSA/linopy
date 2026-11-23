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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
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
)
from linopy.spec.builder import build
from linopy.spec.context import Context, Memo, uses_of
from linopy.spec.errors import SpecDataError, shown, unknown
from linopy.spec.evaluator import fold
from linopy.spec.where import admitted

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec
Sources: TypeAlias = Mapping[str, Any] | xr.Dataset
Rename: TypeAlias = Mapping[str, str] | Callable[[str], str]
Uses: TypeAlias = Mapping[ms.Expression, int]

EVOLVING_MESSAGE = (
    "spec: Model.from_spec and linopy.spec.evaluate are newly added "
    "and their details may change in minor releases. Silence with "
    '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.'
)


def build_into(
    model: Model,
    spec: SpecLike,
    sources: Sources,
    store: bool,
    expressions: bool,
    rename: Rename | None = None,
    dim_order: Sequence[str] | None = None,
) -> None:
    """
    Build *spec* with *sources* into the fresh *model*.

    With *store* its data goes into ``model.parameters``, with
    *expressions* its named expressions into ``model.expressions``, and
    each variable and constraint under the name *rename* gives it, its dims
    in the order *dim_order* sets.

    Raises
    ------
    TypeError, ValueError, FileNotFoundError, mathspec.errors.LanguageError
        What :func:`normalize_spec` raises for a spec it cannot lower.
    SpecDataError
        The spec declares nothing, or the data does not fit it: a missing
        or malformed source, a label no dimension holds, an assumption the
        numbers break, or a constraint row the data leaves without a
        variable term where that row does not hold. A parameter
        is read and checked when a built declaration first reads it, so the
        data of one that none reads is not checked, only its key. With
        *store*, every parameter is read, and a relation ``model.parameters``
        cannot hold is refused before anything is stored.
    NotImplementedError
        A constraint multiplies two variable-carrying operands.
    KeyError, ValueError
        What :func:`_renamer` raises for *rename*.
    TypeError, ValueError
        What :func:`_dim_order` raises for *dim_order*.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE, stacklevel=4)
    order = _dim_order(dim_order)
    program, uses = _normalized(spec)
    _build(model, program, uses, sources, store, expressions, rename, order)


def _build(
    model: Model,
    program: ms.Program,
    uses: Uses,
    sources: Sources,
    store: bool,
    expressions: bool,
    rename: Rename | None,
    dim_order: tuple[str, ...],
) -> None:
    """
    The body of :func:`build_into`, in a frame of its own.

    Lowering a spec leaves reference cycles in pyparsing's exceptions that
    keep the frame of the caller of :func:`_normalized` alive until the
    garbage collector runs. That frame holds no data, so the parameter grids
    read here are freed as soon as the build returns.
    """
    named = _renamer(program, rename)
    attached = attach(program, sources)
    relations = _storable(attached) if store else {}
    attached.require(program.parameters)
    raw = functools.cache(attached.read)
    if store:
        unfilled = {name: raw(name).values for name in program.parameters}
        model.parameters = xr.Dataset(unfilled | relations, coords=attached.coords)
    ctx = _context(
        attached, uses, model, raw, solved=False, rename=named, dim_order=dim_order
    )
    build(ctx, expressions)


def _dim_order(dim_order: Sequence[str] | None) -> tuple[str, ...]:
    """
    *dim_order* as the tuple :meth:`Context.frame` reads, empty for ``None``.

    Raises
    ------
    TypeError
        *dim_order* is a ``str``, which would read as one dim per character.
    ValueError
        *dim_order* names a dim twice.
    """
    if dim_order is None:
        return ()
    if isinstance(dim_order, str):
        raise TypeError(
            f"dim_order takes a sequence of dim names, not the str {dim_order!r}; "
            f"pass [{dim_order!r}]"
        )
    order = tuple(dim_order)
    twice = sorted(d for d, count in Counter(order).items() if count > 1)
    if twice:
        raise ValueError(f"dim_order names the dim {twice[0]!r} twice")
    return order


def _renamer(program: ms.Program, rename: Rename | None) -> Callable[[str], str]:
    """
    The model name of each variable and constraint of *program*: what *rename* gives, its own name where it gives none.

    *rename* is a mapping or a function, as :meth:`pandas.DataFrame.rename`
    takes it. The entries a ``piecewise:`` curve expands into are named by
    it too.

    Raises
    ------
    KeyError
        A key of a mapping *rename* names no variable or constraint of *program*.
    ValueError
        Two variables, or two constraints, take one name.
    """
    if rename is None:
        return str
    named = _mapped(program, rename) if isinstance(rename, Mapping) else rename
    for section, entries in (
        ("variables", program.variables),
        ("constraints", program.constraints),
    ):
        taken = Counter(named(name) for name in entries)
        twice = sorted(name for name, count in taken.items() if count > 1)
        if twice:
            raise ValueError(f"rename gives two {section} the name {twice[0]!r}")
    return named


def _mapped(program: ms.Program, rename: Mapping[str, str]) -> Callable[[str], str]:
    declared = [*program.variables, *program.constraints]
    stray = sorted(set(rename) - set(declared))
    if stray:
        raise unknown("variable or constraint", stray[0], declared)
    mapping = dict(rename)

    def named(name: str) -> str:
        return mapping.get(name, name)

    return named


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
        if not _is_storable(rel):
            raise SpecDataError(
                f"relation '{name}' has the key column(s) {list(rel.key)} and the value "
                f"column(s) {list(rel.values)}, and model.parameters stores a relation as "
                f"one array over its one key column, holding its one value column. Build "
                f"with store=False and keep the sources to rebuild from."
            )
        stored[name] = table.column(rel.values[0]).rename(name)
    return stored


def _is_storable(relation: ms.Relation) -> bool:
    return len(relation.key) == 1 and len(relation.values) == 1


def evaluate(
    spec: SpecLike,
    model: Model,
    *names: str,
    sources: Sources | None = None,
    rename: Rename | None = None,
    dim_order: Sequence[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    The named expressions *names* of *spec*, all in name order when none is named, folded over *model*.

    Each folds over ``model.solution`` and its duals, with the data read from
    *sources*, or from ``model.parameters`` when none is given, as
    ``Model.from_spec(..., store=True)`` writes it: a body over data alone
    needs no solve.
    A model built with ``Model.from_spec(..., rename=...)`` is read with the
    same *rename*, and each expression is laid out in the order *dim_order*
    sets, as ``Model.from_spec(..., dim_order=...)`` lays out what it builds.

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
    TypeError
        *dim_order* is a ``str``.
    ValueError
        *dim_order* names a dim twice.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE)
    order = _dim_order(dim_order)
    program, uses = _normalized(spec)
    return _folded(program, uses, model, names, sources, rename, order)


def _folded(
    program: ms.Program,
    uses: Uses,
    model: Model,
    names: tuple[str, ...],
    sources: Sources | None,
    rename: Rename | None,
    dim_order: tuple[str, ...],
) -> dict[str, xr.DataArray]:
    """The body of :func:`evaluate`, in a frame of its own for the reason :func:`_build` gives."""
    for name in names:
        if name not in program.expressions:
            raise unknown("expression", name, program.expressions)
    named = _renamer(program, rename)
    attached = attach(program, _stored(program, model) if sources is None else sources)
    ctx = _context(
        attached,
        uses,
        model,
        attached.read,
        solved=True,
        rename=named,
        dim_order=dim_order,
    )
    return {name: fold(name, ctx) for name in names or program.expressions}


def _stored(program: ms.Program, model: Model) -> xr.Dataset:
    lacking = [
        name
        for name in (*program.parameters, *program.relations)
        if name not in model.parameters
    ] + sorted(reached_dimensions(program) - set(model.parameters.indexes))
    unstorable = [
        n
        for n in lacking
        if n in program.relations and not _is_storable(program.relations[n])
    ]
    if unstorable:
        raise KeyError(
            f"model.parameters holds no {shown(unstorable)}, and store=True stores no "
            f"bare, multi-key or multi-value relation: pass the data as "
            f"evaluate(spec, model, sources=...)"
        )
    if lacking:
        raise KeyError(
            f"model.parameters holds no {shown(lacking)} that the spec declares: pass the "
            f"data as evaluate(spec, model, sources=...), or build the model with "
            f"Model.from_spec(spec, sources, store=True)"
        )
    return model.parameters


def _context(
    attached: Attached,
    uses: Uses,
    model: Model,
    raw: Callable[[str], Parameter],
    *,
    solved: bool,
    rename: Callable[[str], str],
    dim_order: tuple[str, ...],
) -> Context:
    program = attached.program
    tables = (model, program, attached.coords, attached.relations)

    @functools.cache
    def unmasked(name: str) -> Parameter:
        parameter = raw(name)
        return Parameter(fill(program, name, parameter), parameter.present)

    conditions = Context(
        *tables, unmasked, solved=solved, rename=rename, dim_order=dim_order
    )

    def read(name: str) -> Parameter:
        if program.parameters[name].where is None:
            return unmasked(name)
        parameter = raw(name)
        inside = admitted(name, conditions)
        filled = fill(program, name, parameter, inside)
        return Parameter(filled, (parameter.present & inside).transpose(*filled.dims))

    return Context(
        *tables,
        functools.cache(read),
        solved=solved,
        memo=Memo(uses=dict(uses)),
        rename=rename,
        dim_order=dim_order,
    )


def lower(spec: SpecLike) -> ms.Program:
    """
    *spec* lowered to the program linopy builds, its piecewise curves written out.

    A ``str`` is YAML text, never a file name, so a one-line one gets the
    newline that keeps ``to_spec`` from reading it as a path. Only
    ``piecewise:`` is expanded: a special-ordered set stays a set, which
    linopy builds natively rather than as binaries.
    """
    return _lower(spec)[0]


def _lower(spec: SpecLike) -> tuple[ms.Program, Uses]:
    """What :func:`lower` gives, with how often a build uses each node of it, as :func:`uses_of` counts."""
    if isinstance(spec, str):
        spec = f"{spec}\n"
    elif isinstance(spec, Mapping):
        spec = dict(spec)
    return _lowered(to_spec(spec))


@functools.lru_cache(maxsize=8)
def _lowered(spec: Spec) -> tuple[ms.Program, Uses]:
    """
    The program of *spec* and the uses of its nodes, kept for the next build of an equal spec: a spec and its program are immutable.

    The uses are read-only: a build consumes a copy of them.
    """
    program = spec.expand("piecewise").program
    return program, MappingProxyType(uses_of(program))


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
    return _normalized(spec)[0]


def _normalized(spec: SpecLike) -> tuple[ms.Program, Uses]:
    """What :func:`normalize_spec` gives, with the uses :func:`_lowered` keeps beside it."""
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
    program, uses = _lower(spec)
    if not (program.dimensions or program.parameters or program.variables):
        raise SpecDataError(
            "the spec declares nothing: no dimension, no parameter and no variable. "
            "There is nothing to attach data to and nothing to build."
        )
    return program, uses
