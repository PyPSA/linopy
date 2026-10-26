"""
``model.spec``: the program a model was built from, and its named expressions as data.

The spec owns its data. The spec text, the retained parameters, the relations
and the master coordinates sit on the accessor rather than in
``model.parameters``, which stays the caller's: a spec never overwrites what
was put there, and nothing reading a spec-built model has to guess which of
its parameters the spec owns. All of it round trips through a file, written
under the ``spec-`` prefix.

Only the parameters a named expression left unbuilt reads are kept: one over
data alone, or one reading a dual. A built one sits in ``model.expressions``
and reads its solution off it, so the sources are dropped once the build is
done.
"""

from __future__ import annotations

import functools
import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr
from mathspec import FormatName, to_spec, typeset, typeset_line
from mathspec import program as ms
from mathspec.spec import Spec

from linopy.constants import SOS_TYPE_ATTR, _warn_evolving_api
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec.attach import (
    EVOLVING_MESSAGE,
    Parameter,
    _everywhere,
    parameters_read,
)
from linopy.spec.attach import attach as attach_data
from linopy.spec.builder import build, unbuilt_expressions
from linopy.spec.context import Context, Value
from linopy.spec.errors import SpecDataError, unknown
from linopy.spec.evaluate import evaluate_named, fold

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec


def build_into(
    model: Model, spec: SpecLike, sources: Mapping[str, Any] | xr.Dataset
) -> ModelSpec:
    """
    Build *spec* with *sources* into the empty *model* and return its accessor.

    The spec is named after its file's stem, else ``"spec"``.

    Raises
    ------
    ValueError
        The model already holds variables or constraints.
    TypeError
        *spec* is none of the forms :data:`SpecLike` names.
    FileNotFoundError
        *spec* is a ``Path`` and there is no file there.
    """
    _warn_evolving_api("spec", EVOLVING_MESSAGE, stacklevel=4)
    if len(model.variables) or len(model.constraints):
        raise ValueError(
            "add_spec builds into an empty model, and this one already holds "
            f"{len(model.variables)} variable(s) and {len(model.constraints)} constraint(s)."
        )
    text, program, name = normalize_spec(spec)
    attached = attach_data(program, sources)
    read = functools.cache(attached.read)
    kept = {p: read(p) for p in parameters_read(program, unbuilt_expressions(program))}
    build(Context(model, program, attached.coords, attached.relations, read, name))
    parameters = xr.Dataset(
        {p: k.filled for p, k in kept.items()} | dict(attached.relations),
        coords=attached.coords,
    )
    presence = {p: k.present for p, k in kept.items() if not bool(k.present.all())}
    return ModelSpec(model, name, program, text, parameters, xr.Dataset(presence))


def lower(spec: SpecLike) -> ms.Program:
    """
    *spec* lowered to the program linopy builds, its piecewise curves written out.

    Only ``piecewise:`` is expanded: a special-ordered set stays a set, which
    linopy builds natively rather than as binaries.
    """
    return to_spec(spec).expand("piecewise").program


def lower_text(text: str) -> ms.Program:
    """YAML *text* lowered, as :func:`lower` would lower it."""
    return lower(_read(text))


def _read(text: str) -> Spec:
    """YAML *text* as a Spec, a one-line text included, which ``to_spec`` would read as a path."""
    return to_spec(f"{text}\n")


def normalize_spec(spec: SpecLike) -> tuple[str, ms.Program, str]:
    """
    *spec* as the YAML text kept on the model, lowered, and its name: the file's stem where it came from one, else ``spec``.

    A ``Path``, or a ``str`` naming an existing file, is read; any other
    ``str`` is YAML text; a mapping is a spec as a dict; a ``Spec`` is used
    as it is.

    Raises
    ------
    TypeError
        *spec* is none of these, such as a lowered ``Program`` or an open
        file, neither of which has a YAML form to keep on the model.
    FileNotFoundError
        *spec* is a ``Path`` and there is no file there.
    mathspec.errors.LanguageError
        The spec is not valid under the installed mathspec.
    SpecDataError
        The spec declares no dimension, parameter or variable.
    """
    if isinstance(spec, str) and os.path.isfile(spec):
        spec = Path(spec)
    if isinstance(spec, Path):
        text = spec.read_text()
    elif isinstance(spec, str):
        text = spec
    elif isinstance(spec, Mapping | Spec):
        text = to_spec(dict(spec) if isinstance(spec, Mapping) else spec).to_yaml()
    else:
        raise TypeError(
            "add_spec takes the spec as a path, YAML text, a mapping or a mathspec.Spec, "
            f"not a {type(spec).__name__}."
        )
    program = lower_text(text)
    if not (program.dimensions or program.parameters or program.variables):
        raise SpecDataError(
            "the spec declares nothing: no dimension, no parameter and no variable. "
            "There is nothing to attach data to and nothing to build."
        )
    return text, program, spec.stem if isinstance(spec, Path) else "spec"


def _dimension(dim: str, coords: Mapping[str, pd.Index]) -> str:
    """A dimension and how many labels it holds; a declared one nothing reaches holds none."""
    return f"{dim} ({len(coords[dim])})" if dim in coords else f"{dim} (unreached)"


def _row(label: str, items: list[str], cap: int = 8) -> str:
    """One aligned summary line, capped with a ``(+N more)`` tail."""
    shown = items[:cap]
    if len(items) > cap:
        shown = shown + [f"(+{len(items) - cap} more)"]
    return f"  {label + ':':<13}{', '.join(shown) if shown else '—'}"


@dataclass(eq=False)
class ModelSpec:
    """
    The spec a model was built from.

    Attributes
    ----------
    model
        The model the spec built into.
    name
        The spec's name, its file's stem or ``"spec"``: the stamp everything
        it built carries in ``.spec``.
    program
        The lowered spec.
    text
        The spec as YAML, verbatim where a file or text was passed.
    parameters
        The relations, and the parameters an unbuilt named expression reads,
        on the master coordinates. Each parameter is filled the way its
        ``missing:`` says, see :class:`~linopy.spec.attach.Parameter`.
    presence
        Where the data has a row, for each parameter in ``parameters`` some
        row is missing from.
    """

    model: Model
    name: str
    program: ms.Program
    text: str
    parameters: xr.Dataset
    presence: xr.Dataset

    def __repr__(self) -> str:
        p = self.program
        coords = self.coords
        lines = (p.description or "").strip().splitlines()
        head = f"ModelSpec: {lines[0]}" if lines else "ModelSpec"
        rows = [
            head,
            _row("Dimensions", [_dimension(d, coords) for d in p.dimensions]),
            _row("Variables", list(p.variables)),
            _row("Constraints", list(p.constraints)),
        ]
        if p.objective is not None:
            rows.append(_row("Objective", [p.objective.sense]))
        rows.append(_row("Expressions", list(p.expressions)))
        return "\n".join(rows)

    def _reattach(self, model: Model, deep: bool = True) -> ModelSpec:
        """The same spec, read off *model*, holding its own copy of the parameters."""
        parameters = self.parameters.copy(deep=deep)
        presence = self.presence.copy(deep=deep)
        return replace(self, model=model, parameters=parameters, presence=presence)

    def _parameter(self, name: str) -> Parameter:
        """The kept parameter *name*, beside where the data has a row for it."""
        if name not in self.parameters:
            raise SpecDataError(
                f"parameter '{name}' is not kept on the spec: only the parameters a named "
                f"expression the build left unbuilt reads are, and a built one reads its "
                f"solution off model.expressions. Rebuild the model to fold it again."
            )
        filled = self.parameters[name]
        if name in self.presence:
            return Parameter(filled, self.presence[name])
        return _everywhere(filled)

    @property
    def coords(self) -> dict[str, pd.Index]:
        """Master coordinates by dimension, as the model was built on them."""
        return {str(d): index for d, index in self.parameters.indexes.items()}

    @property
    def relations(self) -> dict[str, xr.DataArray]:
        """By name, each relation as an array over its key dimension."""
        return {name: self.parameters[name] for name in self.program.relations}

    @functools.cached_property
    def expressions(self) -> dict[str, NamedExpression]:
        """Each named expression as a :class:`NamedExpression`: its math, its linopy fold and its solution."""
        return {name: NamedExpression(self, name) for name in self.program.expressions}

    def declaration(self, name: str) -> Declaration:
        """
        One declaration typeset on its own: the :attr:`expressions` handle for a named expression, a :class:`Declaration` for a constraint or a variable.

        Raises
        ------
        KeyError
            Nothing of the spec is called *name*.
        """
        p = self.program
        if name in p.expressions:
            return self.expressions[name]
        if name in p.constraints or name in p.variables:
            return Declaration(self, name)
        raise unknown(
            "declaration", name, [*p.expressions, *p.constraints, *p.variables]
        )

    def typeset(self, fmt: FormatName, **options: Any) -> str:
        """
        The spec this model was built from, typeset in *fmt* as a document.

        The spec, and so not necessarily the whole model: what was added
        beside the spec carries no declaration to typeset.

        Parameters
        ----------
        fmt : {"latex", "markdown", "typst"}
            What spells the math, as ``mathspec.typeset`` takes it.
        **options
            Passed on to ``mathspec.typeset``.

        Warns
        -----
        UserWarning
            The model holds a variable, constraint, expression, objective or
            special ordered set the spec does not declare, which is not in
            the rendered text.
        """
        return self._document(fmt, options)

    def _document(self, fmt: FormatName, options: dict[str, Any]) -> str:
        """The spec typeset in *fmt*, warning the caller of :meth:`typeset` or a ``to_*`` alias."""
        m, p = self.model, self.program
        stamps = [v.spec for v in m.variables.data.values()]
        stamps += [c.spec for c in m.constraints.data.values()]
        stamps += [e.spec for e in m.expressions.data.values()]
        if p.objective is not None or not m.objective.expression.empty:
            stamps.append(m.objective.expression.spec)
        sos = {n for n, v in m.variables.items() if SOS_TYPE_ATTR in v.attrs}
        undeclared_sos = sos - {s.variable for s in p.sos.values()}
        if None in stamps or undeclared_sos:
            warnings.warn(
                "the model holds variables, constraints, expressions, an objective or "
                "special ordered sets the spec does not declare. What is typeset is "
                "the spec, so it is not this model.",
                UserWarning,
                stacklevel=3,
            )
        return typeset(self._schema, fmt, **options)

    def to_latex(self, **options: Any) -> str:
        """The spec typeset as a LaTeX document, see :meth:`typeset`."""
        return self._document("latex", options)

    def to_markdown(self, **options: Any) -> str:
        """The spec typeset as Markdown, its equations in ``$$`` blocks, see :meth:`typeset`."""
        return self._document("markdown", options)

    def to_typst(self, **options: Any) -> str:
        """The spec typeset as Typst, see :meth:`typeset`."""
        return self._document("typst", options)

    def _repr_markdown_(self) -> str:
        return _notebook_math(self._document("markdown", {}))

    @functools.cached_property
    def _schema(self) -> Spec:
        """The spec as the typesetter reads it."""
        return _read(self.text)

    def _context(self, *, solved: bool) -> Context:
        return Context(
            self.model,
            self.program,
            self.coords,
            self.relations,
            functools.cache(self._parameter),
            self.name,
            solved=solved,
        )


def _notebook_math(markdown: str) -> str:
    r"""
    GitHub's verbatim math delimiters as the ``$``-pairs a notebook's MathJax reads.

    mathspec prints ``$\`...\`$`` and ```` ```math ```` fences because GitHub
    runs Markdown's escape pass inside ``$...$``; Jupyter does not, and renders
    only the classic pair.
    """
    fenced = re.sub(r"```math\n(.*?)\n```", r"$$\n\1\n$$", markdown, flags=re.S)
    return re.sub(r"\$`(.*?)`\$", r"$\1$", fenced)


class Declaration:
    """One declaration of the spec, a variable, a constraint or a named expression, typeset on its own."""

    def __init__(self, spec: ModelSpec, name: str) -> None:
        self._spec = spec
        self._name = name

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self._name}')"

    def typeset(self, fmt: FormatName, **options: Any) -> str:
        """This declaration typeset in *fmt* as a single line, no document around it."""
        return typeset_line(self._spec._schema, self._name, fmt, **options)

    def to_latex(self, **options: Any) -> str:
        """This declaration typeset as a single LaTeX line."""
        return self.typeset("latex", **options)

    def to_markdown(self, **options: Any) -> str:
        """This declaration typeset as a single Markdown math line, no ``$$`` around it."""
        return self.typeset("markdown", **options)

    def to_typst(self, **options: Any) -> str:
        """This declaration typeset as a single Typst line."""
        return self.typeset("typst", **options)

    def _repr_markdown_(self) -> str:
        return f"$$\n{self.to_markdown()}\n$$"


class NamedExpression(Declaration):
    """
    One named expression, in three views: its math, its linopy fold and its solution.

    Where the spec built the expression into ``model.expressions``, both
    ``expression`` and ``solution`` are read off that stored object; else
    they fold the body over the parameters the spec kept.
    """

    def __repr__(self) -> str:
        return f"NamedExpression('{self._name}', dims={self.dims})"

    @property
    def _declared(self) -> ms.NamedExpression:
        return self._spec.program.expressions[self._name]

    @property
    def node(self) -> ms.Expression:
        """The expression body as lowered, mathspec's own AST handle."""
        return self._declared.expression

    @property
    def dims(self) -> tuple[str, ...]:
        """The frame the entry is read over, in the declared order."""
        return self._declared.dims

    @property
    def _stored(self) -> LinearExpression | QuadraticExpression | None:
        """The entry the spec built into ``model.expressions``, if it is still there."""
        held = self._spec.model.expressions.data.get(self._name)
        return held if held is not None and held.spec == self._spec.name else None

    @functools.cached_property
    def expression(self) -> Value:
        """
        The linopy symbolic expression, its variables unsolved.

        The entry ``model.expressions`` holds where the spec built one, else
        folded here: a ``LinearExpression`` where the body carries variables,
        a bare ``Variable``, a ``DataArray`` for a data-only body or a
        ``float`` for a constant.
        """
        stored = self._stored
        if stored is not None:
            return stored
        return evaluate_named(self._name, self._spec._context(solved=False))

    @property
    def solution(self) -> xr.DataArray:
        """
        The expression over the model's solution, as data.

        Read afresh on every access, so it follows the model: a body over
        data alone reads without a solve at all.

        Raises
        ------
        AttributeError
            The model is not solved: the expression is built into
            ``model.expressions``, or its body reads a variable the model
            holds no solution for or a constraint it holds no dual for.
        SpecDataError
            The expression was built and has since been removed from
            ``model.expressions``, so the parameters it reads are not kept.
        """
        stored = self._stored
        if stored is not None:
            return stored.solution.rename(self._name)
        return fold(self._name, self._spec._context(solved=True))
