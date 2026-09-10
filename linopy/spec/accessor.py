"""
``model.spec``: the programs a model was built from or extended by, and their named expressions as data.

The spec owns its data. The spec text, the retained parameters, the lookups
and the master coordinates sit on the accessor rather than in
``model.parameters``, which stays the caller's: a spec never overwrites what
was put there, and nothing reading a spec-built model has to guess which of
its parameters the spec owns. All of it round trips through a file, written
under the ``spec-`` prefix.

A model holds an ordered set of spec *layers*. The first may be the whole
model, built into an empty one; any layer may extend a model that already
holds variables, binding the ones it reads through ``sources``. Each
:class:`Layer` is one program with its data; :class:`ModelSpec` is the
model-level view over all of them.

A parameter is resolved the same way however much of it was retained: from
the retained dataset, else from the sources the model was built with, which
the accessor keeps for as long as the model lives. So ``retain`` decides what
a *file* holds, not what a session can read, and it is only after a round trip
that a parameter can be out of reach.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import xarray as xr
import yaml
from math_spec import (
    Spec,
    did_you_mean,
    to_program,
    to_spec,
    typeset,
    typeset_declaration,
)
from math_spec import program as ms
from math_spec.typesetting import FormatName

from linopy.constants import SOS_TYPE_ATTR, warn_evolving_api
from linopy.model import Model
from linopy.semantics import is_v1
from linopy.spec import terms
from linopy.spec.attach import EVOLVING_MESSAGE, Attached, Retain
from linopy.spec.attach import attach as attach_data
from linopy.spec.builder import build
from linopy.spec.context import Context, Views
from linopy.spec.errors import SpecDataError
from linopy.spec.evaluate import evaluate_named, fold
from linopy.spec.nodes import dims_of
from linopy.spec.parameters import Parameters, Resolve

SpecLike: TypeAlias = str | Path | Mapping[str, Any] | Spec

# A note about what is missing, spelled as a comment of the format's own. A
# format math-spec grows later renders without one rather than with a wrong one.
_DRIFTED = "This model has drifted from the spec typeset here: {}."
_EXTENDS = "This spec extends a model it does not describe: {}."

_COMMENT: dict[str, str] = {
    "latex": "% {}",
    "markdown": "<!-- {} -->",
    "typst": "// {}",
}


@dataclass(frozen=True)
class Unspecified:
    """
    How a spec-built model has drifted from the spec it was built from.

    A model goes on taking everything linopy can add to it, and none of that
    carries a math-spec declaration to typeset. A spec's own ``piecewise:``
    and ``sos:`` are not drift: math-spec lowers them into ordinary
    declarations, so they sit in the program like any other.

    Attributes
    ----------
    variables, constraints
        Added beside the spec, a piecewise formulation's own aside.
    expressions
        Everything in ``model.expressions``: a spec's named expressions are
        read lazily off ``model.spec`` and never live there.
    sos
        Variables given a special-ordered set the spec does not declare.
        ``add_sos_constraints`` writes attributes onto a variable rather than
        adding a name of its own, so nothing else here would show it.
    piecewise
        Formulations added by ``add_piecewise_formulation``, named as
        formulations rather than as the variables and constraints they hold.
    objective
        Whether the model's objective is one no spec layer declared. The one
        entry here that a render gets *wrong* rather than leaves out: the
        typeset objective is the spec's, and the model's is another.
    """

    variables: tuple[str, ...]
    constraints: tuple[str, ...]
    expressions: tuple[str, ...]
    sos: tuple[str, ...]
    piecewise: tuple[str, ...]
    objective: bool

    def __bool__(self) -> bool:
        return bool(
            self.variables
            or self.constraints
            or self.expressions
            or self.sos
            or self.piecewise
            or self.objective
        )


def _joined(parts: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(parts) < 3:
        return " and ".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _counted(names: tuple[str, ...], kind: str, cap: int = 5) -> str:
    """``2 constraints (a, b)``, capped with a ``+N more`` tail; empty for no names."""
    if not names:
        return ""
    shown = list(names[:cap])
    if len(names) > cap:
        shown.append(f"+{len(names) - cap} more")
    plural = kind if len(names) == 1 else f"{kind}s"
    return f"{len(names)} {plural} ({', '.join(shown)})"


def attach(
    model: Model,
    spec: SpecLike,
    sources: Mapping[str, Any] | xr.Dataset,
    retain: Retain,
    name: str | None = None,
) -> ModelSpec:
    """
    Build *spec* with *sources* into *model* as a layer and return the accessor.

    The layer is named *name*, else the file's stem, else ``"spec"``.

    Raises
    ------
    ValueError
        The model runs under legacy semantics; a layer of that name is
        already on the model; a variable the spec introduces, a constraint
        or a named expression collides with a name the model already holds;
        or the spec declares an objective and the model already has one.
    TypeError
        *spec* is a lowered ``Program``, which has no YAML form to
        keep on the model.
    """
    warn_evolving_api("spec", EVOLVING_MESSAGE, stacklevel=4)
    if not is_v1():
        raise ValueError(
            "a spec-built model uses linopy's v1 semantics, and the current setting is "
            "'legacy'. Set linopy.options['semantics'] = 'v1' before building from a spec."
        )
    text, program = _source(spec)
    attached: Attached = attach_data(
        program, sources, retain=retain, given=_given(model)
    )
    _check_collisions(model, program, attached)
    layer_name = _layer_name(spec, name)
    if model._spec is not None and layer_name in model._spec.layers:
        raise ValueError(
            f"a spec layer named '{layer_name}' is already on this model; pass another name."
        )
    # Resolved before the build, so a parameter no declaration reads cannot fail
    # halfway through one and leave a model too full to build into again.
    parameters = attached.retained().assign_coords(dict(attached.coords))
    whole = not len(model.variables) and not len(model.constraints)
    build(model, attached)
    layer = Layer(
        model, layer_name, program, text, parameters, attached, attached.names
    )
    spec_ = model._spec if model._spec is not None else ModelSpec(model, [], whole)
    spec_.layers[layer.name] = layer
    if program.objective is not None:
        spec_.objective_owner = layer.name
    return spec_


def _layers(model: Model) -> list[Layer]:
    """The layers already on *model*, in order."""
    return [] if model._spec is None else list(model._spec.layers.values())


def _given(model: Model) -> dict[str, pd.Index]:
    """The master coordinates of every layer already on *model*; they agree wherever they meet."""
    return {d: index for layer in _layers(model) for d, index in layer.coords.items()}


def _layer_name(spec: SpecLike, name: str | None) -> str:
    if name is not None:
        return name
    if isinstance(spec, str) and "\n" not in spec:
        spec = Path(spec)
    return spec.stem if isinstance(spec, Path) else "spec"


def _check_collisions(model: Model, program: ms.Program, attached: Attached) -> None:
    introduced = [
        n for n in program.variables if n not in attached.bound and n in model.variables
    ]
    if introduced:
        raise ValueError(
            f"the spec introduces variable(s) {introduced} and the model already holds "
            f"them: bind it or rename it. A binding passes the model variable under the "
            f"declared name in sources."
        )
    constraints = [n for n in program.constraints if n in model.constraints]
    if constraints:
        raise ValueError(
            f"the spec declares constraint(s) {constraints} and the model already holds them."
        )
    on = {attached.names.get(s.variable, s.variable) for s in program.sos.values()}
    sos = [
        n
        for n in on
        if n in model.variables and SOS_TYPE_ATTR in model.variables[n].attrs
    ]
    if sos:
        raise ValueError(
            f"the spec declares a special-ordered set on variable(s) {sos} and the model "
            f"already holds one on them."
        )
    if program.objective is not None and not model.objective.expression.empty:
        raise ValueError(
            "the spec declares an objective and the model already has one. Add extra cost "
            "terms through a named expression: "
            "`m.objective += m.spec.expressions[name].expression`."
        )
    earlier = {n for layer in _layers(model) for n in layer.program.named_expressions}
    expressions = [n for n in program.named_expressions if n in earlier]
    if expressions:
        raise ValueError(
            f"the spec declares named expression(s) {expressions} and an earlier spec on "
            f"this model already does."
        )


def restore_layer(
    model: Model,
    name: str,
    text: str,
    parameters: xr.Dataset,
    names: Mapping[str, str],
) -> Layer:
    """
    One layer of *model*, with the program lowered afresh from *text*.

    Read from a file, so the sources the model was built with are gone and
    only what ``retain`` kept can be read back.
    """
    program = to_program(yaml.safe_load(text))
    return Layer(model, name, program, text, parameters, None, dict(names))


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


def _dimension(dim: str, coords: Mapping[str, pd.Index]) -> str:
    """A dimension and how many labels it holds; a declared one nothing reaches holds none."""
    return f"{dim} ({len(coords[dim])})" if dim in coords else f"{dim} (unreached)"


def _row(label: str, items: list[str], cap: int = 8) -> str:
    """One aligned summary line, capped with a ``(+N more)`` tail."""
    shown = items[:cap]
    if len(items) > cap:
        shown = shown + [f"(+{len(items) - cap} more)"]
    return f"  {label + ':':<13}{', '.join(shown) if shown else '—'}"


@dataclass(frozen=True, eq=False, repr=False)
class Layer:
    """
    One spec on a model: its program, its text and its data.

    Attributes
    ----------
    name
        What the layer was attached as, ``model.spec[name]``.
    program
        The lowered spec.
    text
        The spec as YAML, verbatim where a file or text was passed.
    parameters
        The parameters and lookups the layer retained, on the master coordinates.
    names
        Spec name to model name for every variable the layer reads instead
        of building.
    """

    model: Model
    name: str
    program: ms.Program
    text: str
    parameters: xr.Dataset
    attached: Attached | None
    names: Mapping[str, str]
    _views: Views = field(default_factory=dict)

    def __repr__(self) -> str:
        return "\n".join(self._rows(f"Layer '{self.name}'"))

    def _rows(self, head: str) -> list[str]:
        p = self.program
        coords = self.coords
        if self.description:
            head = f"{head}: {self.description}"
        rows = [
            head,
            _row("Dimensions", [_dimension(d, coords) for d in p.dimensions]),
            _row("Variables", list(p.variables)),
            _row("Constraints", list(p.constraints)),
        ]
        if p.objective is not None:
            rows.append(_row("Objective", [p.objective.sense]))
        rows.append(_row("Expressions", list(p.named_expressions)))
        return rows

    def _reattach(self, model: Model, deep: bool = True) -> Layer:
        """The same layer, read off *model*, holding its own copy of the parameters."""
        parameters = self.parameters.copy(deep=deep)
        return replace(self, model=model, parameters=parameters, _views={})

    @property
    def description(self) -> str:
        """The spec's own description, its first line, or an empty string."""
        lines = str(self._schema.get("description", "")).strip().splitlines()
        return lines[0] if lines else ""

    @property
    def coords(self) -> dict[str, pd.Index]:
        """Master coordinates by dimension, as the layer was built on them."""
        return {str(d): index for d, index in self.parameters.indexes.items()}

    @property
    def variables(self) -> set[str]:
        """The model variables the layer declares, by model name: built as declared, bound as bound."""
        return {self.names.get(n, n) for n in self.program.variables}

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
        return NamedExpressions({n: self for n in self.program.named_expressions})

    def declaration(self, name: str) -> Declaration:
        """
        One declaration typeset on its own: a named expression, constraint or variable.

        Its math as a single line, no document around it. A named expression
        also carries its linopy fold and solution through :attr:`expressions`;
        this handle is the typesetting one every declaration shares.
        """
        if name not in self._declarations:
            raise KeyError(
                f"unknown declaration '{name}'. "
                + did_you_mean(name, self._declarations)
            )
        return Declaration(self, name)

    @property
    def _declarations(self) -> list[str]:
        p = self.program
        return [*p.named_expressions, *p.constraints, *p.variables]

    def typeset(self, fmt: FormatName, **options: Any) -> str:
        """
        This layer's spec typeset in *fmt* as a document, the spec alone.

        Parameters
        ----------
        fmt : {"latex", "markdown", "typst"}
            What spells the math, as ``math_spec.typeset`` takes it.
        **options
            Passed on to ``math_spec.typeset``: ``symbols``, ``standalone``,
            ``legend``, ``numbered``, ``inline_expressions``.
        """
        return typeset(self._schema, fmt, **options)

    def to_latex(self, **options: Any) -> str:
        """The layer typeset as a LaTeX document, see :meth:`typeset`."""
        return self.typeset("latex", **options)

    def to_markdown(self, **options: Any) -> str:
        """The layer typeset as Markdown, its equations in ``$$`` blocks, see :meth:`typeset`."""
        return self.typeset("markdown", **options)

    def to_typst(self, **options: Any) -> str:
        """The layer typeset as Typst, see :meth:`typeset`."""
        return self.typeset("typst", **options)

    @property
    def _schema(self) -> dict[str, Any]:
        """The spec as the mapping the typesetter reads (a bare string it reads as a path)."""
        return yaml.safe_load(self.text)

    def evaluate(
        self, name: str, sources: Mapping[str, Any] | xr.Dataset
    ) -> NamedExpression:
        """
        The named expression *name*, with its parameters attached afresh from *sources*.

        For reading the spec against other data than the model was built with,
        and for a model read from a file, whose own sources are gone.
        ``expressions`` needs neither. *sources* is read the way ``add_spec``
        read it, and must describe the coordinates the model was built on.

        Raises
        ------
        SpecDataError
            *sources* label a dimension differently than the
            model was built on.
        """
        attached = attach_data(self.program, sources, retain="none")
        coords = self.coords
        for dim, index in attached.coords.items():
            if dim in coords and not index.equals(coords[dim]):
                raise SpecDataError(
                    f"sources describe dimension '{dim}' as {index.tolist()[:5]}, and the model "
                    f"was built on {coords[dim].tolist()[:5]}. evaluate() reads the solution the "
                    f"model holds, so the data must be attached on the same labels in the same order."
                )
        return NamedExpression(self, name, self._context(attached.parameter))

    def _resolve(self, name: str) -> xr.DataArray:
        """The parameter *name*: retained if it was kept, else read from the sources again."""
        if name in self.parameters:
            return self.parameters[name]
        if self.attached is not None:
            return self.attached.parameter(name)
        raise SpecDataError(
            f"parameter '{name}' was not retained and this model no longer holds the sources "
            f"it was built with, which is what a model read from a file looks like. Build with "
            f"retain='all' before writing it out, or read the expression with "
            f"evaluate(name, sources)."
        )

    def _context(self, resolve: Resolve) -> Context:
        return Context(
            self.model,
            self.program,
            self.coords,
            self.lookups,
            Parameters(self.program, resolve),
            solved=True,
            names=self.names,
            views=self._views,
        )


class ModelSpec:
    """
    The spec layers of a model, and the model-level view over them.

    ``model.spec[name]`` is one :class:`Layer`. With a single layer the
    layer's ``program``, ``text``, ``parameters``, ``coords``, ``lookups``,
    ``name`` and ``names`` read through here as well; with several they are
    each layer's own.

    Attributes
    ----------
    layers
        By name, in the order they were attached.
    whole
        Whether the first layer was built into an empty model, so the layers
        together describe the model rather than extend one.
    objective_owner
        The layer whose objective the model holds, ``None`` where the model's
        objective is none of theirs.
    """

    def __init__(
        self,
        model: Model,
        layers: Iterable[Layer],
        whole: bool,
        objective_owner: str | None = None,
    ) -> None:
        self._model = model
        self.layers: dict[str, Layer] = {layer.name: layer for layer in layers}
        self.whole = whole
        self.objective_owner = objective_owner

    def __getitem__(self, name: str) -> Layer:
        if name not in self.layers:
            raise KeyError(
                f"unknown spec layer '{name}'. " + did_you_mean(name, self.layers)
            )
        return self.layers[name]

    def __repr__(self) -> str:
        if len(self.layers) == 1:
            return "\n".join(self._only()._rows("ModelSpec"))
        head = f"ModelSpec: layers {', '.join(self.layers)}"
        return "\n\n".join([head, *map(repr, self.layers.values())])

    def _reattach(self, model: Model, deep: bool = True) -> ModelSpec:
        """The same layers, read off *model*, each holding its own copy of the parameters."""
        layers = [layer._reattach(model, deep) for layer in self.layers.values()]
        return ModelSpec(model, layers, self.whole, self.objective_owner)

    def _only(self) -> Layer:
        if len(self.layers) != 1:
            raise ValueError(
                f"this model holds spec layers {list(self.layers)}; read one through "
                f"model.spec[name]."
            )
        return next(iter(self.layers.values()))

    def _owner(self, name: str, declared: Callable[[Layer], Collection[str]]) -> Layer:
        for layer in self.layers.values():
            if name in declared(layer):
                return layer
        known = [n for layer in self.layers.values() for n in declared(layer)]
        raise KeyError(f"unknown declaration '{name}'. " + did_you_mean(name, known))

    @property
    def name(self) -> str:
        """The single layer's name, see :attr:`Layer.name`."""
        return self._only().name

    @property
    def names(self) -> Mapping[str, str]:
        """The single layer's bound names, see :attr:`Layer.names`."""
        return self._only().names

    @property
    def program(self) -> ms.Program:
        """The single layer's lowered spec."""
        return self._only().program

    @property
    def text(self) -> str:
        """The single layer's spec as YAML."""
        return self._only().text

    @property
    def parameters(self) -> xr.Dataset:
        """The single layer's retained parameters and lookups, on the master coordinates."""
        return self._only().parameters

    @property
    def description(self) -> str:
        """The single layer's description, see :attr:`Layer.description`."""
        return self._only().description

    @property
    def coords(self) -> dict[str, pd.Index]:
        """The single layer's master coordinates by dimension."""
        return self._only().coords

    @property
    def lookups(self) -> dict[str, dict[str, xr.DataArray]]:
        """The single layer's lookups, by dimension, by name."""
        return self._only().lookups

    @property
    def expressions(self) -> NamedExpressions:
        """Every layer's named expressions as :class:`NamedExpression` objects, by name."""
        owners = {
            n: layer
            for layer in self.layers.values()
            for n in layer.program.named_expressions
        }
        return NamedExpressions(owners)

    def declaration(self, name: str) -> Declaration:
        """One declaration typeset on its own, from whichever layer declares it, see :meth:`Layer.declaration`."""
        return self._owner(name, lambda layer: layer._declarations).declaration(name)

    def evaluate(
        self, name: str, sources: Mapping[str, Any] | xr.Dataset
    ) -> NamedExpression:
        """The named expression *name* on fresh *sources*, from the layer that declares it, see :meth:`Layer.evaluate`."""
        owner = self._owner(name, lambda layer: layer.program.named_expressions)
        return owner.evaluate(name, sources)

    @property
    def unspecified(self) -> Unspecified:
        """
        How the model has drifted from its spec layers, see :class:`Unspecified`.

        Falsy for a model that is only what its layers say; everything added
        beside them lands here, and is what typesetting cannot show.
        """
        from linopy.piecewise import _get_piecewise_groups

        model = self._model
        layers = list(self.layers.values())
        pw_variables, pw_constraints = _get_piecewise_groups(model)
        declared_sos = {
            layer.names.get(sos.variable, sos.variable)
            for layer in layers
            for sos in layer.program.sos.values()
        }
        variables = {n for layer in layers for n in layer.variables}
        constraints = {n for layer in layers for n in layer.program.constraints}
        return Unspecified(
            variables=tuple(
                n
                for n in model.variables
                if n not in variables and n not in pw_variables
            ),
            constraints=tuple(
                n
                for n in model.constraints
                if n not in constraints and n not in pw_constraints
            ),
            expressions=tuple(model.expressions),
            sos=tuple(
                n
                for n in model.variables
                if SOS_TYPE_ATTR in model.variables[n].attrs and n not in declared_sos
            ),
            piecewise=tuple(model._piecewise_formulations),
            objective=self.objective_owner is None
            and not model.objective.expression.empty,
        )

    def typeset(self, fmt: FormatName, **options: Any) -> str:
        """
        The spec layers typeset in *fmt* as a document, one rendering after another.

        The spec, and so not necessarily the whole model: what was added
        beside the spec carries no declaration to typeset. Where the model
        holds such a thing, :attr:`unspecified` names it, a warning says so,
        and the rendered text opens with the same tally as a comment of
        *fmt*'s own -- gone once compiled, there in the source.

        Parameters
        ----------
        fmt : {"latex", "markdown", "typst"}
            What spells the math, as ``math_spec.typeset`` takes it.
        **options
            Passed on to ``math_spec.typeset``: ``symbols``, ``standalone``,
            ``legend``, ``numbered``, ``inline_expressions``. Several layers
            refuse ``standalone``: one document cannot hold two preambles.

        Warns
        -----
        UserWarning
            The model holds variables or constraints the spec does not
            declare, which are not in the rendered text.
        """
        return self._render(fmt, options, 3)

    def to_latex(self, **options: Any) -> str:
        """The spec typeset as a LaTeX document, see :meth:`typeset`."""
        return self._render("latex", options, 3)

    def to_markdown(self, **options: Any) -> str:
        """The spec typeset as Markdown, its equations in ``$$`` blocks, see :meth:`typeset`."""
        return self._render("markdown", options, 3)

    def to_typst(self, **options: Any) -> str:
        """The spec typeset as Typst, see :meth:`typeset`."""
        return self._render("typst", options, 3)

    def _render(
        self, fmt: FormatName, options: Mapping[str, Any], stacklevel: int
    ) -> str:
        """Typeset in *fmt*, warned and commented where the model holds more than the spec."""
        if len(self.layers) > 1 and options.get("standalone", False):
            raise ValueError(
                f"a standalone document holds one spec, and this model holds layers "
                f"{list(self.layers)}. Typeset one with model.spec[name].typeset(fmt, "
                f"standalone=True)."
            )
        rendered = "\n\n".join(
            layer.typeset(fmt, **options) for layer in self.layers.values()
        )
        tally = self._tally()
        if tally is None:
            return rendered
        note = self._note(tally)
        warnings.warn(
            f"{note} What is typeset is the spec, so it is not this model.",
            UserWarning,
            stacklevel=stacklevel,
        )
        comment = _COMMENT.get(fmt)
        if comment is None:
            return rendered
        return f"{comment.format(note)}\n{rendered}"

    def _note(self, tally: str) -> str:
        return (_DRIFTED if self.whole else _EXTENDS).format(tally)

    def _tally(self) -> str | None:
        """How the model has drifted, counted and named; ``None`` when it has not."""
        found = self.unspecified
        if not found:
            return None
        parts = [
            _counted(found.variables, "variable"),
            _counted(found.constraints, "constraint"),
            _counted(found.expressions, "expression"),
            _counted(found.sos, "SOS set"),
            _counted(found.piecewise, "piecewise formulation"),
        ]
        objective = (
            "a replaced objective"
            if self.whole
            else "an objective this spec does not declare"
        )
        return _joined([p for p in parts if p] + [objective] * found.objective)

    def _repr_markdown_(self) -> str:
        """The spec as Markdown, with a *visible* note where a notebook would swallow the warning."""
        rendered = self._render("markdown", {}, 3)
        tally = self._tally()
        if tally is None:
            return rendered
        return f"{rendered}\n\n*{self._note(tally)}*"


class NamedExpressions(Mapping[str, "NamedExpression"]):
    """The named expressions of one or more layers, each a :class:`NamedExpression` on read."""

    def __init__(self, owners: Mapping[str, Layer]) -> None:
        self._owners = owners

    def __getitem__(self, name: str) -> NamedExpression:
        if name not in self._owners:
            raise KeyError(
                f"unknown named expression '{name}'. "
                + did_you_mean(name, self._owners)
            )
        layer = self._owners[name]
        return NamedExpression(layer, name, layer._context(layer._resolve))

    def __iter__(self) -> Iterator[str]:
        return iter(self._owners)

    def __len__(self) -> int:
        return len(self._owners)

    def __repr__(self) -> str:
        return f"NamedExpressions({list(self)})"


class Declaration:
    """
    One declaration of a spec, typeset on its own: math only, no document.

    A named expression, a constraint or a variable, reached by name through
    :meth:`ModelSpec.declaration`. :class:`NamedExpression` adds the linopy
    fold and the solution on top of this.
    """

    def __init__(self, layer: Layer, name: str) -> None:
        self._layer = layer
        self._name = name

    def typeset(self, fmt: FormatName, **options: Any) -> str:
        """
        This declaration typeset in *fmt* as a single line, no document around it.

        Nothing here can be out of step with the model the way
        :meth:`ModelSpec.typeset` can: a declaration is reached by name
        through the spec, so there is only ever the spec's own math to show.
        """
        return typeset_declaration(self._layer._schema, self._name, fmt, **options)

    def to_latex(self, **options: Any) -> str:
        """This declaration typeset as a single LaTeX line, no document around it."""
        return self.typeset("latex", **options)

    def to_markdown(self, **options: Any) -> str:
        """This declaration typeset as a single Markdown math line, no ``$$`` around it."""
        return self.typeset("markdown", **options)

    def to_typst(self, **options: Any) -> str:
        """This declaration typeset as a single Typst line, no document around it."""
        return self.typeset("typst", **options)

    def _repr_markdown_(self) -> str:
        return f"$$\n{self.to_markdown()}\n$$"


class NamedExpression(Declaration):
    """
    One named expression, in three views: its math, its linopy fold and its solution.

    The object pins the data sources it was made with for its lifetime, so the
    three views agree. ``expressions[name]`` reads the model's own data --
    what ``retain`` kept, and the sources behind it for the rest;
    ``evaluate(name, sources)`` attaches fresh data instead.

    Attributes
    ----------
    node
        The lowered expression body, math-spec's own AST handle.
    """

    def __init__(self, layer: Layer, name: str, ctx: Context) -> None:
        super().__init__(layer, name)
        self._ctx = ctx

    @property
    def node(self) -> ms.ExpressionNode:
        """The expression body as lowered, math-spec's own AST handle."""
        return self._layer.program.named_expressions[self._name].expression

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the expression spans, read off the spec without binding data."""
        return dims_of(self.node, self._layer.program)

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

        Raises
        ------
        RuntimeError
            The model reads a variable but holds no solution yet.
        SpecDataError
            A parameter the body reads was neither retained nor
            still reachable through the model's sources.
        """
        return fold(self._name, self._ctx)

    def __repr__(self) -> str:
        value = self.__dict__.get("solution", self.__dict__.get("expression"))
        if isinstance(value, xr.DataArray):
            return f"NamedExpression('{self._name}', dims={tuple(value.dims)})"
        return f"NamedExpression('{self._name}')"
