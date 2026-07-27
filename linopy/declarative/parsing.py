"""
Linopy declarative math parsing module.

This module turns a validated math component definition into a list of :class:`Equation` objects.
The objects hold the parsed expression/mask ASTs with all `$name` sub-expression and slicer references resolved.
They also provide the typed entry points that evaluate an equation to a boolean mask array, a linopy expression, a constraint tuple, or a LaTeX math string.

This module is adapted from the calliope Apache-2.0 licensed equation parsing module:
https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/parsing.py
"""

from __future__ import annotations

import functools
import itertools
import logging
import operator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pyparsing as pp
import xarray as xr

from linopy.declarative import grammar
from linopy.declarative.nodes import (
    MODE_T,
    TRUE_ARRAY,
    Component,
    Context,
    Node,
    SliceRef,
    SubExprRef,
    find_refs,
    to_linexpr,
)
from linopy.declarative.schema import (
    BUILD_ORDER,
    COMPONENTS_T,
    EQUATION_GROUP_T,
    MATH_DEFS_T,
    ConstraintDef,
    ExpressionDef,
    MathModel,
    ObjectiveDef,
    _Equations,
)
from linopy.expressions import LinearExpression

LOGGER = logging.getLogger(__name__)

EQUATION_DEFS_T = ConstraintDef | ExpressionDef | ObjectiveDef
"""Math component definitions that carry `equations`/`sub_expressions`/`slices` keys."""

_ERR_BULLET = " * "


@dataclass(frozen=True)
class Equation:
    """
    One fully-resolved equation of a math component.

    Produced by :func:`parse_component`: each combination of sub-expression and
    slicer variants referenced by a user-defined equation yields one `Equation`.
    """

    name: str
    """Unique equation name, including the chosen sub-expression/slicer variants."""

    sets: tuple[str, ...]
    """The component's `foreach` dimensions."""

    expression: Node
    """Parsed expression AST."""

    masks: tuple[Node, ...]
    """Parsed mask ASTs: the equation's own mask plus those of the chosen variants."""

    sub_expressions: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` sub-expression AST per name."""

    slices: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` slicer AST per name."""

    def references(self) -> set[str]:
        """Return the names of all math components referenced by this equation."""
        trees = [
            self.expression,
            *self.masks,
            *self.sub_expressions.values(),
            *self.slices.values(),
        ]
        return set().union(*(find_refs(tree, Component) for tree in trees))


@dataclass(frozen=True)
class ParsedComponent:
    """The parsed math strings of one active component."""

    mask: Node
    """Parsed top-level `mask` AST (the trivial "True" AST for objectives)."""

    equations: list[Equation] = field(default_factory=list)
    """Fully-resolved equations (empty for variables, which carry none)."""


@dataclass(frozen=True)
class ParsedMath:
    """All parsed math strings of a math definition's active components."""

    components: dict[str, dict[str, ParsedComponent]]
    """Per build group, the parsed strings of each active component by name."""

    checks: dict[str, Node]
    """Parsed mask AST of each active input-data check by name."""

    def __getitem__(self, group: str) -> dict[str, ParsedComponent]:
        """Return the parsed components of a build group."""
        return self.components[group]


def parse_math(math: MathModel) -> ParsedMath:
    """
    Parse every math string of the active components, aggregating ALL failures.

    Walks the masks, equations, sub-expressions, and slices of active variables,
    expressions, constraints, and objectives, plus active check masks. Inactive
    components and groups without a build path (`piecewise_constraints`,
    `postprocessed`) are deliberately not parsed: definitions there are
    declarations of intent, and a stale string in them must never block a build.

    Parameters
    ----------
    math : MathModel
        The validated math definition.

    Returns
    -------
    ParsedMath
        Every math string of the walked components, parsed exactly once.

    Raises
    ------
    ValueError
        One error listing every parse failure (syntax errors and undefined
        `$name` references), grouped by component.
    """
    collector = _ParsingCollector()
    mask_parser = _mask_grammar(math)
    raw: dict[str, dict[str, tuple[Node | None, list[Equation]]]] = {}
    for group in BUILD_ORDER:
        raw[group] = {}
        for name in getattr(math, group)._active:
            definition = getattr(math, group)[name]
            component_name = f"{group}:{name}"
            mask_node = collector.parse(
                mask_parser, definition.mask, component_name, "mask"
            )
            equations = (
                parse_component(group, name, definition, math, collector)
                if group != "variables"
                else []
            )
            raw[group][name] = (mask_node, equations)
    raw_checks = {
        name: collector.parse(
            mask_parser, math.checks[name].mask, f"checks:{name}", "mask"
        )
        for name in math.checks._active
    }
    collector.raise_errors()
    components = {
        group: {
            name: ParsedComponent(mask=mask_node, equations=equations)
            for name, (mask_node, equations) in group_raw.items()
            if mask_node is not None  # always true after raise_errors
        }
        for group, group_raw in raw.items()
    }
    checks = {name: node for name, node in raw_checks.items() if node is not None}
    return ParsedMath(components=components, checks=checks)


def _expression_names(math: MathModel) -> frozenset[str]:
    """Return the valid component names for expression-string parsing."""
    return frozenset(set().union(*math.parsing_components["expression"].values()))


def _mask_grammar(math: MathModel) -> pp.ParserElement:
    """Return the mask-string grammar for the math definition's component names."""
    names = math.parsing_components["mask"]
    return grammar.mask_grammar(
        frozenset(names["dimensions"]),
        frozenset(names["inputs"]),
        frozenset(names["results"]),
    )


class _ParsingCollector:
    """
    Collect parsing results and errors per component.

    Collected errors can be raised as a single ValueError at the end.
    """

    def __init__(self) -> None:
        self.errors: dict[str, list[str]] = {}

    def add(self, component_name: str, message: str) -> None:
        """Store a plain (caret-free) error message against a component."""
        self.errors.setdefault(component_name, []).append(f"{_ERR_BULLET}{message}")

    def parse(
        self, parser: pp.ParserElement, string: str, component_name: str, position: str
    ) -> Node | None:
        """
        Parse `string`, returning its AST root or None if parsing fails.

        Failures are stored against `component_name` with a caret marker pointing
        at the parse position, for raising later via :meth:`raise_errors`.
        """
        try:
            return parser.parse_string(string, parse_all=True)[0]
        except pp.ParseException as excinfo:
            pointer = (
                f"{_ERR_BULLET}{position} (line {excinfo.lineno}, char {excinfo.col}): "
            )
            marker_pos = " " * (len(pointer) + excinfo.col - 1)
            self.errors.setdefault(component_name, []).append(
                f"{pointer}{excinfo.line}\n{marker_pos}^"
            )
            return None

    def raise_errors(self) -> None:
        """Raise all collected errors as one ValueError, grouped by component."""
        if self.errors:
            sections = "\n".join(
                f"{component}:\n" + "\n".join(bullets)
                for component, bullets in self.errors.items()
            )
            raise ValueError(sections)


def parse_mask(mask_string: str, math: MathModel, name: str = "") -> Node:
    """
    Parse a standalone mask string, raising on invalid syntax.

    Parameters
    ----------
    mask_string : str
        The mask ("where"-condition) string to parse.
    math : MathModel
        Math definition providing the valid component names.
    name : str, optional
        Name to identify the string by in error messages.
    """
    collector = _ParsingCollector()
    parsed = collector.parse(_mask_grammar(math), mask_string, name, "mask")
    collector.raise_errors()
    assert parsed is not None
    return parsed


def _parse_variants(
    collector: _ParsingCollector,
    component_name: str,
    parser: pp.ParserElement,
    mask_parser: pp.ParserElement,
    expression_list: _Equations,
    sets: tuple[str, ...],
    position: str,
    name_prefix: str,
) -> list[Equation]:
    """Parse a list of `{mask, expression}` items into one Equation per item."""
    equations = []
    for idx, item in enumerate(expression_list):
        position_id = f"{position}[{idx}]"
        mask = collector.parse(
            mask_parser, item.mask, component_name, f"{position_id}.mask"
        )
        expression = collector.parse(
            parser, item.expression, component_name, f"{position_id}.expression"
        )
        if expression is not None and mask is not None:
            equations.append(
                Equation(
                    name=f"{name_prefix}:{idx}",
                    sets=sets,
                    expression=expression,
                    masks=(mask,),
                )
            )
    return equations


def _expand(
    component_name: str,
    equations: list[Equation],
    candidates: dict[str, list[Equation]],
    kind: Literal["sub_expressions", "slices"],
    collector: _ParsingCollector,
) -> list[Equation]:
    """
    Expand equations with all combinations of their referenced `$name` variants.

    Each `$name` reference maps to a list of `{mask, expression}` variants.
    An equation referencing them is replaced by one equation per element of the
    cartesian product of those variant lists.
    The chosen variants' masks and ASTs are merged in.
    Undefined `$name` references are collected (not raised) so they aggregate with any other parse failures.
    """
    expanded = []
    for equation in equations:
        ref_type = SubExprRef if kind == "sub_expressions" else SliceRef
        trees = [equation.expression, *equation.sub_expressions.values()]
        refs = set().union(*(find_refs(tree, ref_type) for tree in trees))
        if not refs:
            expanded.append(equation)
            continue
        undefined = refs.difference(candidates.keys())
        if undefined:
            collector.add(
                component_name,
                f"Undefined {kind} found in equation: {sorted(undefined)}",
            )
            continue
        for combination in itertools.product(*(candidates[ref] for ref in refs)):
            new_name = "-".join([equation.name, *(v.name for v in combination)])
            new_masks = (
                *equation.masks,
                *(mask for variant in combination for mask in variant.masks),
            )
            resolved = {
                variant.name.split(":")[0]: variant.expression
                for variant in combination
            }
            if kind == "sub_expressions":
                new_equation = replace(
                    equation, name=new_name, masks=new_masks, sub_expressions=resolved
                )
            else:
                new_equation = replace(
                    equation, name=new_name, masks=new_masks, slices=resolved
                )
            expanded.append(new_equation)
    return expanded


def parse_component(
    group: EQUATION_GROUP_T,
    name: str,
    definition: EQUATION_DEFS_T,
    math: MathModel,
    collector: _ParsingCollector | None = None,
) -> list[Equation]:
    """
    Parse a math component's equations into fully-resolved :class:`Equation` objects.

    All `expression` and `mask` strings of the component's equations, sub-expressions, and slicers are parsed.
    Then every equation is expanded with the cartesian product of the sub-expression and slicer variants it references.

    Parameters
    ----------
    group : EQUATION_GROUP_T
        Component group the definition belongs to (defines the equation grammar:
        comparisons for constraints, arithmetic otherwise).
    name : str
        Name of the math component.
    definition : EQUATION_DEFS_T
        The component's (already schema-validated) definition.
    math : MathModel
        The full math definition, providing valid component names.
    collector : _ParsingCollector, optional
        Shared error collector (used by :func:`parse_math` to aggregate failures across components).
        If not given, all failures of this component are raised at the end of the call.

    Returns
    -------
    list[Equation]
        One equation per user-defined equation and referenced variant combination.
    """
    component_name = f"{group}:{name}"
    own_collector = collector is None
    collector = collector or _ParsingCollector()
    names = _expression_names(math)
    equation_parser = (
        grammar.equation_grammar(names)
        if group == "constraints"
        else grammar.arithmetic_grammar(names)
    )
    mask_parser = _mask_grammar(math)
    sets = tuple(definition.foreach)

    equations = _parse_variants(
        collector,
        component_name,
        equation_parser,
        mask_parser,
        definition.equations,
        sets,
        "equations",
        component_name,
    )
    sub_expressions = {
        sub_name: _parse_variants(
            collector,
            component_name,
            grammar.sub_expression_grammar(names),
            mask_parser,
            sub_list,
            sets,
            f"sub_expressions.{sub_name}",
            sub_name,
        )
        for sub_name, sub_list in definition.sub_expressions.root.items()
    }
    slices = {
        slice_name: _parse_variants(
            collector,
            component_name,
            grammar.slice_grammar(names),
            mask_parser,
            slice_list,
            sets,
            f"slices.{slice_name}",
            slice_name,
        )
        for slice_name, slice_list in definition.slices.root.items()
    }

    equations = _expand(
        component_name, equations, sub_expressions, "sub_expressions", collector
    )
    equations = _expand(component_name, equations, slices, "slices", collector)
    if own_collector:
        collector.raise_errors()
    return equations


# ---------------------------------------------------------------------------
# Component-level masking
# ---------------------------------------------------------------------------


def foreach_mask(sets: tuple[str, ...], input_data: xr.Dataset) -> xr.DataArray:
    """
    Return the initial boolean array spanning a component's `foreach` dimensions.

    Parameters
    ----------
    sets : tuple[str, ...]
        The component's `foreach` dimensions.
    input_data : xr.Dataset
        Model input data providing the dimension coordinates.
    """
    missing_sets = set(sets).difference(input_data.dims)
    if missing_sets:
        LOGGER.debug(
            f"Math parsing | indexed over unidentified set names: `{missing_sets}`."
        )
        return xr.DataArray(False)
    if not sets:
        return TRUE_ARRAY
    exists_and_foreach = [input_data[i].notnull() for i in sets]
    return functools.reduce(operator.and_, exists_and_foreach)


def drop_dims_not_in_foreach(mask: xr.DataArray, sets: tuple[str, ...]) -> xr.DataArray:
    """
    Reduce a mask array to a component's `foreach` dimensions.

    Any dimension not in `sets` is reduced with a boolean any-operation, and the
    result is transposed to the order given by `sets`.
    """
    unwanted_dims = set(mask.dims).difference(sets)
    return (mask.sum(unwanted_dims) > 0).astype(bool).transpose(*sets)


def _mask_is_empty(mask: xr.DataArray, name: str, reason: str) -> bool:
    """Return True (with a debug log) if `mask` leaves no valid data point."""
    if not mask.any():
        LOGGER.debug(f"Math parsing | {name} | Component not added; {reason}.")
        return True
    return False


def component_mask(
    group: COMPONENTS_T,
    name: str,
    definition: MATH_DEFS_T,
    mask_node: Node,
    ctx: Context,
    *,
    align_to_foreach_sets: bool = True,
) -> xr.DataArray:
    """
    Evaluate a component's top-level mask over its `foreach` dimensions.

    Combines the `foreach` existence array with the component's pre-parsed top-level `mask` AST, breaking early if no valid element remains.

    Parameters
    ----------
    group : COMPONENTS_T
        Component group the definition belongs to.
    name : str
        Name of the math component.
    definition : MATH_DEFS_T
        The component's (already schema-validated) definition.
    mask_node : Node
        The component's pre-parsed top-level `mask` AST
        (e.g. from :func:`parse_math` / :class:`ParsedMath`).
    ctx : Context
        Evaluation context.
    align_to_foreach_sets : bool, default: True
        If True, reduce the result to the `foreach` dimensions
        (see :func:`drop_dims_not_in_foreach`).
    """
    component_name = f"{group}:{name}"
    sets = tuple(definition.foreach)
    initial_mask = foreach_mask(sets, ctx.input_data)
    if _mask_is_empty(
        initial_mask, component_name, "'foreach' does not apply anywhere"
    ):
        return initial_mask

    mask_ctx = replace(ctx, mode="mask", equation_name=component_name)
    mask = xr.DataArray(initial_mask & mask_node.evaluate(mask_ctx))
    if _mask_is_empty(mask, component_name, "'mask' does not apply anywhere"):
        return mask

    if align_to_foreach_sets:
        mask = drop_dims_not_in_foreach(mask, sets)
    return mask


def _equation_ctx(
    equation: Equation,
    ctx: Context,
    mode: MODE_T,
    **kwargs: Any,
) -> Context:
    """Return a context copy carrying the equation's name and resolved references."""
    return replace(
        ctx,
        equation_name=equation.name,
        mode=mode,
        sub_expressions=equation.sub_expressions,
        slices=equation.slices,
        **kwargs,
    )


def as_mask(
    equation: Equation, ctx: Context, *, initial_mask: xr.DataArray = TRUE_ARRAY
) -> xr.DataArray:
    """
    Evaluate an equation's mask strings to a boolean array.

    Parameters
    ----------
    equation : Equation
        Parsed equation.
    ctx : Context
        Evaluation context.
    initial_mask : xr.DataArray, optional
        Mask to combine (boolean AND) with the equation's own masks.
        E.g., the component-level mask from :func:`component_mask`.

    Returns
    -------
    xr.DataArray
        Boolean array defining on which index items the equation applies.
    """
    mask_ctx = _equation_ctx(equation, ctx, "mask")
    evaluated = [mask.evaluate(mask_ctx) for mask in equation.masks]
    mask = xr.DataArray(functools.reduce(operator.and_, [initial_mask, *evaluated]))
    _mask_is_empty(mask, equation.name, "'mask' does not apply anywhere")
    return mask


def as_expression(
    equation: Equation, ctx: Context, *, mask: xr.DataArray = TRUE_ARRAY
) -> LinearExpression:
    """
    Evaluate an equation's arithmetic expression to a linopy expression.

    Parameters
    ----------
    equation : Equation
        Parsed equation (from an `expressions`/`objectives` component).
    ctx : Context
        Evaluation context.
    mask : xr.DataArray, optional
        Boolean array with which to mask the produced arrays.

    Returns
    -------
    LinearExpression
        The evaluated expression; a pure-parameter expression (evaluated to an
        `xr.DataArray`) is coerced to a `LinearExpression`.
    """
    expr_ctx = _equation_ctx(equation, ctx, "expr", mask=mask)
    return to_linexpr(equation.expression.evaluate(expr_ctx), ctx.model)


def as_constraint(
    equation: Equation, ctx: Context, *, mask: xr.DataArray = TRUE_ARRAY
) -> tuple[LinearExpression, xr.DataArray, LinearExpression]:
    """
    Evaluate an equation of the form `LHS OP RHS` to a constraint tuple.

    Parameters
    ----------
    equation : Equation
        Parsed equation (from a `constraints` component).
    ctx : Context
        Evaluation context.
    mask : xr.DataArray, optional
        Boolean array with which to mask the produced arrays.

    Returns
    -------
    tuple[LinearExpression, xr.DataArray, LinearExpression]
        `(lhs, sign, rhs)` for constraint assembly.
        `lhs` and `rhs` are coerced to `LinearExpression`.
        `sign` is an array of the comparison operator.
    """
    expr_ctx = _equation_ctx(equation, ctx, "expr", mask=mask)
    lhs, sign, rhs = equation.expression.evaluate(expr_ctx)
    return lhs, sign, rhs


def as_latex_mask(
    equation: Equation,
    ctx: Context,
) -> str:
    """
    Render an equation's mask as a LaTeX math string.

    Parameters
    ----------
    equation : Equation
        Parsed equation.
    ctx : Context
        Evaluation context.
    what : Literal["expression", "mask"], default: "expression"
        Whether to render the equation's expression (including an equation's comparison operator) or its combined mask conditions.

    Returns
    -------
    str
        A valid LaTeX math string.
    """
    mask_ctx = _equation_ctx(equation, ctx, "mask")
    strings = [mask.to_latex(mask_ctx) for mask in equation.masks]
    return r"\land{}".join(f"({s})" for s in strings if s != "true")


def as_latex_expression(
    equation: Equation,
    ctx: Context,
) -> str:
    """
    Render an equation's expression as a LaTeX math string.

    Parameters
    ----------
    equation : Equation
        Parsed equation.
    ctx : Context
        Evaluation context.

    Returns
    -------
    str
        A valid LaTeX math string.
    """
    expr_ctx = _equation_ctx(equation, ctx, "raw")
    return equation.expression.to_latex(expr_ctx)
