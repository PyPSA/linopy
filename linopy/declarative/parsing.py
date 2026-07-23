"""
Linopy declarative math parsing module.

This module turns a validated math component definition into a list of
:class:`Equation` objects — pure data holding the parsed expression/mask ASTs
with all `$name` sub-expression and slicer references resolved — and provides
the typed entry points that evaluate an equation to a boolean mask array, a
linopy expression, a constraint tuple, or a LaTeX math string.
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
from linopy.declarative.evaluate import (
    TRUE_ARRAY,
    Context,
    evaluate,
    to_math_string,
)
from linopy.declarative.grammar import Node
from linopy.declarative.schema import (
    MATH_DEFS_T,
    ConstraintDef,
    ExpressionDef,
    MathModel,
    ObjectiveDef,
    _Equations,
)
from linopy.expressions import LinearExpression

LOGGER = logging.getLogger(__name__)

GROUP_T = Literal[
    "variables",
    "expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
    "postprocessed",
]

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
        return set().union(
            *(grammar.find_refs(tree, grammar.Component) for tree in trees)
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


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


class _ErrorCollector:
    """Collect parse errors with their positions, to raise a single error at the end."""

    def __init__(self, component_name: str) -> None:
        self.component_name = component_name
        self.errors: list[str] = []

    def parse(
        self, parser: pp.ParserElement, string: str, position: str
    ) -> Node | None:
        """
        Parse `string`, returning its AST root or None if parsing fails.

        Failures are stored with a caret marker pointing at the parse position,
        for raising later via :meth:`raise_errors`.
        """
        try:
            return parser.parse_string(string, parse_all=True)[0]
        except pp.ParseException as excinfo:
            pointer = f"{position} (line {excinfo.lineno}, char {excinfo.col}): "
            marker_pos = " " * (len(pointer) + 2 * len(_ERR_BULLET) + excinfo.col - 1)
            self.errors.append(f"{pointer}{excinfo.line}\n{marker_pos}^")
            return None

    def raise_errors(self) -> None:
        """Raise all collected parse errors as a single bullet-point ValueError."""
        if self.errors:
            raise ValueError(f"- {self.component_name}: {self.errors}")


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
    collector = _ErrorCollector(name)
    parsed = collector.parse(_mask_grammar(math), mask_string, "mask")
    collector.raise_errors()
    assert parsed is not None
    return parsed


def _parse_variants(
    collector: _ErrorCollector,
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
        mask = collector.parse(mask_parser, item.mask, f"{position_id}.mask")
        expression = collector.parse(
            parser, item.expression, f"{position_id}.expression"
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
) -> list[Equation]:
    """
    Expand equations with all combinations of their referenced `$name` variants.

    Each `$name` reference maps to a list of `{mask, expression}` variants; an
    equation referencing them is replaced by one equation per element of the
    cartesian product of those variant lists, with the chosen variants' masks and
    ASTs merged in.
    """
    expanded = []
    for equation in equations:
        ref_type = grammar.SubExprRef if kind == "sub_expressions" else grammar.SliceRef
        trees = [equation.expression, *equation.sub_expressions.values()]
        refs = set().union(*(grammar.find_refs(tree, ref_type) for tree in trees))
        if not refs:
            expanded.append(equation)
            continue
        undefined = refs.difference(candidates.keys())
        if undefined:
            raise KeyError(
                f"{component_name}: Undefined {kind} found in equation: {undefined}"
            )
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
    group: GROUP_T, name: str, definition: EQUATION_DEFS_T, math: MathModel
) -> list[Equation]:
    """
    Parse a math component's equations into fully-resolved :class:`Equation` objects.

    All `expression` and `mask` strings of the component's equations,
    sub-expressions, and slicers are parsed (syntax errors across all of them are
    collected and raised together), then every equation is expanded with the
    cartesian product of the sub-expression and slicer variants it references.

    Parameters
    ----------
    group : GROUP_T
        Component group the definition belongs to (defines the equation grammar:
        comparisons for constraints, arithmetic otherwise).
    name : str
        Name of the math component.
    definition : EQUATION_DEFS_T
        The component's (already schema-validated) definition.
    math : MathModel
        The full math definition, providing valid component names.

    Returns
    -------
    list[Equation]
        One equation per user-defined equation and referenced variant combination.
    """
    component_name = f"{group}:{name}"
    names = _expression_names(math)
    equation_parser = (
        grammar.equation_grammar(names)
        if group == "constraints"
        else grammar.arithmetic_grammar(names)
    )
    mask_parser = _mask_grammar(math)
    # Objectives are adimensional: they carry no `foreach` key.
    sets = tuple(getattr(definition, "foreach", ()))
    collector = _ErrorCollector(component_name)

    equations = _parse_variants(
        collector,
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
            grammar.slice_grammar(names),
            mask_parser,
            slice_list,
            sets,
            f"slices.{slice_name}",
            slice_name,
        )
        for slice_name, slice_list in definition.slices.root.items()
    }
    collector.raise_errors()

    equations = _expand(component_name, equations, sub_expressions, "sub_expressions")
    return _expand(component_name, equations, slices, "slices")


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


def component_mask(
    group: GROUP_T,
    name: str,
    definition: MATH_DEFS_T,
    ctx: Context,
    *,
    align_to_foreach_sets: bool = True,
) -> xr.DataArray:
    """
    Evaluate a component's top-level mask over its `foreach` dimensions.

    Combines the `foreach` existence array with the component's (optional)
    top-level `mask` string, breaking early if no valid element remains.

    Parameters
    ----------
    group : GROUP_T
        Component group the definition belongs to.
    name : str
        Name of the math component.
    definition : MATH_DEFS_T
        The component's (already schema-validated) definition.
    ctx : Context
        Evaluation context.
    align_to_foreach_sets : bool, default: True
        If True, reduce the result to the `foreach` dimensions
        (see :func:`drop_dims_not_in_foreach`).
    """
    component_name = f"{group}:{name}"
    # Objectives are adimensional: they carry no `foreach` or `mask` keys.
    sets = tuple(getattr(definition, "foreach", ()))
    mask_string = getattr(definition, "mask", "True")
    initial_mask = foreach_mask(sets, ctx.input_data)
    if not initial_mask.any():
        LOGGER.debug(
            f"Math parsing | {component_name} | Component not added; "
            "'foreach' does not apply anywhere."
        )
        return initial_mask

    mask_node = parse_mask(mask_string, ctx.math, component_name)
    mask_ctx = replace(ctx, route="mask", equation_name=component_name)
    mask = xr.DataArray(initial_mask & evaluate(mask_node, mask_ctx))
    if not mask.any():
        LOGGER.debug(
            f"Math parsing | {component_name} | Component not added; "
            "'mask' does not apply anywhere."
        )
        return mask

    if align_to_foreach_sets:
        mask = drop_dims_not_in_foreach(mask, sets)
    return mask


# ---------------------------------------------------------------------------
# Typed evaluation entry points
# ---------------------------------------------------------------------------


def _equation_ctx(
    equation: Equation,
    ctx: Context,
    route: Literal["expression", "mask"],
    **kwargs: Any,
) -> Context:
    """Return a context copy carrying the equation's name and resolved references."""
    return replace(
        ctx,
        equation_name=equation.name,
        route=route,
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
        Mask to combine (boolean AND) with the equation's own masks, e.g. the
        component-level mask from :func:`component_mask`.

    Returns
    -------
    xr.DataArray
        Boolean array defining on which index items the equation applies.
    """
    mask_ctx = _equation_ctx(equation, ctx, "mask")
    evaluated = [evaluate(mask, mask_ctx) for mask in equation.masks]
    mask = xr.DataArray(functools.reduce(operator.and_, [initial_mask, *evaluated]))
    if not mask.any():
        LOGGER.debug(
            f"Math parsing | {equation.name} | Component not added; "
            "'mask' does not apply anywhere."
        )
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
    expr_ctx = _equation_ctx(equation, ctx, "expression", mask=mask)
    evaluated = evaluate(equation.expression, expr_ctx, expr=True)
    if isinstance(evaluated, xr.DataArray):
        evaluated = LinearExpression(evaluated, ctx.model)
    return evaluated


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
        `(lhs, sign, rhs)` for constraint assembly; pure-parameter sides are
        coerced to `LinearExpression` and `sign` is an array of the comparison
        operator.
    """
    expr_ctx = _equation_ctx(equation, ctx, "expression", mask=mask)
    lhs, sign, rhs = evaluate(equation.expression, expr_ctx, expr=True)
    if isinstance(lhs, xr.DataArray):
        lhs = LinearExpression(lhs, ctx.model)
    if isinstance(rhs, xr.DataArray):
        rhs = LinearExpression(rhs, ctx.model)
    return lhs, sign, rhs


def as_latex(
    equation: Equation,
    ctx: Context,
    *,
    what: Literal["expression", "mask"] = "expression",
) -> str:
    """
    Render an equation's expression or mask as a LaTeX math string.

    Parameters
    ----------
    equation : Equation
        Parsed equation.
    ctx : Context
        Evaluation context.
    what : Literal["expression", "mask"], default: "expression"
        Whether to render the equation's expression (including an equation's
        comparison operator) or its combined mask conditions.

    Returns
    -------
    str
        A valid LaTeX math string.
    """
    if what == "mask":
        mask_ctx = _equation_ctx(equation, ctx, "mask")
        strings = [to_math_string(mask, mask_ctx) for mask in equation.masks]
        return r"\land{}".join(f"({s})" for s in strings if s != "true")
    expr_ctx = _equation_ctx(equation, ctx, "expression")
    return to_math_string(equation.expression, expr_ctx)


def check_mask_expr_consistency(
    name: str, expression: xr.DataArray, mask: xr.DataArray
) -> None:
    """
    Check that an evaluated expression is consistent with its mask array.

    Parameters
    ----------
    name : str
        Name to identify the equation by in error messages.
    expression : xr.DataArray
        Array of linear expressions or one side of a constraint equation.
    mask : xr.DataArray
        Boolean mask; there should be a valid expression value wherever it is True.

    Raises
    ------
    ValueError
        If the expression is indexed over dimensions not present in the mask, or
        has missing (NaN) entries where the mask applies.
    """
    broadcast_dims_mask = set(expression.dims).difference(set(mask.dims))
    if broadcast_dims_mask:
        raise ValueError(
            f"{name} | The linear expression array is indexed over dimensions "
            f"not present in `foreach`: {broadcast_dims_mask}"
        )
    incomplete_constraints = expression.isnull() & mask
    if incomplete_constraints.any():
        raise ValueError(
            f"{name} | Missing a linear expression for some coordinates selected "
            "by 'mask'. Adapting 'mask' might help."
        )
