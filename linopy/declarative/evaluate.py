"""
Linopy declarative math evaluation module.

This module contains the evaluation context and the two tree walkers that turn a
parsed math AST (see :mod:`linopy.declarative.grammar`) into either a LaTeX math
string (:func:`to_math_string`) or data — an `xr.DataArray` or a linopy
expression (:func:`evaluate`).
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pyparsing as pp
import xarray as xr

from linopy.declarative.grammar import (
    Arith,
    Call,
    Compare,
    Component,
    ConfigRef,
    Constant,
    ListNode,
    Node,
    Sliced,
    SliceRef,
    SubExprRef,
    Subset,
    Unary,
)
from linopy.declarative.helpers import KIND_T, HelperFunction, dim_iterator
from linopy.declarative.schema import ConfigModel, MathModel
from linopy.expressions import LinearExpression
from linopy.variables import Variable

if TYPE_CHECKING:
    from linopy.model import Model

TRUE_ARRAY = xr.DataArray(True)

ROUTE_T = Literal["expression", "mask"]

_INPUT_GROUPS = ("parameters", "lookups", "dimensions")

_OPERATIONS = {
    "**": operator.pow,
    "*": operator.mul,
    "/": operator.truediv,
    "+": operator.add,
    "-": operator.sub,
    "and": operator.and_,
    "or": operator.or_,
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}

_LATEX_OPERATORS = {
    "**": "{val}^{{{operand}}}",
    "*": r"{val} \times {operand}",
    "/": r"\frac{{ {val} }}{{ {operand} }}",
    "+": "{val} + {operand}",
    "-": "{val} - {operand}",
    "and": r"{val} \land {operand}",
    "or": r"{val} \lor {operand}",
}

_LATEX_IDENTITIES = {"+": "0", "-": "0", "and": "true", "or": "true"}
"""Operands that add nothing to a LaTeX operator chain (e.g. `0 + flow` is just `flow`)."""

_LATEX_EQUATION_OPERATORS = {"<=": r" \leq ", ">=": r" \geq ", "=": " = "}

_LATEX_MASK_OPERATORS = {
    "<=": r"\mathord{\leq}",
    ">=": r"\mathord{\geq}",
    "==": r"\mathord{==}",
    "<": r"\mathord{<}",
    ">": r"\mathord{>}",
}


@dataclass
class Context:
    """
    Context against which a parsed math AST is evaluated.

    The first five fields describe the model being built and are fixed for a
    whole build; the remainder are evaluation state, set per equation and only
    ever modified on immutable copies (via :func:`dataclasses.replace`).
    """

    model: Model
    """Linopy model containing any already-built variables and expressions."""

    input_data: xr.Dataset = field(default_factory=xr.Dataset)
    """Model input data (parameters, lookups, dimensions)."""

    math: MathModel = field(default_factory=MathModel)
    """Declarative math definition."""

    config: ConfigModel = field(default_factory=ConfigModel)
    """Build configuration options."""

    helpers: dict[KIND_T, dict[str, type[HelperFunction]]] = field(default_factory=dict)
    """Helper-function registry (see :func:`linopy.declarative.helpers.build_registry`)."""

    equation_name: str = ""
    """Name of the equation being evaluated (used in error messages)."""

    route: ROUTE_T = "expression"
    """Whether the AST being evaluated came from an expression or a mask string."""

    apply_mask: bool = True
    """On the mask route, whether component references coerce to existence booleans."""

    sub_expressions: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` sub-expression AST per name."""

    slices: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` slicer AST per name."""

    mask: xr.DataArray = field(default_factory=lambda: TRUE_ARRAY)
    """Boolean array defining where the evaluated expression applies."""

    math_reprs: dict[str, str] = field(default_factory=dict)
    """Custom LaTeX representations per component name, taking precedence when
    rendering math strings (used by :class:`linopy.declarative.latex.LatexModelBuilder`)."""


def error(ctx: Context, node: Node, message: str) -> ValueError:
    """Return a ValueError contextualised with the equation name and source string."""
    return ValueError(f"({ctx.equation_name}, {node.instring}) | {message}")


def get_dot_attr(var: Any, attr: str) -> Any:
    """
    Get a nested attribute in dot notation (e.g. "foo.bar").

    Works for nested objects: dictionaries, pydantic models, etc.
    """
    levels = attr.split(".", 1)
    value = var[levels[0]] if isinstance(var, dict) else getattr(var, levels[0])
    if len(levels) > 1:
        value = get_dot_attr(value, levels[1])
    return value


def _to_linexpr(obj: Any) -> Any:
    """
    Normalise a model object to a linopy expression.

    `Variable` objects are converted to `LinearExpression`; everything else is
    returned unchanged. This is the single place where the `Variable` ->
    `LinearExpression` coercion is performed on the expression route.
    """
    if isinstance(obj, Variable):
        return obj.to_linexpr()
    return obj


def _apply_mask(evaluated: Any, mask: xr.DataArray) -> Any:
    """Mask an evaluated operand, broadcasting first if it cannot be masked directly."""
    try:
        return evaluated.where(mask)
    except AttributeError:
        return evaluated.broadcast_like(mask).where(mask)


def _unwrap(value: Any) -> Any:
    """Extract the scalar from a dimensionless DataArray, e.g. for use in `.sel`/`.isin`."""
    return (
        value.item() if isinstance(value, xr.DataArray) and value.ndim == 0 else value
    )


# ---------------------------------------------------------------------------
# Data evaluation
# ---------------------------------------------------------------------------


def evaluate(node: Node, ctx: Context, expr: bool = False) -> Any:
    """
    Evaluate a math AST node to data.

    Parameters
    ----------
    node : Node
        AST node to evaluate.
    ctx : Context
        Evaluation context.
    expr : bool, default: False
        If False ("raw" mode), return the underlying data without route-specific
        transformation: an `xr.DataArray` for parameters/lookups/dimensions, the
        raw model object (`Variable`/`LinearExpression`) for model entries, and a
        boolean array on the mask route. If True ("expr" mode), guarantee a
        linopy-expression-compatible result: `Variable` objects are coerced to
        `LinearExpression` and a top-level :class:`Compare` returns a masked
        `(lhs, sign, rhs)` tuple for constraint assembly.
    """
    match node:
        case Constant(value=bool() as val):
            return xr.DataArray(np.bool_(val))
        case Constant(value=str() as val):
            return val
        case Constant(value=val):
            return xr.DataArray(float(val), name=float(val))
        case ListNode(items=items):
            return [evaluate(item, ctx) for item in items]
        case Component():
            return _evaluate_component(node, ctx, expr)
        case ConfigRef():
            return _evaluate_config(node, ctx)
        case SubExprRef(name=name):
            return evaluate(ctx.sub_expressions[name], ctx, expr)
        case SliceRef(name=name):
            return evaluate(ctx.slices[name], ctx)
        case Sliced(obj=obj, slices=slices):
            evaluated_slices = {
                dim: [_unwrap(i) for i in vals]
                if isinstance(vals := evaluate(slicer, ctx), list)
                else vals
                for dim, slicer in slices.items()
            }
            return evaluate(obj, ctx, expr).sel(**evaluated_slices)
        case Call():
            return _evaluate_call(node, ctx, expr)
        case Unary(op=op, operand=operand):
            if op == "not":
                return ~evaluate(operand, ctx)
            evaluated = evaluate(operand, ctx, expr)
            return -1 * evaluated if op == "-" else evaluated
        case Arith(first=first, rest=rest):
            boolean = rest[0][0] in ("and", "or")
            val = evaluate(first, ctx, expr)
            if not boolean:
                val = _apply_mask(val, ctx.mask)
            for op, operand in rest:
                evaluated = evaluate(operand, ctx, expr)
                if not boolean:
                    evaluated = _apply_mask(evaluated, ctx.mask)
                val = _OPERATIONS[op](val, evaluated)
            return val
        case Compare() if expr:
            return _evaluate_equation(node, ctx)
        case Compare(lhs=lhs, op=op, rhs=rhs):
            unmasked_ctx = replace(ctx, apply_mask=False)
            comparison = _OPERATIONS[op](
                evaluate(lhs, unmasked_ctx), evaluate(rhs, unmasked_ctx)
            )
            return xr.DataArray(comparison)
        case Subset(items=items, dim=dim):
            subset = [_unwrap(evaluate(item, ctx)) for item in items]
            dim_array = evaluate(dim, replace(ctx, apply_mask=False))
            return dim_array.isin(subset)
        case _:
            raise error(
                ctx, node, f"Cannot evaluate node of type {type(node).__name__}"
            )


def _evaluate_component(node: Component, ctx: Context, expr: bool) -> Any:
    """Evaluate a component reference according to its category and the evaluation mode."""
    name = node.name
    if node.category == "dimension":
        # The mask string should evaluate successfully even if a dimension isn't defined.
        return ctx.input_data.get(name, xr.DataArray())
    if node.category == "input":
        da = ctx.input_data.get(name, xr.DataArray(False))
        if ctx.apply_mask and da.dtype.kind != "b":
            da = da.notnull() & (da != np.inf) & (da != -np.inf)
        elif da.isnull().any() and pd.notnull(
            default := ctx.math.find(name)["default"]
        ):
            da = da.fillna(default)
        return da
    if node.category == "result":
        result: Any = ctx.model[name]
        if ctx.apply_mask:
            result = ~result.isnull()
        return result

    # category "any": resolve the component group from the math definition.
    math_def = ctx.math.find(name)
    group = math_def._group
    evaluated: Any
    if group in _INPUT_GROUPS:
        # A parameter/lookup/dimension defined in the math but absent from the
        # input data resolves to its default (NaN if none is set).
        evaluated = ctx.input_data.get(name, xr.DataArray(np.nan))
    else:
        # Model entries (variables / expressions): a model entry that was never
        # built (e.g. skipped because its mask was empty) resolves to a NaN
        # expression rather than raising. On the expression route the object is
        # normalised to a linopy expression; in raw mode it is returned unchanged.
        try:
            evaluated = getattr(ctx.model, group)[name]
        except KeyError:
            return LinearExpression(xr.DataArray(np.nan), ctx.model)
        if not expr:
            return evaluated
        evaluated = _to_linexpr(evaluated)
    if evaluated.isnull().any() and pd.notna(default := math_def["default"]):
        evaluated = evaluated.fillna(default)
    return evaluated


def _evaluate_config(node: ConfigRef, ctx: Context) -> xr.DataArray:
    """Evaluate a config option reference to a dimensionless array."""
    config_val = get_dot_attr(ctx.config, node.option)
    if not isinstance(config_val, int | float | str | bool | np.bool_):
        raise error(
            ctx,
            node,
            f"Configuration option resolves to invalid type "
            f"`{type(config_val).__name__}`, expected a number, string, or boolean.",
        )
    return xr.DataArray(config_val)


def _lookup_helper(node: Call, ctx: Context) -> type[HelperFunction]:
    """Return the helper class for a function call, validating it exists in the registry."""
    kind: KIND_T = "mask" if ctx.route == "mask" else "expression"
    helpers = ctx.helpers.get(kind, {})
    if node.func not in helpers:
        raise error(ctx, node, f"Invalid helper function defined: {node.func}")
    helper_cls = helpers[node.func]
    if not (isinstance(helper_cls, type) and issubclass(helper_cls, HelperFunction)):
        raise error(
            ctx,
            node,
            "Helper function must be subclassed from "
            f"linopy.declarative.helpers.HelperFunction: {node.func}",
        )
    return helper_cls


def _evaluate_call(node: Call, ctx: Context, expr: bool) -> Any:
    """
    Evaluate a helper-function call.

    The helper itself is instantiated with the enclosing mode so that
    expression-route helpers can dispatch to their `as_expr` implementation.
    Its arguments, however, are always evaluated in raw mode: helpers must
    receive un-normalised inputs (`xr.DataArray` for parameters/lookups/
    dimensions and the raw model object for variables/expressions) rather than
    values that have been coerced to boolean masks or `LinearExpression`.
    """
    helper_cls = _lookup_helper(node, ctx)
    helper = helper_cls("expr" if expr else "raw", ctx)
    if helper_cls.ignore_mask:
        ctx = replace(ctx, mask=TRUE_ARRAY)
    args = [evaluate(arg, ctx) for arg in node.args]
    kwargs = {name: evaluate(val, ctx) for name, val in node.kwargs.items()}
    return helper(*args, **kwargs)


def _evaluate_equation(node: Compare, ctx: Context) -> tuple[Any, xr.DataArray, Any]:
    """Evaluate an equation to a masked `(lhs, sign, rhs)` tuple for constraint assembly."""
    lhs = evaluate(node.lhs, ctx, expr=True)
    rhs = evaluate(node.rhs, ctx, expr=True)
    for side, arr in {"left": lhs, "right": rhs}.items():
        extra_dims = set(arr.dims).difference(set(ctx.mask.dims) | {"_term"})
        if extra_dims:
            raise error(
                ctx,
                node,
                f"The {side}-hand side of the equation is indexed over "
                f"dimensions not present in `foreach`: {extra_dims}",
            )
    lhs_masked = _to_linexpr(lhs.where(ctx.mask))
    rhs_masked = _to_linexpr(rhs.where(ctx.mask))
    sign_masked = xr.DataArray(node.op).where(ctx.mask)
    return lhs_masked, sign_masked, rhs_masked


# ---------------------------------------------------------------------------
# LaTeX math-string evaluation
# ---------------------------------------------------------------------------


def to_math_string(node: Node, ctx: Context) -> str:
    """
    Evaluate a math AST node to a LaTeX math string.

    Parameters
    ----------
    node : Node
        AST node to evaluate.
    ctx : Context
        Evaluation context.
    """
    match node:
        case Constant(value=bool() as val):
            return str(val).lower()
        case Constant(value=str() as val):
            return val
        case Constant(value=val):
            return re.sub(
                r"([\d]+?)e([+-])([\d]+)",
                r"\1\\mathord{\\times}10^{\2\3}",
                f"{float(val):.6g}",
            )
        case ListNode(items=items):
            return "[" + ",".join(_plain_string(item, ctx) for item in items) + "]"
        case Component():
            return _component_math_string(node, ctx)
        case ConfigRef(option=option):
            return rf"\text{{config.{option}}}"
        case SubExprRef(name=name):
            return to_math_string(ctx.sub_expressions[name], ctx)
        case SliceRef(name=name):
            return to_math_string(ctx.slices[name], ctx)
        case Sliced():
            return _sliced_math_string(node, ctx)
        case Call():
            helper = _lookup_helper(node, ctx)("math_string", ctx)
            args = [_call_arg_math_string(arg, ctx) for arg in node.args]
            kwargs = {
                name: _call_arg_math_string(val, ctx)
                for name, val in node.kwargs.items()
            }
            return helper(*args, **kwargs)
        case Unary(op="not", operand=operand):
            return rf"\neg ({to_math_string(operand, ctx)})"
        case Unary(op=op, operand=operand):
            return op + to_math_string(operand, ctx)
        case Arith(first=first, rest=rest):
            val = to_math_string(first, ctx)
            for op, operand in rest:
                evaluated = to_math_string(operand, ctx)
                # We ignore identity elements that do nothing (e.g. `0 + flow` is `flow`)
                if evaluated == _LATEX_IDENTITIES.get(op):
                    continue
                if isinstance(first, Arith):
                    val = f"({val})"
                if isinstance(operand, Arith):
                    evaluated = f"({evaluated})"
                if val == _LATEX_IDENTITIES.get(op):
                    val = evaluated
                else:
                    val = _LATEX_OPERATORS[op].format(val=val, operand=evaluated)
            return val
        case Compare(lhs=lhs, op=op, rhs=rhs) if ctx.route == "expression":
            lhs_str = to_math_string(lhs, ctx)
            rhs_str = to_math_string(rhs, ctx)
            return lhs_str + _LATEX_EQUATION_OPERATORS[op] + rhs_str
        case Compare(lhs=lhs, op=op, rhs=rhs):
            unmasked_ctx = replace(ctx, apply_mask=False)
            lhs_str = to_math_string(lhs, unmasked_ctx)
            rhs_str = to_math_string(rhs, unmasked_ctx)
            if r"\text" not in rhs_str:
                rhs_str = rf"\text{{{rhs_str}}}"
            return lhs_str + _LATEX_MASK_OPERATORS[op] + rhs_str
        case Subset(items=items, dim=dim):
            subset = [_unwrap(evaluate(item, ctx)) for item in items]
            # Subsets can range over lookups as well as dimensions; dim_iterator
            # falls back to the plain name when there is no dimension iterator.
            dim_name = (
                dim.name if isinstance(dim, Component) else to_math_string(dim, ctx)
            )
            iterator = dim_iterator(ctx.math, dim_name)
            subset_string = "[" + ",".join(str(i) for i in subset) + "]"
            return rf"\text{{{iterator}}} \in \text{{{subset_string}}}"
        case _:
            raise error(
                ctx, node, f"Cannot render node of type {type(node).__name__} as LaTeX"
            )


def _plain_string(item: Node, ctx: Context) -> str:
    """Return a plain-text representation of a list item for LaTeX rendering."""
    evaluated = evaluate(item, ctx)
    return (
        str(evaluated.name) if isinstance(evaluated, xr.DataArray) else str(evaluated)
    )


def _call_arg_math_string(arg: Node, ctx: Context) -> Any:
    """
    Evaluate one helper-function argument for LaTeX rendering.

    List arguments are passed as raw item lists (so helpers can extract names);
    all other arguments are passed as LaTeX strings.
    """
    if isinstance(arg, ListNode):
        return evaluate(arg, ctx)
    return to_math_string(arg, ctx)


def _component_math_string(node: Component, ctx: Context) -> str:
    """
    Render a component reference as LaTeX.

    A representation registered in `ctx.math_reprs` takes precedence (this is how
    :class:`linopy.declarative.latex.LatexModelBuilder` decorates references),
    followed by a `math_repr` data attribute, then the bare component name.
    """
    name = node.name
    custom = ctx.math_reprs.get(name)
    if node.category == "dimension":
        return name
    if node.category == "input":
        if custom is None:
            array = ctx.input_data.get(name)
            attr = array.attrs.get("math_repr") if array is not None else None
            custom = str(attr) if attr is not None else rf"\textit{{{name}}}"
        if ctx.apply_mask:
            custom = rf"\exists ({custom})"
        return custom
    if node.category == "result":
        if custom is not None:
            return rf"\exists ({custom})" if ctx.apply_mask else custom
        try:
            return str(
                ctx.model[name].attrs.get("math_repr", rf"\exists (\textbf{{{name}}})")
            )
        except KeyError:
            return rf"\exists (\textbf{{{name}}})"
    if custom is not None:
        return custom
    evaluated = evaluate(node, ctx)
    attrs = getattr(evaluated, "attrs", {})
    return str(attrs.get("math_repr", name))


def _sliced_math_string(node: Sliced, ctx: Context) -> str:
    r"""
    Render a sliced component as LaTeX.

    If the component's LaTeX representation carries an iterator substring (from a
    `math_repr` data attribute, e.g. `\textbf{flow}_\text{n}`), the slices are
    injected into it by re-parsing (e.g. `\textbf{flow}_\text{n=a}` when sliced
    with `node=a`); otherwise the slices are appended as a subscript.
    """
    slice_strings = {
        dim_iterator(ctx.math, dim): to_math_string(slicer, ctx)
        for dim, slicer in node.slices.items()
    }
    obj_string = to_math_string(node.obj, ctx)

    def _replace(term: pp.ParseResults) -> Any:
        if len(term) == 1:
            return term
        replacers = {k: f"{k}={v}" for k, v in slice_strings.items()}
        return (
            term[0] + term[1] + ",".join(replacers.get(k, k) for k in term[2]) + term[3]
        )

    id_ = pp.Combine(
        pp.Word(pp.alphas, pp.alphanums)
        + pp.ZeroOrMore("_" + pp.Word(pp.alphanums))
        + pp.Opt("_")
    )
    id_formatted = pp.Combine("\\" + pp.Word(pp.alphas) + "{" + id_ + "}")
    obj_parser = id_formatted + pp.Opt(
        r"_\text{" + pp.Group(pp.DelimitedList(id_)) + "}"
    )
    obj_parser.set_parse_action(_replace)
    try:
        return obj_parser.parse_string(obj_string, parse_all=True)[0]
    except pp.ParseException:
        subscript = ",".join(f"{k}={v}" for k, v in slice_strings.items())
        return rf"{obj_string}_\text{{{subscript}}}"
