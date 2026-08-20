"""
Linopy declarative math AST module.

This module contains the AST node types produced when parsing declarative math strings.
For each node, this module contains the `pyparsing` parse action(s) that build it, the data evaluator, and the LaTeX math renderer.
The evaluation context and shared utilities live here too.
The pyparsing grammars that produce the nodes live in :mod:`linopy.declarative.grammar`.

This module is adapted from the calliope Apache-2.0 licensed math parsers:
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/expression_parser.py
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/where_parser.py

"""

from __future__ import annotations

import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pyparsing as pp
import xarray as xr

from linopy.declarative.helpers import (
    KIND_T,
    HelperFunction,
    _update_iterator,
    dim_iterator,
)
from linopy.declarative.schema import ConfigModel, MathModel
from linopy.expressions import LinearExpression
from linopy.variables import Variable

if TYPE_CHECKING:
    from linopy.model import Model

TRUE_ARRAY = xr.DataArray(True)

MODE_T = Literal["mask", "raw", "expr"]

COMPONENT_CATEGORY_T = Literal["any", "dimension", "input", "result"]

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

    mode: MODE_T = "raw"
    """Evaluation mode: "mask" (boolean-array route), "raw" (un-normalised
    data), or "expr" (results coerced to linopy expressions)."""

    apply_mask: bool = True
    """In mask mode, whether component references coerce to existence booleans."""

    sub_expressions: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` sub-expression AST per name."""

    slices: dict[str, Node] = field(default_factory=dict)
    """Resolved `$name` slicer AST per name."""

    mask: xr.DataArray = field(default_factory=lambda: TRUE_ARRAY)
    """Boolean array defining where the evaluated expression applies."""

    math_reprs: dict[str, str] = field(default_factory=dict)
    """Custom LaTeX representations per component name, taking precedence when
    rendering math strings (used by :class:`linopy.declarative.latex.LatexModelBuilder`)."""

    def demote(self) -> Context:
        """Copy in raw mode if in expr mode (helper args, slicers, list items)."""
        return replace(self, mode="raw") if self.mode == "expr" else self

    def helper_kind(self) -> KIND_T:
        """Return the helper-registry kind matching the evaluation mode."""
        return "mask" if self.mode == "mask" else "expression"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def error(ctx: Context, node: Node, message: str) -> ValueError:
    """Return a ValueError with the equation name and a caret at the node's position."""
    marker = " " * (pp.col(node.loc, node.instring) - 1) + "^"
    return ValueError(
        f"({ctx.equation_name}) | {message}\n  {node.instring}\n  {marker}"
    )


def _unwrap(value: Any) -> Any:
    """Extract the scalar from a dimensionless DataArray, e.g. for use in `.sel`/`.isin`."""
    return (
        value.item() if isinstance(value, xr.DataArray) and value.ndim == 0 else value
    )


def to_linexpr(obj: Any, model: Model) -> Any:
    """
    Normalise an evaluated object to a linopy expression.

    `Variable` objects are converted via `to_linexpr` and `xr.DataArray` objects
    are wrapped in a constant `LinearExpression`; everything else is returned
    unchanged. This is the single place where expression-route coercion happens.
    """
    if isinstance(obj, Variable):
        return obj.to_linexpr()
    if isinstance(obj, xr.DataArray):
        return LinearExpression(obj, model)
    return obj


def latex_number(value: float | int) -> str:
    r"""Format a number for LaTeX, mapping infinities to `\infty`."""
    if value == float("inf"):
        return r"\infty"
    if value == float("-inf"):
        return r"-\infty"
    return re.sub(
        r"([\d]+?)e([+-])([\d]+)", r"\1\\mathord{\\times}10^{\2\3}", f"{value:.6g}"
    )


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Node(ABC):
    """
    Base class of all declarative math AST nodes.

    Nodes are immutable data produced by the grammars in
    :mod:`linopy.declarative.grammar`; each concrete node carries its own parse
    action(s) (`from_tokens`), data evaluator (`evaluate`), and LaTeX renderer
    (`to_latex`).
    """

    instring: str = field(repr=False, compare=False)
    """The full source string this node was parsed from (used in error messages)."""

    loc: int = field(default=0, repr=False, compare=False)
    """Character offset of this node in `instring` (used in error messages)."""

    @abstractmethod
    def evaluate(self, ctx: Context) -> Any:
        """Evaluate this node to data (an `xr.DataArray` or a linopy expression)."""

    @abstractmethod
    def to_latex(self, ctx: Context) -> str:
        """Render this node as a LaTeX math string."""


@dataclass(frozen=True, kw_only=True)
class Constant(Node):
    """A literal number (including `inf`), boolean, or generic string."""

    value: float | bool | str

    @classmethod
    def number(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
        """Parse action for numeric literals."""
        return cls(value=float(tokens[0]), instring=instring, loc=loc)

    @classmethod
    def string(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
        """Parse action for generic string literals."""
        return cls(value=str(tokens[0]), instring=instring, loc=loc)

    @classmethod
    def boolean(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
        """Parse action for boolean literals."""
        return cls(value=str(tokens[0]).lower() == "true", instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate to a dimensionless array (or the plain string for string literals)."""
        if isinstance(self.value, bool):
            return xr.DataArray(np.bool_(self.value))
        if isinstance(self.value, str):
            return self.value
        return xr.DataArray(float(self.value), name=float(self.value))

    def to_latex(self, ctx: Context) -> str:
        """Render the literal value."""
        if isinstance(self.value, bool):
            return str(self.value).lower()
        if isinstance(self.value, str):
            return self.value
        return latex_number(float(self.value))


@dataclass(frozen=True, kw_only=True)
class ListNode(Node):
    """A literal list of items, e.g. `[a, b, 1]`."""

    items: tuple[Node, ...]

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> ListNode:
        """Parse action for `[item, item, ...]` lists."""
        return cls(items=tuple(tokens), instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> list[Any]:
        """Evaluate to a plain list of the evaluated items."""
        return [item.evaluate(ctx.demote()) for item in self.items]

    def to_latex(self, ctx: Context) -> str:
        """Render as a bracketed plain-text list."""
        return "[" + ",".join(_plain_string(item, ctx) for item in self.items) + "]"


@dataclass(frozen=True, kw_only=True)
class Component(Node):
    """
    A reference to a named math component (parameter, lookup, dimension, variable or expression).

    `category` records which name set matched during parsing: mask-string grammars
    distinguish dimensions / inputs / results at parse time, while expression
    grammars leave it as "any" (resolved from the math definition at evaluation).
    """

    name: str
    category: COMPONENT_CATEGORY_T = "any"

    @classmethod
    def from_tokens_as(
        cls, category: COMPONENT_CATEGORY_T
    ) -> Callable[[str, int, pp.ParseResults], Component]:
        """Return a parse action building a component reference of `category`."""

        def _action(instring: str, loc: int, tokens: pp.ParseResults) -> Component:
            return cls(
                name=str(tokens[0]), category=category, instring=instring, loc=loc
            )

        return _action

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the reference according to its category and the evaluation mode."""
        name = self.name
        if self.category == "dimension":
            # The mask string should evaluate successfully even if a dimension isn't defined.
            return ctx.input_data.get(name, xr.DataArray())
        if self.category == "input":
            da = ctx.input_data.get(name, xr.DataArray(False))
            if ctx.apply_mask and da.dtype.kind != "b":
                da = da.notnull() & (da != np.inf) & (da != -np.inf)
            elif da.isnull().any() and pd.notnull(
                default := ctx.math.find(name)["default"]
            ):
                da = da.fillna(default)
            return da
        if self.category == "result":
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
            if hasattr(math_def, "dims") and math_def.dims is not None:
                evaluated = evaluated.reindex(
                    {dim: ctx.input_data.coords[dim] for dim in math_def.dims},
                    fill_value=np.nan,
                )
        else:
            # Model entries (variables / expressions): a model entry that was never
            # built (e.g. skipped because its mask was empty) resolves to a NaN
            # expression rather than raising. In expr mode the object is
            # normalised to a linopy expression; in raw mode it is returned unchanged.
            try:
                evaluated = getattr(ctx.model, group)[name]
            except KeyError:
                return LinearExpression(xr.DataArray(np.nan), ctx.model)
            if ctx.mode != "expr":
                return evaluated
            evaluated = to_linexpr(evaluated, ctx.model)
        if evaluated.isnull().any() and pd.notna(default := math_def["default"]):
            evaluated = evaluated.fillna(default)
        return evaluated

    def to_latex(self, ctx: Context) -> str:
        r"""
        Render the component reference as LaTeX.

        A representation registered in `ctx.math_reprs` takes precedence (this is how
        :class:`linopy.declarative.latex.LatexModelBuilder` decorates references),
        followed by a `math_repr` data attribute, then the bare component name.
        """
        name = self.name
        custom = ctx.math_reprs.get(name)
        if self.category == "dimension":
            return name
        if self.category == "input":
            if custom is None:
                array = ctx.input_data.get(name)
                attr = array.attrs.get("math_repr") if array is not None else None
                custom = str(attr) if attr is not None else rf"\textit{{{name}}}"
            if ctx.apply_mask:
                custom = rf"\exists ({custom})"
            return custom
        if self.category == "result":
            if custom is not None:
                return rf"\exists ({custom})" if ctx.apply_mask else custom
            try:
                return str(
                    ctx.model[name].attrs.get(
                        "math_repr", rf"\exists (\textbf{{{name}}})"
                    )
                )
            except KeyError:
                return rf"\exists (\textbf{{{name}}})"
        if custom is not None:
            return custom
        evaluated = self.evaluate(ctx)
        attrs = getattr(evaluated, "attrs", {})
        return str(attrs.get("math_repr", name))


@dataclass(frozen=True, kw_only=True)
class Sliced(Node):
    """A sliced component, e.g. `flow[node=a]` or `flow[node=$n]`."""

    obj: Component
    slices: dict[str, Node]

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Sliced:
        """Parse action for `name[dim=slicer, ...]`."""
        slices = {str(grp["set_name"][0]): grp["slicer"][0] for grp in tokens["slices"]}
        return cls(obj=tokens["obj"], slices=slices, instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the component and select the evaluated slices from it."""
        slicer_ctx = ctx.demote()
        evaluated_slices = {
            dim: [_unwrap(i) for i in vals]
            if isinstance(vals := slicer.evaluate(slicer_ctx), list)
            else vals
            for dim, slicer in self.slices.items()
        }
        return self.obj.evaluate(ctx).sel(**evaluated_slices)

    def to_latex(self, ctx: Context) -> str:
        r"""
        Render the sliced component as LaTeX.

        If the component's LaTeX representation carries an iterator substring
        (e.g. `\textbf{flow}_\text{n}` from a `math_repr`), the slices are
        injected into it (`\textbf{flow}_\text{n=a}` when sliced with `node=a`);
        otherwise the slices are appended as a subscript.
        """
        slice_strings = {
            dim_iterator(ctx.math, dim): slicer.to_latex(ctx)
            for dim, slicer in self.slices.items()
        }
        obj_string = self.obj.to_latex(ctx)
        if re.search(r"_\\text\{", obj_string):
            return _update_iterator(
                obj_string, {it: f"={v}" for it, v in slice_strings.items()}, "add"
            )
        subscript = ",".join(f"{k}={v}" for k, v in slice_strings.items())
        return rf"{obj_string}_\text{{{subscript}}}"


@dataclass(frozen=True, kw_only=True)
class SliceRef(Node):
    """A `$name` reference to a named slicer, valid only inside slice brackets."""

    name: str

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> SliceRef:
        """Parse action for `$name` slicer references."""
        return cls(name=str(tokens[0]), instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the resolved slicer AST."""
        return ctx.slices[self.name].evaluate(ctx.demote())

    def to_latex(self, ctx: Context) -> str:
        """Render the resolved slicer AST."""
        return ctx.slices[self.name].to_latex(ctx)


@dataclass(frozen=True, kw_only=True)
class SubExprRef(Node):
    """A `$name` reference to a named sub-expression."""

    name: str

    @classmethod
    def from_tokens(
        cls, instring: str, loc: int, tokens: pp.ParseResults
    ) -> SubExprRef:
        """Parse action for `$name` sub-expression references."""
        return cls(name=str(tokens[0]), instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the resolved sub-expression AST."""
        return ctx.sub_expressions[self.name].evaluate(ctx)

    def to_latex(self, ctx: Context) -> str:
        """Render the resolved sub-expression AST."""
        return ctx.sub_expressions[self.name].to_latex(ctx)


@dataclass(frozen=True, kw_only=True)
class Call(Node):
    """A helper-function call, e.g. `sum(flow, over=node)`."""

    func: str
    args: tuple[Node, ...]
    kwargs: dict[str, Node]

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Call:
        """Parse action for `name(*args, **kwargs)` helper calls."""
        token_dict = tokens.as_dict()
        args = tuple(
            arg[0] if isinstance(arg, (pp.ParseResults, list)) else arg
            for arg in token_dict.get("args", [])
        )
        kwargs = {
            name: val[0] if isinstance(val, (pp.ParseResults, list)) else val
            for name, val in token_dict.get("kwargs", {}).items()
        }
        return cls(
            func=token_dict["func"],
            args=args,
            kwargs=kwargs,
            instring=instring,
            loc=loc,
        )

    def _helper(self, ctx: Context) -> type[HelperFunction]:
        """Return the helper class for this call, validating it exists in the registry."""
        helpers = ctx.helpers.get(ctx.helper_kind(), {})
        if self.func not in helpers:
            raise error(ctx, self, f"Invalid helper function defined: {self.func}")
        return helpers[self.func]

    def evaluate(self, ctx: Context) -> Any:
        """
        Evaluate the helper-function call.

        The helper dispatches to `as_expr` in expr mode and `as_raw` otherwise.
        Its arguments, however, are always evaluated in a demoted (raw) mode:
        helpers must receive un-normalised inputs (`xr.DataArray` for parameters/
        lookups/dimensions and the raw model object for variables/expressions)
        rather than values coerced to boolean masks or `LinearExpression`.
        """
        helper_cls = self._helper(ctx)
        helper = helper_cls(ctx)
        if helper_cls.ignore_mask:
            ctx = replace(ctx, mask=TRUE_ARRAY)
        arg_ctx = ctx.demote()
        args = [arg.evaluate(arg_ctx) for arg in self.args]
        kwargs = {name: val.evaluate(arg_ctx) for name, val in self.kwargs.items()}
        if ctx.mode == "expr":
            return helper.as_expr(*args, **kwargs)
        return helper.as_raw(*args, **kwargs)

    def to_latex(self, ctx: Context) -> str:
        """Render the call via the helper's `as_math_string`."""
        helper = self._helper(ctx)(ctx)
        args = [_call_arg_math_string(arg, ctx) for arg in self.args]
        kwargs = {
            name: _call_arg_math_string(val, ctx) for name, val in self.kwargs.items()
        }
        return helper.as_math_string(*args, **kwargs)


@dataclass(frozen=True, kw_only=True)
class Unary(Node):
    """A unary operation: leading `+`/`-` sign or boolean `not`."""

    op: str
    operand: Node

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Unary:
        """Parse action for unary `+`/`-`/`not` operations."""
        op, operand = tokens[0]
        return cls(op=str(op), operand=operand, instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the operand and apply the unary operator."""
        if self.op == "not":
            return ~self.operand.evaluate(ctx.demote())
        evaluated = self.operand.evaluate(ctx)
        return -1 * evaluated if self.op == "-" else evaluated

    def to_latex(self, ctx: Context) -> str:
        """Render the unary operation."""
        if self.op == "not":
            return rf"\neg ({self.operand.to_latex(ctx)})"
        return self.op + self.operand.to_latex(ctx)


@dataclass(frozen=True, kw_only=True)
class Arith(Node):
    """
    A chain of same-precedence binary operations.

    Covers arithmetic (`**`, `*`, `/`, `+`, `-`) and boolean (`and`, `or`)
    operator chains: `first OP operand OP operand ...`.
    """

    first: Node
    rest: tuple[tuple[str, Node], ...]

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Arith:
        """Parse action for infix operator chains."""
        items = tokens[0]
        rest = tuple(
            (str(op), operand)
            for op, operand in zip(items[1::2], items[2::2], strict=True)
        )
        return cls(first=items[0], rest=rest, instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate the operands (masked, unless boolean) and fold the operator chain."""
        boolean = self.rest[0][0] in ("and", "or")
        val = self.first.evaluate(ctx)
        if not boolean:
            val = val.where(ctx.mask)
        for op, operand in self.rest:
            evaluated = operand.evaluate(ctx)
            if not boolean:
                evaluated = evaluated.where(ctx.mask)
            val = _OPERATIONS[op](val, evaluated)
        return val

    def to_latex(self, ctx: Context) -> str:
        """Render the operator chain, skipping identity operands."""
        val = self.first.to_latex(ctx)
        for op, operand in self.rest:
            evaluated = operand.to_latex(ctx)
            # We ignore identity elements that do nothing (e.g. `0 + flow` is `flow`)
            if evaluated == _LATEX_IDENTITIES.get(op):
                continue
            if isinstance(self.first, Arith):
                val = f"({val})"
            if isinstance(operand, Arith):
                evaluated = f"({evaluated})"
            if val == _LATEX_IDENTITIES.get(op):
                val = evaluated
            else:
                val = _LATEX_OPERATORS[op].format(val=val, operand=evaluated)
        return val


@dataclass(frozen=True, kw_only=True)
class Compare(Node):
    """A comparison `lhs OP rhs`: an equation in expressions, a condition in masks."""

    lhs: Node
    op: str
    rhs: Node

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Compare:
        """Parse action for `lhs OP rhs` comparisons."""
        lhs, op, rhs = tokens
        return cls(lhs=lhs, op=str(op), rhs=rhs, instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """
        Evaluate the comparison.

        In expr mode, return a masked `(lhs, sign, rhs)` tuple for constraint
        assembly; otherwise return the boolean comparison array.
        """
        if ctx.mode == "expr":
            return self._evaluate_equation(ctx)
        unmasked_ctx = replace(ctx, apply_mask=False)
        comparison = _OPERATIONS[self.op](
            self.lhs.evaluate(unmasked_ctx), self.rhs.evaluate(unmasked_ctx)
        )
        return xr.DataArray(comparison)

    def _evaluate_equation(self, ctx: Context) -> tuple[Any, xr.DataArray, Any]:
        """Evaluate the equation to a masked `(lhs, sign, rhs)` tuple for constraint assembly."""
        lhs = self.lhs.evaluate(ctx)
        rhs = self.rhs.evaluate(ctx)
        for side, arr in {"left": lhs, "right": rhs}.items():
            extra_dims = set(arr.dims).difference(set(ctx.mask.dims) | {"_term"})
            if extra_dims:
                raise error(
                    ctx,
                    self,
                    f"The {side}-hand side of the equation is indexed over "
                    f"dimensions not present in `foreach`: {extra_dims}",
                )
        lhs_masked = to_linexpr(lhs.where(ctx.mask), ctx.model)
        rhs_masked = to_linexpr(rhs.where(ctx.mask), ctx.model)
        sign_masked = xr.DataArray(self.op).where(ctx.mask)
        return lhs_masked, sign_masked, rhs_masked

    def to_latex(self, ctx: Context) -> str:
        """Render the comparison with mask or equation operator tables per mode."""
        if ctx.mode == "mask":
            unmasked_ctx = replace(ctx, apply_mask=False)
            lhs_str = self.lhs.to_latex(unmasked_ctx)
            rhs_str = self.rhs.to_latex(unmasked_ctx)
            # Wrap plain-text tokens (coordinate labels, bare numbers, booleans) in `\text{}` so they render upright
            if "\\" not in rhs_str:
                rhs_str = rf"\text{{{rhs_str}}}"
            return lhs_str + _LATEX_MASK_OPERATORS[self.op] + rhs_str
        lhs_str = self.lhs.to_latex(ctx)
        rhs_str = self.rhs.to_latex(ctx)
        return lhs_str + _LATEX_EQUATION_OPERATORS[self.op] + rhs_str


@dataclass(frozen=True, kw_only=True)
class Subset(Node):
    """A dimension subset condition, e.g. `[a, b] in node`."""

    items: tuple[Node, ...]
    dim: Node

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> Subset:
        """Parse action for `[item, ...] in dim` subsets."""
        items, dim = tokens
        return cls(items=tuple(items), dim=dim, instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> Any:
        """Evaluate to a boolean array flagging dimension items in the subset."""
        subset = [_unwrap(item.evaluate(ctx)) for item in self.items]
        dim_array = self.dim.evaluate(replace(ctx, apply_mask=False))
        return dim_array.isin(subset)

    def to_latex(self, ctx: Context) -> str:
        """Render the subset condition."""
        subset = [_unwrap(item.evaluate(ctx)) for item in self.items]
        # Subsets can range over lookups as well as dimensions; dim_iterator
        # falls back to the plain name when there is no dimension iterator.
        dim_name = (
            self.dim.name if isinstance(self.dim, Component) else self.dim.to_latex(ctx)
        )
        iterator = dim_iterator(ctx.math, dim_name)
        subset_string = "[" + ",".join(str(i) for i in subset) + "]"
        return rf"\text{{{iterator}}} \in \text{{{subset_string}}}"


@dataclass(frozen=True, kw_only=True)
class ConfigRef(Node):
    """A reference to a build-configuration option, e.g. `config.foo`."""

    option: str

    @classmethod
    def from_tokens(cls, instring: str, loc: int, tokens: pp.ParseResults) -> ConfigRef:
        """Parse action for `config.option` references."""
        return cls(option=str(tokens[0]), instring=instring, loc=loc)

    def evaluate(self, ctx: Context) -> xr.DataArray:
        """Evaluate the config option to a dimensionless array."""
        try:
            config_val = getattr(ctx.config, self.option)
        except AttributeError:
            raise error(
                ctx, self, f"Unknown configuration option: {self.option}"
            ) from None
        if not isinstance(config_val, int | float | str | bool | np.bool_):
            raise error(
                ctx,
                self,
                f"Configuration option resolves to invalid type "
                f"`{type(config_val).__name__}`, expected a number, string, or boolean.",
            )
        return xr.DataArray(config_val)

    def to_latex(self, ctx: Context) -> str:
        """Render the config option reference."""
        return rf"\text{{config.{self.option}}}"


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def iter_nodes(node: Node) -> list[Node]:
    """
    Return `node` and all its descendant nodes, depth first.

    Parameters
    ----------
    node : Node
        Root of the (sub-)tree to walk.
    """
    found = [node]
    for f in fields(node):
        val = getattr(node, f.name)
        items: list = []
        if isinstance(val, Node):
            items = [val]
        elif isinstance(val, tuple):
            items = [
                i
                for pair in val
                for i in (pair if isinstance(pair, tuple) else (pair,))
            ]
        elif isinstance(val, dict):
            items = list(val.values())
        for item in items:
            if isinstance(item, Node):
                found.extend(iter_nodes(item))
    return found


def find_refs(node: Node, of_type: type[Node]) -> set[str]:
    """
    Return the names of all nodes of `of_type` found in the tree rooted at `node`.

    Parameters
    ----------
    node : Node
        Root of the (sub-)tree to search.
    of_type : type[Node]
        Node type with a `name` attribute to collect
        (e.g. :class:`SubExprRef`, :class:`SliceRef`, :class:`Component`).
    """
    return {n.name for n in iter_nodes(node) if isinstance(n, of_type)}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LaTeX helpers and functional walker wrappers
# ---------------------------------------------------------------------------


def _plain_string(item: Node, ctx: Context) -> str:
    """Return a plain-text representation of a list item for LaTeX rendering."""
    evaluated = item.evaluate(ctx.demote())
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
        return arg.evaluate(ctx)
    return arg.to_latex(ctx)


def evaluate(node: Node, ctx: Context) -> Any:
    """Evaluate a math AST node to data (see :meth:`Node.evaluate`)."""
    return node.evaluate(ctx)


def to_math_string(node: Node, ctx: Context) -> str:
    """Render a math AST node as a LaTeX math string (see :meth:`Node.to_latex`)."""
    return node.to_latex(ctx)
