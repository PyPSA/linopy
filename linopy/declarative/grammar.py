"""
Linopy declarative math grammar module.

This module contains the AST node types produced when parsing declarative math
strings, and the pyparsing grammars that produce them. Nodes are pure data;
all evaluation logic lives in :mod:`linopy.declarative.evaluate`.

The infix-notation grammar structure is adapted from the pyparsing `eval_arith.py`
example (https://github.com/pyparsing/pyparsing/blob/master/examples/eval_arith.py,
MIT licensed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import cache
from typing import Literal

import pyparsing as pp

pp.ParserElement.enable_packrat()

REFERENCE_CLASSIFIER = "$"
"""Prefix marking a reference to a sub-expression (`$foo`) or slicer (`x[dim=$foo]`)."""

EQUATION_OPERATORS = ("<=", ">=", "=")
"""Comparison operators allowed in constraint equations."""

MASK_OPERATORS = ("<", ">", "==", ">=", "<=")
"""Comparison operators allowed in mask strings."""

COMPONENT_CATEGORY_T = Literal["any", "dimension", "input", "result"]


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Node:
    """
    Base class of all declarative math AST nodes.

    Nodes are immutable data produced by the grammars in this module and consumed
    by the walkers in :mod:`linopy.declarative.evaluate`.
    """

    instring: str = field(repr=False, compare=False)
    """The full source string this node was parsed from (used in error messages)."""


@dataclass(frozen=True, kw_only=True)
class Constant(Node):
    """A literal number (including `inf`), boolean, or generic string."""

    value: float | bool | str


@dataclass(frozen=True, kw_only=True)
class ListNode(Node):
    """A literal list of items, e.g. `[a, b, 1]`."""

    items: tuple[Node, ...]


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


@dataclass(frozen=True, kw_only=True)
class Sliced(Node):
    """A sliced component, e.g. `flow[node=a]` or `flow[node=$n]`."""

    obj: Component
    slices: dict[str, Node]


@dataclass(frozen=True, kw_only=True)
class SliceRef(Node):
    """A `$name` reference to a named slicer, valid only inside slice brackets."""

    name: str


@dataclass(frozen=True, kw_only=True)
class SubExprRef(Node):
    """A `$name` reference to a named sub-expression."""

    name: str


@dataclass(frozen=True, kw_only=True)
class Call(Node):
    """A helper-function call, e.g. `sum(flow, over=node)`."""

    func: str
    args: tuple[Node, ...]
    kwargs: dict[str, Node]


@dataclass(frozen=True, kw_only=True)
class Unary(Node):
    """A unary operation: leading `+`/`-` sign or boolean `not`."""

    op: str
    operand: Node


@dataclass(frozen=True, kw_only=True)
class Arith(Node):
    """
    A chain of same-precedence binary operations.

    Covers arithmetic (`**`, `*`, `/`, `+`, `-`) and boolean (`and`, `or`)
    operator chains: `first OP operand OP operand ...`.
    """

    first: Node
    rest: tuple[tuple[str, Node], ...]


@dataclass(frozen=True, kw_only=True)
class Compare(Node):
    """A comparison `lhs OP rhs`: an equation in expressions, a condition in masks."""

    lhs: Node
    op: str
    rhs: Node


@dataclass(frozen=True, kw_only=True)
class Subset(Node):
    """A dimension subset condition, e.g. `[a, b] in node`."""

    items: tuple[Node, ...]
    dim: Node


@dataclass(frozen=True, kw_only=True)
class ConfigRef(Node):
    """A reference to a build-configuration option, e.g. `config.foo`."""

    option: str


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
# Parse actions
# ---------------------------------------------------------------------------


def _number_action(instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
    return Constant(value=float(tokens[0]), instring=instring)


def _string_action(instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
    return Constant(value=str(tokens[0]), instring=instring)


def _bool_action(instring: str, loc: int, tokens: pp.ParseResults) -> Constant:
    return Constant(value=str(tokens[0]).lower() == "true", instring=instring)


def _list_action(instring: str, loc: int, tokens: pp.ParseResults) -> ListNode:
    return ListNode(items=tuple(tokens), instring=instring)


def _component_action(
    category: COMPONENT_CATEGORY_T,
) -> Callable[[str, int, pp.ParseResults], Component]:
    def _action(instring: str, loc: int, tokens: pp.ParseResults) -> Component:
        return Component(name=str(tokens[0]), category=category, instring=instring)

    return _action


def _sliced_action(instring: str, loc: int, tokens: pp.ParseResults) -> Sliced:
    slices = {str(grp["set_name"][0]): grp["slicer"][0] for grp in tokens["slices"]}
    return Sliced(obj=tokens["obj"], slices=slices, instring=instring)


def _slice_ref_action(instring: str, loc: int, tokens: pp.ParseResults) -> SliceRef:
    return SliceRef(name=str(tokens[0]), instring=instring)


def _sub_expr_ref_action(
    instring: str, loc: int, tokens: pp.ParseResults
) -> SubExprRef:
    return SubExprRef(name=str(tokens[0]), instring=instring)


def _call_action(instring: str, loc: int, tokens: pp.ParseResults) -> Call:
    token_dict = tokens.as_dict()
    args = tuple(
        arg[0] if isinstance(arg, (pp.ParseResults, list)) else arg
        for arg in token_dict.get("args", [])
    )
    kwargs = {
        name: val[0] if isinstance(val, (pp.ParseResults, list)) else val
        for name, val in token_dict.get("kwargs", {}).items()
    }
    return Call(func=token_dict["func"], args=args, kwargs=kwargs, instring=instring)


def _unary_action(instring: str, loc: int, tokens: pp.ParseResults) -> Unary:
    op, operand = tokens[0]
    return Unary(op=str(op), operand=operand, instring=instring)


def _arith_action(instring: str, loc: int, tokens: pp.ParseResults) -> Arith:
    items = tokens[0]
    rest = tuple(
        (str(op), operand) for op, operand in zip(items[1::2], items[2::2], strict=True)
    )
    return Arith(first=items[0], rest=rest, instring=instring)


def _compare_action(instring: str, loc: int, tokens: pp.ParseResults) -> Compare:
    lhs, op, rhs = tokens
    return Compare(lhs=lhs, op=str(op), rhs=rhs, instring=instring)


def _subset_action(instring: str, loc: int, tokens: pp.ParseResults) -> Subset:
    items, dim = tokens
    return Subset(items=tuple(items), dim=dim, instring=instring)


def _config_action(instring: str, loc: int, tokens: pp.ParseResults) -> ConfigRef:
    return ConfigRef(option=str(tokens[0]), instring=instring)


# ---------------------------------------------------------------------------
# Grammar primitives
# ---------------------------------------------------------------------------


def _base_elements() -> tuple[pp.ParserElement, pp.ParserElement]:
    """Return the (number, identifier) primitives shared by all grammars."""
    inf_kw = pp.Combine(pp.Opt(pp.Suppress(".")) + pp.Keyword("inf", caseless=True))
    number = (pp.pyparsing_common.number | inf_kw).set_parse_action(_number_action)
    identifier = ~inf_kw + pp.Word(pp.alphas, pp.alphanums + "_")
    return number, identifier


def _names_parser(names: frozenset[str]) -> pp.ParserElement:
    """Return a keyword parser matching any of `names` (or nothing if empty)."""
    if not names:
        return pp.NoMatch()
    return pp.one_of(sorted(names), as_keyword=True)


def _component_parser(
    names: frozenset[str], category: COMPONENT_CATEGORY_T = "any"
) -> pp.ParserElement:
    """Return a parser matching any name in `names` as a :class:`Component`."""
    return _names_parser(names).set_parse_action(_component_action(category))


def _string_parser(
    identifier: pp.ParserElement, excluded_names: frozenset[str]
) -> pp.ParserElement:
    """Return a parser for generic strings that are not in `excluded_names`."""
    return (~_names_parser(excluded_names) + identifier).set_parse_action(
        _string_action
    )


def _list_parser(*items: pp.ParserElement) -> pp.ParserElement:
    """Return a parser for `[item, item, ...]` lists of the given item parsers."""
    element = pp.MatchFirst(items)
    id_list = pp.Suppress("[") + pp.DelimitedList(element) + pp.Suppress("]")
    return id_list.set_parse_action(_list_action)


def _call_parser(
    *args: pp.ParserElement,
    identifier: pp.ParserElement,
    allow_nested_calls: bool = False,
) -> pp.ParserElement:
    """
    Return a parser for helper-function calls `name(*args, **kwargs)`.

    Parameters
    ----------
    *args : pp.ParserElement
        Parsers for allowed argument values, matched in the given order.
    identifier : pp.ParserElement
        Parser for the function name (no parse action attached).
    allow_nested_calls : bool, default: False
        If True, calls may appear directly as arguments of other calls.
        (Calls nested via arithmetic are enabled by passing the arithmetic
        parser in `args` instead.)
    """
    call = pp.Forward()
    allowed_args = list(args)
    if allow_nested_calls:
        allowed_args.insert(0, call)

    func_name = pp.Combine(identifier + pp.Suppress("("))("func")
    arg_value = pp.MatchFirst(allowed_args) + pp.NotAny("=")
    arg_list = pp.Group(pp.DelimitedList(arg_value.copy()))("args")
    key = identifier + pp.Suppress("=")
    kwarg_list = pp.Group(pp.DelimitedList(pp.dict_of(key, arg_value)))("kwargs")
    call_args = arg_list + pp.Suppress(",") + kwarg_list | pp.Opt(
        arg_list, default=[]
    ) + pp.Opt(kwarg_list, default={})

    call <<= func_name + call_args + pp.Suppress(")")
    return call.set_parse_action(_call_action)


def _sliced_component_parser(
    slicers: list[pp.ParserElement],
    identifier: pp.ParserElement,
    component: pp.ParserElement,
    allow_slice_references: bool = True,
) -> pp.ParserElement:
    """
    Return a parser for sliced components `name[dim=slicer, ...]`.

    Parameters
    ----------
    slicers : list[pp.ParserElement]
        Parsers for allowed slicer values, matched in the given order.
    identifier : pp.ParserElement
        Parser for the slice dimension name (no parse action attached).
    component : pp.ParserElement
        Parser for the sliced component name.
    allow_slice_references : bool, default: True
        If True, allow `$name` slicer references (e.g. `$bar` in `foo[bars=$bar]`).
    """
    slicer: pp.ParserElement = pp.MatchFirst(slicers)
    if allow_slice_references:
        slice_ref = pp.Suppress(REFERENCE_CLASSIFIER) + identifier
        slice_ref.set_parse_action(_slice_ref_action)
        slicer = slice_ref | slicer

    one_slice = pp.Group(
        identifier("set_name") + pp.Suppress("=") + pp.Group(slicer)("slicer")
    )
    slices = pp.Group(pp.DelimitedList(one_slice))("slices")
    sliced = pp.Combine(component("obj") + pp.Suppress("[")) + slices + pp.Suppress("]")
    return sliced.set_parse_action(_sliced_action)


def _sub_expression_ref_parser(identifier: pp.ParserElement) -> pp.ParserElement:
    """Return a parser for `$name` sub-expression references."""
    ref = pp.Combine(pp.Suppress(REFERENCE_CLASSIFIER) + identifier)
    return ref.set_parse_action(_sub_expr_ref_action)


def _arithmetic_rules(
    *operands: pp.ParserElement, arithmetic: pp.Forward | None = None
) -> pp.Forward:
    """
    Return an infix-notation parser combining `operands` with `+ - * / **`.

    Parameters
    ----------
    *operands : pp.ParserElement
        Parsers for allowed operands, matched in the given order.
    arithmetic : pp.Forward, optional
        If given, attach the rules to this existing forward-declared rule so
        that operands (e.g. function calls) can recursively contain arithmetic.
    """
    signop = pp.one_of(["+", "-"])
    multop = pp.one_of(["*", "/"])
    expop = pp.Literal("**")
    if arithmetic is None:
        arithmetic = pp.Forward()
    arithmetic <<= pp.infix_notation(
        # the order matters if two could capture the same string, e.g. "inf".
        pp.MatchFirst(operands),
        [
            (signop, 1, pp.opAssoc.RIGHT, _unary_action),
            (expop, 2, pp.opAssoc.LEFT, _arith_action),
            (multop, 2, pp.opAssoc.LEFT, _arith_action),
            (signop, 2, pp.opAssoc.LEFT, _arith_action),
        ],
    )
    return arithmetic


# ---------------------------------------------------------------------------
# Grammar entry points
# ---------------------------------------------------------------------------


@cache
def slice_grammar(component_names: frozenset[str]) -> pp.ParserElement:
    """
    Return the grammar for named slicer expressions.

    Slicers are linked into equations by `$name` references inside slice brackets
    (e.g. `$bar` in `foo[bars=$bar]`). Unlike sub-expressions and equations,
    slicer strings allow neither arithmetic nor references to other slicers.

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    """
    number, identifier = _base_elements()
    string = _string_parser(identifier, component_names)
    component = _component_parser(component_names)
    call_list = _list_parser(number, string, component)
    slicer_list = _list_parser(number, string)
    sliced = _sliced_component_parser(
        [number, string, slicer_list],
        identifier,
        component,
        allow_slice_references=False,
    )
    call = _call_parser(
        sliced,
        component,
        number,
        call_list,
        string,
        identifier=identifier,
        allow_nested_calls=True,
    )
    return call | sliced | component | number | slicer_list | string


@cache
def sub_expression_grammar(component_names: frozenset[str]) -> pp.Forward:
    """
    Return the grammar for named sub-expressions.

    Sub-expressions are linked into equations by `$name` references. They allow
    arbitrarily nested arithmetic and function calls and `$name` slicer
    references, but no references to other sub-expressions.

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    """
    number, identifier = _base_elements()
    string = _string_parser(identifier, component_names)
    component = _component_parser(component_names)
    call_list = _list_parser(number, string, component)
    slicer_list = _list_parser(number, string)
    sliced = _sliced_component_parser(
        [number, string, slicer_list], identifier, component
    )
    arithmetic = pp.Forward()
    call = _call_parser(arithmetic, call_list, string, identifier=identifier)
    return _arithmetic_rules(call, sliced, number, component, arithmetic=arithmetic)


@cache
def arithmetic_grammar(component_names: frozenset[str]) -> pp.Forward:
    """
    Return the grammar for arithmetic expressions (`+ - * / **`).

    Allows arbitrarily nested arithmetic and function calls, and references to
    sub-expressions (`$name`) and slicers (`x[dim=$name]`).

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    """
    number, identifier = _base_elements()
    string = _string_parser(identifier, component_names)
    component = _component_parser(component_names)
    call_list = _list_parser(number, string, component)
    slicer_list = _list_parser(number, string)
    sliced = _sliced_component_parser(
        [number, string, slicer_list], identifier, component
    )
    sub_expression = _sub_expression_ref_parser(identifier)
    arithmetic = pp.Forward()
    call = _call_parser(arithmetic, call_list, string, identifier=identifier)
    return _arithmetic_rules(
        call, sub_expression, sliced, number, component, arithmetic=arithmetic
    )


@cache
def equation_grammar(component_names: frozenset[str]) -> pp.ParserElement:
    """
    Return the grammar for equations of the form `LHS OPERATOR RHS`.

    Each side is an arithmetic expression (see :func:`arithmetic_grammar`) and
    the operator is one of `<=`, `>=`, `=`.

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    """
    arithmetic = arithmetic_grammar(component_names)
    equation = arithmetic + pp.one_of(list(EQUATION_OPERATORS)) + arithmetic
    return equation.set_parse_action(_compare_action)


@cache
def mask_grammar(
    dimensions: frozenset[str], inputs: frozenset[str], results: frozenset[str]
) -> pp.ParserElement:
    """
    Return the grammar for boolean mask ("where") strings.

    Masks combine existence conditions on model data, comparisons, dimension
    subsets, helper functions, booleans, and config options with
    `not` / `and` / `or` operators.

    Parameters
    ----------
    dimensions : frozenset[str]
        Valid dimension names.
    inputs : frozenset[str]
        Valid parameter/lookup names.
    results : frozenset[str]
        Valid variable/expression names.
    """
    all_names = dimensions | inputs | results
    number, identifier = _base_elements()
    dimension = _component_parser(dimensions, "dimension")
    input_ = _component_parser(inputs, "input")
    result = _component_parser(results, "result")
    config_option = (pp.Suppress("config.") + identifier).set_parse_action(
        _config_action
    )
    bool_operand = (
        pp.Keyword("True", caseless=True) | pp.Keyword("False", caseless=True)
    ).set_parse_action(_bool_action)
    unique_string = _string_parser(identifier, all_names)
    general_string = _string_parser(identifier, frozenset())
    id_list = _list_parser(number, unique_string, dimension)

    subset_items = pp.Group(
        pp.DelimitedList(pp.MatchFirst([config_option, number, general_string]))
    )
    subset = (
        pp.Suppress("[")
        + subset_items
        + pp.Suppress("]")
        + pp.Suppress(pp.White(" ", min=1))
        + pp.Suppress("in")
        + pp.Suppress(pp.White(" ", min=1))
        + pp.MatchFirst([dimension, input_])
    ).set_parse_action(_subset_action)

    arithmetic = pp.Forward()
    comparison_call = _call_parser(
        unique_string, number, id_list, arithmetic, identifier=identifier
    )
    _arithmetic_rules(
        comparison_call, number, dimension, input_, config_option, arithmetic=arithmetic
    )
    comparison = (
        arithmetic
        + pp.one_of(list(MASK_OPERATORS))
        + pp.MatchFirst([comparison_call, bool_operand, number, general_string])
    ).set_parse_action(_compare_action)

    call = _call_parser(
        unique_string,
        number,
        id_list,
        dimension,
        input_,
        result,
        config_option,
        identifier=identifier,
    )

    notop = pp.Keyword("not", caseless=True)
    andorop = pp.Keyword("and", caseless=True) | pp.Keyword("or", caseless=True)
    return pp.infix_notation(
        pp.MatchFirst([bool_operand, comparison, call, subset, input_, result]),
        [
            (notop, 1, pp.opAssoc.RIGHT, _unary_action),
            (andorop, 2, pp.opAssoc.LEFT, _arith_action),
        ],
    )
