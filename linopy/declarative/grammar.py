"""
Linopy declarative math grammar module.

This module contains the pyparsing grammars that turn declarative math strings into ASTs of :mod:`linopy.declarative.nodes` node objects.
Parse actions are the node classes' `from_tokens` classmethods; all evaluation logic lives on the nodes themselves.

The infix-notation grammar structure is adapted from the pyparsing MIT licensed `eval_arith.py` example:
https://github.com/pyparsing/pyparsing/blob/master/examples/eval_arith.py.

This module is adapted from the calliope Apache-2.0 licensed math parsers:
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/expression_parser.py
- https://github.com/calliope-project/calliope/blob/9916116a06ec8c1feaf3c2606bdb8941b916ce85/src/calliope/backend/where_parser.py
"""

from __future__ import annotations

from functools import cache

import pyparsing as pp

from linopy.declarative.nodes import (
    COMPONENT_CATEGORY_T,
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
    find_refs,
    iter_nodes,
)

__all__ = [
    "COMPONENT_CATEGORY_T",
    "EQUATION_OPERATORS",
    "MASK_OPERATORS",
    "REFERENCE_CLASSIFIER",
    "Arith",
    "Call",
    "Compare",
    "Component",
    "ConfigRef",
    "Constant",
    "ListNode",
    "Node",
    "SliceRef",
    "Sliced",
    "SubExprRef",
    "Subset",
    "Unary",
    "arithmetic_grammar",
    "equation_grammar",
    "find_refs",
    "iter_nodes",
    "mask_grammar",
    "slice_grammar",
    "sub_expression_grammar",
]

pp.ParserElement.enable_packrat()

REFERENCE_CLASSIFIER = "$"
"""Prefix marking a reference to a sub-expression (`$foo`) or slicer (`x[dim=$foo]`)."""

EQUATION_OPERATORS = ("<=", ">=", "=")
"""Comparison operators allowed in constraint equations."""

MASK_OPERATORS = ("<", ">", "==", ">=", "<=")
"""Comparison operators allowed in mask strings."""


# ---------------------------------------------------------------------------
# Grammar primitives
# ---------------------------------------------------------------------------


def _base_elements() -> tuple[pp.ParserElement, pp.ParserElement]:
    """Return the (number, identifier) primitives shared by all grammars."""
    inf_kw = pp.Combine(pp.Opt(pp.Suppress(".")) + pp.Keyword("inf", caseless=True))
    number = (pp.pyparsing_common.number | inf_kw).set_parse_action(Constant.number)
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
    return _names_parser(names).set_parse_action(Component.from_tokens_as(category))


def _string_parser(
    identifier: pp.ParserElement, excluded_names: frozenset[str]
) -> pp.ParserElement:
    """Return a parser for generic strings that are not in `excluded_names`."""
    return (~_names_parser(excluded_names) + identifier).set_parse_action(
        Constant.string
    )


def _list_parser(*items: pp.ParserElement) -> pp.ParserElement:
    """Return a parser for `[item, item, ...]` lists of the given item parsers."""
    element = pp.MatchFirst(items)
    id_list = pp.Suppress("[") + pp.DelimitedList(element) + pp.Suppress("]")
    return id_list.set_parse_action(ListNode.from_tokens)


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
    return call.set_parse_action(Call.from_tokens)


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
        slice_ref.set_parse_action(SliceRef.from_tokens)
        slicer = slice_ref | slicer

    one_slice = pp.Group(
        identifier("set_name") + pp.Suppress("=") + pp.Group(slicer)("slicer")
    )
    slices = pp.Group(pp.DelimitedList(one_slice))("slices")
    sliced = pp.Combine(component("obj") + pp.Suppress("[")) + slices + pp.Suppress("]")
    return sliced.set_parse_action(Sliced.from_tokens)


def _sub_expression_ref_parser(identifier: pp.ParserElement) -> pp.ParserElement:
    """Return a parser for `$name` sub-expression references."""
    ref = pp.Combine(pp.Suppress(REFERENCE_CLASSIFIER) + identifier)
    return ref.set_parse_action(SubExprRef.from_tokens)


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
            (signop, 1, pp.opAssoc.RIGHT, Unary.from_tokens),
            (expop, 2, pp.opAssoc.LEFT, Arith.from_tokens),
            (multop, 2, pp.opAssoc.LEFT, Arith.from_tokens),
            (signop, 2, pp.opAssoc.LEFT, Arith.from_tokens),
        ],
    )
    return arithmetic


# ---------------------------------------------------------------------------
# Grammar entry points
# ---------------------------------------------------------------------------


@cache
def _expression_grammar(
    component_names: frozenset[str],
    *,
    arithmetic: bool = True,
    slice_refs: bool = True,
    sub_expr_refs: bool = False,
) -> pp.ParserElement:
    """
    Return an expression-family grammar for the given component names.

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    arithmetic : bool, default: True
        If True, combine operands with infix `+ - * / **` rules (with calls
        allowed to recursively contain arithmetic). If False, return a flat
        single-operand grammar (used for slicer strings).
    slice_refs : bool, default: True
        If True, allow `$name` slicer references inside slice brackets.
    sub_expr_refs : bool, default: False
        If True, allow `$name` sub-expression references as operands.
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
        allow_slice_references=slice_refs,
    )
    if not arithmetic:
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
    arith = pp.Forward()
    call = _call_parser(arith, call_list, string, identifier=identifier)
    operands = [call]
    if sub_expr_refs:
        operands.append(_sub_expression_ref_parser(identifier))
    operands += [sliced, number, component]
    return _arithmetic_rules(*operands, arithmetic=arith)


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
    return _expression_grammar(component_names, arithmetic=False, slice_refs=False)


def sub_expression_grammar(component_names: frozenset[str]) -> pp.ParserElement:
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
    return _expression_grammar(component_names)


def arithmetic_grammar(component_names: frozenset[str]) -> pp.ParserElement:
    """
    Return the grammar for arithmetic expressions (`+ - * / **`).

    Allows arbitrarily nested arithmetic and function calls, and references to
    sub-expressions (`$name`) and slicers (`x[dim=$name]`).

    Parameters
    ----------
    component_names : frozenset[str]
        Valid math component names, to separate them from generic strings.
    """
    return _expression_grammar(component_names, sub_expr_refs=True)


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
    return equation.set_parse_action(Compare.from_tokens)


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
        ConfigRef.from_tokens
    )
    bool_operand = (
        pp.Keyword("True", caseless=True) | pp.Keyword("False", caseless=True)
    ).set_parse_action(Constant.boolean)
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
    ).set_parse_action(Subset.from_tokens)

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
    ).set_parse_action(Compare.from_tokens)

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
            (notop, 1, pp.opAssoc.RIGHT, Unary.from_tokens),
            (andorop, 2, pp.opAssoc.LEFT, Arith.from_tokens),
        ],
    )
