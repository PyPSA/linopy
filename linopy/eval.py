#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 18:15:50 2021

@author: fabian
"""

import re
from warnings import warn

plus_minus_signs = re.compile(r"(\+|\-)")
equation_signs = re.compile(r"(>=|==|<=|=)")

disallowed_tokens = ["^", "**"]


class Expr(str):
    "Subclass of string to extract mathematical relations for the linopy model."

    def __init__(self, expr: str):
        """
        Initialize a Expression with a string.

        Parameters
        ----------
        expr : str
            Non-empty expression.

        """
        if not expr:
            raise ValueError("expr cannot be an empty string")
        for token in disallowed_tokens:
            if token in self:
                raise ValueError(
                    f"The expression contains a disallowed symbol '{token}'."
                )
        return

    def to_string_tuples(self) -> list:
        """
        Restructure an expression that can be interpreted as a linear expression to a
        list of coefficient-variable-tuples, where all values are kept as strings.

        Returns
        -------
        list
            List of tuples.

        """
        return [separate_coeff_and_var(e) for e in separate_terms(self)]

    def to_constraint_args_kwargs(self) -> tuple:
        """
        Convert expression that can be interpreted as a constraint definition
        to arguments and keyword-arguments for the function ``Model.add_constraints``
        where all values are kept as strings.

        Returns
        -------
        tuple :
            args, kwargs.

        """
        exprs = [e.strip() for e in equation_signs.split(self) if e != ""]
        if len(exprs) != 3:
            raise ValueError("The passed (in)equality is not correctly formatted.")

        lhs, sign, rhs = [Expr(e) for e in exprs]
        kwargs = {}
        # lhs may contain name definition with ":" separator
        if ":" in lhs:
            kwargs["name"] = lhs.split(":")[0].strip()
            lhs = Expr(lhs.split(":")[-1].strip())
        lhs = lhs.to_string_tuples()
        args = (lhs, sign, rhs)
        return args, kwargs

    def to_variable_kwargs(self) -> dict:
        """
        Convert expression that can be interpreted as a variable definition
        to keyword arguments for the function ``Model.add_variables`` where
        all values are kept as strings.

        Returns
        -------
        dict
            kwargs

        """
        if "==" in self:
            warn(
                'Encountered "==" operator in variable, this is not processed by linopy.'
            )
        exprs = [e.strip() for e in re.split(r"(>=|<=)", self)]
        if len(exprs) == 1:
            # assume only variable name is given
            return dict(name=exprs[0])
        elif len(exprs) == 3:
            # assume that one bound is given, variable comes first
            if exprs[1] == ">=":
                return dict(name=exprs[0], lower=exprs[2])
            else:
                return dict(name=exprs[0], upper=exprs[2])
        elif len(exprs) == 5:
            return dict(lower=exprs[0], name=exprs[2], upper=exprs[4])
        else:
            raise ValueError("Your variable expression is not correctly formatted.")


def separate_terms(expr: Expr) -> list:
    """
    Subdivide expression that can be interpreted as a linear expression into
    separate terms.

    Parameters
    ----------
    expr : Expr

    Returns
    -------
    List of strings where each string represents a mathematical term.

    """
    exprs = plus_minus_signs.split(expr)
    exprs = [e.strip() for e in exprs]

    res = []
    prefix = ""
    for e in exprs:
        temp = prefix + e
        if plus_minus_signs.match(e):
            prefix = e
        else:
            if temp:
                res.append(temp)
            prefix = ""
    res = [e if plus_minus_signs.match(e[0]) else "+" + e for e in res]
    return res


def separate_coeff_and_var(term: Expr) -> tuple:
    """
    Subdivide a Exprs that can be interpreted as a mathematical term into
    a coefficient-variable tuple. The expression is required to start with a sign.

    If the expression contains more than one "*"-token, the last one is takes as
    a separator.

    Parameters
    ----------
    term : Expr

    Returns
    -------
    tuple of strings :
        coefficient, variable

    """
    if not plus_minus_signs.match(term[0]):
        raise ValueError("Term does not start with a sign.")
    if not "*" in term:
        term = plus_minus_signs.sub(r"\g<0>1*", term)
    t = [s.strip() for s in re.split(r"\*", term)]
    if len(t) > 2:
        t = "*".join(t[:-1]), t[-1]
    return tuple(t)
