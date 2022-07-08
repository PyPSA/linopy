#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:47:04 2021.

@author: fabian
"""

import numpy as np
import pytest
import xarray as xr

from linopy import Model
from linopy.eval import Expr, separate_coeff_and_var, separate_terms


@pytest.fixture()
def model():
    m = Model()
    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper, name="x")
    m.add_variables(lower, upper, name="y")
    return m


def test_separate_terms():
    assert separate_terms("45") == ["+45"]
    assert separate_terms("45 + 4") == ["+45", "+4"]
    assert separate_terms("-45+ 1*a - @c") == ["-45", "+1*a", "-@c"]


def test_separate_coeff_and_var():
    assert separate_coeff_and_var("+12 * @df") == ("+12", "@df")
    assert separate_coeff_and_var("+12 * @df * a") == ("+12*@df", "a")
    assert separate_coeff_and_var("- @df") == ("-1", "@df")
    assert separate_coeff_and_var("- @(lasd.asd[]) ") == ("-1", "@(lasd.asd[])")
    assert separate_coeff_and_var("+a ") == ("+1", "a")

    with pytest.raises(ValueError):
        separate_coeff_and_var("a ")


def test_Expr_init():
    with pytest.raises(ValueError):
        Expr("")
    with pytest.raises(ValueError):
        Expr("5**a")


def test_expr_to_string_tuples():
    assert Expr("5*b - @ds * b").to_string_tuples() == [("+5", "b"), ("-@ds", "b")]
    assert Expr("5*a*b - @ds * b").to_string_tuples() == [("+5*a", "b"), ("-@ds", "b")]


def test_expr_to_constraint_args_kwargs():
    args, kwargs = Expr("5*b - @ds * b >= 2").to_constraint_args_kwargs()

    assert len(args) == 3
    lhs, sign, rhs = args
    assert lhs == [("+5", "b"), ("-@ds", "b")]
    assert sign == ">="
    assert rhs == "2"

    args, kwargs = Expr("con : 5*b - @ds * b >= 2").to_constraint_args_kwargs()

    assert len(args) == 3
    lhs, sign, rhs = args
    assert lhs == [("+5", "b"), ("-@ds", "b")]
    assert sign == ">="
    assert rhs == "2"
    assert kwargs["name"] == "con"

    with pytest.raises(ValueError):
        Expr("a + b").to_constraint_args_kwargs()


def test_expr_to_variable_kwargs():
    kwargs = Expr("x >= 5").to_variable_kwargs()
    assert len(kwargs) == 2
    assert kwargs["name"] == "x"
    assert kwargs["lower"] == "5"

    kwargs = Expr("x <= 5*@d").to_variable_kwargs()
    assert len(kwargs) == 2
    assert kwargs["name"] == "x"
    assert kwargs["upper"] == "5*@d"

    kwargs = Expr("xasd").to_variable_kwargs()
    assert len(kwargs) == 1
    assert kwargs["name"] == "xasd"

    kwargs = Expr("1 <= myvariable <= 150").to_variable_kwargs()
    assert len(kwargs) == 3
    assert kwargs["name"] == "myvariable"
    assert kwargs["lower"] == "1"
    assert kwargs["upper"] == "150"

    with pytest.raises(ValueError):
        Expr("1 <= 150 <= x >= 3").to_variable_kwargs()

    with pytest.warns(UserWarning):
        Expr("1 == 150").to_variable_kwargs()


def test_model_eval(model):
    assert model["x"].equals(model._eval("x"))


def test_var_eval(model):
    model.vareval("z <= 0")
    assert "z" in model.variables
    assert model.variables["z"].upper.item() == 0

    model.vareval("a <= 0", eval_kw={})
    assert "a" in model.variables


def test_lin_eval(model):
    c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])
    target = model.linexpr((c, model["x"]), (-1, model["y"]))
    assert model.lineval("@c * x - y").equals(target)
    model.lineval("@c * x - y", eval_kw={})


def test_con_eval(model):
    c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])
    model.coneval("@c * x - y >= 0")
    assert len(model.constraints.labels)


def test_con_eval_repeated(model):
    c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])
    model.coneval("@c * x - y >= 0")
    model.coneval("@c * x - y >= 0")
    assert len(model.constraints.labels)

    model.coneval("@c * x - y >= 0", eval_kw={})


def test_con_eval_with_name(model):
    c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])
    model.coneval("Con1: @c * x - y >= 0")
    assert "Con1" in model.constraints
