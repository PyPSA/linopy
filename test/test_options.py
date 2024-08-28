#!/usr/bin/env python3

import pytest

from linopy.config import OptionSettings


@pytest.fixture
def options():
    return OptionSettings(a=1, b=2, c=3)


def test_set_value(options):
    options.set_value(a=10)
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options.set_value(d=20)


def test_get_value(options):
    assert options.get_value("a") == 1

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options.get_value("d")


def test_call(options):
    options(a=10)
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options(d=20)


def test_getitem(options):
    assert options["a"] == 1

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options["d"]


def test_setitem(options):
    options["a"] = 10
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options["d"] = 20


def test_repr(options):
    repr(options)


def test_with_statement(options):
    with options as o:
        o.set_value(a=3)
        assert o.get_value("a") == 3
    assert options.get_value("a") == 1


def test_reset(options):
    options(a=10)
    options.reset()
    assert options._current_values == {"a": 1, "b": 2, "c": 3}
