#!/usr/bin/env python3


import pytest

from linopy.config import OptionSettings


@pytest.fixture
def options() -> OptionSettings:
    return OptionSettings(a=1, b=2, c=3)


def test_set_value(options: OptionSettings) -> None:
    options.set_value(a=10)
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options.set_value(d=20)


def test_get_value(options: OptionSettings) -> None:
    assert options.get_value("a") == 1

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options.get_value("d")


def test_call(options: OptionSettings) -> None:
    options(a=10)
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options(d=20)


def test_getitem(options: OptionSettings) -> None:
    assert options["a"] == 1

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options["d"]


def test_setitem(options: OptionSettings) -> None:
    options["a"] = 10
    assert options._current_values == {"a": 10, "b": 2, "c": 3}

    with pytest.raises(KeyError, match="d is not a valid setting."):
        options["d"] = 20


def test_repr(options: OptionSettings) -> None:
    repr(options)


def test_with_statement(options: OptionSettings) -> None:
    with options as o:
        o.set_value(a=3)
        assert o.get_value("a") == 3
    assert options.get_value("a") == 1


def test_reset(options: OptionSettings) -> None:
    options(a=10)
    options.reset()
    assert options._current_values == {"a": 1, "b": 2, "c": 3}


def test_context_restores_prior_value(options: OptionSettings) -> None:
    options(a=5)
    with options:
        options(a=10)
    assert options._current_values == {"a": 5, "b": 2, "c": 3}


def test_context_undoes_inner_changes(options: OptionSettings) -> None:
    with options as o:
        o(b=20)
        assert o.get_value("b") == 20
    assert options.get_value("b") == 2


def test_nested_context_restores_level_by_level(options: OptionSettings) -> None:
    options(a=1)
    with options:
        options(a=2)
        with options:
            options(a=3)
            assert options.get_value("a") == 3
        assert options.get_value("a") == 2
    assert options.get_value("a") == 1


def test_context_restores_on_exception(options: OptionSettings) -> None:
    options(a=1)
    with pytest.raises(ValueError):
        with options:
            options(a=99)
            raise ValueError("boom")
    assert options.get_value("a") == 1
