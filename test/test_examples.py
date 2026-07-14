"""
Test module for the examples module.
"""

from linopy import Model
from linopy.examples import (
    benchmark_model,
    simple_two_array_variables_model,
    simple_two_single_variables_model,
)


def test_simple_two_single_variables_model() -> None:
    """
    Test function for the simple_two_single_variables_model.
    """
    model = simple_two_single_variables_model()
    assert isinstance(model, Model)


def test_simple_two_array_variables_model() -> None:
    """
    Test function for the simple_two_array_variables_model.
    """
    model = simple_two_array_variables_model()
    assert isinstance(model, Model)


def test_benchmark_model() -> None:
    """
    Test function for the benchmark_model.
    """
    model = benchmark_model()
    assert isinstance(model, Model)


def test_benchmark_model_with_integer_labels() -> None:
    """
    Test function for the benchmark_model with integer labels.
    """
    model = benchmark_model(integerlabels=True)
    assert isinstance(model, Model)
