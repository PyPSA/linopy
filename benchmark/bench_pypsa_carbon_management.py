import pypsa
import pytest

import linopy as lp


@pytest.fixture(scope="module")
def network():
    return pypsa.examples.carbon_management()


def test_create_model_frozen(benchmark, network):
    benchmark(network.optimize.create_model, freeze_constraints=True)


def test_create_model_mutable(benchmark, network):
    benchmark(network.optimize.create_model, freeze_constraints=False)


@pytest.fixture(scope="module")
def model_frozen(network):
    return network.optimize.create_model(freeze_constraints=True)


@pytest.fixture(scope="module")
def model_mutable(network):
    return network.optimize.create_model(freeze_constraints=False)


def test_to_highspy_frozen(benchmark, model_frozen):
    benchmark(lp.io.to_highspy, model_frozen)


def test_to_highspy_mutable(benchmark, model_mutable):
    benchmark(lp.io.to_highspy, model_mutable)


def test_to_highspy_mutable_no_names(benchmark, model_mutable):
    benchmark(lp.io.to_highspy, model_mutable, set_names=False)


def test_to_highspy_frozen_no_names(benchmark, model_frozen):
    benchmark(lp.io.to_highspy, model_frozen, set_names=False)
