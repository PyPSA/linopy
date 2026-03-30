import pypsa
import pytest

import linopy as lp

lp.options["freeze_constraints"] = True


@pytest.fixture(scope="module")
def network():
    return pypsa.examples.carbon_management()


@pytest.fixture(scope="module")
def model(network):
    return network.optimize.create_model()


def test_create_model(benchmark, network):
    benchmark(network.optimize.create_model)


def test_to_highspy(benchmark, model):
    benchmark(lp.io.to_highspy, model)


def test_to_highspy_no_names(benchmark, model):
    benchmark(lp.io.to_highspy, model, set_names=False)
