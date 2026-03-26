import pypsa
import pytest

import linopy as lp


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


def _to_highspy_no_names(model: lp.Model) -> None:
    import highspy

    h = highspy.Highs()
    M = model.matrices
    h.addVars(len(M.vlabels), M.lb, M.ub)
    import numpy as np

    h.changeColsCost(len(M.c), np.arange(len(M.c), dtype=np.int32), M.c)
    A = M.A
    if A is not None:
        A = A.tocsr()
        num_cons = A.shape[0]
        lower = np.where(M.sense != "<", M.b, -np.inf)
        upper = np.where(M.sense != ">", M.b, np.inf)
        h.addRows(num_cons, lower, upper, A.nnz, A.indptr, A.indices, A.data)
    return h


def test_to_highspy_no_names(benchmark, model):
    benchmark(_to_highspy_no_names, model)
