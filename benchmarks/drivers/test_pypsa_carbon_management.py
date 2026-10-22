from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import linopy as lp

# pypsa is an optional benchmark dep. Skip the whole module if it's missing
# so the rest of the suite stays collectable without it.
pypsa = pytest.importorskip("pypsa", minversion="1.4")


@pytest.fixture(scope="module")
def network() -> Any:
    try:
        return pypsa.examples.carbon_management()
    except Exception as exc:  # network / example-data drift, not a linopy signal
        pytest.skip(f"pypsa example data unavailable: {exc}")


@pytest.fixture(scope="module", params=[False, True], ids=["dense", "sparse"])
def sparse(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_create_model(
    benchmark: Callable[..., object], network: Any, sparse: bool
) -> None:
    benchmark(network.optimize.create_model, sparse=sparse)


@pytest.fixture(scope="module")
def model(network: Any, sparse: bool) -> Any:
    return network.optimize.create_model(sparse=sparse)


@pytest.mark.parametrize("set_names", [None, False], ids=["default", "no_names"])
def test_to_highspy(
    benchmark: Callable[..., object], model: Any, set_names: bool | None
) -> None:
    benchmark(lp.io.to_highspy, model, set_names=set_names)
