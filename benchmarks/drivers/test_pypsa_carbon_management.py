from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import linopy as lp

# pypsa is an optional benchmark dep. Skip the whole module if it's missing
# so the rest of the suite stays collectable without it.
pypsa = pytest.importorskip("pypsa")


@pytest.fixture(autouse=True)
def _skip_under_v1(_bench_semantics: None) -> None:
    if lp.options["semantics"] == "v1":
        pytest.skip("PyPSA emits NaN-valued constants that v1 rejects by design.")


@pytest.fixture(scope="module")
def network() -> Any:
    try:
        return pypsa.examples.carbon_management()
    except Exception as exc:  # network / example-data drift, not a linopy signal
        pytest.skip(f"pypsa example data unavailable: {exc}")


def test_create_model_frozen(benchmark: Callable[..., object], network: Any) -> None:
    benchmark(network.optimize.create_model, freeze_constraints=True)


def test_create_model_mutable(benchmark: Callable[..., object], network: Any) -> None:
    benchmark(network.optimize.create_model, freeze_constraints=False)


@pytest.fixture(scope="module")
def model_frozen(network: Any) -> Any:
    return network.optimize.create_model(freeze_constraints=True)


@pytest.fixture(scope="module")
def model_mutable(network: Any) -> Any:
    return network.optimize.create_model(freeze_constraints=False)


def test_to_highspy_frozen(benchmark: Callable[..., object], model_frozen: Any) -> None:
    benchmark(lp.io.to_highspy, model_frozen)


def test_to_highspy_mutable(
    benchmark: Callable[..., object], model_mutable: Any
) -> None:
    benchmark(lp.io.to_highspy, model_mutable)


def test_to_highspy_mutable_no_names(
    benchmark: Callable[..., object], model_mutable: Any
) -> None:
    benchmark(lp.io.to_highspy, model_mutable, set_names=False)


def test_to_highspy_frozen_no_names(
    benchmark: Callable[..., object], model_frozen: Any
) -> None:
    benchmark(lp.io.to_highspy, model_frozen, set_names=False)
