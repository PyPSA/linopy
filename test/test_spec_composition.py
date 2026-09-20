"""
Composing spec fragments into one whole model.

A fragment declares under ``given:`` the variables it reads but does not build.
:func:`linopy.spec.merge` folds those reads into the fragment that introduces
them and :func:`linopy.spec.override` lays a patch over a base, both producing
one whole model with no ``given:`` left. A program that still carries a
``given:`` name is a fragment, not a whole model, and building it is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import xarray as xr

pytest.importorskip("math_spec")

import linopy  # noqa: E402
from conftest import DISPATCH_DATA, DISPATCH_P, solved  # noqa: E402
from linopy import Model, read_netcdf  # noqa: E402
from linopy.spec import merge, override  # noqa: E402
from linopy.testing import assert_model_equal  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

GENERATION: dict[str, Any] = {
    "dimensions": {"snapshot": {"dtype": "int"}, "generator": {}},
    "parameters": {"p_max": {"dims": ["generator"]}, "cost": {"dims": ["generator"]}},
    "variables": {
        "p": {
            "dims": ["snapshot", "generator"],
            "where": "p_max > 0",
            "bounds": {"lower": 0, "upper": "p_max"},
        }
    },
    "objective": {"sense": "minimize", "expression": "sum(p * cost)"},
}
BALANCE: dict[str, Any] = {
    "dimensions": {"snapshot": {"dtype": "int"}, "generator": {}},
    "given": {"variables": {"p": {"dims": ["snapshot", "generator"]}}},
    "parameters": {"load": {"dims": ["snapshot"]}},
    "constraints": {
        "power_balance": {
            "dims": ["snapshot"],
            "expression": "sum(p, over=generator) == load",
        }
    },
}
FRAGMENTS = {"generation": GENERATION, "balance": BALANCE}


def test_merge_reexport_folds_given_into_a_whole_model_dict() -> None:
    import linopy.spec as spec

    assert {"merge", "override"} <= set(spec.__all__)
    merged = spec.merge(FRAGMENTS)
    assert isinstance(merged, dict) and not merged.get("given")
    assert {"p"} <= set(merged["variables"])
    assert "power_balance" in merged["constraints"]


def test_override_reexport_lays_a_patch_onto_the_dict() -> None:
    import linopy.spec as spec

    patched = spec.override(
        spec.merge(FRAGMENTS), {"cap": {"parameters": {"budget": {"dims": []}}}}
    )
    assert isinstance(patched, dict) and "budget" in patched["parameters"]
    assert not patched.get("given")


def test_merged_fragments_build_the_whole_model_and_solve() -> None:
    m = solved(merge(FRAGMENTS), DISPATCH_DATA)
    assert set(m.variables) == {"p"} and set(m.constraints) == {"power_balance"}
    assert m.objective.value == pytest.approx(2500.0)
    xr.testing.assert_allclose(m.solution["p"], DISPATCH_P)


def test_a_merged_model_carries_no_given() -> None:
    m = Model.from_spec(merge(FRAGMENTS), DISPATCH_DATA)
    assert not m.spec.program.given.variables
    assert not m.spec.program.given.constraints


def test_override_lays_a_patch_over_a_base() -> None:
    peak = {
        "constraints": {
            "peak": {
                "dims": ["generator"],
                "expression": "sum(p, over=snapshot) <= p_max",
            }
        }
    }
    m = Model.from_spec(override(merge(FRAGMENTS), {"cap": peak}), DISPATCH_DATA)
    assert set(m.constraints) == {"power_balance", "peak"}


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(BALANCE, id="lone-fragment"),
        pytest.param(merge({"balance": BALANCE}), id="merge-without-introducer"),
    ],
)
def test_a_program_that_still_reads_a_given_name_is_refused(
    spec: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"under `given:` that it does not build"):
        Model.from_spec(spec, DISPATCH_DATA)


@pytest.mark.parametrize("engine", ["netcdf4", "scipy"])
def test_a_composed_model_round_trips_through_netcdf(
    tmp_path: Path, engine: str
) -> None:
    m = Model.from_spec(merge(FRAGMENTS), DISPATCH_DATA)
    path = tmp_path / f"composed-{engine}.nc"
    m.to_netcdf(path, engine=engine)
    p = read_netcdf(path)
    assert_model_equal(m, p)
    assert p.spec.text == m.spec.text
    assert not p.spec.program.given.variables
