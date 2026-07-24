"""Benchmark configuration and shared test helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from benchmarks.registry import iter_params, spec_param_id

if TYPE_CHECKING:
    import linopy
    from benchmarks.registry import BenchSpec


@pytest.fixture(autouse=True)
def _bench_semantics() -> Iterator[None]:
    """
    Force the arithmetic semantics for the whole run from
    ``LINOPY_BENCH_SEMANTICS`` (``legacy`` / ``v1``), restoring after.

    The memory-report job runs the build twice — once at the default (legacy),
    once with this set to ``v1`` — to A/B the v1 cost. Leaving it unset keeps the
    default, so the node ids the legacy run and the CodSpeed run produce are
    identical (``test_build[…]``, no suffix): CodSpeed keeps its per-id history
    against master, and the two report JSONs line up under ``benchmem compare``.
    """
    import linopy

    mode = os.environ.get("LINOPY_BENCH_SEMANTICS")
    if not mode:
        yield
        return
    old = linopy.options["semantics"]
    linopy.options["semantics"] = mode
    try:
        yield
    finally:
        linopy.options["semantics"] = old


# Test modules the CodSpeed instruments measure (edit to change coverage).
# build + the two export paths: to_lp (LP text) and to_solver (direct handoff,
# which also exercises matrix-gen). matrices is dropped — a subset of to_solver;
# netcdf excluded — disk I/O, noisy. All still run under the smoke job.
CODSPEED_MODULES = (
    "test_build",
    "test_to_lp",
    "test_to_solver",
    "test_ops",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pipeline",
        action="store_true",
        default=False,
        help=(
            "Include the opt-in end-to-end pipeline benchmark (build → matrices "
            "→ lp in one measured region). Off by default — it re-runs the "
            "per-phase work and includes the build."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    ``test_pipeline`` (end-to-end) is opt-in — deselected unless ``--pipeline``.
    ``--codspeed`` narrows the run to ``CODSPEED_MODULES`` (drops netcdf/matrices).
    """
    if not config.getoption("--pipeline"):
        dropped = [i for i in items if i.path.stem == "test_pipeline"]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = [i for i in items if i.path.stem != "test_pipeline"]

    if getattr(config.option, "codspeed", False):
        deselected = [i for i in items if i.path.stem not in CODSPEED_MODULES]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [i for i in items if i.path.stem in CODSPEED_MODULES]


def cases(phase: str) -> pytest.MarkDecorator:
    """Parametrize a phase driver over every ``(spec, n)`` that phase runs."""
    params = iter_params(phase)
    return pytest.mark.parametrize(
        ("spec", "n"),
        params,
        ids=[spec_param_id(s.name, s.axis, v) for s, v in params],
    )


def require(spec: BenchSpec) -> None:
    """``importorskip`` a spec's optional dependencies before it runs."""
    for mod in spec.requires:
        pytest.importorskip(mod)


def build_model(spec: BenchSpec, n: int) -> linopy.Model:
    """Build ``spec`` at ``n`` — the untimed setup, after the requires-skip."""
    require(spec)
    return spec.build(n)


@pytest.fixture(autouse=True)
def _benchmem_dims(request: pytest.FixtureRequest, benchmark: object) -> None:
    """
    Mirror each case's ``spec``/``phase``/``axis`` into pytest-benchmark
    ``extra_info`` as analysis dims, so a ``--benchmark-json`` run plots cleanly
    under pytest-benchmem — which reads dims from ``params``/``extra_info`` and
    can see neither the (unserialisable) spec param nor the phase, which lives in
    the test-function name. The numeric ``n`` is already a clean param. No-op
    under CodSpeed, whose fixture carries no ``extra_info``.
    """
    callspec = getattr(request.node, "callspec", None)
    info = getattr(benchmark, "extra_info", None)
    func = getattr(request, "function", None)
    if (
        callspec is None
        or info is None
        or func is None
        or "spec" not in callspec.params
    ):
        return
    spec = callspec.params["spec"]
    info.update(
        spec=spec.name, phase=func.__name__.removeprefix("test_"), axis=spec.axis
    )
