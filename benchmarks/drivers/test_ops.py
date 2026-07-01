"""
Arithmetic operation micro-benchmarks.

One benchmark per ``(operation, size profile)`` — the operands are built in
setup (not measured) and the fixture measures a single ``op(*operands)``. See
:mod:`benchmarks.ops` for the op registry and the size / alignment axes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.ops import OpSpec, Profile, iter_op_params

_CASES = iter_op_params()


@pytest.mark.parametrize(
    "op, profile",
    _CASES,
    ids=[f"{op.name}[{profile.key}]" for op, profile in _CASES],
)
def test_op(benchmark: Callable[..., object], op: OpSpec, profile: Profile) -> None:
    operands = op.setup(profile)
    benchmark(op.op, *operands)
