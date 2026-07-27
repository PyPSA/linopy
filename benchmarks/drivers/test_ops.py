"""
Arithmetic operation micro-benchmarks.

One benchmark per ``(operation, size profile)`` — the operands are built in
setup (not measured) and the fixture measures a single ``op(*operands)``. See
:mod:`benchmarks.ops` for the op registry and the size / alignment axes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.ops import GRID, OpSpec, iter_ops

_OPS = iter_ops()


@pytest.mark.parametrize("op", _OPS, ids=[op.name for op in _OPS])
def test_op(benchmark: Callable[..., object], op: OpSpec) -> None:
    operands = op.setup(GRID)
    benchmark(op.op, *operands)
