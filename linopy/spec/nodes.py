"""Walks over expression nodes that descend into every operand, a ``Power``'s included."""

from __future__ import annotations

from collections.abc import Iterator

from math_spec import program as ms


def children(node: ms.ExpressionNode) -> tuple[ms.ExpressionNode, ...]:
    """The operands of *node*: ``math_spec.program.children`` plus a power's base and exponent."""
    if isinstance(node, ms.Power):
        return (node.base, node.exponent)
    return ms.children(node)


def walk(*nodes: ms.ExpressionNode) -> Iterator[ms.ExpressionNode]:
    """Every node under *nodes*, each of them included, parents first."""
    for node in nodes:
        yield node
        yield from walk(*children(node))


def parameters_of(*nodes: ms.ExpressionNode) -> frozenset[str]:
    """Every parameter named anywhere under *nodes*."""
    return frozenset(n.name for n in walk(*nodes) if isinstance(n, ms.Parameter))
