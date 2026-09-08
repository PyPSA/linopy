"""Walks over a program's expression nodes, and the dimensions a node spans."""

from __future__ import annotations

from collections.abc import Iterator

from math_spec import program as ms


def walk(*nodes: ms.ExpressionNode) -> Iterator[ms.ExpressionNode]:
    """Every node under *nodes*, each of them included, parents first."""
    for node in nodes:
        yield node
        yield from walk(*ms.children(node))


def amounts_of(node: ms.ExpressionNode) -> Iterator[str]:
    """The parameters *node* names as an amount: a translation's offset or a window's width."""
    if isinstance(node, ms.Translate) and isinstance(node.offset, str):
        yield node.offset
    elif isinstance(node, ms.Window) and isinstance(node.width, str):
        yield node.width


def parameters_of(*nodes: ms.ExpressionNode) -> frozenset[str]:
    """Every parameter named anywhere under *nodes*."""
    return frozenset(n.name for n in walk(*nodes) if isinstance(n, ms.Parameter))


def dims_of(node: ms.ExpressionNode, program: ms.Program) -> tuple[str, ...]:
    """The dimensions *node* spans, in the program's dimension order, before any data is bound."""
    spanned = _dims(node, program)
    return tuple(d for d in program.dimensions if d in spanned)


def _dims(node: ms.ExpressionNode, program: ms.Program) -> frozenset[str]:
    if isinstance(node, ms.Constant):
        return frozenset()
    if isinstance(node, ms.Variable):
        return frozenset(program.variables[node.name].dims)
    if isinstance(node, ms.Parameter):
        return frozenset(program.parameters[node.name].dims)
    if isinstance(node, ms.Dual):
        return frozenset(program.constraints[node.constraint].dims)
    if isinstance(node, ms.Sum):
        return _dims(node.operand, program) - set(node.over)
    if isinstance(node, ms.GroupSum | ms.At):
        return (_dims(node.operand, program) - {node.over}) | set(node.into)
    if isinstance(node, ms.Cases):
        return frozenset().union(*(_dims(r.value, program) for r in node.regions))
    return frozenset().union(*(_dims(c, program) for c in ms.children(node)))
