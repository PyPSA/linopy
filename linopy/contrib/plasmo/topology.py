# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
"""
Subgraph topology for a :class:`~linopy.contrib.plasmo.PlasmoModel`.

The partition produces *cells* -- one node per partition assignment, each holding
an LP block. A :class:`Topology` decides how those cells are arranged into the
Plasmo subgraph tree: which cell's subgraph each other cell is nested under.
There are no empty container subgraphs -- every subgraph holds exactly one cell's
OptiNode, and nesting means "attach these cells directly under that cell".

A topology resolves, given the ordered ``node_keys``, a ``parent`` node index per
cell (``-1`` for "attach under the root graph"). That vector drives the streaming
builder (:func:`linopy.contrib.plasmo.build.iter_blocks` -> ``add_block!``).

- :func:`flat` -- every cell is a sibling under the root. The default.
- ``"manual"`` -- build the tree yourself: ``pm.add_subgraph(name)`` returns a
  handle for the (single) cell with that key; ``handle.add_subgraph(name)``
  attaches every cell whose key starts with ``name`` under that handle's cell.
  Chaining is restricted to *singular* receivers (a handle that resolved to
  exactly one cell); nesting under a multi-cell level raises.
"""

from __future__ import annotations

from linopy.contrib.plasmo.partition import NodeKey


def _key_head(key: NodeKey):
    """The selector head of a node key: the whole scalar key, or its first slot."""
    return key[0] if isinstance(key, tuple) else key


def _node_name(key: NodeKey) -> str:
    """Human-readable subgraph/node name from a partition node key."""
    if isinstance(key, tuple):
        head, *rest = key
        return f"{head}[{', '.join(map(str, rest))}]"
    return str(key)


class Topology:
    """Base: resolve ``node_keys`` to a ``parent`` node index per cell."""

    def parents(self, node_keys: list[NodeKey]) -> list[int]:
        raise NotImplementedError

    def describe(
        self,
        node_keys: list[NodeKey],
        parents: list[int] | None = None,
        label=None,
    ) -> str:
        """
        Render the subgraph tree as indented ``- name`` lines, one per cell.

        ``parents`` lets the caller pass an already-resolved (e.g. cached,
        post-build) parent vector; otherwise this calls :meth:`parents`
        itself, falling back to naming the topology class if that raises
        (e.g. an incomplete :class:`ManualTopology`). ``label(i, key)`` formats
        each cell (default: just its :func:`_node_name`) -- pass one to add
        e.g. per-node variable/constraint counts without this module needing
        to know about them.
        """
        if parents is None:
            try:
                parents = self.parents(node_keys)
            except Exception:
                return f"topology: {type(self).__name__} (not resolved yet)"

        if label is None:
            label = lambda i, key: _node_name(key)  # noqa: E731

        children: dict[int, list[int]] = {}
        for i, p in enumerate(parents):
            children.setdefault(p, []).append(i)

        lines: list[str] = []

        def walk(parent: int, depth: int) -> None:
            for i in children.get(parent, []):
                lines.append("  " * depth + f"- {label(i, node_keys[i])}")
                walk(i, depth + 1)

        walk(-1, 0)
        return "\n".join(lines)


class _Flat(Topology):
    def parents(self, node_keys):
        return [-1] * len(node_keys)


def flat() -> Topology:
    """Every cell a sibling under the root graph (no nesting)."""
    return _Flat()


class ManualTopology(Topology):
    """
    User-built subgraph tree. Not constructed directly -- ``PlasmoModel`` exposes
    :meth:`~PlasmoModel.add_subgraph` which delegates here.
    """

    def __init__(self, node_keys: list[NodeKey]):
        self._node_keys = node_keys
        self._parent = [-1] * len(node_keys)  # default: root
        self._placed = [False] * len(node_keys)

    def _cells_with_head(self, head) -> list[int]:
        return [i for i, k in enumerate(self._node_keys) if _key_head(k) == head]

    def _attach(self, name, parent_idx: int) -> _Handle:
        cells = self._cells_with_head(name)
        if not cells:
            raise KeyError(f"no partition cell has key head {name!r}")
        for i in cells:
            if self._placed[i]:
                raise ValueError(
                    f"cell {self._node_keys[i]!r} is already attached elsewhere"
                )
            self._parent[i] = parent_idx
            self._placed[i] = True
        return _Handle(self, name, cells)

    def parents(self, node_keys):
        # every cell must be placed (attached somewhere), else the tree is
        # incomplete -- surface it rather than silently rooting stragglers.
        unplaced = [node_keys[i] for i, p in enumerate(self._placed) if not p]
        if unplaced:
            raise ValueError(
                f"manual topology is incomplete: cells {unplaced[:5]} were never "
                "attached. Add them via add_subgraph(...), or use topology=flat()."
            )
        return list(self._parent)


class _Handle:
    """
    A placed selection of cells. Chaining ``.add_subgraph`` nests further cells
    under this one -- only allowed when the selection is a single cell.
    """

    def __init__(self, topo: ManualTopology, name, cells: list[int]):
        self._topo = topo
        self._name = name
        self._cells = cells

    def add_subgraph(self, name) -> _Handle:
        if len(self._cells) != 1:
            raise ValueError(
                f"cannot nest under {self._name!r}: it resolved to "
                f"{len(self._cells)} cells. Nesting is only supported under a "
                "single cell in this version."
            )
        return self._topo._attach(name, self._cells[0])
