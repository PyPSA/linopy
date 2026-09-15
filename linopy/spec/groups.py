"""How a lookup partitions an axis: the shape every group-wise operator reads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


def unmapped(key: object) -> bool:
    """Whether a lookup left this member in no group: ``None``, or the NaN that never equals itself."""
    return key is None or key != key


@dataclass(frozen=True)
class Groups:
    labels: np.ndarray
    grouped: xr.DataArray
    belongs: xr.DataArray
    within: xr.DataArray
    size: xr.DataArray
    roster: np.ndarray
    names: tuple[object, ...]
    counts: tuple[int, ...]


def grouped(over: str, labels: np.ndarray, groups: xr.DataArray) -> Groups:
    """
    How the lookup *groups* partitions the axis *over*.

    A coordinate the lookup sends nowhere belongs to no group: its ``within``
    is 0, its ``size`` 1 and its ``grouped`` False.
    """
    keys = np.asarray(groups.sel({over: labels}).values, dtype=object)
    peers: dict[object, list[int]] = {}
    within = np.zeros(len(labels), dtype=int)
    held = np.zeros(len(labels), dtype=bool)
    for k, key in enumerate(keys):
        if unmapped(key):
            continue
        held[k] = True
        beside = peers.setdefault(key, [])
        within[k] = len(beside)
        beside.append(k)
    order = {key: g for g, key in enumerate(peers)}
    widest = max((len(beside) for beside in peers.values()), default=1)
    roster = np.zeros((max(len(peers), 1), widest), dtype=int)
    for key, beside in peers.items():
        roster[order[key], : len(beside)] = beside
    belongs = np.array([order.get(key, 0) for key in keys], dtype=int)
    span = np.array(
        [len(peers[key]) if inside else 1 for key, inside in zip(keys, held)], dtype=int
    )

    def on_axis(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, coords={over: labels}, dims=[over])

    return Groups(
        labels,
        on_axis(held),
        on_axis(belongs),
        on_axis(within),
        on_axis(span),
        roster,
        tuple(peers),
        tuple(len(beside) for beside in peers.values()),
    )
