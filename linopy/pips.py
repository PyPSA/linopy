from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from linopy.model import Model

LAUNCHER_RANK_FLAG = {"mpirun": "-np", "srun": "-n"}


def assign_blocks(
    m: Model, dim: str, n_blocks: int, boundary: str = "contiguous"
) -> Model:
    if boundary != "contiguous":
        raise NotImplementedError(f"boundary {boundary!r} not supported")
    if dim not in m.variables.indexes:
        raise ValueError(f"dimension {dim!r} not found in model variables")
    index = m.variables.indexes[dim]
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    if n_blocks > len(index):
        raise ValueError(
            f"n_blocks ({n_blocks}) exceeds length of {dim!r} ({len(index)})"
        )
    ids = np.empty(len(index), dtype=np.int64)
    for i, part in enumerate(np.array_split(np.arange(len(index)), n_blocks)):
        ids[part] = i + 1
    m.blocks = xr.DataArray(ids, coords=[index], dims=[dim])
    return m


def _fmt(value: int) -> str:
    return f"{value:,}".replace(",", " ")


@dataclass
class BlockReport:
    n_blocks: int
    n_vars: int
    n_cons: int
    nnz: int

    n_global_cols: int
    block_cols: dict[int, int]

    block_nnz: dict[int, int]
    balance_min: int
    balance_median: float
    balance_max: int
    balance_ratio: float

    n_local_rows: int
    n_global_rows: int
    n_linking_rows: int
    n_adjacent_rows: int
    n_border_rows: int

    border_nnz: int
    border_fraction: float

    max_ranks: int
    target_cores: int | None
    rec_ranks: int
    rec_threads: int

    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        cols_med = (
            float(np.median(list(self.block_cols.values()))) if self.block_cols else 0.0
        )
        col_min = min(self.block_cols.values()) if self.block_cols else 0
        col_max = max(self.block_cols.values()) if self.block_cols else 0
        header = (
            f"BlockReport: {self.n_blocks} blocks | {_fmt(self.n_vars)} vars | "
            f"{_fmt(self.n_cons)} cons | {_fmt(self.nnz)} nnz"
        )
        columns = (
            f"  columns    global={_fmt(self.n_global_cols)}   per-block min/med/max "
            f"= {_fmt(col_min)} / {_fmt(int(cols_med))} / {_fmt(col_max)}"
        )
        block = (
            f"  block nnz  min/med/max = {_fmt(self.balance_min)} / "
            f"{_fmt(int(self.balance_median))} / {_fmt(self.balance_max)}   "
            f"(max/med ratio {self.balance_ratio:.2f})"
        )
        rows = (
            f"  rows       local={_fmt(self.n_local_rows)}  "
            f"global={_fmt(self.n_global_rows)}  linking={_fmt(self.n_linking_rows)}  "
            f"(adjacent={_fmt(self.n_adjacent_rows)}  border={_fmt(self.n_border_rows)})"
        )
        border = (
            f"  border     nnz={_fmt(self.border_nnz)} / {_fmt(self.nnz)} = "
            f"{self.border_fraction:.1%}"
        )
        if self.target_cores is None:
            parallel = (
                f"  parallel   max_ranks={self.max_ranks}  "
                f"ranks={self.rec_ranks} threads={self.rec_threads}"
            )
        else:
            parallel = (
                f"  parallel   max_ranks={self.max_ranks}  "
                f"target_cores={self.target_cores} -> ranks={self.rec_ranks} "
                f"threads={self.rec_threads}"
            )
        if self.warnings:
            warn = "\n".join(f"    {w}" for w in self.warnings)
            warnings = f"  warnings\n{warn}"
        else:
            warnings = "  warnings   (none)"
        return "\n".join([header, columns, block, rows, border, parallel, warnings])

    __repr__ = __str__


def diagnose(m: Model, target_cores: int | None = None) -> BlockReport:
    if m.blocks is None:
        raise ValueError(
            "no blocks assigned; call assign_blocks(m, dim, n_blocks) first"
        )

    m.calculate_block_maps()

    if m.matrices.A is None:
        raise ValueError("model has no regular constraints to diagnose")

    N = int(m.blocks.max())
    block_map = m.variables.get_blockmap(m.blocks.dtype.type)
    vlabels = m.matrices.vlabels
    col_blocks = block_map[vlabels]
    A = m.matrices.A

    row_blocks = np.concatenate(
        [
            c.data["blocks"].values.ravel()[c.active_row_mask()]
            for _, c in m.constraints.items()
            if not c.is_indicator
        ]
    )
    assert len(row_blocks) == A.shape[0]

    coo = A.tocoo()
    rb = row_blocks[coo.row]
    cb = col_blocks[coo.col]

    is_border = (rb == N + 1) | (cb == 0)
    border_nnz = int(is_border.sum())
    border_fraction = border_nnz / A.nnz if A.nnz else 0.0

    counts = np.bincount(np.clip(rb, 0, N + 1), minlength=N + 2)
    block_nnz = {n: int(counts[n]) for n in range(1, N + 1)}
    nonempty = np.array([v for v in block_nnz.values() if v > 0])
    if nonempty.size:
        balance_min = int(nonempty.min())
        balance_max = int(nonempty.max())
        balance_median = float(np.median(nonempty))
        balance_ratio = balance_max / balance_median if balance_median else 0.0
    else:
        balance_min = balance_max = 0
        balance_median = balance_ratio = 0.0

    n_global_cols = int((col_blocks == 0).sum())
    col_counts = np.bincount(np.clip(col_blocks, 0, N), minlength=N + 1)
    block_cols = {n: int(col_counts[n]) for n in range(1, N + 1)}

    n_local_rows = int(((row_blocks >= 1) & (row_blocks <= N)).sum())
    n_global_rows = int((row_blocks == 0).sum())
    linking_mask = row_blocks == N + 1
    n_linking_rows = int(linking_mask.sum())

    indptr, indices = A.indptr, A.indices
    n_adjacent_rows = 0
    n_border_rows = 0
    for i in np.nonzero(linking_mask)[0]:
        cols = indices[indptr[i] : indptr[i + 1]]
        local = np.unique(col_blocks[cols])
        n_local = int((local >= 1).sum())
        if n_local == 2:
            n_adjacent_rows += 1
        else:
            n_border_rows += 1

    max_ranks = N
    if target_cores is None:
        rec_ranks, rec_threads = max_ranks, 1
    else:
        rec_ranks = min(max_ranks, target_cores)
        rec_threads = max(1, target_cores // rec_ranks)

    warnings: list[str] = []
    if N == 1:
        warnings.append("model is not decomposed (n_blocks == 1)")
    if border_fraction > 0.15:
        warnings.append(
            f"high border fraction {border_fraction:.1%} (> 15%): root Schur "
            "complement will dominate; reduce K or reformulate"
        )
    if balance_ratio > 3:
        warnings.append(
            f"block imbalance max/median = {balance_ratio:.1f} (> 3): stragglers "
            "will stall synchronous IPM iterations"
        )
    empty = [n for n, v in block_nnz.items() if v == 0]
    if empty:
        warnings.append(f"empty local blocks (no rows): {empty}")
    if target_cores is not None and target_cores > max_ranks:
        warnings.append(
            f"target_cores {target_cores} > n_blocks {max_ranks}: MPI width is "
            "capped by blocks; raise K or add threads_per_rank"
        )

    return BlockReport(
        n_blocks=N,
        n_vars=len(vlabels),
        n_cons=A.shape[0],
        nnz=A.nnz,
        n_global_cols=n_global_cols,
        block_cols=block_cols,
        block_nnz=block_nnz,
        balance_min=balance_min,
        balance_median=balance_median,
        balance_max=balance_max,
        balance_ratio=balance_ratio,
        n_local_rows=n_local_rows,
        n_global_rows=n_global_rows,
        n_linking_rows=n_linking_rows,
        n_adjacent_rows=n_adjacent_rows,
        n_border_rows=n_border_rows,
        border_nnz=border_nnz,
        border_fraction=border_fraction,
        max_ranks=max_ranks,
        target_cores=target_cores,
        rec_ranks=rec_ranks,
        rec_threads=rec_threads,
        warnings=warnings,
    )


@dataclass
class PipsConfig:
    launcher: str = "mpirun"
    n_ranks: int | None = None
    threads_per_rank: int = 1
    launcher_args: list[str] = field(default_factory=list)
    linear_solver: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def build_pips_command(
    binary: str,
    export_dir: str,
    config: PipsConfig,
    n_blocks: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    if config.launcher not in LAUNCHER_RANK_FLAG:
        raise ValueError(
            f"launcher {config.launcher!r} not supported; use one of "
            f"{sorted(LAUNCHER_RANK_FLAG)}"
        )
    ranks = config.n_ranks if config.n_ranks is not None else (n_blocks or 1)
    if ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    if n_blocks is not None and ranks > n_blocks:
        ranks = n_blocks
    command = [
        config.launcher,
        LAUNCHER_RANK_FLAG[config.launcher],
        str(ranks),
        *config.launcher_args,
        binary,
        export_dir,
    ]
    driver_options = dict(config.options)
    if config.linear_solver is not None:
        driver_options.setdefault("linear-solver", config.linear_solver)
    for key, value in driver_options.items():
        command += [f"--{key}", str(value)]
    threads = str(config.threads_per_rank)
    env = {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
    return command, env
