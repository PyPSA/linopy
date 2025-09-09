#!/usr/bin/env python3
"""
Module containing all import/export functionalities.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from io import BufferedWriter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from numpy import ones_like, zeros_like
from scipy.sparse import tril, triu
from tqdm import tqdm

from linopy import solvers
from linopy.constants import CONCAT_DIM
from linopy.objective import Objective

if TYPE_CHECKING:
    from highspy.highs import Highs

    from linopy.model import Model


logger = logging.getLogger(__name__)


ufunc_kwargs = dict(vectorize=True)
concat_kwargs = dict(dim=CONCAT_DIM, coords="minimal")

TQDM_COLOR = "#80bfff"


name_sanitizer = str.maketrans("-+*^[] ", "_______")


def clean_name(name: str) -> str:
    return name.translate(name_sanitizer)


coord_sanitizer = str.maketrans("[,]", "(,)", " ")


def print_coord(coord: str) -> str:
    from linopy.common import print_coord

    coord = print_coord(coord).translate(coord_sanitizer)
    return coord


def get_printers_scalar(
    m: Model, explicit_coordinate_names: bool = False
) -> tuple[Callable, Callable]:
    """Get printer functions for scalar values (non-polars)."""
    if explicit_coordinate_names:

        def print_variable(var: Any) -> str:
            name, coord = m.variables.get_label_position(var)
            name = clean_name(name)
            return f"{name}{print_coord(coord)}#{var}"

        def print_constraint(cons: Any) -> str:
            name, coord = m.constraints.get_label_position(cons)
            name = clean_name(name)  # type: ignore
            return f"{name}{print_coord(coord)}#{cons}"  # type: ignore

        return print_variable, print_constraint
    else:

        def print_variable(var: Any) -> str:
            return f"x{var}"

        def print_constraint(cons: Any) -> str:
            return f"c{cons}"

        return print_variable, print_constraint


def get_printers(
    m: Model, explicit_coordinate_names: bool = False
) -> tuple[Callable, Callable]:
    """Get printer functions for polars dataframes."""
    if explicit_coordinate_names:

        def print_variable(var: Any) -> str:
            name, coord = m.variables.get_label_position(var)
            name = clean_name(name)
            return f"{name}{print_coord(coord)}#{var}"

        def print_constraint(cons: Any) -> str:
            name, coord = m.constraints.get_label_position(cons)
            name = clean_name(name)  # type: ignore
            return f"{name}{print_coord(coord)}#{cons}"  # type: ignore

        def print_variable_series(series: pl.Series) -> tuple[pl.Expr, pl.Series]:
            return pl.lit(" "), series.map_elements(
                print_variable, return_dtype=pl.String
            )

        def print_constraint_series(series: pl.Series) -> tuple[pl.Expr, pl.Series]:
            return pl.lit(None), series.map_elements(
                print_constraint, return_dtype=pl.String
            )

        return print_variable_series, print_constraint_series

    else:

        def print_variable_series(series: pl.Series) -> tuple[pl.Expr, pl.Series]:
            return pl.lit(" x").alias("x"), series.cast(pl.String)

        def print_constraint_series(series: pl.Series) -> tuple[pl.Expr, pl.Series]:
            return pl.lit("c").alias("c"), series.cast(pl.String)

        return print_variable_series, print_constraint_series


def objective_write_linear_terms(
    f: BufferedWriter, df: pl.DataFrame, print_variable: Callable
) -> None:
    cols = [
        pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
        pl.col("coeffs").cast(pl.String),
        *print_variable(pl.col("vars")),
    ]
    df = df.select(pl.concat_str(cols, ignore_nulls=True))
    df.write_csv(
        f, separator=" ", null_value="", quote_style="never", include_header=False
    )


def objective_write_quadratic_terms(
    f: BufferedWriter, df: pl.DataFrame, print_variable: Callable
) -> None:
    cols = [
        pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
        pl.col("coeffs").mul(2).cast(pl.String),
        *print_variable(pl.col("vars1")),
        pl.lit(" *"),
        *print_variable(pl.col("vars2")),
    ]
    f.write(b"+ [\n")
    df = df.select(pl.concat_str(cols, ignore_nulls=True))
    df.write_csv(
        f, separator=" ", null_value="", quote_style="never", include_header=False
    )
    f.write(b"] / 2\n")


def objective_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out the objective of a model to a lp file.
    """
    if progress:
        logger.info("Writing objective.")

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    sense = m.objective.sense
    f.write(f"{sense}\n\nobj:\n\n".encode())
    df = m.objective.to_polars()

    if m.is_linear:
        objective_write_linear_terms(f, df, print_variable)

    elif m.is_quadratic:
        linear_terms = df.filter(pl.col("vars1").eq(-1) | pl.col("vars2").eq(-1))
        linear_terms = linear_terms.with_columns(
            pl.when(pl.col("vars1").eq(-1))
            .then(pl.col("vars2"))
            .otherwise(pl.col("vars1"))
            .alias("vars")
        )
        objective_write_linear_terms(f, linear_terms, print_variable)

        quads = df.filter(pl.col("vars1").ne(-1) & pl.col("vars2").ne(-1))
        objective_write_quadratic_terms(f, quads, print_variable)


def bounds_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out variables of a model to a lp file.
    """
    names = list(m.variables.continuous) + list(m.variables.integers)
    if not len(list(names)):
        return

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\nbounds\n\n")
    if progress:
        names = tqdm(
            list(names),
            desc="Writing continuous variables.",
            colour=TQDM_COLOR,
        )

    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.to_polars()

            columns = [
                pl.when(pl.col("lower") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
                pl.col("lower").cast(pl.String),
                pl.lit(" <= "),
                *print_variable(pl.col("labels")),
                pl.lit(" <= "),
                pl.when(pl.col("upper") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
                pl.col("upper").cast(pl.String),
            ]

            kwargs: Any = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def binaries_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out binaries of a model to a lp file.
    """
    names = m.variables.binaries
    if not len(list(names)):
        return

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\nbinary\n\n")
    if progress:
        names = tqdm(
            list(names),
            desc="Writing binary variables.",
            colour=TQDM_COLOR,
        )

    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.to_polars()

            columns = [
                *print_variable(pl.col("labels")),
            ]

            kwargs: Any = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def integers_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    integer_label: str = "general",
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out integers of a model to a lp file.
    """
    names = m.variables.integers
    if not len(list(names)):
        return

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(f"\n\n{integer_label}\n\n".encode())
    if progress:
        names = tqdm(
            list(names),
            desc="Writing integer variables.",
            colour=TQDM_COLOR,
        )

    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.to_polars()

            columns = [
                *print_variable(pl.col("labels")),
            ]

            kwargs: Any = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def constraints_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    lazy: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    if not len(m.constraints):
        return

    print_variable, print_constraint = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\ns.t.\n\n")
    names = m.constraints
    if progress:
        names = tqdm(
            list(names),
            desc="Writing constraints.",
            colour=TQDM_COLOR,
        )

    # to make this even faster, we can use polars expression
    # https://docs.pola.rs/user-guide/expressions/plugins/#output-data-types
    for name in names:
        con = m.constraints[name]
        for con_slice in con.iterate_slices(slice_size):
            df = con_slice.to_polars()

            if df.height == 0:
                continue

            # Ensure each constraint has both coefficient and RHS terms
            analysis = df.group_by("labels").agg(
                [
                    pl.col("coeffs").is_not_null().sum().alias("coeff_rows"),
                    pl.col("sign").is_not_null().sum().alias("rhs_rows"),
                ]
            )

            valid = analysis.filter(
                (pl.col("coeff_rows") > 0) & (pl.col("rhs_rows") > 0)
            )

            if valid.height == 0:
                continue

            # Keep only constraints that have both parts
            df = df.join(valid.select("labels"), on="labels", how="inner")

            # Sort by labels and mark first/last occurrences
            df = df.sort("labels").with_columns(
                [
                    pl.when(pl.col("labels").is_first_distinct())
                    .then(pl.col("labels"))
                    .otherwise(pl.lit(None))
                    .alias("labels_first"),
                    (pl.col("labels") != pl.col("labels").shift(-1))
                    .fill_null(True)
                    .alias("is_last_in_group"),
                ]
            )

            row_labels = print_constraint(pl.col("labels_first"))
            col_labels = print_variable(pl.col("vars"))
            columns = [
                pl.when(pl.col("labels_first").is_not_null()).then(row_labels[0]),
                pl.when(pl.col("labels_first").is_not_null()).then(row_labels[1]),
                pl.when(pl.col("labels_first").is_not_null())
                .then(pl.lit(":\n"))
                .alias(":"),
                pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")),
                pl.col("coeffs").cast(pl.String),
                pl.when(pl.col("vars").is_not_null()).then(col_labels[0]),
                pl.when(pl.col("vars").is_not_null()).then(col_labels[1]),
                pl.when(pl.col("is_last_in_group")).then(pl.col("sign")),
                pl.when(pl.col("is_last_in_group")).then(pl.lit(" ")),
                pl.when(pl.col("is_last_in_group")).then(pl.col("rhs").cast(pl.String)),
            ]

            kwargs: Any = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)

            # in the future, we could use lazy dataframes when they support appending
            # tp existent files
            # formatted = df.lazy().select(pl.concat_str(columns, ignore_nulls=True))
            # formatted.sink_csv(f,  **kwargs)


def to_lp_file(
    m: Model,
    fn: Path,
    integer_label: str = "general",
    slice_size: int = 2_000_000,
    progress: bool = True,
    explicit_coordinate_names: bool = False,
) -> None:
    with open(fn, mode="wb") as f:
        start = time.time()

        objective_to_file(
            m, f, progress=progress, explicit_coordinate_names=explicit_coordinate_names
        )
        constraints_to_file(
            m,
            f=f,
            progress=progress,
            slice_size=slice_size,
            explicit_coordinate_names=explicit_coordinate_names,
        )
        bounds_to_file(
            m,
            f=f,
            progress=progress,
            slice_size=slice_size,
            explicit_coordinate_names=explicit_coordinate_names,
        )
        binaries_to_file(
            m,
            f=f,
            progress=progress,
            slice_size=slice_size,
            explicit_coordinate_names=explicit_coordinate_names,
        )
        integers_to_file(
            m,
            integer_label=integer_label,
            f=f,
            progress=progress,
            slice_size=slice_size,
            explicit_coordinate_names=explicit_coordinate_names,
        )
        f.write(b"end\n")

        logger.info(f" Writing time: {round(time.time() - start, 2)}s")


def to_file(
    m: Model,
    fn: Path | str | None,
    io_api: str | None = None,
    integer_label: str = "general",
    slice_size: int = 2_000_000,
    progress: bool | None = None,
    explicit_coordinate_names: bool = False,
) -> Path:
    """
    Write out a model to a lp or mps file.
    """
    if fn is None:
        fn = Path(m.get_problem_file())
    if isinstance(fn, str):
        fn = Path(fn)
    if fn.exists():
        fn.unlink()

    if io_api is None:
        io_api = fn.suffix[1:]

    if progress is None:
        progress = m._xCounter > 10_000

    if io_api == "lp" or io_api == "lp-polars":
        to_lp_file(
            m,
            fn,
            integer_label,
            slice_size=slice_size,
            progress=progress,
            explicit_coordinate_names=explicit_coordinate_names,
        )

    elif io_api == "mps":
        if "highs" not in solvers.available_solvers:
            raise RuntimeError(
                "Package highspy not installed. This is required to exporting to MPS file."
            )

        # Use very fast highspy implementation
        # Might be replaced by custom writer, however needs C/Rust bindings for performance
        h = m.to_highspy(explicit_coordinate_names=explicit_coordinate_names)
        h.writeModel(str(fn))
    else:
        raise ValueError(
            f"Invalid io_api '{io_api}'. Choose from 'lp', 'lp-polars' or 'mps'."
        )

    return fn


def to_mosek(
    m: Model, task: Any | None = None, explicit_coordinate_names: bool = False
) -> Any:
    """
    Export model to MOSEK.

    Export the model directly to MOSEK without writing files.

    Parameters
    ----------
    m : linopy.Model
    task : empty MOSEK task

    Returns
    -------
    task : MOSEK Task object
    """

    import mosek

    print_variable, print_constraint = get_printers_scalar(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    if task is None:
        task = mosek.Task()

    task.appendvars(m.nvars)
    task.appendcons(m.ncons)

    M = m.matrices
    # for j, n in enumerate(("x" + M.vlabels.astype(str).astype(object))):
    #    task.putvarname(j, n)

    labels = np.vectorize(print_variable)(M.vlabels).astype(object)
    task.generatevarnames(
        np.arange(0, len(labels)), "%0", [len(labels)], None, [0], list(labels)
    )

    ## Variables

    # MOSEK uses bound keys (free, bounded below or above, ranged and fixed)
    # plus bound values (lower and upper), and it is considered an error to
    # input an infinite value for a finite bound.
    # bkx and bkc define the boundkeys based on upper and lower bound, and blx,
    # bux, blc and buc define the finite bounds. The numerical value of a bound
    # indicated to be infinite by the bound key is ignored by MOSEK.
    bkx = [
        (
            (
                (mosek.boundkey.ra if lb < ub else mosek.boundkey.fx)
                if ub < np.inf
                else mosek.boundkey.lo
            )
            if (lb > -np.inf)
            else (mosek.boundkey.up if (ub < np.inf) else mosek.boundkey.fr)
        )
        for (lb, ub) in zip(M.lb, M.ub)
    ]
    blx = [b if b > -np.inf else 0.0 for b in M.lb]
    bux = [b if b < np.inf else 0.0 for b in M.ub]
    task.putvarboundslice(0, m.nvars, bkx, blx, bux)

    if len(m.binaries.labels) + len(m.integers.labels) > 0:
        idx = [i for (i, v) in enumerate(M.vtypes) if v in ["B", "I"]]
        task.putvartypelist(idx, [mosek.variabletype.type_int] * len(idx))
        if len(m.binaries.labels) > 0:
            bidx = [i for (i, v) in enumerate(M.vtypes) if v == "B"]
            task.putvarboundlistconst(bidx, mosek.boundkey.ra, 0.0, 1.0)

    ## Constraints

    if len(m.constraints) > 0:
        names = np.vectorize(print_constraint)(M.clabels).astype(object)
        for i, n in enumerate(names):
            task.putconname(i, n)
        bkc = [
            (
                (mosek.boundkey.up if b < np.inf else mosek.boundkey.fr)
                if s == "<"
                else (
                    (mosek.boundkey.lo if b > -np.inf else mosek.boundkey.up)
                    if s == ">"
                    else mosek.boundkey.fx
                )
            )
            for s, b in zip(M.sense, M.b)
        ]
        blc = [b if b > -np.inf else 0.0 for b in M.b]
        buc = [b if b < np.inf else 0.0 for b in M.b]
        # blc = M.b
        # buc = M.b
        if M.A is not None:
            A = M.A.tocsr()
            task.putarowslice(
                0, m.ncons, A.indptr[:-1], A.indptr[1:], A.indices, A.data
            )
            task.putconboundslice(0, m.ncons, bkc, blc, buc)

    ## Objective
    if M.Q is not None:
        Q = (0.5 * tril(M.Q + M.Q.transpose())).tocoo()
        task.putqobj(Q.row, Q.col, Q.data)
    task.putclist(list(np.arange(m.nvars)), M.c)

    if m.objective.sense == "max":
        task.putobjsense(mosek.objsense.maximize)
    else:
        task.putobjsense(mosek.objsense.minimize)
    return task


def to_gurobipy(
    m: Model, env: Any | None = None, explicit_coordinate_names: bool = False
) -> Any:
    """
    Export the model to gurobipy.

    This function does not write the model to intermediate files but directly
    passes it to gurobipy. Note that for large models this is not
    computationally efficient.

    Parameters
    ----------
    m : linopy.Model
    env : gurobipy.Env

    Returns
    -------
    model : gurobipy.Model
    """
    import gurobipy

    print_variable, print_constraint = get_printers_scalar(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    m.constraints.sanitize_missings()
    model = gurobipy.Model(env=env)

    M = m.matrices

    names = np.vectorize(print_variable)(M.vlabels).astype(object)
    kwargs = {}
    if len(m.binaries.labels) + len(m.integers.labels):
        kwargs["vtype"] = M.vtypes
    x = model.addMVar(M.vlabels.shape, M.lb, M.ub, name=list(names), **kwargs)

    if m.is_quadratic:
        model.setObjective(0.5 * x.T @ M.Q @ x + M.c @ x)  # type: ignore
    else:
        model.setObjective(M.c @ x)

    if m.objective.sense == "max":
        model.ModelSense = -1

    if len(m.constraints):
        names = np.vectorize(print_constraint)(M.clabels).astype(object)
        c = model.addMConstr(M.A, x, M.sense, M.b)  # type: ignore
        c.setAttr("ConstrName", list(names))  # type: ignore

    model.update()
    return model


def to_highspy(m: Model, explicit_coordinate_names: bool = False) -> Highs:
    """
    Export the model to highspy.

    This function does not write the model to intermediate files but directly
    passes it to highspy.

    Note, this function does not track variable and constraint labels.

    Parameters
    ----------
    m : linopy.Model

    Returns
    -------
    model : highspy.Highs
    """
    import highspy

    print_variable, print_constraint = get_printers_scalar(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    M = m.matrices
    h = highspy.Highs()
    h.addVars(len(M.vlabels), M.lb, M.ub)
    if len(m.binaries) + len(m.integers):
        vtypes = M.vtypes
        labels = np.arange(len(vtypes))[(vtypes == "B") | (vtypes == "I")]
        n = len(labels)
        h.changeColsIntegrality(n, labels, ones_like(labels))
        if len(m.binaries):
            labels = np.arange(len(vtypes))[vtypes == "B"]
            n = len(labels)
            h.changeColsBounds(n, labels, zeros_like(labels), ones_like(labels))

    # linear objective
    h.changeColsCost(len(M.c), np.arange(len(M.c), dtype=np.int32), M.c)

    # linear constraints
    A = M.A
    if A is not None:
        A = A.tocsr()
        num_cons = A.shape[0]
        lower = np.where(M.sense != "<", M.b, -np.inf)
        upper = np.where(M.sense != ">", M.b, np.inf)
        h.addRows(num_cons, lower, upper, A.nnz, A.indptr, A.indices, A.data)

    lp = h.getLp()
    lp.col_names_ = np.vectorize(print_variable)(M.vlabels).astype(object)
    if len(M.clabels):
        lp.row_names_ = np.vectorize(print_constraint)(M.clabels).astype(object)
    h.passModel(lp)

    # quadrative objective
    Q = M.Q
    if Q is not None:
        Q = triu(Q)
        Q = Q.tocsr()
        num_vars = Q.shape[0]
        h.passHessian(num_vars, Q.nnz, 1, Q.indptr, Q.indices, Q.data)

    # change objective sense
    if m.objective.sense == "max":
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    return h


def to_block_files(m: Model, fn: Path) -> None:
    """
    Write out the linopy model to a block structured output.

    Experimental: This function does not support grouping duplicated variables
    in one constraint row yet!
    """
    if fn is None:
        fn = Path(TemporaryDirectory(prefix="linopy-problem-", dir=m.solver_dir).name)

    path = Path(fn)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)

    m.calculate_block_maps()

    if m.blocks is None:
        raise ValueError("Model does not have blocks defined.")

    N = int(m.blocks.max())
    for n in range(N + 2):
        (path / f"block{n}").mkdir()

    raise NotImplementedError("This function is not yet implemented fully.")

    # # Write out variables
    # blocks = vars.ravel("blocks", filter_missings=True)
    # for key, suffix in zip(["labels", "lower", "upper"], ["x", "xl", "xu"]):
    #     arr = vars.ravel(key, filter_missings=True)
    #     for n in tqdm(range(N + 1), desc=f"Write variable {key}"):
    #         arr[blocks == n].tofile(path / f"block{n}" / suffix, sep="\n")

    # # Write out objective (uses variable blocks from above)
    # coeffs = np.zeros(m._xCounter)
    # coeffs[np.asarray(m.objective.vars)] = np.asarray(m.objective.coeffs)
    # # reorder like non-missing variables
    # coeffs = coeffs[vars.ravel("labels", filter_missings=True)]
    # for n in tqdm(range(N + 1), desc="Write objective"):
    #     coeffs[blocks == n].tofile(path / f"block{n}" / "c", sep="\n")

    # # Write out rhs
    # blocks = cons.ravel("blocks", filter_missings=True)
    # rhs = cons.ravel("rhs", filter_missings=True)
    # is_equality = cons.ravel(cons.sign == EQUAL, filter_missings=True)
    # is_lower_bound = cons.ravel(cons.sign == GREATER_EQUAL, filter_missings=True)

    # for n in tqdm(range(N + 2), desc="Write RHS"):
    #     is_blockn = blocks == n

    #     rhs[is_blockn & is_equality].tofile(path / f"block{n}" / "b", sep="\n")

    #     not_equality = is_blockn & ~is_equality
    #     is_lower_bound_sub = is_lower_bound[not_equality]
    #     rhs_sub = rhs[not_equality]

    #     lower_bound = np.where(is_lower_bound_sub, rhs_sub, -np.inf)
    #     lower_bound.tofile(path / f"block{n}" / "dl", sep="\n")

    #     upper_bound = np.where(~is_lower_bound_sub, rhs_sub, np.inf)
    #     upper_bound.tofile(path / f"block{n}" / "du", sep="\n")

    # # Write out constraints
    # conblocks = cons.ravel("blocks", "vars", filter_missings=True)
    # varblocks = cons.ravel("var_blocks", "vars", filter_missings=True)
    # is_equality = cons.ravel(cons.sign == EQUAL, "vars", filter_missings=True)

    # is_varblock_0 = varblocks == 0
    # is_conblock_L = conblocks == N + 1

    # keys = ["labels", "coeffs", "vars"]

    # def filtered(arr, mask, key):
    #     """
    #     Set coefficients to zero where mask is False, keep others unchanged.

    #     PIPS requires this information to set the shape of sub-matrices.
    #     """
    #     assert key in keys
    #     return np.where(mask, arr, 0) if key == "coeffs" else arr

    # for key, suffix in zip(keys, ["row", "data", "col"]):
    #     arr = cons.ravel(key, "vars", filter_missings=True)
    #     for n in tqdm(range(N + 1), desc=f"Write constraint {key}"):
    #         is_conblock_n = conblocks == n
    #         is_varblock_n = varblocks == n

    #         mask = is_conblock_n & is_equality
    #         filtered(arr[mask], is_varblock_0[mask], key).tofile(
    #             path / f"block{n}" / f"A_{suffix}", sep="\n"
    #         )
    #         mask = is_conblock_n & ~is_equality
    #         filtered(arr[mask], is_varblock_0[mask], key).tofile(
    #             path / f"block{n}" / f"C_{suffix}", sep="\n"
    #         )

    #         mask = is_conblock_L & is_equality
    #         filtered(arr[mask], is_varblock_n[mask], key).tofile(
    #             path / f"block{n}" / f"BL_{suffix}", sep="\n"
    #         )
    #         mask = is_conblock_L & ~is_equality
    #         filtered(arr[mask], is_varblock_n[mask], key).tofile(
    #             path / f"block{n}" / f"DL_{suffix}", sep="\n"
    #         )

    #         if n:
    #             mask = is_conblock_n & is_equality
    #             filtered(arr[mask], is_varblock_n[mask], key).tofile(
    #                 path / f"block{n}" / f"B_{suffix}", sep="\n"
    #             )
    #             mask = is_conblock_n & ~is_equality
    #             filtered(arr[mask], is_varblock_n[mask], key).tofile(
    #                 path / f"block{n}" / f"D_{suffix}", sep="\n"
    #             )


def non_bool_dict(
    d: dict[str, tuple[int, int] | str | bool | list[str] | int],
) -> dict[str, tuple[int, int] | str | int | list[str]]:
    """
    Convert bool to int for netCDF4 storing.
    """
    return {k: int(v) if isinstance(v, bool) else v for k, v in d.items()}


def to_netcdf(m: Model, *args: Any, **kwargs: Any) -> None:
    """
    Write out the model to a netcdf file.

    Parameters
    ----------
    m : linopy.Model
        Model to write out.
    *args
        Arguments passed to ``xarray.Dataset.to_netcdf``.
    **kwargs : TYPE
        Keyword arguments passed to ``xarray.Dataset.to_netcdf``.
    """

    def with_prefix(ds: xr.Dataset, prefix: str) -> xr.Dataset:
        to_rename = set([*ds.dims, *ds.coords, *ds])
        ds = ds.rename({d: f"{prefix}-{d}" for d in to_rename})
        ds.attrs = {f"{prefix}-{k}": v for k, v in ds.attrs.items()}

        # Flatten multiindexes
        for dim in ds.dims:
            if isinstance(ds[dim].to_index(), pd.MultiIndex):
                prefix_len = len(prefix) + 1  # leave original index level name
                names = [n[prefix_len:] for n in ds[dim].to_index().names]
                ds = ds.reset_index(dim)
                ds.attrs[f"{dim}_multiindex"] = list(names)

        return ds

    vars = [
        with_prefix(var.data, f"variables-{name}") for name, var in m.variables.items()
    ]
    cons = [
        with_prefix(con.data, f"constraints-{name}")
        for name, con in m.constraints.items()
    ]
    objective = m.objective.data
    objective = objective.assign_attrs(sense=m.objective.sense)
    if m.objective.value is not None:
        objective = objective.assign_attrs(value=m.objective.value)
    obj = [with_prefix(objective, "objective")]
    params = [with_prefix(m.parameters, "parameters")]

    scalars = {k: getattr(m, k) for k in m.scalar_attrs}
    ds = xr.merge(vars + cons + obj + params, combine_attrs="drop_conflicts")
    ds = ds.assign_attrs(scalars)
    ds.attrs = non_bool_dict(ds.attrs)

    for k in ds:
        ds[k].attrs = non_bool_dict(ds[k].attrs)

    ds.to_netcdf(*args, **kwargs)


def read_netcdf(path: Path | str, **kwargs: Any) -> Model:
    """
    Read in a model from a netcdf file.

    Parameters
    ----------
    path : path_like
        Path of the stored model.
    **kwargs
        Keyword arguments passed to ``xarray.load_dataset``.

    Returns
    -------
    m : linopy.Model
    """
    from linopy.model import (
        Constraint,
        Constraints,
        LinearExpression,
        Model,
        Variable,
        Variables,
    )

    if isinstance(path, str):
        path = Path(path)

    m = Model()
    ds = xr.load_dataset(path, **kwargs)

    def has_prefix(k: str, prefix: str) -> bool:
        return k.rsplit("-", 1)[0] == prefix

    def remove_prefix(k: str, prefix: str) -> str:
        return k[len(prefix) + 1 :]

    def get_prefix(ds: xr.Dataset, prefix: str) -> xr.Dataset:
        ds = ds[[k for k in ds if has_prefix(str(k), prefix)]]
        multiindexes = []
        for dim in ds.dims:
            for name in ds.attrs.get(f"{dim}_multiindex", []):
                multiindexes.append(prefix + "-" + name)
        ds = ds.drop_vars(set(ds.coords) - set(ds.dims) - set(multiindexes))
        to_rename = set([*ds.dims, *ds.coords, *ds])
        ds = ds.rename({d: remove_prefix(d, prefix) for d in to_rename})
        ds.attrs = {
            remove_prefix(k, prefix): v
            for k, v in ds.attrs.items()
            if has_prefix(k, prefix)
        }

        for dim in ds.dims:
            if f"{dim}_multiindex" in ds.attrs:
                names = ds.attrs.pop(f"{dim}_multiindex")
                ds = ds.set_index({dim: names})

        return ds

    vars = [str(k) for k in ds if str(k).startswith("variables")]
    var_names = list({str(k).rsplit("-", 1)[0] for k in vars})
    variables = {}
    for k in sorted(var_names):
        name = remove_prefix(k, "variables")
        variables[name] = Variable(get_prefix(ds, k), m, name)

    m._variables = Variables(variables, m)

    cons = [str(k) for k in ds if str(k).startswith("constraints")]
    con_names = list({str(k).rsplit("-", 1)[0] for k in cons})
    constraints = {}
    for k in sorted(con_names):
        name = remove_prefix(k, "constraints")
        constraints[name] = Constraint(get_prefix(ds, k), m, name)
    m._constraints = Constraints(constraints, m)

    objective = get_prefix(ds, "objective")
    m.objective = Objective(
        LinearExpression(objective, m), m, objective.attrs.pop("sense")
    )
    m.objective._value = objective.attrs.pop("value", None)

    m.parameters = get_prefix(ds, "parameters")

    for k in m.scalar_attrs:
        setattr(m, k, ds.attrs.get(k))

    return m
