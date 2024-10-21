#!/usr/bin/env python3
"""
Module containing all import/export functionalities.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from numpy import ones_like, zeros_like
from pandas.core.frame import DataFrame
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


def handle_batch(batch: list[str], f: TextIOWrapper, batch_size: int) -> list[str]:
    """
    Write out a batch to a file and reset the batch.
    """
    if len(batch) >= batch_size:
        f.writelines(batch)  # write out a batch
        batch = []  # reset batch
    return batch


def objective_write_linear_terms(
    df: DataFrame, f: TextIOWrapper, batch: list[Any], batch_size: int
) -> list[str | Any]:
    """
    Write the linear terms of the objective to a file.
    """
    coeffs = df.coeffs.values
    vars = df.vars.values
    for idx in range(len(coeffs)):
        coeff = coeffs[idx]
        var = vars[idx]
        batch.append(f"{coeff:+.12g} x{var}\n")
        batch = handle_batch(batch, f, batch_size)
    return batch


def objective_write_quad_terms(
    quadratic: DataFrame, f: TextIOWrapper, batch: list[str], batch_size: int
) -> list[str]:
    """
    Write the cross terms of the objective to a file.
    """
    coeffs = quadratic.coeffs.values
    vars1 = quadratic.vars1.values
    vars2 = quadratic.vars2.values
    for idx in range(len(coeffs)):
        coeff = coeffs[idx]
        var1 = vars1[idx]
        var2 = vars2[idx]
        batch.append(f"{coeff:+.12g} x{var1} * x{var2}\n")
        batch = handle_batch(batch, f, batch_size)
    return batch


def objective_to_file(
    m: Model, f: TextIOWrapper, log: bool = False, batch_size: int = 10000
) -> None:
    """
    Write out the objective of a model to a lp file.
    """
    if log:
        logger.info("Writing objective.")

    sense = m.objective.sense
    f.write(f"{sense}\n\nobj:\n\n")
    df = m.objective.flat

    if np.isnan(df.coeffs).any():
        logger.warning(
            "Objective coefficients are missing (nan) where variables are not (-1)."
        )

    if m.is_linear:
        batch = objective_write_linear_terms(df, f, [], batch_size)

    elif m.is_quadratic:
        is_linear = (df.vars1 == -1) | (df.vars2 == -1)
        linear = df[is_linear]
        linear = linear.assign(
            vars=linear.vars1.where(linear.vars1 != -1, linear.vars2)
        )
        batch = objective_write_linear_terms(linear, f, [], batch_size)

        if not is_linear.all():
            batch.append("+ [\n")
            quadratic = df[~is_linear]
            quadratic = quadratic.assign(coeffs=2 * quadratic.coeffs)
            batch = objective_write_quad_terms(quadratic, f, batch, batch_size)
            batch.append("] / 2\n")

    if batch:  # write the remaining lines
        f.writelines(batch)


def constraints_to_file(
    m: Model,
    f: TextIOWrapper,
    log: bool = False,
    batch_size: int = 50_000,
    slice_size: int = 100_000,
) -> None:
    if not len(m.constraints):
        return

    f.write("\n\ns.t.\n\n")
    names: Iterable = m.constraints
    if log:
        names = tqdm(
            list(names),
            desc="Writing constraints.",
            colour=TQDM_COLOR,
        )

    batch = []
    for name in names:
        con = m.constraints[name]
        for con_slice in con.iterate_slices(slice_size):
            df = con_slice.flat

            labels = df.labels.values
            vars = df.vars.values
            coeffs = df.coeffs.values
            rhs = df.rhs.values
            sign = df.sign.values

            len_df = len(df)  # compute length once
            if not len_df:
                continue

            # write out the start to enable a fast loop afterwards
            idx = 0
            label = labels[idx]
            coeff = coeffs[idx]
            var = vars[idx]
            batch.append(f"c{label}:\n{coeff:+.12g} x{var}\n")
            prev_label = label
            prev_sign = sign[idx]
            prev_rhs = rhs[idx]

            for idx in range(1, len_df):
                label = labels[idx]
                coeff = coeffs[idx]
                var = vars[idx]

                if label != prev_label:
                    batch.append(
                        f"{prev_sign} {prev_rhs:+.12g}\n\nc{label}:\n{coeff:+.12g} x{var}\n"
                    )
                    prev_sign = sign[idx]
                    prev_rhs = rhs[idx]
                else:
                    batch.append(f"{coeff:+.12g} x{var}\n")

                batch = handle_batch(batch, f, batch_size)

                prev_label = label

            batch.append(f"{prev_sign} {prev_rhs:+.12g}\n")

    if batch:  # write the remaining lines
        f.writelines(batch)


def bounds_to_file(
    m: Model,
    f: TextIOWrapper,
    log: bool = False,
    batch_size: int = 10000,
    slice_size: int = 100_000,
) -> None:
    """
    Write out variables of a model to a lp file.
    """
    names: Iterable = list(m.variables.continuous) + list(m.variables.integers)
    if not len(list(names)):
        return

    f.write("\n\nbounds\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing continuous variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.flat

            labels = df.labels.values
            lowers = df.lower.values
            uppers = df.upper.values

            for idx in range(len(df)):
                label = labels[idx]
                lower = lowers[idx]
                upper = uppers[idx]
                batch.append(f"{lower:+.12g} <= x{label} <= {upper:+.12g}\n")
                batch = handle_batch(batch, f, batch_size)

    if batch:  # write the remaining lines
        f.writelines(batch)


def binaries_to_file(
    m: Model,
    f: TextIOWrapper,
    log: bool = False,
    batch_size: int = 1000,
    slice_size: int = 100_000,
) -> None:
    """
    Write out binaries of a model to a lp file.
    """
    names: Iterable = m.variables.binaries
    if not len(list(names)):
        return

    f.write("\n\nbinary\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing binary variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.flat

            for label in df.labels.values:
                batch.append(f"x{label}\n")
                batch = handle_batch(batch, f, batch_size)

    if batch:  # write the remaining lines
        f.writelines(batch)


def integers_to_file(
    m: Model,
    f: TextIOWrapper,
    log: bool = False,
    batch_size: int = 1000,
    slice_size: int = 100_000,
    integer_label: str = "general",
) -> None:
    """
    Write out integers of a model to a lp file.
    """
    names: Iterable = m.variables.integers
    if not len(list(names)):
        return

    f.write(f"\n\n{integer_label}\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing integer variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.flat

            for label in df.labels.values:
                batch.append(f"x{label}\n")
                batch = handle_batch(batch, f, batch_size)

    if batch:  # write the remaining lines
        f.writelines(batch)


def to_lp_file(m: Model, fn: Path, integer_label: str, slice_size: int = 10_000_000):
    log = m._xCounter > 10_000

    batch_size = 5000

    with open(fn, mode="w") as f:
        start = time.time()

        if isinstance(f, int):
            raise ValueError("File not found.")

        objective_to_file(m, f, log=log)
        constraints_to_file(
            m, f=f, log=log, batch_size=batch_size, slice_size=slice_size
        )
        bounds_to_file(m, f=f, log=log, batch_size=batch_size, slice_size=slice_size)
        binaries_to_file(m, f=f, log=log, batch_size=batch_size, slice_size=slice_size)
        integers_to_file(
            m,
            integer_label=integer_label,
            f=f,
            log=log,
            batch_size=batch_size,
            slice_size=slice_size,
        )
        f.write("end\n")

        logger.info(f" Writing time: {round(time.time()-start, 2)}s")


def objective_write_linear_terms_polars(f, df):
    cols = [
        pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
        pl.col("coeffs").cast(pl.String),
        pl.lit(" x"),
        pl.col("vars").cast(pl.String),
    ]
    df = df.select(pl.concat_str(cols, ignore_nulls=True))
    df.write_csv(
        f, separator=" ", null_value="", quote_style="never", include_header=False
    )


def objective_write_quadratic_terms_polars(f, df):
    cols = [
        pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
        pl.col("coeffs").mul(2).cast(pl.String),
        pl.lit(" x"),
        pl.col("vars1").cast(pl.String),
        pl.lit(" * x"),
        pl.col("vars2").cast(pl.String),
    ]
    f.write(b"+ [\n")
    df = df.select(pl.concat_str(cols, ignore_nulls=True))
    df.write_csv(
        f, separator=" ", null_value="", quote_style="never", include_header=False
    )
    f.write(b"] / 2\n")


def objective_to_file_polars(m, f, log=False):
    """
    Write out the objective of a model to a lp file.
    """
    if log:
        logger.info("Writing objective.")

    sense = m.objective.sense
    f.write(f"{sense}\n\nobj:\n\n".encode())
    df = m.objective.to_polars()

    if m.is_linear:
        objective_write_linear_terms_polars(f, df)

    elif m.is_quadratic:
        lins = df.filter(pl.col("vars1").eq(-1) | pl.col("vars2").eq(-1))
        lins = lins.with_columns(
            pl.when(pl.col("vars1").eq(-1))
            .then(pl.col("vars2"))
            .otherwise(pl.col("vars1"))
            .alias("vars")
        )
        objective_write_linear_terms_polars(f, lins)

        quads = df.filter(pl.col("vars1").ne(-1) & pl.col("vars2").ne(-1))
        objective_write_quadratic_terms_polars(f, quads)


def bounds_to_file_polars(m, f, log=False, slice_size=2_000_000):
    """
    Write out variables of a model to a lp file.
    """
    names = list(m.variables.continuous) + list(m.variables.integers)
    if not len(list(names)):
        return

    f.write(b"\n\nbounds\n\n")
    if log:
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
                pl.lit(" <= x"),
                pl.col("labels").cast(pl.String),
                pl.lit(" <= "),
                pl.when(pl.col("upper") >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
                pl.col("upper").cast(pl.String),
            ]

            kwargs = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def binaries_to_file_polars(m, f, log=False, slice_size=2_000_000):
    """
    Write out binaries of a model to a lp file.
    """
    names = m.variables.binaries
    if not len(list(names)):
        return

    f.write(b"\n\nbinary\n\n")
    if log:
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
                pl.lit("x"),
                pl.col("labels").cast(pl.String),
            ]

            kwargs = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def integers_to_file_polars(
    m, f, log=False, integer_label="general", slice_size=2_000_000
):
    """
    Write out integers of a model to a lp file.
    """
    names = m.variables.integers
    if not len(list(names)):
        return

    f.write(f"\n\n{integer_label}\n\n".encode())
    if log:
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
                pl.lit("x"),
                pl.col("labels").cast(pl.String),
            ]

            kwargs = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)


def constraints_to_file_polars(m, f, log=False, lazy=False, slice_size=2_000_000):
    if not len(m.constraints):
        return

    f.write(b"\n\ns.t.\n\n")
    names = m.constraints
    if log:
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

            # df = df.lazy()
            # filter out repeated label values
            df = df.with_columns(
                pl.when(pl.col("labels").is_first_distinct())
                .then(pl.col("labels"))
                .otherwise(pl.lit(None))
                .alias("labels")
            )

            columns = [
                pl.when(pl.col("labels").is_not_null()).then(pl.lit("c")).alias("c"),
                pl.col("labels").cast(pl.String),
                pl.when(pl.col("labels").is_not_null()).then(pl.lit(":\n")).alias(":"),
                pl.when(pl.col("coeffs") >= 0).then(pl.lit("+")),
                pl.col("coeffs").cast(pl.String),
                pl.when(pl.col("vars").is_not_null()).then(pl.lit(" x")).alias("x"),
                pl.col("vars").cast(pl.String),
                "sign",
                pl.lit(" "),
                pl.col("rhs").cast(pl.String),
            ]

            kwargs = dict(
                separator=" ", null_value="", quote_style="never", include_header=False
            )
            formatted = df.select(pl.concat_str(columns, ignore_nulls=True))
            formatted.write_csv(f, **kwargs)

            # in the future, we could use lazy dataframes when they support appending
            # tp existent files
            # formatted = df.lazy().select(pl.concat_str(columns, ignore_nulls=True))
            # formatted.sink_csv(f,  **kwargs)


def to_lp_file_polars(m, fn, integer_label="general", slice_size=2_000_000):
    log = m._xCounter > 10_000

    with open(fn, mode="wb") as f:
        start = time.time()

        objective_to_file_polars(m, f, log=log)
        constraints_to_file_polars(m, f=f, log=log, slice_size=slice_size)
        bounds_to_file_polars(m, f=f, log=log, slice_size=slice_size)
        binaries_to_file_polars(m, f=f, log=log, slice_size=slice_size)
        integers_to_file_polars(
            m, integer_label=integer_label, f=f, log=log, slice_size=slice_size
        )
        f.write(b"end\n")

        logger.info(f" Writing time: {round(time.time()-start, 2)}s")


def to_file(
    m: Model,
    fn: Path | str | None,
    io_api: str | None = None,
    integer_label: str = "general",
    slice_size: int = 2_000_000,
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

    if io_api == "lp":
        to_lp_file(m, fn, integer_label, slice_size=slice_size)
    elif io_api == "lp-polars":
        to_lp_file_polars(m, fn, integer_label, slice_size=slice_size)

    elif io_api == "mps":
        if "highs" not in solvers.available_solvers:
            raise RuntimeError(
                "Package highspy not installed. This is required to exporting to MPS file."
            )

        # Use very fast highspy implementation
        # Might be replaced by custom writer, however needs C/Rust bindings for performance
        h = m.to_highspy()
        h.writeModel(str(fn))
    else:
        raise ValueError(
            f"Invalid io_api '{io_api}'. Choose from 'lp', 'lp-polars' or 'mps'."
        )

    return fn


def to_mosek(m: Model, task: Any | None = None) -> Any:
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

    if task is None:
        task = mosek.Task()

    task.appendvars(m.nvars)
    task.appendcons(m.ncons)

    M = m.matrices
    # for j, n in enumerate(("x" + M.vlabels.astype(str).astype(object))):
    #    task.putvarname(j, n)

    labels = M.vlabels.astype(str).astype(object)
    task.generatevarnames(
        np.arange(0, len(labels)), "x%0", [len(labels)], None, [0], list(labels)
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
        names = "c" + M.clabels.astype(str).astype(object)
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


def to_gurobipy(m: Model, env: Any | None = None) -> Any:
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

    m.constraints.sanitize_missings()
    model = gurobipy.Model(env=env)

    M = m.matrices

    names = "x" + M.vlabels.astype(str).astype(object)
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
        names = "c" + M.clabels.astype(str).astype(object)
        c = model.addMConstr(M.A, x, M.sense, M.b)  # type: ignore
        c.setAttr("ConstrName", list(names))  # type: ignore

    model.update()
    return model


def to_highspy(m: Model) -> Highs:
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
    lp.col_names_ = "x" + M.vlabels.astype(str).astype(object)
    if len(M.clabels):
        lp.row_names_ = "c" + M.clabels.astype(str).astype(object)
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


def to_block_files(m: Model, fn: Path):
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


def to_netcdf(m: Model, *args, **kwargs) -> None:
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

    def with_prefix(ds, prefix):
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


def read_netcdf(path: Path | str, **kwargs) -> Model:
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

    def has_prefix(k, prefix):
        return k.rsplit("-", 1)[0] == prefix

    def remove_prefix(k, prefix):
        return k[len(prefix) + 1 :]

    def get_prefix(ds, prefix):
        ds = ds[[k for k in ds if has_prefix(k, prefix)]]
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
