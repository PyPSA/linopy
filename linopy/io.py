#!/usr/bin/env python3
"""
Module containing all import/export functionalities.
"""

from __future__ import annotations

import copy as _copy
import json
import logging
import shutil
import time
import warnings
from collections.abc import Callable, Iterable
from importlib.metadata import version
from io import BufferedWriter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from tqdm import tqdm

from linopy import solvers
from linopy.common import to_polars
from linopy.constants import CONCAT_DIM, SOS_DIM_ATTR, SOS_TYPE_ATTR
from linopy.objective import Objective

if TYPE_CHECKING:
    from cupdlpx import Model as cupdlpxModel
    from highspy.highs import Highs

    from linopy.model import Model
    from linopy.variables import Variable


logger = logging.getLogger(__name__)

NETCDF_VERSION_ATTR = "_linopy_version"


ufunc_kwargs = dict(vectorize=True)
concat_kwargs = dict(dim=CONCAT_DIM, coords="minimal")

TQDM_COLOR = "#80bfff"


name_sanitizer = str.maketrans("-+*^[] ", "_______")


def clean_name(name: str) -> str:
    return name.translate(name_sanitizer)


coord_sanitizer = str.maketrans("[,]", "(,)", " ")


def _format_and_write(
    df: pl.DataFrame, columns: list[pl.Expr], f: BufferedWriter
) -> None:
    """
    Format columns via concat_str and write to file.

    Uses Polars streaming engine for better memory efficiency.
    """
    df.lazy().select(pl.concat_str(columns, ignore_nulls=True)).collect(
        engine="streaming"
    ).write_csv(
        f, separator=" ", null_value="", quote_style="never", include_header=False
    )


def signed_number(expr: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """
    Return polars expressions for a signed number string, handling -0.0 correctly.

    Parameters
    ----------
    expr : pl.Expr
        Numeric value

    Returns
    -------
    tuple[pl.Expr, pl.Expr]
        value_string with sign
    """
    return (
        pl.when(expr >= 0).then(pl.lit("+")).otherwise(pl.lit("")),
        pl.when(expr == 0).then(pl.lit("0.0")).otherwise(expr.cast(pl.String)),
    )


def format_coord(coord: str) -> str:
    from linopy.common import format_coord

    coord = format_coord(coord).translate(coord_sanitizer)
    return coord


def get_printers_scalar(
    m: Model, explicit_coordinate_names: bool = False
) -> tuple[Callable, Callable]:
    """
    Get batch printer functions for numpy label arrays (non-polars).

    Returns two callables that take an int64 numpy array of labels and return
    a list of name strings.
    """
    if explicit_coordinate_names:

        def _fmt_var(var: Any) -> str:
            name, coord = m.variables.get_label_position(var)
            name = clean_name(name)
            return f"{name}{format_coord(coord)}#{var}"

        def _fmt_con(cons: Any) -> str:
            name, coord = m.constraints.get_label_position(cons)
            name = clean_name(name)  # type: ignore
            return f"{name}{format_coord(coord)}#{cons}"  # type: ignore

        def print_variables(labels: np.ndarray) -> list[str]:
            return np.vectorize(_fmt_var)(labels).tolist()

        def print_constraints(labels: np.ndarray) -> list[str]:
            return np.vectorize(_fmt_con)(labels).tolist()

        return print_variables, print_constraints
    else:

        def print_variables(labels: np.ndarray) -> list[str]:
            return ("x" + pl.Series(labels).cast(pl.String)).to_list()

        def print_constraints(labels: np.ndarray) -> list[str]:
            return ("c" + pl.Series(labels).cast(pl.String)).to_list()

        return print_variables, print_constraints


def get_printers(
    m: Model, explicit_coordinate_names: bool = False
) -> tuple[Callable, Callable]:
    """Get printer functions for polars dataframes."""
    if explicit_coordinate_names:

        def print_variable(var: Any) -> str:
            name, coord = m.variables.get_label_position(var)
            name = clean_name(name)
            return f"{name}{format_coord(coord)}#{var}"

        def print_constraint(cons: Any) -> str:
            name, coord = m.constraints.get_label_position(cons)
            name = clean_name(name)  # type: ignore
            return f"{name}{format_coord(coord)}#{cons}"  # type: ignore

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
        *signed_number(pl.col("coeffs")),
        *print_variable(pl.col("vars")),
    ]
    _format_and_write(df, cols, f)


def objective_write_quadratic_terms(
    f: BufferedWriter, df: pl.DataFrame, print_variable: Callable
) -> None:
    cols = [
        *signed_number(pl.col("coeffs").mul(2)),
        *print_variable(pl.col("vars1")),
        pl.lit(" *"),
        *print_variable(pl.col("vars2")),
    ]
    f.write(b"+ [\n")
    _format_and_write(df, cols, f)
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


def _binary_has_nondefault_bounds(var: Variable) -> bool:
    """
    Whether a binary variable carries bounds other than the implied (0, 1).

    Scans the raw bound values (a single vectorised pass each), so masked
    slots are tolerated: a false positive only routes the variable through
    the bounds loop, where masked labels are dropped before writing.
    """
    return bool((var.lower.values != 0).any() or (var.upper.values != 1).any())


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
    names = (
        list(m.variables.continuous)
        + list(m.variables.integers)
        + list(m.variables.semi_continuous)
        + [
            n
            for n in m.variables.binaries
            if _binary_has_nondefault_bounds(m.variables[n])
        ]
    )
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
                *signed_number(pl.col("lower")),
                pl.lit(" <= "),
                *print_variable(pl.col("labels")),
                pl.lit(" <= "),
                *signed_number(pl.col("upper")),
            ]

            _format_and_write(df, columns, f)


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

            _format_and_write(df, columns, f)


def semi_continuous_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out semi-continuous variables of a model to a lp file.
    """
    names = m.variables.semi_continuous
    if not len(list(names)):
        return

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\nsemi-continuous\n\n")
    if progress:
        names = tqdm(
            list(names),
            desc="Writing semi-continuous variables.",
            colour=TQDM_COLOR,
        )

    for name in names:
        var = m.variables[name]
        for var_slice in var.iterate_slices(slice_size):
            df = var_slice.to_polars()

            columns = [
                *print_variable(pl.col("labels")),
            ]

            _format_and_write(df, columns, f)


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

            _format_and_write(df, columns, f)


def sos_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write out SOS constraints of a model to an LP file.
    """
    names = m.variables.sos
    if not len(list(names)):
        return

    print_variable, _ = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\nsos\n\n")
    if progress:
        names = tqdm(
            list(names),
            desc="Writing sos constraints.",
            colour=TQDM_COLOR,
        )

    for name in names:
        var = m.variables[name]
        sos_type = int(var.attrs[SOS_TYPE_ATTR])  # type: ignore[call-overload]
        sos_dim = str(var.attrs[SOS_DIM_ATTR])

        other_dims = [dim for dim in var.labels.dims if dim != sos_dim]
        for var_slice in var.iterate_slices(slice_size, other_dims):
            ds = var_slice.labels.to_dataset()
            # Per-set id = max member label: unique per set (labels are globally
            # unique); a fully-masked set reduces to -1 and is dropped below.
            ds["sos_labels"] = ds["labels"].max(sos_dim)
            ds["weights"] = ds.coords[sos_dim]
            df = to_polars(ds)

            # Drop masked members
            df = df.filter((pl.col("labels") != -1) & (pl.col("sos_labels") != -1))
            if df.is_empty():
                continue

            df = df.group_by("sos_labels").agg(
                pl.concat_str(
                    *print_variable(pl.col("labels")), pl.lit(":"), pl.col("weights")
                )
                .str.join(" ")
                .alias("var_weights")
            )

            columns = [
                pl.lit("s"),
                pl.col("sos_labels"),
                pl.lit(f": S{sos_type} :: "),
                pl.col("var_weights"),
            ]

            _format_and_write(df, columns, f)


def indicator_constraints_to_file(
    m: Model,
    f: BufferedWriter,
    explicit_coordinate_names: bool = False,
) -> None:
    """
    Write indicator constraints to the s.t. section of an LP file.

    Indicator constraints appear in the Subject To section with the format:
    ``ic0: x0 = 1 -> +1.0 x1 <= 5.0``
    """
    if not len(m.constraints.indicator):
        return

    if not len(m.constraints.regular):
        f.write(b"\n\ns.t.\n\n")

    print_variable_scalar, _ = get_printers_scalar(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    for con in m.constraints.indicator.data.values():
        ic_data = con.data
        labels_flat = ic_data.labels.values.flatten()
        binary_var_flat = ic_data.binary_var.values.flatten()
        binary_val_flat = np.broadcast_to(
            ic_data.binary_val.values, labels_flat.shape
        ).flatten()
        coeffs_flat = ic_data.coeffs.values.reshape(len(labels_flat), -1)
        vars_flat = ic_data.vars.values.reshape(len(labels_flat), -1)
        sign_flat = np.broadcast_to(ic_data.sign.values, labels_flat.shape).flatten()
        rhs_flat = np.broadcast_to(ic_data.rhs.values, labels_flat.shape).flatten()

        for i in range(len(labels_flat)):
            if labels_flat[i] == -1:
                continue

            bvar_name = print_variable_scalar(binary_var_flat[i : i + 1])[0]
            valid = vars_flat[i] != -1
            var_names = print_variable_scalar(vars_flat[i][valid])

            terms = []
            for coeff, var_name in zip(coeffs_flat[i][valid], var_names):
                coeff = float(coeff)
                prefix = "+" if coeff >= 0 else ""
                terms.append(f"{prefix}{coeff} {var_name}")

            lhs_str = " ".join(terms)
            line = (
                f"ic{labels_flat[i]}: {bvar_name} = {int(binary_val_flat[i])} -> "
                f"{lhs_str} {sign_flat[i]} {float(rhs_flat[i])}\n"
            )
            f.write(line.encode())


def constraints_to_file(
    m: Model,
    f: BufferedWriter,
    progress: bool = False,
    lazy: bool = False,
    slice_size: int = 2_000_000,
    explicit_coordinate_names: bool = False,
) -> None:
    regular = m.constraints.regular
    if not len(regular):
        return

    print_variable, print_constraint = get_printers(
        m, explicit_coordinate_names=explicit_coordinate_names
    )

    f.write(b"\n\ns.t.\n\n")
    names = list(regular)
    if progress:
        names = tqdm(
            names,
            desc="Writing constraints.",
            colour=TQDM_COLOR,
        )

    # to make this even faster, we can use polars expression
    # https://docs.pola.rs/user-guide/expressions/plugins/#output-data-types
    for name in names:
        con = regular[name]
        for con_slice in con.iterate_slices(slice_size):
            df = con_slice.to_polars()

            if df.height == 0:
                continue

            # Sort by labels and mark first/last occurrences
            df = df.sort("labels").with_columns(
                [
                    pl.col("labels").is_first_distinct().alias("is_first_in_group"),
                    (pl.col("labels") != pl.col("labels").shift(-1))
                    .fill_null(True)
                    .alias("is_last_in_group"),
                ]
            )

            row_labels = print_constraint(pl.col("labels"))
            col_labels = print_variable(pl.col("vars"))
            columns = [
                pl.when(pl.col("is_first_in_group")).then(row_labels[0]),
                pl.when(pl.col("is_first_in_group")).then(row_labels[1]),
                pl.when(pl.col("is_first_in_group")).then(pl.lit(":\n")).alias(":"),
                *signed_number(pl.col("coeffs")),
                col_labels[0],
                col_labels[1],
                pl.when(pl.col("is_last_in_group")).then(pl.lit("\n")),
                pl.when(pl.col("is_last_in_group")).then(pl.col("sign")),
                pl.when(pl.col("is_last_in_group")).then(pl.lit(" ")),
                pl.when(pl.col("is_last_in_group")).then(pl.col("rhs").cast(pl.String)),
            ]

            _format_and_write(df, columns, f)

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
        indicator_constraints_to_file(
            m,
            f=f,
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
        semi_continuous_to_file(
            m,
            f=f,
            progress=progress,
            slice_size=slice_size,
            explicit_coordinate_names=explicit_coordinate_names,
        )
        sos_to_file(
            m,
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
        h = solvers.Highs._build_solver_model(
            m, explicit_coordinate_names=explicit_coordinate_names
        )
        h.writeModel(str(fn))
    else:
        raise ValueError(
            f"Invalid io_api '{io_api}'. Choose from 'lp', 'lp-polars' or 'mps'."
        )

    return fn


def to_mosek(
    m: Model,
    task: Any | None = None,
    explicit_coordinate_names: bool = False,
    set_names: bool = True,
) -> Any:
    """Build the MOSEK task for `m`."""
    import mosek

    if task is None:
        task = mosek.Task()
    return solvers.Mosek._build_solver_model(
        m,
        task,
        explicit_coordinate_names=explicit_coordinate_names,
        set_names=set_names,
    )


def to_gurobipy(
    m: Model,
    env: Any | None = None,
    explicit_coordinate_names: bool = False,
    set_names: bool = True,
) -> Any:
    """Build the gurobipy.Model for `m`."""
    solver = solvers.Gurobi.from_model(
        m,
        io_api="direct",
        explicit_coordinate_names=explicit_coordinate_names,
        set_names=set_names,
        env=env,
    )
    return solver.solver_model


def to_highspy(
    m: Model,
    explicit_coordinate_names: bool = False,
    set_names: bool = True,
) -> Highs:
    """Build the highspy.Highs instance for `m`."""
    solver = solvers.Highs.from_model(
        m,
        io_api="direct",
        explicit_coordinate_names=explicit_coordinate_names,
        set_names=set_names,
    )
    return solver.solver_model


def to_xpress(
    m: Model,
    explicit_coordinate_names: bool = False,
    set_names: bool = True,
) -> Any:
    """Build the xpress.problem instance for `m`."""
    return solvers.Xpress._build_solver_model(
        m,
        explicit_coordinate_names=explicit_coordinate_names,
        set_names=set_names,
    )


def to_cupdlpx(m: Model) -> cupdlpxModel:
    """Build the cupdlpx.Model for `m`."""
    solver = solvers.cuPDLPx.from_model(m, io_api="direct")
    return solver.solver_model


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

    Notes
    -----
    The SOS reformulation lifecycle token lives only on the in-memory
    Model and is not persisted. If the model has an active SOS
    reformulation at serialization time, the netcdf contains the
    reformulated MILP form (aux binaries and cardinality constraints)
    and a :class:`UserWarning` is emitted to flag that the deserialized
    copy will not be able to undo the reformulation.

    ``Model.solve(remote=...)`` invokes ``to_netcdf`` internally on the
    reformulated model and suppresses this warning.
    """
    if m._sos_reformulation_state is not None:
        warnings.warn(
            "Serializing a model with an active SOS reformulation. The "
            "netcdf will contain the reformulated MILP form; the "
            "reformulation lifecycle token is not persisted, so a "
            "reader cannot undo it. Call `model.undo_sos_reformulation()` "
            "first if you want the original SOS form on disk.",
            UserWarning,
            stacklevel=2,
        )

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
                # scipy netCDF3 backend cannot write unicode-array attrs.
                ds.attrs[f"{dim}_multiindex"] = json.dumps(list(names))

        return ds

    vars = [
        with_prefix(var.data, f"variables-{name}") for name, var in m.variables.items()
    ]
    cons = [
        with_prefix(con.to_netcdf_ds(), f"constraints-{name}")
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
    ds.attrs[NETCDF_VERSION_ATTR] = version("linopy")
    if m._relaxed_registry:
        ds.attrs["_relaxed_registry"] = json.dumps(m._relaxed_registry)
    if m._piecewise_formulations:
        ds.attrs["_piecewise_formulations"] = json.dumps(
            {
                name: {
                    "method": pwl.method,
                    "variable_names": pwl.variable_names,
                    "constraint_names": pwl.constraint_names,
                    "convexity": pwl.convexity,
                }
                for name, pwl in m._piecewise_formulations.items()
            }
        )
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

    Notes
    -----
    The SOS reformulation lifecycle token is not persisted by
    :func:`to_netcdf`. If the saved model was in reformulated form,
    the deserialized Model is too, but
    :meth:`Model.undo_sos_reformulation` is a no-op on it.
    """
    from linopy.constraints import (
        Constraint,
        ConstraintBase,
        Constraints,
        CSRConstraint,
    )
    from linopy.expressions import LinearExpression
    from linopy.model import Model
    from linopy.variables import Variable, Variables

    if isinstance(path, str):
        path = Path(path)

    m = Model()
    ds = xr.load_dataset(path, **kwargs)

    def has_prefix(k: str, prefix: str) -> bool:
        return k.rsplit("-", 1)[0] == prefix

    def remove_prefix(k: str, prefix: str) -> str:
        return k[len(prefix) + 1 :]

    def parse_multiindex_attr(value: str | Iterable[str]) -> list[str]:
        # str = JSON (new); iterable = legacy list from older linopy.
        if isinstance(value, str):
            return [str(n) for n in json.loads(value)]
        return [str(n) for n in value]

    def get_prefix(ds: xr.Dataset, prefix: str) -> xr.Dataset:
        ds = ds[[k for k in ds if has_prefix(str(k), prefix)]]
        multiindexes = []
        for dim in ds.dims:
            attr = ds.attrs.get(f"{dim}_multiindex")
            if attr is None:
                continue
            for name in parse_multiindex_attr(attr):
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
                names = parse_multiindex_attr(ds.attrs.pop(f"{dim}_multiindex"))
                ds = ds.set_index({dim: names})  # type: ignore[dict-item]

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
    constraints: dict[str, ConstraintBase] = {}
    for k in sorted(con_names):
        name = remove_prefix(k, "constraints")
        con_ds = get_prefix(ds, k)
        if con_ds.attrs.get("_linopy_format") == "csr":
            constraints[name] = CSRConstraint.from_netcdf_ds(con_ds, m, name)
        else:
            constraints[name] = Constraint(con_ds, m, name)
    m._constraints = Constraints(constraints, m)

    objective = get_prefix(ds, "objective")
    m.objective = Objective(
        LinearExpression(objective, m), m, objective.attrs.pop("sense")
    )
    m.objective._value = objective.attrs.pop("value", None)

    m.parameters = get_prefix(ds, "parameters")

    for k in m.scalar_attrs:
        if k in ds.attrs:
            setattr(m, k, ds.attrs[k])

    if "_relaxed_registry" in ds.attrs:
        m._relaxed_registry = json.loads(ds.attrs["_relaxed_registry"])

    if "_piecewise_formulations" in ds.attrs:
        from linopy.piecewise import PiecewiseFormulation

        for name, d in json.loads(ds.attrs["_piecewise_formulations"]).items():
            m._piecewise_formulations[name] = PiecewiseFormulation(
                name=name,
                method=d["method"],
                variable_names=d["variable_names"],
                constraint_names=d["constraint_names"],
                model=m,
                convexity=d["convexity"],
            )

    return m


def copy(m: Model, include_solution: bool = False, deep: bool = True) -> Model:
    """
    Return a copy of this model.

    With ``deep=True`` (default), variables, constraints, objective,
    parameters, blocks, and scalar attributes are copied to a fully
    independent model. With ``deep=False``, returns a shallow copy.

    :meth:`Model.copy` defaults to deep copy for workflow safety.
    In contrast, ``copy.copy(model)`` is shallow via ``__copy__``, and
    ``copy.deepcopy(model)`` is deep via ``__deepcopy__``.

    Solver runtime metadata (for example, ``solver_name`` and
    ``solver_model``) is intentionally not copied. Solver backend state
    is recreated on ``solve()``.

    Parameters
    ----------
    m : Model
        The model to copy.
    include_solution : bool, optional
        Whether to include solution and dual values in the copy.
        If False (default), solve artifacts are excluded: solution/dual data,
        objective value, and solve status are reset to initialized state.
        If True, these values are copied when present. For unsolved models,
        this has no additional effect.
    deep : bool, optional
        Whether to return a deep copy (default) or shallow copy. If False,
        the returned model uses independent wrapper objects that share
        underlying data buffers with the source model.

    Returns
    -------
    Model
        A deep or shallow copy of the model.
    """
    from linopy.constraints import Constraint, ConstraintBase, Constraints
    from linopy.expressions import LinearExpression
    from linopy.model import Model, Objective
    from linopy.variables import Variable, Variables

    SOLVE_STATE_ATTRS = {"status", "termination_condition"}

    new_model = Model(
        chunk=m._chunk,
        force_dim_names=m._force_dim_names,
        auto_mask=m._auto_mask,
        freeze_constraints=m.freeze_constraints,
        set_names_in_solver_io=m.set_names_in_solver_io,
        solver_dir=str(m._solver_dir),
    )

    new_model._variables = Variables(
        {
            name: Variable(
                var.data.copy(deep=deep)
                if include_solution
                else var.data[m.variables.dataset_attrs].copy(deep=deep),
                new_model,
                name,
            )
            for name, var in m.variables.items()
        },
        new_model,
    )

    def _copy_con_data(con: ConstraintBase) -> xr.Dataset:
        d = con.mutable().data
        if include_solution:
            return d.copy(deep=deep)
        return d[con.data_attrs].copy(deep=deep)

    new_model._constraints = Constraints(
        {
            name: Constraint(_copy_con_data(con), new_model, name)
            for name, con in m.constraints.items()
        },
        new_model,
    )

    obj_expr = LinearExpression(m.objective.expression.data.copy(deep=deep), new_model)
    new_model._objective = Objective(obj_expr, new_model, m.objective.sense)
    new_model._objective._value = (
        float(m.objective.value)
        if (include_solution and m.objective.value is not None)
        else None
    )

    new_model._parameters = m._parameters.copy(deep=deep)
    new_model._blocks = m._blocks.copy(deep=deep) if m._blocks is not None else None

    for attr in m.scalar_attrs:
        if include_solution or attr not in SOLVE_STATE_ATTRS:
            setattr(new_model, attr, getattr(m, attr))

    if m._sos_reformulation_state is not None:
        new_model._sos_reformulation_state = _copy.deepcopy(m._sos_reformulation_state)

    return new_model


def shallowcopy(m: Model) -> Model:
    """
    Support Python's ``copy.copy`` protocol for ``Model``.

    Returns a shallow copy with independent wrapper objects that share
    underlying array buffers with ``m``. Solve artifacts are excluded,
    matching :meth:`Model.copy` defaults.
    """
    return copy(m, include_solution=False, deep=False)


def deepcopy(m: Model, memo: dict[int, Any]) -> Model:
    """
    Support Python's ``copy.deepcopy`` protocol for ``Model``.

    Returns a deep, structurally independent copy and records it in ``memo``
    as required by Python's copy protocol. Solve artifacts are excluded,
    matching :meth:`Model.copy` defaults.
    """
    new_model = copy(m, include_solution=False, deep=True)
    memo[id(m)] = new_model
    return new_model
