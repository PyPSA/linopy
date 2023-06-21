#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing all import/export functionalities.
"""
import logging
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr
from numpy import asarray, concatenate, ones_like, zeros_like
from tqdm import tqdm

from linopy import solvers
from linopy.constants import EQUAL, GREATER_EQUAL

logger = logging.getLogger(__name__)


ufunc_kwargs = dict(vectorize=True)
concat_dim = "_concat_dim"
concat_kwargs = dict(dim=concat_dim, coords="minimal")

TQDM_COLOR = "#80bfff"


def objective_to_file(m, f, log=False, batch_size=10000):
    """
    Write out the objective of a model to a lp file.
    """
    if log:
        logger.info("Writing objective.")

    f.write("min\n\nobj:\n\n")
    df = m.objective.flat

    if np.isnan(df.coeffs).any():
        raise ValueError(
            "Objective coefficients are missing (nan) where variables are not (-1)."
        )

    coeffs = df.coeffs.values
    vars = df.vars.values

    batch = []  # to store batch of lines
    for idx in range(len(df)):
        coeff = coeffs[idx]
        var = vars[idx]

        batch.append(f"{coeff:+.12g} x{var}\n")

        if len(batch) >= batch_size:
            f.writelines(batch)  # write out a batch
            batch = []  # reset batch

    if batch:  # write the remaining lines
        f.writelines(batch)


def constraints_to_file(m, f, log=False, batch_size=50000):
    f.write("\n\ns.t.\n\n")
    names = m.constraints
    if log:
        names = tqdm(
            list(names),
            desc="Writing constraints.",
            colour=TQDM_COLOR,
        )

    batch = []
    for name in names:
        df = m.constraints[name].flat

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

            if len(batch) >= batch_size:
                f.writelines(batch)
                batch = []  # reset batch

            prev_label = label

        batch.append(f"{prev_sign} {prev_rhs:+.12g}\n")

    if batch:  # write the remaining lines
        f.writelines(batch)


def bounds_to_file(m, f, log=False, batch_size=10000):
    """
    Write out variables of a model to a lp file.
    """
    names = list(m.variables.continuous) + list(m.variables.integers)
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
        df = m.variables[name].flat

        labels = df.labels.values
        lowers = df.lower.values
        uppers = df.upper.values

        for idx in range(len(df)):
            label = labels[idx]
            lower = lowers[idx]
            upper = uppers[idx]
            batch.append(f"{lower:+.12g} <= x{label} <= {upper:+.12g}\n")

            if len(batch) >= batch_size:
                f.writelines(batch)  # write out a batch
                batch = []  # reset batch

    if batch:  # write the remaining lines
        f.writelines(batch)


def binaries_to_file(m, f, log=False, batch_size=1000):
    """
    Write out binaries of a model to a lp file.
    """
    names = m.variables.binaries
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
        df = m.variables[name].flat

        for label in df.labels.values:
            batch.append(f"x{label}\n")

            if len(batch) >= batch_size:
                f.writelines(batch)  # write out a batch
                batch = []  # reset batch

    if batch:  # write the remaining lines
        f.writelines(batch)


def integers_to_file(m, f, log=False, batch_size=1000):
    """
    Write out integers of a model to a lp file.
    """
    names = m.variables.integers
    if not len(list(names)):
        return

    f.write("\n\ninteger\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing integer variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        df = m.variables[name].flat

        for label in df.labels.values:
            batch.append(f"x{label}\n")

            if len(batch) >= batch_size:
                f.writelines(batch)  # write out a batch
                batch = []  # reset batch

    if batch:  # write the remaining lines
        f.writelines(batch)


def to_file(m, fn):
    """
    Write out a model to a lp or mps file.
    """
    fn = Path(m.get_problem_file(fn))
    if fn.exists():
        fn.unlink()

    if fn.suffix == ".lp":
        log = m._xCounter > 10_000

        batch_size = 5000

        with open(fn, mode="w") as f:
            start = time.time()

            objective_to_file(m, f, log=log)
            constraints_to_file(m, f, log=log, batch_size=batch_size)
            bounds_to_file(m, f, log=log, batch_size=batch_size)
            binaries_to_file(m, f, log=log, batch_size=batch_size)
            integers_to_file(m, f, log=log, batch_size=batch_size)
            f.write("end\n")

            logger.info(f" Writing time: {round(time.time()-start, 2)}s")

    elif fn.suffix == ".mps":
        if "highs" not in solvers.available_solvers:
            raise RuntimeError(
                "Package highspy not installed. This is required to exporting to MPS file."
            )

        # Use very fast highspy implementation
        # Might be replaced by custom writer, however needs C bindings for performance
        h = m.to_highspy()
        h.writeModel(str(fn))
    else:
        raise ValueError(
            f"Cannot write problem to {fn}, file format `{fn.suffix}` not supported."
        )

    return fn


def to_gurobipy(m):
    """
    Export the model to gurobipy.

    This function does not write the model to intermediate files but directly
    passes it to gurobipy. Note that for large models this is not
    computationally efficient.

    Parameters
    ----------
    m : linopy.Model

    Returns
    -------
    model : gurobipy.Model
    """
    import gurobipy

    m.constraints.sanitize_missings()
    model = gurobipy.Model()

    M = m.matrices

    names = "x" + M.vlabels.astype(str).astype(object)
    kwargs = {}
    if len(m.binaries.labels) + len(m.integers.labels):
        kwargs["vtype"] = M.vtypes
    x = model.addMVar(M.vlabels.shape, M.lb, M.ub, name=list(names), **kwargs)

    model.setObjective(M.c @ x)

    names = "c" + M.clabels.astype(str).astype(object)
    c = model.addMConstr(M.A, x, M.sense, M.b)
    c.setAttr("ConstrName", list(names))

    model.update()
    return model


def to_highspy(m):
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
    h.changeColsCost(len(M.c), np.arange(len(M.c), dtype=np.int32), M.c)
    A = M.A.tocsr()
    num_cons = A.shape[0]
    lower = np.where(M.sense != "<", M.b, -np.inf)
    upper = np.where(M.sense != ">", M.b, np.inf)
    h.addRows(num_cons, lower, upper, A.nnz, A.indptr, A.indices, A.data)
    lp = h.getLp()
    lp.row_names_ = "c" + M.clabels.astype(str).astype(object)
    lp.col_names_ = "x" + M.vlabels.astype(str).astype(object)
    h.passModel(lp)
    return h


def to_block_files(m, fn):
    """
    Write out the linopy model to a block structured output.

    Experimental: This function does not support grouping duplicated variables
    in one constraint row yet!
    """
    if fn is None:
        fn = TemporaryDirectory(prefix="linopy-problem-", dir=m.solver_dir).name

    path = Path(fn)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)

    m.calculate_block_maps()

    N = int(m.blocks.max())
    for n in range(N + 2):
        (path / f"block{n}").mkdir()

    vars = m.variables
    cons = m.constraints

    # Write out variables
    blocks = vars.ravel("blocks", filter_missings=True)
    for key, suffix in zip(["labels", "lower", "upper"], ["x", "xl", "xu"]):
        arr = vars.ravel(key, filter_missings=True)
        for n in tqdm(range(N + 1), desc=f"Write variable {key}"):
            arr[blocks == n].tofile(path / f"block{n}" / suffix, sep="\n")

    # Write out objective (uses variable blocks from above)
    coeffs = np.zeros(m._xCounter)
    coeffs[np.asarray(m.objective.vars)] = np.asarray(m.objective.coeffs)
    # reorder like non-missing variables
    coeffs = coeffs[vars.ravel("labels", filter_missings=True)]
    for n in tqdm(range(N + 1), desc="Write objective"):
        coeffs[blocks == n].tofile(path / f"block{n}" / "c", sep="\n")

    # Write out rhs
    blocks = cons.ravel("blocks", filter_missings=True)
    rhs = cons.ravel("rhs", filter_missings=True)
    is_equality = cons.ravel(cons.sign == EQUAL, filter_missings=True)
    is_lower_bound = cons.ravel(cons.sign == GREATER_EQUAL, filter_missings=True)

    for n in tqdm(range(N + 2), desc="Write RHS"):
        is_blockn = blocks == n

        rhs[is_blockn & is_equality].tofile(path / f"block{n}" / "b", sep="\n")

        not_equality = is_blockn & ~is_equality
        is_lower_bound_sub = is_lower_bound[not_equality]
        rhs_sub = rhs[not_equality]

        lower_bound = np.where(is_lower_bound_sub, rhs_sub, -np.inf)
        lower_bound.tofile(path / f"block{n}" / "dl", sep="\n")

        upper_bound = np.where(~is_lower_bound_sub, rhs_sub, np.inf)
        upper_bound.tofile(path / f"block{n}" / "du", sep="\n")

    # Write out constraints
    conblocks = cons.ravel("blocks", "vars", filter_missings=True)
    varblocks = cons.ravel("var_blocks", "vars", filter_missings=True)
    is_equality = cons.ravel(cons.sign == EQUAL, "vars", filter_missings=True)

    is_varblock_0 = varblocks == 0
    is_conblock_L = conblocks == N + 1

    keys = ["labels", "coeffs", "vars"]

    def filtered(arr, mask, key):
        """
        Set coefficients to zero where mask is False, keep others unchanged.

        PIPS requires this information to set the shape of sub-matrices.
        """
        assert key in keys
        return np.where(mask, arr, 0) if key == "coeffs" else arr

    for key, suffix in zip(keys, ["row", "data", "col"]):
        arr = cons.ravel(key, "vars", filter_missings=True)
        for n in tqdm(range(N + 1), desc=f"Write constraint {key}"):
            is_conblock_n = conblocks == n
            is_varblock_n = varblocks == n

            mask = is_conblock_n & is_equality
            filtered(arr[mask], is_varblock_0[mask], key).tofile(
                path / f"block{n}" / f"A_{suffix}", sep="\n"
            )
            mask = is_conblock_n & ~is_equality
            filtered(arr[mask], is_varblock_0[mask], key).tofile(
                path / f"block{n}" / f"C_{suffix}", sep="\n"
            )

            mask = is_conblock_L & is_equality
            filtered(arr[mask], is_varblock_n[mask], key).tofile(
                path / f"block{n}" / f"BL_{suffix}", sep="\n"
            )
            mask = is_conblock_L & ~is_equality
            filtered(arr[mask], is_varblock_n[mask], key).tofile(
                path / f"block{n}" / f"DL_{suffix}", sep="\n"
            )

            if n:
                mask = is_conblock_n & is_equality
                filtered(arr[mask], is_varblock_n[mask], key).tofile(
                    path / f"block{n}" / f"B_{suffix}", sep="\n"
                )
                mask = is_conblock_n & ~is_equality
                filtered(arr[mask], is_varblock_n[mask], key).tofile(
                    path / f"block{n}" / f"D_{suffix}", sep="\n"
                )


def non_bool_dict(d):
    """
    Convert bool to int for netCDF4 storing.
    """
    return {k: int(v) if isinstance(v, bool) else v for k, v in d.items()}


def to_netcdf(m, *args, **kwargs):
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
        ds = ds.rename({d: f"{prefix}-{d}" for d in [*ds.dims, *ds]})
        ds.attrs = {f"{prefix}-{k}": v for k, v in ds.attrs.items()}
        return ds

    vars = [
        with_prefix(var.data, f"variables-{name}") for name, var in m.variables.items()
    ]
    cons = [
        with_prefix(con.data, f"constraints-{name}")
        for name, con in m.constraints.items()
    ]
    obj = [with_prefix(m.objective.data, "objective")]
    params = [with_prefix(m.parameters, "params")]

    scalars = {k: getattr(m, k) for k in m.scalar_attrs}
    ds = xr.merge(vars + cons + obj + params)
    ds = ds.assign_attrs(scalars)
    ds.attrs = non_bool_dict(ds.attrs)

    for k in ds:
        ds[k].attrs = non_bool_dict(ds[k].attrs)

    ds.to_netcdf(*args, **kwargs)


def read_netcdf(path, **kwargs):
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

    m = Model()
    ds = xr.load_dataset(path, **kwargs)

    def get_prefix(ds, prefix):
        ds = ds[[k for k in ds if k.startswith(prefix)]]
        ds = ds.rename({d: d.split(prefix + "-", 1)[1] for d in [*ds.dims, *ds]})
        ds.attrs = {
            k.split(prefix + "-", 1)[1]: v
            for k, v in ds.attrs.items()
            if k.startswith(prefix + "-")
        }
        return ds

    vars = get_prefix(ds, "variables")
    var_names = list({k.split("-", 1)[0] for k in vars})
    variables = {k: Variable(get_prefix(vars, k), m, k) for k in var_names}
    m._variables = Variables(variables, m)

    cons = get_prefix(ds, "constraints")
    con_names = list({k.split("-", 1)[0] for k in cons})
    constraints = {k: Constraint(get_prefix(cons, k), m, k) for k in con_names}
    m._constraints = Constraints(constraints, m)

    objective = get_prefix(ds, "objective")
    m._objective = LinearExpression(objective, m)

    m._parameters = get_prefix(ds, "parameter")

    for k in m.scalar_attrs:
        setattr(m, k, ds.attrs.get(k))

    return m
