from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack
from xarray import DataArray, Dataset

from .model import Model

# Methods to convert a full model into a vector and sparse matrix representation
# (that's basically the original playground makes sense to understand, especially the
# index_map renumbering is probably crucial for creating the block matrices)


def index_map(ds, num_variables=None):
    """ """
    if num_variables is None:
        num_variables = max(ds.max().values()).item() + 1
    indmap = np.empty(num_variables, dtype=np.integer)
    ind = 0
    for var in ds.values():
        size = var.size
        indmap[np.ravel(var)] = np.arange(ind, ind + size)
        ind += size
    return indmap, ind


def to_matrix(constraints, vars, coeffs, var_map, nvar):
    matrix = []
    for con, labels in constraints.items():
        dims = labels.dims + (f"{con}_term",)
        cols = var_map[np.ravel(vars[con].transpose(*dims))]
        data = np.ravel(coeffs[con].transpose(*dims))
        nterms = vars.dims[f"{con}_term"]
        indptr = np.arange(0, labels.size * nterms + 1, nterms)

        matrix.append(csr_matrix((data, cols, indptr), shape=(labels.size, nvar)))

    return vstack(matrix)


def to_bounds(constraints, vars, sign, rhs):
    lo = []
    hi = []
    for con, labels in constraints.items():
        dims = labels.dims + (f"{con}_term",)
        data = np.ravel(rhs[con].broadcast_like(vars[con]).transpose(*dims))
        s = sign[con].item()
        if s == "<=":
            lo.append(np.full_like(data, -np.inf))
            hi.append(data)
        elif s == ">=":
            lo.append(data)
            hi.append(np.full_like(data, np.inf))

    return np.concatenate(lo), np.concatenate(hi)


def to_rhs(constraints, vars, rhs):
    r = []
    for con, labels in constraints.items():
        dims = labels.dims + (f"{con}_term",)
        data = np.ravel(rhs[con].broadcast_like(vars[con]).transpose(*dims))
        r.append(data)

    return np.concatenate(r)


def to_constraint_matrix(m: Model):
    """
    builds the matrices and variables representing equality and inequality constraints
    in a full model.

    TODO
    needs to be written such that it can be tied into the Constraint(s) classes below,
    which might also represent only individual blocks.
    """
    constraints = m.constraints
    vars = m.constraints_lhs_vars
    coeffs = m.constraints_lhs_coeffs
    sign = m.constraints_sign
    rhs = m.constraints_rhs

    var_map, nvar = index_map(m.variables)

    ineqs = [n for n, s in m.constraints_sign.items() if s in ("<=", ">=")]
    ineq_matrix = to_matrix(
        constraints[ineqs], vars[ineqs], coeffs[ineqs], var_map, nvar
    )
    ineq_lo, ineq_hi = to_bounds(
        constraints[ineqs], vars[ineqs], sign[ineqs], rhs[ineqs]
    )

    eqs = [n for n, s in m.constraints_sign.items() if s in ("=", "==")]
    eq_matrix = to_matrix(constraints[eqs], vars[eqs], coeffs[eqs], var_map, nvar)
    eq_rhs = to_rhs(constraints[eqs], vars[eqs], rhs[eqs])

    return {
        "ineq_matrix": ineq_matrix,
        "ineq_lo": ineq_lo,
        "ineq_hi": ineq_hi,
        "eq_matrix": eq_matrix,
        "eq_rhs": eq_rhs,
    }


def best_uint(max_value):
    for t in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(t).max:
            return t


## Dataclass helpers into which I would want to tie the blocking logic


@dataclass
class Constraint:
    name: str
    defs: DataArray
    coeffs: DataArray
    vars: DataArray
    sign: DataArray
    rhs: DataArray

    def block_indicator(self, num_blocks, block_map):
        """
        Constructs the block_indicator for this set of constraints

        The block_indicator is a num_blocks x dim1 x dim2 boolean array, where
        indicator[block, d1, d2] indicates that for constraint `name` d1, d2 is part of
        block. The same constraint is normally part of several blocks.

        TODO
        ----
        Unclear whether it fits into dask here, maybe pull out of class. I think it does,
        the best dask way in my mind would be to make use of xarray's high-level map_blocks
        method, which wraps the different dask blocks into DataArrays again, so that indices
        are available. The involved overhead is probably necessary.
        """
        constr_block_map = block_map[
            self.vars.transpose(f"{self.name}_term", *self.defs.dims).values
        ]
        indicator = np.zeros((num_blocks,) + self.defs.shape, dtype=bool)
        indicator[
            (constr_block_map,)
            + tuple(np.ogrid[tuple(slice(None, s) for s in self.defs.shape)])
        ] = True
        return indicator

    def block_sizes(self, num_blocks, block_map):
        """ "
        TODO
        ----
        Unclear whether it fits into dask here, maybe pull out of class
        """
        sizes = np.zeros(num_blocks + 1, dtype=int)
        indicator = self.block_indicator(num_blocks, block_map)

        num_of_nonzero_blocks = indicator[1:].sum(axis=0)
        sizes[0] += (num_of_nonzero_blocks == 0).sum()

        onlyone_b = num_of_nonzero_blocks == 1
        if onlyone_b.any():
            sizes[1:num_blocks] = indicator[1:, onlyone_b].sum(axis=1)

        sizes[num_blocks] += (num_of_nonzero_blocks > 1).sum()
        return sizes


@dataclass
class Constraints:
    """
    A slightly more helpful representation of all constraints in a model
    which aims at providing easy block writing methods for the constraints.
    """

    defs: Dataset
    coeffs: Dataset
    vars: Dataset
    sign: Dataset
    rhs: Dataset

    @classmethod
    def from_model(cls, model: Model) -> "Constraints":
        return cls(
            model.constraints,
            model.constraints_lhs_coeffs,
            model.constraints_lhs_vars,
            model.constraints_sign,
            model.constraints_rhs,
        )

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Constraint, "Constraints"]:
        if isinstance(names, str):
            return Constraint(
                names,
                self.defs[names],
                self.coeffs[names],
                self.vars[names],
                self.sign[names],
                self.rhs[names],
            )

        return self.__class__(
            self.defs[names],
            self.coeffs[names],
            self.vars[names],
            self.sign[names],
            self.rhs[names],
        )

    @property
    def inequalities(self):
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

    def block_sizes(self, num_blocks, block_map) -> np.ndarray:
        sizes = np.zeros(num_blocks + 1, dtype=int)
        for name in self.defs:
            sizes += self[name].block_sizes(num_blocks, block_map)
        return sizes


class PipsSerialiser:
    """
    Breaks a `Model` down into blocks based on variable annotations and creates a
    binary representation to be fed into PIPS-IPM++.

    """

    def __init__(self, model: Model, blocks: Optional[Dataset] = None):
        self.model = model
        self.constraints = Constraints.from_model(model)

        if blocks is None:
            self.blocks = model.blocks
        self.num_blocks = self.blocks.max().item() + 1

    @staticmethod
    def build_block_map(
        num_blocks: int, blocks: Dataset, num_variables: int, variables: Dataset
    ):
        """
        Split variables into blocks based on `blocks` definition

        Returns
        -------
        block_map : ndarray[shape=(num_variables,), dtype=int]
            indexable array that turns variables into blocks

        TODO
        ----
        Rewrite for dask
        """
        block_map = np.empty(num_variables, dtype=best_uint(num_blocks))
        for name, variable in variables.items():
            variable_block = blocks.get(name, blocks["general"])
            block_map[np.ravel(variable)] = np.ravel(
                variable_block.broadcast_like(variable)
            )
        return block_map

    def to_blocks(self):
        # Split variables into blocks
        block_map = self.build_block_map(
            self.num_blocks, self.blocks, self.model._xCounter, self.model.variables
        )

        # Split constraints into equalities and inequalities
        eq_constr = self.constraints.equalities
        ineq_constr = self.constraints.inequalities

        # Determine blocks and sizes across constraints, first build the block indicator
        eq_block_sizes = eq_constr.block_sizes(self.num_blocks, block_map)
        ineq_block_sizes = ineq_constr.block_sizes(self.num_blocks, block_map)
