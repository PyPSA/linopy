import numpy as np
from scipy.sparse import csr_matrix, vstack

from .model import Model

def index_map(ds):
    """
    """
    last_ind = max(ds.max().values()).item()
    indmap = np.empty(last_ind+1, dtype=np.integer)
    ind = 0
    for var in ds.values():
        size = var.size
        indmap[np.ravel(var)] = np.arange(ind, ind+size)
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
    eq_matrix = to_matrix(
        constraints[eqs], vars[eqs], coeffs[eqs], var_map, nvar
    )
    eq_rhs = to_rhs(constraints[eqs], vars[eqs], rhs[eqs])

    return {
        "ineq_matrix": ineq_matrix,
        "ineq_lo": ineq_lo,
        "ineq_hi": ineq_hi,
        "eq_matrix": eq_matrix,
        "eq_rhs": eq_rhs
    }
