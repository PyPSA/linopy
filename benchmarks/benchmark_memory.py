"""Benchmark memory usage of int32 vs int64 labels."""

import numpy as np

import linopy.common
import linopy.constants
import linopy.expressions
import linopy.model
from linopy import Model
from linopy.constants import DEFAULT_LABEL_DTYPE


def build_model(n_vars: int) -> Model:
    m = Model()
    coords = [range(n_vars)]
    x = m.add_variables(lower=0, upper=1, coords=coords, name="x")
    m.add_constraints(x >= 0.5, name="c")
    m.add_objective(x.sum())
    return m


def report_nbytes(m: Model, label: str) -> None:
    var_bytes = sum(v.nbytes for v in m.variables["x"].data.data_vars.values())
    con_bytes = sum(v.nbytes for v in m.constraints["c"].data.data_vars.values())
    total = var_bytes + con_bytes
    print(
        f"  {label}: variables={var_bytes:,} B, constraints={con_bytes:,} B, total={total:,} B"
    )


def main() -> None:
    print(f"DEFAULT_LABEL_DTYPE = {DEFAULT_LABEL_DTYPE}")
    print()
    for n in [10_000, 100_000, 1_000_000]:
        print(f"n_vars = {n:,}")
        m = build_model(n)
        report_nbytes(m, "int32 (default)")

        # Compare: override to int64
        orig = linopy.constants.DEFAULT_LABEL_DTYPE
        for mod in [linopy.constants, linopy.model, linopy.expressions, linopy.common]:
            mod.DEFAULT_LABEL_DTYPE = np.int64

        m64 = build_model(n)
        report_nbytes(m64, "int64 (comparison)")

        # Restore
        for mod in [linopy.constants, linopy.model, linopy.expressions, linopy.common]:
            mod.DEFAULT_LABEL_DTYPE = orig
        print()


if __name__ == "__main__":
    main()
