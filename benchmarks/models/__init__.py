"""
Model builders for benchmarks.

Importing this package registers every model in :data:`benchmarks.registry.REGISTRY`.
Each module exposes a ``build_<name>(size) -> linopy.Model`` callable and a
module-level ``SPEC`` :class:`~benchmarks.registry.ModelSpec`.
"""

from benchmarks.models.basic import SIZES as BASIC_SIZES
from benchmarks.models.basic import build_basic
from benchmarks.models.expression_arithmetic import SIZES as EXPR_SIZES
from benchmarks.models.expression_arithmetic import build_expression_arithmetic
from benchmarks.models.knapsack import SIZES as KNAPSACK_SIZES
from benchmarks.models.knapsack import build_knapsack
from benchmarks.models.masked import SIZES as MASKED_SIZES
from benchmarks.models.masked import build_masked
from benchmarks.models.milp import SIZES as MILP_SIZES
from benchmarks.models.milp import build_milp
from benchmarks.models.piecewise import SIZES as PIECEWISE_SIZES
from benchmarks.models.piecewise import build_piecewise
from benchmarks.models.pypsa_scigrid import SIZES as PYPSA_SIZES
from benchmarks.models.pypsa_scigrid import build_pypsa_scigrid
from benchmarks.models.qp import SIZES as QP_SIZES
from benchmarks.models.qp import build_qp
from benchmarks.models.sos import SIZES as SOS_SIZES
from benchmarks.models.sos import build_sos
from benchmarks.models.sparse_network import SIZES as SPARSE_SIZES
from benchmarks.models.sparse_network import build_sparse_network

__all__ = [
    "BASIC_SIZES",
    "EXPR_SIZES",
    "KNAPSACK_SIZES",
    "MASKED_SIZES",
    "MILP_SIZES",
    "PIECEWISE_SIZES",
    "PYPSA_SIZES",
    "QP_SIZES",
    "SOS_SIZES",
    "SPARSE_SIZES",
    "build_basic",
    "build_expression_arithmetic",
    "build_knapsack",
    "build_masked",
    "build_milp",
    "build_piecewise",
    "build_pypsa_scigrid",
    "build_qp",
    "build_sos",
    "build_sparse_network",
]
