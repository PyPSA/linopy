# Benchmark Workflow


This directory contains a Snakemake workflow benchmarking the performance of `linopy` against `pyomo` and `JuMP`. The linear problem

$$ \min \;\; 2 x_{i,j} \; y_{i,j} \qquad \forall \; i,j \in \{1,...,N\} $$
s.t.
$$ x_{i,j} - y_{i,j} \; \ge \; i \qquad \forall \; i,j \in \{1,...,N\} $$
$$ x_{i,j} + y_{i,j} \; \ge \; 0 \qquad \forall \; i,j \in \{1,...,N\} $$

is initialized and solved for different values of `N` with each of the API's.
