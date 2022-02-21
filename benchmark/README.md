# Benchmark Workflow


This directory contains a Snakemake workflow benchmarking the performance of `linopy` against `pyomo` and `JuMP`. The problem used for benchmarking is

\begin{align*}
    & \min \;\; 2 x_{i,j} \; y_{i,j} \qquad \forall \; i,j \in \{1,...,N\} \\
    s.t. & \\
    & x_{i,j} - y_{i,j} \; \ge \; i \qquad \forall \; i,j \in \{1,...,N\}
\end{align*}
