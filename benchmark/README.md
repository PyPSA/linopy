# Benchmark Workflow


This directory contains a Snakemake workflow benchmarking the performance of `linopy` against `pyomo` and `JuMP`. The linear problem


<p><span class="math display">min   2<em>x</em><sub><em>i</em>, <em>j</em></sub> <em>y</em><sub><em>i</em>, <em>j</em></sub>   ∀ <em>i</em>, <em>j</em> ∈ {1, ..., <em>N</em>}</span>
s.t. <span class="math display"><em>x</em><sub><em>i</em>, <em>j</em></sub> − <em>y</em><sub><em>i</em>, <em>j</em></sub> ≥ <em>i</em>   ∀ <em>i</em>, <em>j</em> ∈ {1, ..., <em>N</em>}</span>
<span class="math display"><em>x</em><sub><em>i</em>, <em>j</em></sub> + <em>y</em><sub><em>i</em>, <em>j</em></sub> ≥ 0   ∀ <em>i</em>, <em>j</em> ∈ {1, ..., <em>N</em>}</span></p>

is initialized and solved for different values of `N` with each of the API's.
