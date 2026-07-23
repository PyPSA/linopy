from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from test_io import _pips_time_plant_model

import linopy.pips as pips
from linopy import Model, available_solvers


def _time_model(n_time: int) -> Model:
    m = Model()
    time = pd.RangeIndex(n_time, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x <= 1, name="c")
    m.add_objective(x.sum())
    return m


def _synthetic_storage_model() -> Model:
    m = Model()
    time = pd.RangeIndex(6, name="time")
    plant = pd.Index([0, 1], name="plant")
    x = m.add_variables(lower=0, coords=[time, plant], name="x")
    soc = m.add_variables(coords=[time], name="soc")
    cap = m.add_variables(lower=0, coords=[plant], name="cap")
    m.add_constraints(x - cap <= 0, name="capacity")
    continuity = (soc - soc.shift(time=1) - x.sum("plant")).isel(time=slice(1, None))
    m.add_constraints(continuity == 0, name="storage")
    m.add_constraints(x.sum() <= 100, name="budget")
    m.add_objective(x.sum() + soc.sum() + cap.sum())
    return m


@pytest.mark.parametrize(
    "n_time, n_blocks, sizes",
    [(6, 2, [3, 3]), (6, 4, [2, 2, 1, 1]), (10, 3, [4, 3, 3])],
)
def test_assign_blocks_contiguous(n_time: int, n_blocks: int, sizes: list[int]) -> None:
    m = _time_model(n_time)
    out = pips.assign_blocks(m, "time", n_blocks)
    expected = np.repeat(np.arange(1, n_blocks + 1), sizes)
    assert out is m
    assert m.blocks.dims == ("time",)
    assert list(m.blocks.coords["time"].values) == list(range(n_time))
    np.testing.assert_array_equal(m.blocks.values, expected)


def test_assign_blocks_dtype_is_best_int() -> None:
    m = _time_model(6)
    m.assign_blocks("time", 2)
    assert m.blocks.dtype == np.dtype("int8")


def test_assign_blocks_method_matches_function() -> None:
    a = _time_model(6).assign_blocks("time", 3)
    b = pips.assign_blocks(_time_model(6), "time", 3)
    np.testing.assert_array_equal(a.blocks.values, b.blocks.values)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"dim": "nope", "n_blocks": 2}, ValueError),
        ({"dim": "time", "n_blocks": 0}, ValueError),
        ({"dim": "time", "n_blocks": 7}, ValueError),
        ({"dim": "time", "n_blocks": 2, "boundary": "custom"}, NotImplementedError),
    ],
)
def test_assign_blocks_fail_fast(kwargs: dict, exc: type[Exception]) -> None:
    m = _time_model(6)
    with pytest.raises(exc):
        pips.assign_blocks(m, **kwargs)


def test_diagnose_synthetic_exact_counts() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    r = pips.diagnose(m)

    assert r.n_blocks == 3
    assert r.n_vars == 20
    assert r.n_cons == 18
    assert r.nnz == 56

    assert r.n_global_cols == 2
    assert r.block_cols == {1: 6, 2: 6, 3: 6}
    assert r.n_global_cols + sum(r.block_cols.values()) == r.n_vars

    assert r.block_nnz == {1: 12, 2: 12, 3: 12}
    assert r.balance_ratio == 1.0

    assert (r.n_local_rows, r.n_global_rows, r.n_linking_rows) == (15, 0, 3)
    assert r.n_local_rows + r.n_global_rows + r.n_linking_rows == r.n_cons
    assert (r.n_adjacent_rows, r.n_border_rows) == (2, 1)
    assert r.n_adjacent_rows + r.n_border_rows == r.n_linking_rows

    assert r.border_nnz == 32
    assert r.border_fraction == pytest.approx(32 / 56)


def test_diagnose_border_nnz_bruteforce() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    r = pips.diagnose(m)

    N = int(m.blocks.max())
    block_map = m.variables.get_blockmap(m.blocks.dtype.type)
    col_blocks = block_map[m.matrices.vlabels]
    row_blocks = np.concatenate(
        [
            c.data["blocks"].values.ravel()[c.active_row_mask()]
            for _, c in m.constraints.items()
            if not c.is_indicator
        ]
    )
    coo = m.matrices.A.tocoo()
    is_border = (row_blocks[coo.row] == N + 1) | (col_blocks[coo.col] == 0)
    assert r.border_nnz == int(is_border.sum())
    assert 0.0 <= r.border_fraction <= 1.0


def test_diagnose_preconditions() -> None:
    m = _synthetic_storage_model()
    with pytest.raises(ValueError, match="no blocks assigned"):
        pips.diagnose(m)


def test_diagnose_no_regular_constraints() -> None:
    m = Model()
    time = pd.RangeIndex(4, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_objective(x.sum())
    m.assign_blocks("time", 2)
    with pytest.raises(ValueError, match="no regular constraints"):
        pips.diagnose(m)


@pytest.mark.parametrize(
    "target_cores, rec_ranks, rec_threads, capped",
    [(None, 50, 1, False), (50, 50, 1, False), (200, 50, 4, True), (30, 30, 1, False)],
)
def test_diagnose_recommendation(
    target_cores: int | None, rec_ranks: int, rec_threads: int, capped: bool
) -> None:
    m = _time_model(50)
    m.assign_blocks("time", 50)
    r = pips.diagnose(m, target_cores=target_cores)
    assert r.max_ranks == 50
    assert r.rec_ranks == rec_ranks
    assert r.rec_threads == rec_threads
    assert any("capped" in w for w in r.warnings) == capped
    assert r.rec_ranks <= r.max_ranks
    assert r.rec_threads >= 1


def _imbalanced_model() -> Model:
    m = Model()
    time = pd.RangeIndex(9, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x <= 1, name="c")
    for i in range(3):
        m.add_constraints(x.isel(time=slice(0, 3)) >= -5, name=f"extra{i}")
    m.add_objective(x.sum())
    m.assign_blocks("time", 3)
    return m


def _empty_block_model() -> Model:
    m = Model()
    time = pd.RangeIndex(9, name="time")
    x = m.add_variables(coords=[time], name="x")
    m.add_constraints(x.isel(time=slice(0, 6)) <= 1, name="c")
    m.add_objective(x.sum())
    m.assign_blocks("time", 3)
    return m


@pytest.mark.parametrize(
    "builder, substring",
    [
        (_synthetic_storage_model, "high border fraction"),
        (_imbalanced_model, "block imbalance"),
        (_empty_block_model, "empty local blocks"),
        (lambda: _time_model(6).assign_blocks("time", 1), "not decomposed"),
    ],
)
def test_diagnose_warnings(builder, substring: str) -> None:
    m = builder()
    if m.blocks is None:
        m.assign_blocks("time", 3)
    r = pips.diagnose(m)
    assert any(substring in w for w in r.warnings)


def test_diagnose_str_renders_all_groups() -> None:
    m = _synthetic_storage_model()
    m.assign_blocks("time", 3)
    text = str(pips.diagnose(m, target_cores=8))
    for token in [
        "BlockReport",
        "columns",
        "block nnz",
        "rows",
        "border",
        "parallel",
        "warnings",
    ]:
        assert token in text


@pytest.mark.parametrize("masked", [False, True])
def test_diagnose_realistic_consistency(masked: bool) -> None:
    m = _pips_time_plant_model(masked=masked)
    m.assign_blocks("time", 2)
    r = pips.diagnose(m, target_cores=8)
    assert r.n_global_cols + sum(r.block_cols.values()) == r.n_vars
    assert r.n_local_rows + r.n_global_rows + r.n_linking_rows == r.n_cons
    assert r.n_adjacent_rows + r.n_border_rows == r.n_linking_rows
    assert r.border_nnz == pytest.approx(r.border_fraction * r.nnz)
    assert r.rec_ranks <= r.max_ranks == r.n_blocks


def test_build_pips_command_defaults_ranks_to_blocks() -> None:
    cmd, env = pips.build_pips_command("drv", "dir", pips.PipsConfig(), n_blocks=8)
    assert cmd == ["mpirun", "-np", "8", "drv", "dir"]
    assert env == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def test_build_pips_command_caps_ranks_at_blocks() -> None:
    cfg = pips.PipsConfig(n_ranks=200)
    cmd, _ = pips.build_pips_command("drv", "dir", cfg, n_blocks=50)
    assert cmd[:3] == ["mpirun", "-np", "50"]


def test_build_pips_command_srun_threads_solver_and_args() -> None:
    cfg = pips.PipsConfig(
        launcher="srun",
        n_ranks=4,
        threads_per_rank=8,
        launcher_args=["--exclusive"],
        linear_solver="pardiso",
        options={"tol": 1e-8},
    )
    cmd, env = pips.build_pips_command("drv", "dir", cfg)
    assert cmd[:4] == ["srun", "-n", "4", "--exclusive"]
    assert cmd[4:6] == ["drv", "dir"]
    assert cmd[6:] == ["--tol", "1e-08", "--linear-solver", "pardiso"]
    assert env == {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}


@pytest.mark.parametrize(
    "cfg, exc",
    [
        (pips.PipsConfig(launcher="poe"), ValueError),
        (pips.PipsConfig(n_ranks=0), ValueError),
    ],
)
def test_build_pips_command_fail_fast(
    cfg: pips.PipsConfig, exc: type[Exception]
) -> None:
    with pytest.raises(exc):
        pips.build_pips_command("drv", "dir", cfg, n_blocks=4)


def test_pips_config_env_defaults_and_option_overrides(monkeypatch) -> None:
    from linopy.solvers import _pips_config

    monkeypatch.setenv("PIPS_LAUNCHER", "srun")
    monkeypatch.setenv("PIPS_THREADS", "4")
    monkeypatch.setenv("PIPS_LINEAR_SOLVER", "mumps")
    env_cfg = _pips_config({"presolve": "on"})
    assert env_cfg.launcher == "srun"
    assert env_cfg.threads_per_rank == 4
    assert env_cfg.linear_solver == "mumps"
    assert env_cfg.options == {"presolve": "on"}

    override = _pips_config(
        {"launcher": "mpirun", "n_ranks": 12, "linear_solver": "ma57"}
    )
    assert override.launcher == "mpirun"
    assert override.n_ranks == 12
    assert override.linear_solver == "ma57"


def _exported(tmp_path, n_blocks: int = 3) -> Model:
    m = _time_model(6)
    m.assign_blocks("time", n_blocks)
    m.to_pips_files(tmp_path)
    return m


def test_write_job_slurm_script(tmp_path) -> None:
    _exported(tmp_path, 3)
    cfg = pips.PipsConfig(threads_per_rank=4, linear_solver="mumps")
    script = pips.write_job(
        tmp_path,
        cfg,
        binary="/opt/pips/drv",
        nodes=2,
        time="01:00:00",
        partition="p",
        account="a",
    )
    assert script == tmp_path / "pips.slurm"
    text = script.read_text()
    assert text.startswith("#!/bin/bash")
    assert "#SBATCH --ntasks=3" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --partition=p" in text
    assert "#SBATCH --account=a" in text
    assert "export OMP_NUM_THREADS=4" in text
    assert "export MKL_NUM_THREADS=4" in text
    assert "srun -n 3" in text
    assert "/opt/pips/drv" in text
    assert str(tmp_path.resolve()) in text
    assert "--linear-solver mumps" in text


def test_write_job_binary_from_env(tmp_path, monkeypatch) -> None:
    _exported(tmp_path, 2)
    monkeypatch.setenv("PIPS_BINARY", "/env/drv")
    text = pips.write_job(tmp_path).read_text()
    assert "/env/drv" in text
    assert "#SBATCH --ntasks=2" in text


def test_write_job_custom_path_and_sbatch_args(tmp_path) -> None:
    _exported(tmp_path, 2)
    out = tmp_path / "custom.slurm"
    script = pips.write_job(
        tmp_path, binary="/d", path=out, sbatch_args=["--qos=long", "--mem=0"]
    )
    assert script == out
    text = out.read_text()
    assert "#SBATCH --qos=long" in text
    assert "#SBATCH --mem=0" in text


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"scheduler": "pbs", "binary": "/d"}, NotImplementedError),
        ({}, ValueError),
    ],
)
def test_write_job_fail_fast(tmp_path, monkeypatch, kwargs: dict, exc) -> None:
    _exported(tmp_path, 2)
    monkeypatch.delenv("PIPS_BINARY", raising=False)
    with pytest.raises(exc):
        pips.write_job(tmp_path, **kwargs)


def test_write_job_emits_output_and_strict_mode_and_manifest(tmp_path) -> None:
    _exported(tmp_path, 2)
    text = pips.write_job(tmp_path, binary="/d").read_text()
    assert "#SBATCH --output=" in text
    assert "set -euo pipefail" in text
    assert "module load" not in text
    assert (tmp_path / "pips.run.json").exists()


def test_write_job_modules_env_setup_and_custom_output(tmp_path) -> None:
    _exported(tmp_path, 2)
    text = pips.write_job(
        tmp_path,
        binary="/d",
        modules=["gcc", "openmpi"],
        env_setup=["source /opt/x/setvars.sh"],
        output="/x/%j.out",
    ).read_text()
    assert "module purge" in text
    assert "module load gcc openmpi" in text
    assert "source /opt/x/setvars.sh" in text
    assert "#SBATCH --output=/x/%j.out" in text
    assert "set +u" in text and "set -u" in text


def test_write_job_run_manifest_content(tmp_path) -> None:
    _exported(tmp_path, 2)
    cfg = pips.PipsConfig(threads_per_rank=4, linear_solver="mumps")
    pips.write_job(
        tmp_path,
        cfg,
        binary="/d",
        modules=["gcc"],
        env_setup=["source /opt/x/setvars.sh"],
        output="/x/%j.out",
    )
    manifest = json.loads((tmp_path / "pips.run.json").read_text())
    assert set(manifest) == {
        "linopy_version",
        "created",
        "n_blocks",
        "config",
        "command",
        "env",
        "job",
    }
    assert manifest["config"] == asdict(replace(cfg, launcher="srun"))
    assert manifest["command"][0] == "srun"
    assert manifest["env"]["OMP_NUM_THREADS"] == "4"
    assert manifest["n_blocks"] == 2
    job = manifest["job"]
    assert job["modules"] == ["gcc"]
    assert job["env_setup"] == ["source /opt/x/setvars.sh"]
    assert job["output"] == "/x/%j.out"


def test_write_run_manifest_inline_shape(tmp_path) -> None:
    cfg = pips.PipsConfig(threads_per_rank=2)
    command, env = pips.build_pips_command("/d", str(tmp_path), cfg, n_blocks=2)
    pips.write_run_manifest(tmp_path, config=cfg, command=command, env=env, n_blocks=2)
    manifest = json.loads((tmp_path / "pips.run.json").read_text())
    assert set(manifest) == {
        "linopy_version",
        "created",
        "n_blocks",
        "config",
        "command",
        "env",
    }
    assert manifest["config"] == asdict(cfg)
    assert manifest["command"] == command
    assert manifest["env"] == env


@pytest.mark.skipif(
    "pips" not in available_solvers,
    reason="requires the PIPS driver (set PIPS_BINARY)",
)
def test_doctor_ok() -> None:
    report = pips.doctor()
    assert "OK" in report
    assert "objective" in report
