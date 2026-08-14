# DREAM-MPC

Official implementation of **DREAM-MPC**, a dual-responsive edge-time risk
allocation model predictive control method for collision-free coordination of
heterogeneous multi-robot systems.

本仓库提供 DREAM-MPC 的完整可复现实验代码，包括五种对比方法、消融与统计检验、
规模及参数敏感性实验，以及 MuJoCo 刚体动力学验证。

## Contents

- `code/dream_mpc/`: controllers, risk allocator, solver, environments, metrics,
  MuJoCo benchmark, and visualization utilities.
- `code/run_experiments.py`: main system-level experiments.
- `code/run_strict_ablation.py`: strictly paired risk-allocation ablations.
- `code/run_mujoco.py`: MuJoCo rigid-body experiments.
- `code/results/`: generated records, tables, and publication figures (created
  locally by the experiment scripts and intentionally not versioned).

## Reproduction

Install the main dependencies and run from the repository root:

```bash
python -m pip install -r code/requirements.txt
python code/run_experiments.py
python code/run_strict_ablation.py
python code/make_figures.py
```

For the rigid-body benchmark, additionally install the MuJoCo dependencies:

```bash
python -m pip install -r code/requirements-mujoco.txt
python code/run_mujoco.py
```

Detailed experiment counts, output descriptions, and the snapshot reproduction
pipeline are documented in [`code/README.md`](code/README.md).

## Citation

The citation entry will be added after the associated paper is formally
published.
