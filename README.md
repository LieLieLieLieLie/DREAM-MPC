# DREAM-MPC

Official implementation of **DREAM-MPC**, a dual-responsive edge-time risk
allocation model predictive control method for collision-free coordination of
heterogeneous multi-robot systems.

本仓库提供 DREAM-MPC 的完整可复现实验代码，包括五种对比方法、严格消融与统计检验、
规模及参数敏感性实验，以及 MuJoCo 刚体动力学验证。

## Repository structure

- `dream_mpc/`: controllers, risk allocator, solver, environments, metrics,
  MuJoCo benchmark, and visualization utilities.
- `run_experiments.py`: main system-level experiments.
- `run_strict_ablation.py`: strictly paired risk-allocation ablations.
- `run_mujoco.py`: MuJoCo rigid-body experiments.
- `make_figures.py`: regenerate all publication figures from saved records.
- `results/`: generated records, tables, and figures; created locally and not
  versioned.

## Environment

- Python 3.10+
- Main experiments: NumPy, SciPy, pandas, Matplotlib, seaborn, OSQP, and
  statsmodels.
- MuJoCo experiments: MuJoCo and the packages in
  `requirements-mujoco.txt`.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-mujoco.txt
```

## Reproduction

Run the full system-level benchmark, strict ablation, MuJoCo benchmark, and
figure generation from the repository root:

```bash
python run_experiments.py
python run_strict_ablation.py
python run_mujoco.py
python make_figures.py
```

To rebuild the MuJoCo state montage after the full MuJoCo suite:

```bash
python render_mujoco_snapshots.py
python update_snapshot_metrics.py
python update_dream_failure_snapshot.py capture
python update_dream_failure_snapshot.py render
python make_figures.py
```

The main command runs 678 reported system-level/auxiliary kinematic trials plus
30 preliminary diagnostic ablations retained for backward compatibility. Only
the 678 reported trials and the separate 384-run strict ablation enter the
paper. The MuJoCo suite contains 192 rigid-body trials. All random experiments
use explicit seeds recorded in the generated files.

The HOCBF rows use the implemented current-command coefficient
`lambda_i * beta_i`, including condition-dependent actuator gain. Risk vectors
are keyed by `(edge_type, object_1, object_2, prediction_stage)`. Missing keys
are zero-extended only for temporal-variation auditing; inactive historical
shares remain cached for later reactivation. The water-filling routine leaves
unusable upper-bounded budget unallocated and relaxes the component lower bound
to `alpha / m` only when required, so total allocation never exceeds `alpha`
for any active-set size.

In MuJoCo, the force command is
`m_nominal * scenario_gain * desired_acceleration`, while the simulated body
mass is `m_nominal * mass_scale`. The HOCBF therefore uses the equivalent
acceleration-domain effectiveness `scenario_gain / mass_scale`.

## Generated outputs

- `results/models/`: episode-level JSONL records, representative traces, run
  metadata, and logs.
- `results/tables/`: ablations, controller parameters, and corrected
  statistical tests.
- `results/figures/`: publication-ready Chinese PDF figures.

## Citation

The citation entry will be added after the associated paper is formally
published.
