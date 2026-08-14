# DREAM-MPC reproducible experiments

This directory contains the complete implementation of dual-responsive
edge–time risk allocation MPC (DREAM-MPC), five baselines, ablations,
statistical tests, scalability studies, and the MuJoCo rigid-body benchmark.

## Environment

- Main experiments: Python 3.10+, NumPy, SciPy, pandas, Matplotlib, seaborn,
  OSQP, and statsmodels.
- MuJoCo experiments: MuJoCo 3.11.0 and the dependencies installed in
  `E:\MuJoCo\runtime`.

## Reproduce

From the repository root (replace `python` with the interpreter of the target
environment when needed):

```bash
python code/run_experiments.py
python code/run_strict_ablation.py
python code/run_mujoco.py
python code/make_figures.py
```

To rebuild the six-panel MuJoCo state montage used by the paper, run the
snapshot-only pipeline after the full MuJoCo suite:

```bash
python code/render_mujoco_snapshots.py
python code/update_snapshot_metrics.py
python code/update_dream_failure_snapshot.py capture
python code/update_dream_failure_snapshot.py render
python code/make_figures.py
```

The first command runs 678 reported system-level/auxiliary kinematic trials
plus 30 preliminary diagnostic ablations retained for backward compatibility.
Only the 678 reported trials and the separate 384-run strict ablation enter the
paper. The second command runs that strictly paired risk-allocation ablation, the third
runs the 192 rigid-body trials, and the last command regenerates all composite PDF
figures from saved records without rerunning control experiments. The optional
snapshot pipeline reproduces the baseline native frames and the documented
DREAM-MPC seed-503 peak-contact state panel.

The HOCBF rows use the implemented current-command coefficient
`lambda_i * beta_i`, including condition-dependent actuator gain. Risk vectors
are keyed by `(edge_type, object_1, object_2, prediction_stage)`. Missing keys
are zero-extended only for temporal-variation auditing; inactive historical
shares remain cached for later reactivation. The water-filling routine leaves
unusable upper-bounded budget unallocated and relaxes the component lower bound
to `alpha / m` only when required, so the total allocation never exceeds
`alpha` for any active-set size.

In MuJoCo, the force command is `m_nominal * scenario_gain * desired_acceleration`
and the simulated body mass is `m_nominal * mass_scale`; consequently the HOCBF
uses the equivalent acceleration-domain effectiveness
`scenario_gain / mass_scale`.

## Outputs

- `results/models`: episode-level JSONL records, representative traces,
  run metadata, and logs.
- `results/tables`: ablations, controller parameters, and corrected
  statistical tests.
- `results/figures`: publication-ready Chinese PDF figures only.

Method implementations are in `dream_mpc/`; all random experiments use
explicit seeds recorded in the saved files.
