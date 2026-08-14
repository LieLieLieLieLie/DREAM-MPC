from __future__ import annotations

import numpy as np

from .envs import MultiRobotArena, minimum_clearance


def summarize_episode(arena: MultiRobotArena, method: str, seed: int, solve_log: list[dict[str, float]],
                      layout: str, condition: str, variant: str = "完整方法") -> dict[str, float | int | str]:
    pair_clear: list[float] = []
    obstacle_clear: list[float] = []
    jerk: list[float] = []
    commands: list[np.ndarray] = []
    for snap in arena.history:
        p = np.asarray(snap["pos"])
        pc, oc = minimum_clearance(p, arena.radii, arena.layout.obstacles)
        pair_clear.append(pc); obstacle_clear.append(oc)
        commands.append(np.asarray(snap["u"]))
    for a, b in zip(commands[:-1], commands[1:]):
        jerk.append(float(np.mean(np.linalg.norm((b - a) / arena.cfg.controller.dt, axis=1))))
    pair_arr = np.asarray(pair_clear)
    obs_arr = np.asarray(obstacle_clear)
    collided = bool(np.any(pair_arr < 0.0) or np.any(obs_arr < 0.0))
    completion = float(np.mean(arena.done))
    success = float(completion == 1.0 and not collided)
    goal_error = float(np.mean(np.linalg.norm(arena.pos - arena.goal, axis=1)))
    solve = np.asarray([row["solve_ms"] for row in solve_log]) if solve_log else np.zeros(1)
    slack = np.asarray([row["max_slack"] for row in solve_log]) if solve_log else np.zeros(1)
    slack_sum = np.asarray([row.get("sum_slack", 0.0) for row in solve_log], dtype=float) if solve_log else np.zeros(1)
    positive_slack = np.asarray([row.get("positive_slack_count", 0) for row in solve_log], dtype=float) if solve_log else np.zeros(1)
    solver_failed = np.asarray([row.get("solver_failed", 0.0) for row in solve_log], dtype=float) if solve_log else np.zeros(1)
    entropy = np.asarray([row["risk_entropy"] for row in solve_log]) if solve_log else np.zeros(1)
    risk_tv = np.asarray([row.get("risk_tv", 0.0) for row in solve_log], dtype=float) if solve_log else np.zeros(1)
    dual_corr = np.asarray([row.get("dual_risk_spearman", np.nan) for row in solve_log], dtype=float) if solve_log else np.asarray([np.nan])
    dual_ratio = np.asarray([row.get("dual_top20_ratio", np.nan) for row in solve_log], dtype=float) if solve_log else np.asarray([np.nan])
    finite_corr = dual_corr[np.isfinite(dual_corr)]
    finite_ratio = dual_ratio[np.isfinite(dual_ratio)]
    return {
        "method": method, "variant": variant, "seed": seed, "layout": layout, "condition": condition,
        "n_agents": arena.n, "success": success, "collision_free": float(not collided), "completion": completion,
        "min_pair_clearance": float(np.min(pair_arr)) if len(pair_arr) else np.nan,
        "min_obstacle_clearance": float(np.min(obs_arr)) if len(obs_arr) else np.nan,
        "clearance_q05": float(np.quantile(pair_arr, 0.05)) if len(pair_arr) else np.nan,
        "collision_steps": int(np.sum(pair_arr < 0.0) + np.sum(obs_arr < 0.0)),
        "completion_time_s": float(arena.t * arena.cfg.controller.dt), "goal_error_m": goal_error,
        "path_length_m": float(np.sum(arena.path_length)), "energy": float(np.sum(arena.energy)),
        "mean_jerk": float(np.mean(jerk)) if jerk else 0.0,
        "solve_ms_mean": float(np.mean(solve)), "solve_ms_p95": float(np.quantile(solve, 0.95)),
        "max_slack": float(np.max(slack)), "risk_entropy": float(np.mean(entropy)),
        "mpc_slack_sum": float(np.sum(slack_sum)),
        "positive_slack_cycle_rate": float(np.mean(positive_slack > 0.0)),
        "solver_failure_count": int(np.sum(solver_failed)),
        "dual_risk_spearman": float(np.mean(finite_corr)) if len(finite_corr) else np.nan,
        "dual_top20_share_ratio": float(np.mean(finite_ratio)) if len(finite_ratio) else np.nan,
        "risk_budget_tv": float(np.sum(risk_tv[1:]) / max(len(risk_tv) - 1, 1)),
    }


def trace_payload(arena: MultiRobotArena, solve_log: list[dict[str, object]]) -> dict[str, object]:
    return {
        "starts": arena.layout.starts.tolist(), "goals": arena.layout.goals.tolist(),
        "radii": arena.radii.tolist(),
        "obstacles": [{"center": list(o.center), "radius": o.radius} for o in arena.layout.obstacles],
        "positions": [np.asarray(s["pos"]).tolist() for s in arena.history],
        "velocities": [np.asarray(s["vel"]).tolist() for s in arena.history],
        "commands": [np.asarray(s["u"]).tolist() for s in arena.history],
        "solver": solve_log,
    }
