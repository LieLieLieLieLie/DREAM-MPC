from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from .config import METHODS, MODELS, TABLES, ControllerConfig, ExperimentConfig
from .controllers import DreamController
from .envs import MultiRobotArena
from .metrics import summarize_episode, trace_payload


ABLATIONS = ("完整方法", "无对偶敏感度", "均匀风险分配", "高斯机会约束", "无安全投影")
STRICT_ABLATIONS = ("完整方法", "均匀风险分配", "无对偶响应", "无KL平滑")


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation without an additional statistics dependency."""
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def run_episode(cfg: ExperimentConfig, method: str, seed: int, layout: str, condition: str,
                n_agents: int = 6, variant: str = "完整方法", keep_trace: bool = False):
    arena = MultiRobotArena(cfg, layout, condition, n_agents, seed)
    controller = DreamController(cfg.controller, method, variant)
    logs: list[dict[str, object]] = []
    previous_allocation: dict[object, float] | None = None
    for _ in range(cfg.steps):
        state = arena.state()
        command = controller.act(state)
        solution = controller.last_solution
        assert solution is not None
        allocation = solution.allocation
        if previous_allocation is None:
            risk_tv = 0.0
        else:
            union = set(previous_allocation) | set(allocation)
            risk_tv = float(sum(abs(allocation.get(key, 0.0) - previous_allocation.get(key, 0.0)) for key in union))
        previous_allocation = dict(allocation)
        top = sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:8]
        pressure_keys = [key for key in allocation if key in solution.allocation_pressure]
        pressure = np.asarray([solution.allocation_pressure[key] for key in pressure_keys], dtype=float)
        shares = np.asarray([allocation[key] for key in pressure_keys], dtype=float)
        dual_risk_spearman = _rank_correlation(pressure, shares)
        if len(pressure) >= 2 and not np.allclose(pressure, pressure[0]):
            top_n = max(1, int(np.ceil(0.20 * len(pressure))))
            order = np.argsort(pressure)
            high = shares[order[-top_n:]]
            rest = shares[order[:-top_n]]
            dual_top20_ratio = float(np.mean(high) / max(np.mean(rest), 1e-12)) if len(rest) else float("nan")
        else:
            dual_top20_ratio = float("nan")
        logs.append({
            "solve_ms": solution.solve_ms, "max_slack": solution.max_slack,
            "sum_slack": solution.sum_slack, "positive_slack_count": solution.positive_slack_count,
            "solver_failed": float("solved" not in solution.status.lower()),
            "risk_entropy": solution.risk_entropy, "active_constraints": solution.active_constraints,
            "uncertainty_scale": controller.uncertainty_scale, "interventions": controller.interventions,
            "risk_tv": risk_tv,
            "dual_risk_spearman": dual_risk_spearman,
            "dual_top20_ratio": dual_top20_ratio,
            "top_risk": [[list(k), v] for k, v in top],
        })
        arena.step(command)
        if arena.done.all():
            break
    row = summarize_episode(arena, method, seed, logs, layout, condition, variant)
    row["interventions"] = controller.interventions
    return row, (trace_payload(arena, logs) if keep_trace else None)


def _holm_pvalues(raw: list[float]) -> list[float]:
    order = np.argsort(raw)
    adjusted = np.zeros(len(raw))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _statistical_tests(records: pd.DataFrame) -> pd.DataFrame:
    main = records[records.phase == "main"].copy()
    dream = main[main.method == "DREAM-MPC"].sort_values(["layout", "condition", "seed"])
    rows = []; raw = []
    for baseline in METHODS[:-1]:
        other = main[main.method == baseline].sort_values(["layout", "condition", "seed"])
        delta = dream.success.to_numpy() - other.success.to_numpy()
        try:
            p = float(wilcoxon(delta, zero_method="zsplit", alternative="greater").pvalue)
        except ValueError:
            p = 1.0
        raw.append(p)
        rows.append({"比较": f"DREAM-MPC vs {baseline}", "安全成功率差值": float(np.mean(delta)), "原始p值": p})
    adjusted = _holm_pvalues(raw)
    for row, value in zip(rows, adjusted):
        row["Holm校正p值"] = value
        row["显著性"] = "是" if value < 0.05 else "否"
    return pd.DataFrame(rows)


def run_all(cfg: ExperimentConfig | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    cfg = cfg or ExperimentConfig(); cfg.ensure_dirs()
    records: list[dict[str, object]] = []
    traces: dict[str, object] = {}
    total_start = time.perf_counter(); counter = 0

    def execute(phase: str, method: str, seed: int, layout: str, condition: str,
                n_agents: int = 6, variant: str = "完整方法", trace_key: str | None = None):
        nonlocal counter
        row, trace = run_episode(cfg, method, seed, layout, condition, n_agents, variant, trace_key is not None)
        row["phase"] = phase; records.append(row)
        if trace_key is not None and trace is not None:
            traces[trace_key] = trace
        counter += 1
        if counter % 24 == 0:
            elapsed = time.perf_counter() - total_start
            print(f"[{counter:04d}] elapsed={elapsed:.1f}s last={phase}/{method}/{layout}/{condition}/seed{seed}", flush=True)

    for layout in cfg.layouts:
        for condition in cfg.conditions:
            for method in cfg.methods:
                for seed in cfg.main_seeds:
                    key = None
                    if seed == 2 and condition == "突发失配" and layout == "仓储通道":
                        key = f"{method}-warehouse"
                    execute("main", method, seed, layout, condition, 6, "完整方法", key)

    for n_agents in (4, 6, 8):
        for method in cfg.methods:
            for seed in cfg.auxiliary_seeds[:4]:
                execute("scalability", method, seed + 100, "交叉口", "重尾扰动", n_agents)

    for budget in (0.10, 0.14, 0.18, 0.22, 0.26):
        local_ctrl = replace(cfg.controller, total_risk=budget)
        local_cfg = replace(cfg, controller=local_ctrl)
        for seed in cfg.auxiliary_seeds:
            row, _ = run_episode(local_cfg, "DREAM-MPC", seed + 200, "交叉口", "混合偏移", 6)
            row["phase"] = "sensitivity"; row["risk_budget"] = budget; records.append(row); counter += 1

    for variant in ABLATIONS:
        for seed in cfg.auxiliary_seeds:
            execute("ablation", "DREAM-MPC", seed + 300, "仓储通道", "突发失配", 6, variant)

    frame = pd.DataFrame(records)
    frame.to_json(MODELS / "episode_records.jsonl", orient="records", lines=True, force_ascii=False)
    with (MODELS / "representative_traces.json").open("w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False)
    summary = frame.groupby(["phase", "method"], dropna=False).mean(numeric_only=True).reset_index()
    payload = {
        "config": asdict(cfg), "episodes": len(frame), "elapsed_s": time.perf_counter() - total_start,
        "summary": summary.to_dict(orient="records"),
    }
    with (MODELS / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    ablation = frame[frame.phase == "ablation"].groupby("variant").agg(
        安全成功率=("success", "mean"), 无碰撞率=("collision_free", "mean"),
        最小间距_m=("min_pair_clearance", "mean"), 完成时间_s=("completion_time_s", "mean"),
        控制跳变=("mean_jerk", "mean"), 平均求解时间_ms=("solve_ms_mean", "mean")
    ).reset_index().rename(columns={"variant": "变体"})
    ablation.to_csv(TABLES / "ablation.csv", index=False, encoding="utf-8-sig")
    stats = _statistical_tests(frame)
    stats.to_csv(TABLES / "statistical_tests.csv", index=False, encoding="utf-8-sig")
    parameters = pd.DataFrame([{"参数": key, "数值": value} for key, value in asdict(cfg.controller).items()])
    parameters.to_csv(TABLES / "controller_parameters.csv", index=False, encoding="utf-8-sig")
    with (MODELS / "run.log").open("w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return frame, traces
