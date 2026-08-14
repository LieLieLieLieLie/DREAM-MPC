from __future__ import annotations

import os

# Some managed Windows sessions expose a broken system entropy provider.  The
# experiments themselves use explicit NumPy seeds, so a deterministic fallback
# is sufficient for library import-time initialization and preserves rerunability.
try:
    os.urandom(1)
except OSError:
    os.urandom = lambda n: bytes((37 + 17 * i) % 256 for i in range(n))  # type: ignore[assignment]

from dataclasses import asdict
import json
from math import sqrt
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from dream_mpc.config import ExperimentConfig, METHODS, MODELS, TABLES
from dream_mpc.experiments import STRICT_ABLATIONS, run_episode


PAIR_KEYS = ["layout", "condition", "seed"]
BINARY_METRICS = ("success", "collision_free")
CONTINUOUS_METRICS = (
    "clearance_q05",
    "completion_time_s",
    "energy",
    "mean_jerk",
    "solve_ms_mean",
    "solve_ms_p95",
    "risk_budget_tv",
    "mpc_slack_sum",
    "positive_slack_cycle_rate",
    "solver_failure_count",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    den = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / den
    half = z / den * sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)


def holm_adjust(raw: list[float]) -> list[float]:
    if not raw:
        return []
    order = np.argsort(raw)
    adjusted = np.zeros(len(raw), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def exact_mcnemar(reference: np.ndarray, comparator: np.ndarray) -> tuple[int, int, float]:
    ref = reference.astype(bool)
    cmp = comparator.astype(bool)
    ref_only = int(np.sum(ref & ~cmp))
    cmp_only = int(np.sum(~ref & cmp))
    discordant = ref_only + cmp_only
    p = 1.0 if discordant == 0 else float(binomtest(ref_only, discordant, 0.5, alternative="two-sided").pvalue)
    return ref_only, cmp_only, p


def paired_frames(frame: pd.DataFrame, group_col: str, reference: str, comparator: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = frame[frame[group_col] == reference].sort_values(PAIR_KEYS).reset_index(drop=True)
    right = frame[frame[group_col] == comparator].sort_values(PAIR_KEYS).reset_index(drop=True)
    if len(left) != len(right) or not left[PAIR_KEYS].equals(right[PAIR_KEYS]):
        raise RuntimeError(f"Paired design mismatch: {reference} vs {comparator}")
    return left, right


def paired_tests(frame: pd.DataFrame, group_col: str, reference: str, comparators: list[str], label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw: list[float] = []
    for comparator in comparators:
        ref, cmp = paired_frames(frame, group_col, reference, comparator)
        for metric in BINARY_METRICS:
            ref_only, cmp_only, p = exact_mcnemar(ref[metric].to_numpy(), cmp[metric].to_numpy())
            rows.append({
                "比较": f"{reference} vs {comparator}", "指标": metric, "检验": "精确McNemar",
                "样本对数": len(ref), "参考方法独有成功数": ref_only, "对照方法独有成功数": cmp_only,
                "配对差值": float((ref[metric] - cmp[metric]).mean()), "秩二列效应量": np.nan, "原始p值": p,
            })
            raw.append(p)
        for metric in CONTINUOUS_METRICS:
            if metric not in ref or metric not in cmp:
                continue
            delta = ref[metric].to_numpy(float) - cmp[metric].to_numpy(float)
            if np.allclose(delta, 0.0):
                p = 1.0
                effect = 0.0
            else:
                p = float(wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto").pvalue)
                nonzero = delta[~np.isclose(delta, 0.0)]
                ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy(float)
                denom = float(np.sum(ranks))
                effect = float((np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / denom) if denom else 0.0
            rows.append({
                "比较": f"{reference} vs {comparator}", "指标": metric, "检验": "配对Wilcoxon符号秩",
                "样本对数": len(ref), "参考方法独有成功数": np.nan, "对照方法独有成功数": np.nan,
                "配对差值": float(np.mean(delta)), "秩二列效应量": effect, "原始p值": p,
            })
            raw.append(p)
    adjusted = holm_adjust(raw)
    for row, adj in zip(rows, adjusted):
        row["Holm校正p值"] = adj
        row["显著性"] = "是" if adj < 0.05 else "否"
        row["分析集"] = label
    return pd.DataFrame(rows)


def make_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in STRICT_ABLATIONS:
        part = frame[frame.variant == variant]
        n = len(part)
        success_n = int(part.success.sum())
        safe_n = int(part.collision_free.sum())
        success_lo, success_hi = wilson_interval(success_n, n)
        safe_lo, safe_hi = wilson_interval(safe_n, n)
        rows.append({
            "变体": variant,
            "样本数": n,
            "成功数": success_n,
            "任务成功率": success_n / n,
            "任务成功率95%CI下限": success_lo,
            "任务成功率95%CI上限": success_hi,
            "无碰撞数": safe_n,
            "无碰撞率": safe_n / n,
            "无碰撞率95%CI下限": safe_lo,
            "无碰撞率95%CI上限": safe_hi,
            "间距5%分位数_m": float(part.clearance_q05.mean()),
            "最小机器人间距_m": float(part.min_pair_clearance.mean()),
            "完成时间_s": float(part.completion_time_s.mean()),
            "控制能耗": float(part.energy.mean()),
            "控制跳变_m_s3": float(part.mean_jerk.mean()),
            "平均求解时间_ms": float(part.solve_ms_mean.mean()),
            "95%求解时间_ms": float(part.solve_ms_p95.mean()),
            "风险分配熵": float(part.risk_entropy.mean()),
            "风险预算抖动": float(part.risk_budget_tv.mean()),
            "MPC安全松弛累计值": float(part.mpc_slack_sum.mean()),
            "正松弛周期比例": float(part.positive_slack_cycle_rate.mean()),
            "子问题失败次数": float(part.solver_failure_count.mean()),
            "对偶压力-风险份额秩相关": float(part.dual_risk_spearman.mean()),
            "高对偶压力Top20%份额比": float(part.dual_top20_share_ratio.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = ExperimentConfig()
    cfg.ensure_dirs()
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    total = len(STRICT_ABLATIONS) * len(cfg.layouts) * len(cfg.conditions) * len(cfg.main_seeds)
    count = 0
    for variant in STRICT_ABLATIONS:
        for layout in cfg.layouts:
            for condition in cfg.conditions:
                for seed in cfg.main_seeds:
                    row, _ = run_episode(cfg, "DREAM-MPC", seed, layout, condition, 6, variant)
                    row["phase"] = "strict_ablation"
                    records.append(row)
                    count += 1
                    if count % 12 == 0:
                        print(f"[{count:03d}/{total}] {variant}/{layout}/{condition}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    frame = pd.DataFrame(records)
    frame.to_json(MODELS / "strict_ablation_records.jsonl", orient="records", lines=True, force_ascii=False)
    summary = make_summary(frame)
    summary.to_csv(TABLES / "strict_risk_ablation.csv", index=False, encoding="utf-8-sig")
    stats = paired_tests(frame, "variant", "完整方法", list(STRICT_ABLATIONS[1:]), "严格风险分配消融")
    stats.to_csv(TABLES / "strict_ablation_statistics.csv", index=False, encoding="utf-8-sig")

    main_path = MODELS / "episode_records.jsonl"
    if main_path.exists():
        old = pd.read_json(main_path, lines=True)
        main_frame = old[old.phase == "main"].copy()
        main_stats = paired_tests(main_frame, "method", "DREAM-MPC", list(METHODS[:-1]), "主实验基线比较")
        main_stats.to_csv(TABLES / "main_paired_statistics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "config": asdict(cfg),
        "paired_design": {"layouts": list(cfg.layouts), "conditions": list(cfg.conditions), "seeds": list(cfg.main_seeds)},
        "variants": list(STRICT_ABLATIONS),
        "episodes": len(frame),
        "elapsed_s": time.perf_counter() - started,
    }
    (MODELS / "strict_ablation_run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
