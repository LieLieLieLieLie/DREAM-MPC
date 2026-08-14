from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.special import ndtri

from .config import ControllerConfig


RiskKey = tuple[str, int, int, int]


def capped_simplex(weights: np.ndarray, total: float, lower: float, upper: float) -> np.ndarray:
    """Water-fill a nonnegative budget without violating the total-risk cap.

    If too few components exist to consume ``total`` under ``upper``, the
    remainder is deliberately left unallocated.  If the configured lower bound
    is infeasible because ``n * lower > total``, the effective lower bound is
    relaxed to ``total / n``.  Thus the returned vector always satisfies
    ``sum(x) <= total`` and the active allocation problem remains feasible.
    """
    n = len(weights)
    if n == 0:
        return np.empty(0)
    total = max(float(total), 0.0)
    lower_eff = min(max(float(lower), 0.0), total / n)
    upper_eff = max(float(upper), lower_eff)
    target_total = min(total, n * upper_eff)
    x = np.full(n, lower_eff, dtype=float)
    remaining = target_total - n * lower_eff
    active = np.ones(n, dtype=bool)
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    for _ in range(n + 2):
        if remaining <= 1e-12 or not np.any(active):
            break
        share = remaining * w[active] / np.sum(w[active])
        room = upper_eff - x[active]
        add = np.minimum(share, room)
        x[active] += add
        used = float(np.sum(add))
        remaining -= used
        idx = np.flatnonzero(active)
        active[idx[x[idx] >= upper_eff - 1e-12]] = False
        if used <= 1e-12:
            break
    return x


@dataclass
class DualResponsiveRiskAllocator:
    cfg: ControllerConfig
    previous: dict[RiskKey, float] = field(default_factory=dict)
    duals: dict[RiskKey, float] = field(default_factory=dict)
    last_pressure: dict[RiskKey, float] = field(default_factory=dict)

    def allocate(self, keys: list[RiskKey], urgency: np.ndarray, sigma: np.ndarray,
                 adaptive: bool, use_dual: bool = True,
                 use_smoothing: bool = True) -> dict[RiskKey, float]:
        if not keys:
            return {}
        if adaptive:
            dual = np.asarray([self.duals.get(k, 0.0) for k in keys]) if use_dual else np.zeros(len(keys))
            self.last_pressure = {k: float(v) for k, v in zip(keys, dual)}
            sensitivity = (0.22 + urgency) * sigma * (0.35 + np.sqrt(dual + 0.04))
            # KKT solution of the dominant a/sqrt(epsilon) term: epsilon proportional to a^(2/3).
            target = capped_simplex(np.power(np.maximum(sensitivity, 1e-10), 2.0 / 3.0), self.cfg.total_risk,
                                    self.cfg.epsilon_min, self.cfg.epsilon_max)
            if use_smoothing:
                prior = np.asarray([self.previous.get(k, self.cfg.total_risk / len(keys)) for k in keys])
                prior = capped_simplex(prior, self.cfg.total_risk, self.cfg.epsilon_min, self.cfg.epsilon_max)
                eta = self.cfg.risk_smoothing
                values = capped_simplex(
                    np.exp((1.0 - eta) * np.log(prior + 1e-12) + eta * np.log(target + 1e-12)),
                    self.cfg.total_risk, self.cfg.epsilon_min, self.cfg.epsilon_max,
                )
            else:
                values = target
        else:
            self.last_pressure = {k: 0.0 for k in keys}
            values = capped_simplex(np.ones(len(keys)), self.cfg.total_risk,
                                    self.cfg.epsilon_min, self.cfg.epsilon_max)
        allocation = {k: float(v) for k, v in zip(keys, values)}
        self.previous.update(allocation)
        return allocation

    def update_duals(self, duals: dict[RiskKey, float]) -> None:
        for key, value in duals.items():
            old = self.duals.get(key, 0.0)
            self.duals[key] = 0.62 * old + 0.38 * min(abs(float(value)), 50.0)


def prediction_sigma(stage: int, cfg: ControllerConfig, relative: bool = True, scale: float = 1.0) -> float:
    h = stage + 1
    base = cfg.covariance_floor + cfg.covariance_growth * np.sqrt(h)
    return float(cfg.moment_inflation * scale * base * (np.sqrt(2.0) if relative else 1.0))


def risk_margin(method: str, epsilon: float, sigma: float, cfg: ControllerConfig) -> float:
    # The lower bound may be relaxed when the active set is too large.  Retain
    # the actual allocation here; clipping it back to epsilon_min would weaken
    # the Cantelli tightening and invalidate the total-risk guarantee.
    epsilon = float(np.clip(epsilon, 1e-9, 0.20))
    if method == "Tube-DMPC":
        return cfg.fixed_tube
    if method in {"CC-MPC", "CVaR-MPC"}:
        gaussian = max(0.0, float(ndtri(1.0 - epsilon))) * sigma
        return gaussian * (1.18 if method == "CVaR-MPC" else 1.0)
    if method in {"Uniform-DRMPC", "DREAM-MPC"}:
        # Exact one-sided Cantelli factor for the moment ambiguity class.
        return float(np.sqrt((1.0 - epsilon) / epsilon) * sigma)
    return 0.0
