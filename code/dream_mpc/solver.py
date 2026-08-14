from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import osqp
from scipy import sparse

from .config import ControllerConfig
from .envs import CircleObstacle
from .risk import DualResponsiveRiskAllocator, RiskKey, prediction_sigma, risk_margin


@dataclass
class MPCSolution:
    sequence: np.ndarray
    predicted_positions: np.ndarray
    solve_ms: float
    status: str
    max_slack: float
    sum_slack: float
    positive_slack_count: int
    duals: dict[RiskKey, float]
    allocation: dict[RiskKey, float]
    risk_entropy: float
    active_constraints: int
    allocation_pressure: dict[RiskKey, float]


def _prediction_maps(n: int, h: int, dt: float, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nu = n * h * 2
    rows = n * h * 2
    g = np.zeros((rows, nu), dtype=float)
    gv = np.zeros((rows, nu), dtype=float)
    base = np.zeros(rows, dtype=float)
    for i in range(n):
        for k in range(h):
            stage = k + 1
            for d in range(2):
                row = (i * h + k) * 2 + d
                base[row] = pos[i, d] + stage * dt * vel[i, d]
                for t in range(k + 1):
                    col = (i * h + t) * 2 + d
                    g[row, col] = dt * dt * (stage - t)
                    gv[row, col] = dt
    return base, g, gv


def _smoothness_matrix(n: int, h: int) -> np.ndarray:
    nu = n * h * 2
    dmat = np.zeros((nu, nu), dtype=float)
    for i in range(n):
        for k in range(h):
            for d in range(2):
                row = (i * h + k) * 2 + d
                dmat[row, row] = 1.0
                if k > 0:
                    prev = (i * h + k - 1) * 2 + d
                    dmat[row, prev] = -1.0
    return dmat


def _row_for_position(g: np.ndarray, i: int, k: int, h: int, normal: np.ndarray) -> np.ndarray:
    ix = (i * h + k) * 2
    return normal[0] * g[ix] + normal[1] * g[ix + 1]


class SequentialConvexMPC:
    def __init__(self, cfg: ControllerConfig, method: str):
        self.cfg = cfg
        self.method = method
        self.allocator = DualResponsiveRiskAllocator(cfg)
        self.warm: np.ndarray | None = None
        self.risk_use_dual = True
        self.risk_use_smoothing = True

    def _initial(self, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray, amax: np.ndarray) -> np.ndarray:
        n, h = len(pos), self.cfg.horizon
        if self.warm is not None and self.warm.shape == (n, h, 2):
            shifted = np.concatenate((self.warm[:, 1:], self.warm[:, -1:]), axis=1)
            return shifted.copy()
        desired = 0.95 * (goal - pos) - 0.85 * vel
        norm = np.linalg.norm(desired, axis=1) + 1e-12
        desired *= np.minimum(1.0, amax / norm)[:, None]
        decay = np.linspace(1.0, 0.45, h)[None, :, None]
        return desired[:, None, :] * decay

    def solve(self, state: dict[str, object], uncertainty_scale: float = 1.0) -> MPCSolution:
        started = time.perf_counter()
        cfg = self.cfg
        pos = np.asarray(state["pos"], dtype=float)
        vel = np.asarray(state["vel"], dtype=float)
        goal = np.asarray(state["goal"], dtype=float)
        radii = np.asarray(state["radii"], dtype=float)
        vmax = np.asarray(state["vmax"], dtype=float)
        amax = np.asarray(state["amax"], dtype=float)
        done = np.asarray(state["done"], dtype=bool)
        obstacles: tuple[CircleObstacle, ...] = state["obstacles"]  # type: ignore[assignment]
        n, h = len(pos), cfg.horizon
        priority = np.asarray(state.get("priority", np.linspace(1.0, 0.15, n)), dtype=float)
        nu = n * h * 2
        base, g, gv = _prediction_maps(n, h, cfg.dt, pos, vel)
        goal_flat = np.repeat(goal[:, None, :], h, axis=1).reshape(-1)
        stage_weights = np.tile(np.repeat(np.linspace(0.65, 1.0, h), 2), n)
        for i in range(n):
            stage_weights[i*h*2:(i+1)*h*2] *= 0.55 + 1.25 * priority[i]
            stage_weights[(i * h + h - 1) * 2:(i * h + h) * 2] *= cfg.terminal_weight / cfg.tracking_weight
        qdiag = cfg.tracking_weight * stage_weights
        smooth = _smoothness_matrix(n, h)
        p_u = 2.0 * (g.T @ (qdiag[:, None] * g) + cfg.control_weight * np.eye(nu) + cfg.smooth_weight * smooth.T @ smooth)
        q_u = 2.0 * g.T @ (qdiag * (base - goal_flat))
        u = self._initial(pos, vel, goal, amax).reshape(-1)
        allocation: dict[RiskKey, float] = {}
        final_slack = np.empty(0)
        final_status = "not solved"
        final_duals: dict[RiskKey, float] = {}
        active_count = 0

        for _ in range(cfg.scp_iterations):
            pred = (base + g @ u).reshape(n, h, 2)
            specs: list[tuple[RiskKey, np.ndarray, float, float, float]] = []
            keys: list[RiskKey] = []
            urgencies: list[float] = []
            sigmas: list[float] = []
            for k in range(h):
                for i in range(n):
                    for j in range(i + 1, n):
                        if done[i] and done[j]:
                            continue
                        rel = pred[i, k] - pred[j, k]
                        dist = float(np.linalg.norm(rel))
                        clearance = dist - radii[i] - radii[j]
                        if clearance > cfg.activation_distance:
                            continue
                        normal = rel / max(dist, 1e-8)
                        key: RiskKey = ("pair", i, j, k)
                        sigma = prediction_sigma(k, cfg, True, uncertainty_scale)
                        urgency = float(np.exp(-max(clearance, -0.2) / 0.72) * (1.0 + 0.30 / (k + 1)))
                        coeff = _row_for_position(g, i, k, h, normal) - _row_for_position(g, j, k, h, normal)
                        base_proj = float(normal @ (base[(i*h+k)*2:(i*h+k)*2+2] - base[(j*h+k)*2:(j*h+k)*2+2]))
                        specs.append((key, coeff, base_proj, radii[i] + radii[j], sigma))
                        keys.append(key); urgencies.append(urgency); sigmas.append(sigma)
                    if done[i]:
                        continue
                    for o, obstacle in enumerate(obstacles):
                        rel = pred[i, k] - np.asarray(obstacle.center)
                        dist = float(np.linalg.norm(rel))
                        clearance = dist - radii[i] - obstacle.radius
                        if clearance > cfg.obstacle_activation_distance:
                            continue
                        normal = rel / max(dist, 1e-8)
                        key = ("obs", i, o, k)
                        sigma = prediction_sigma(k, cfg, False, uncertainty_scale)
                        urgency = float(np.exp(-max(clearance, -0.2) / 0.62) * (1.0 + 0.25 / (k + 1)))
                        coeff = _row_for_position(g, i, k, h, normal)
                        base_i = base[(i*h+k)*2:(i*h+k)*2+2]
                        base_proj = float(normal @ (base_i - np.asarray(obstacle.center)))
                        specs.append((key, coeff, base_proj, radii[i] + obstacle.radius, sigma))
                        keys.append(key); urgencies.append(urgency); sigmas.append(sigma)

            adaptive = self.method == "DREAM-MPC"
            if self.method in {"CC-MPC", "CVaR-MPC", "Uniform-DRMPC", "DREAM-MPC"}:
                allocation = self.allocator.allocate(
                    keys, np.asarray(urgencies), np.asarray(sigmas), adaptive,
                    use_dual=self.risk_use_dual,
                    use_smoothing=self.risk_use_smoothing,
                )
            else:
                allocation = {k: cfg.total_risk / max(1, len(keys)) for k in keys}
            m = len(specs)
            active_count = m
            nv = nu + m
            p = np.zeros((nv, nv), dtype=float)
            p[:nu, :nu] = p_u
            if m:
                p[nu:, nu:] = 2.0 * cfg.slack_quadratic * np.eye(m)
            q = np.r_[q_u, np.full(m, cfg.slack_linear)]
            rows = []
            lower = []
            upper = []
            accel_bound = np.repeat(np.repeat(amax, h), 2)
            for i in np.flatnonzero(done):
                accel_bound[i*h*2:(i+1)*h*2] = 0.0
            rows.append(sparse.hstack((sparse.eye(nu), sparse.csc_matrix((nu, m)))))
            lower.append(-accel_bound); upper.append(accel_bound)
            if m:
                rows.append(sparse.hstack((sparse.csc_matrix((m, nu)), sparse.eye(m))))
                lower.append(np.zeros(m)); upper.append(np.full(m, np.inf))
                safety = np.zeros((m, nv), dtype=float)
                safety_lower = np.zeros(m)
                for idx, (key, coeff, base_proj, physical, sigma) in enumerate(specs):
                    eps = allocation.get(key, cfg.total_risk / max(1, m))
                    margin = risk_margin(self.method, eps, sigma, cfg)
                    safety[idx, :nu] = coeff
                    safety[idx, nu + idx] = 1.0
                    safety_lower[idx] = physical + margin - base_proj
                rows.append(sparse.csc_matrix(safety))
                lower.append(safety_lower); upper.append(np.full(m, np.inf))
            # Componentwise velocity bounds are conservative but preserve a sparse convex QP.
            speed_bound = np.repeat(np.repeat(vmax / np.sqrt(2.0), h), 2)
            vbase = np.repeat(vel[:, None, :], h, axis=1).reshape(-1)
            rows.append(sparse.hstack((sparse.csc_matrix(gv), sparse.csc_matrix((len(vbase), m)))))
            lower.append(-speed_bound - vbase); upper.append(speed_bound - vbase)
            amat = sparse.vstack(rows, format="csc")
            lvec = np.concatenate(lower); uvec = np.concatenate(upper)
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(p)), q=q, A=amat, l=lvec, u=uvec,
                         verbose=False, polish=True, max_iter=2500, eps_abs=2e-4, eps_rel=2e-4,
                         adaptive_rho=True, check_termination=25)
            x0 = np.r_[u, np.zeros(m)]
            solver.warm_start(x=x0)
            result = solver.solve()
            final_status = str(result.info.status)
            if result.x is None or "solved" not in final_status.lower():
                break
            u = np.asarray(result.x[:nu])
            final_slack = np.asarray(result.x[nu:]) if m else np.empty(0)
            if m:
                safety_offset = nu + m
                raw_dual = np.asarray(result.y[safety_offset:safety_offset + m])
                final_duals = {specs[i][0]: abs(float(raw_dual[i])) for i in range(m)}
                self.allocator.update_duals(final_duals)

        sequence = u.reshape(n, h, 2)
        sequence[done] = 0.0
        self.warm = sequence.copy()
        predicted = (base + g @ sequence.reshape(-1)).reshape(n, h, 2)
        values = np.asarray(list(allocation.values()), dtype=float)
        if len(values) and values.sum() > 0:
            probs = values / values.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / max(np.log(len(probs)), 1e-12))
        else:
            entropy = 0.0
        return MPCSolution(sequence, predicted, 1000.0 * (time.perf_counter() - started), final_status,
                           float(np.max(final_slack)) if len(final_slack) else 0.0,
                           float(np.sum(final_slack)) if len(final_slack) else 0.0,
                           int(np.sum(final_slack > 1e-6)) if len(final_slack) else 0,
                           final_duals, allocation, entropy, active_count,
                           dict(self.allocator.last_pressure))
