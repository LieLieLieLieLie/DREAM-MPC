from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse

from .config import ControllerConfig
from .envs import CircleObstacle
from .risk import prediction_sigma, risk_margin
from .solver import MPCSolution, SequentialConvexMPC


class DreamController:
    def __init__(self, cfg: ControllerConfig, method: str, variant: str = "完整方法"):
        self.cfg = cfg
        self.method = method
        self.variant = variant
        self.mpc = SequentialConvexMPC(cfg, method)
        self.previous_vel: np.ndarray | None = None
        self.previous_u: np.ndarray | None = None
        self.uncertainty_scale = 1.0
        self.interventions = 0
        self.last_solution: MPCSolution | None = None

    def _update_uncertainty(self, vel: np.ndarray) -> None:
        if self.previous_vel is None or self.previous_u is None:
            return
        realized = (vel - self.previous_vel) / self.cfg.dt
        residual = realized - self.previous_u
        score = float(np.median(np.linalg.norm(residual, axis=1)) / 0.12)
        self.uncertainty_scale = float(np.clip(0.88 * self.uncertainty_scale + 0.12 * score, 0.82, 1.45))

    def _cbf_project(self, u_nom: np.ndarray, state: dict[str, object]) -> tuple[np.ndarray, bool]:
        pos = np.asarray(state["pos"], dtype=float)
        vel = np.asarray(state["vel"], dtype=float)
        radii = np.asarray(state["radii"], dtype=float)
        amax = np.asarray(state["amax"], dtype=float)
        effectiveness = np.asarray(state.get("control_effectiveness", np.ones(len(pos))), dtype=float)
        accel_drift = np.asarray(state.get("accel_drift", np.zeros_like(pos)), dtype=float)
        done = np.asarray(state["done"], dtype=bool)
        obstacles: tuple[CircleObstacle, ...] = state["obstacles"]  # type: ignore[assignment]
        n = len(pos); dt = self.cfg.dt
        flat = u_nom.reshape(-1).copy()
        inequalities: list[tuple[np.ndarray, float]] = []
        if self.method == "DREAM-MPC":
            sigma = prediction_sigma(0, self.cfg, True, self.uncertainty_scale)
            buffer = 0.07 + min(0.10, risk_margin("DREAM-MPC", 0.040, sigma, self.cfg))
        else:
            buffer = 0.075
        for i in range(n):
            for j in range(i + 1, n):
                if done[i] and done[j]:
                    continue
                rel = pos[i] - pos[j]
                dist = float(np.linalg.norm(rel))
                clearance = dist - radii[i] - radii[j]
                if clearance > 1.35 + buffer:
                    continue
                normal = rel / max(dist, 1e-9)
                row = np.zeros(2 * n)
                if not done[i]:
                    row[2*i:2*i+2] = 2.0 * effectiveness[i] * rel
                if not done[j]:
                    row[2*j:2*j+2] = -2.0 * effectiveness[j] * rel
                target = radii[i] + radii[j] + buffer
                relative_velocity = vel[i] - vel[j]
                relative_drift = accel_drift[i] - accel_drift[j]
                h_value = dist * dist - target * target
                h_dot = 2.0 * float(rel @ relative_velocity)
                rhs = (-2.0 * float(relative_velocity @ relative_velocity)
                       -2.0 * float(rel @ relative_drift) - 2.8 * h_dot - 2.0 * h_value)
                inequalities.append((row, rhs))
            if done[i]:
                continue
            for obstacle in obstacles:
                rel = pos[i] - np.asarray(obstacle.center)
                dist = float(np.linalg.norm(rel))
                clearance = dist - radii[i] - obstacle.radius
                if clearance > 1.15 + buffer:
                    continue
                normal = rel / max(dist, 1e-9)
                row = np.zeros(2 * n); row[2*i:2*i+2] = 2.0 * effectiveness[i] * rel
                target = radii[i] + obstacle.radius + buffer
                h_value = dist * dist - target * target
                h_dot = 2.0 * float(rel @ vel[i])
                rhs = (-2.0 * float(vel[i] @ vel[i]) - 2.0 * float(rel @ accel_drift[i])
                       - 2.8 * h_dot - 2.0 * h_value)
                inequalities.append((row, rhs))
        if not inequalities:
            return u_nom, False
        original = flat.copy()
        constraint_matrix = np.vstack([item[0] for item in inequalities])
        rhs = np.asarray([item[1] for item in inequalities])
        m = len(inequalities); nu = 2 * n; nv = nu + m
        # Slack preserves feasibility while its audited penalty makes violations explicit.
        p = sparse.block_diag((sparse.eye(nu), 2.0e4 * sparse.eye(m)), format="csc")
        q = np.r_[-original, np.full(m, 1200.0)]
        safety = sparse.hstack((sparse.csc_matrix(constraint_matrix), sparse.eye(m)), format="csc")
        slack_positive = sparse.hstack((sparse.csc_matrix((m, nu)), sparse.eye(m)), format="csc")
        directions = np.column_stack((np.cos(np.arange(16) * np.pi / 8.0),
                                      np.sin(np.arange(16) * np.pi / 8.0)))
        polygon = np.zeros((16 * n, nu))
        polygon_bound = np.empty(16 * n)
        for i in range(n):
            polygon[16*i:16*(i+1), 2*i:2*i+2] = directions
            polygon_bound[16*i:16*(i+1)] = amax[i] * np.cos(np.pi / 16.0)
        accel_polygon = sparse.hstack((sparse.csc_matrix(polygon), sparse.csc_matrix((16*n, m))), format="csc")
        matrix = sparse.vstack((safety, slack_positive, accel_polygon), format="csc")
        lower = np.r_[rhs, np.zeros(m), np.full(16*n, -np.inf)]
        upper = np.r_[np.full(m, np.inf), np.full(m, np.inf), polygon_bound]
        problem = osqp.OSQP()
        problem.setup(P=p, q=q, A=matrix, l=lower, u=upper, verbose=False,
                      eps_abs=2e-5, eps_rel=2e-5, max_iter=4000, polish=True)
        result = problem.solve()
        if result.x is None or "solved" not in result.info.status.lower():
            return u_nom, False
        filtered = np.asarray(result.x[:nu]).reshape(n, 2)
        return filtered, bool(np.linalg.norm(filtered.reshape(-1) - original) > 1e-5)

    def act(self, state: dict[str, object]) -> np.ndarray:
        vel = np.asarray(state["vel"], dtype=float)
        self._update_uncertainty(vel)
        scale = self.uncertainty_scale if self.method in {"DREAM-MPC", "Uniform-DRMPC"} else 1.0
        original_solver_method = self.mpc.method
        self.mpc.risk_use_dual = self.variant != "无对偶响应"
        self.mpc.risk_use_smoothing = self.variant != "无KL平滑"
        if self.variant == "均匀风险分配":
            self.mpc.method = "Uniform-DRMPC"
        elif self.variant == "高斯机会约束":
            self.mpc.method = "CC-MPC"
        if self.variant in {"无对偶敏感度", "无对偶响应"}:
            self.mpc.risk_use_dual = False
        solution = self.mpc.solve(state, scale)
        self.mpc.method = original_solver_method
        self.last_solution = solution
        command = solution.sequence[:, 0].copy()
        if self.method in {"DREAM-MPC", "ECBF-NMPC"} and self.variant != "无安全投影":
            command, changed = self._cbf_project(command, state)
            self.interventions += int(changed)
        self.previous_vel = vel.copy()
        self.previous_u = command.copy()
        return command
