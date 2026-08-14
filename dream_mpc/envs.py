from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import AGENT_TYPES, AgentType, ExperimentConfig


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class Layout:
    name: str
    starts: np.ndarray
    goals: np.ndarray
    obstacles: tuple[CircleObstacle, ...]
    world_limit: float


def _ring_points(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    a = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(a), radius * np.sin(a)))


def make_layout(name: str, n_agents: int, seed: int = 0) -> Layout:
    rng = np.random.default_rng(seed + 1709)
    if name == "交叉口":
        starts = _ring_points(n_agents, 5.4, 0.08)
        goals = -starts
        obstacles: tuple[CircleObstacle, ...] = ()
        limit = 6.2
    elif name == "环岛":
        starts = _ring_points(n_agents, 5.8, 0.18)
        goals = -starts
        obstacles = (CircleObstacle((0.0, 0.0), 1.38),)
        limit = 6.6
    elif name == "仓储通道":
        base = np.array([
            [-5.6, -3.7], [-5.6, 0.0], [-5.6, 3.7],
            [5.6, -3.7], [5.6, 0.0], [5.6, 3.7],
            [0.0, -5.4], [0.0, 5.4],
        ], dtype=float)
        starts = base[:n_agents].copy()
        goals = -starts
        obstacles = (
            CircleObstacle((-2.15, -1.65), 0.92),
            CircleObstacle((-2.15, 1.65), 0.92),
            CircleObstacle((2.15, -1.65), 0.92),
            CircleObstacle((2.15, 1.65), 0.92),
        )
        limit = 6.4
    else:
        raise ValueError(f"未知布局: {name}")
    starts += rng.normal(0.0, 0.055, starts.shape)
    goals += rng.normal(0.0, 0.035, goals.shape)
    return Layout(name, starts, goals, obstacles, limit)


CONDITION_PARAMS = {
    "标称": dict(noise=0.030, tail=0.0, gain=1.0, burst=False),
    "重尾扰动": dict(noise=0.050, tail=0.70, gain=1.0, burst=False),
    "混合偏移": dict(noise=0.045, tail=0.25, gain=0.82, burst=False),
    "突发失配": dict(noise=0.040, tail=0.35, gain=0.75, burst=True),
}


class MultiRobotArena:
    def __init__(self, cfg: ExperimentConfig, layout_name: str, condition: str, n_agents: int, seed: int):
        self.cfg = cfg
        self.layout = make_layout(layout_name, n_agents, seed)
        self.condition = condition
        self.params = CONDITION_PARAMS[condition]
        self.rng = np.random.default_rng(seed + 9103)
        self.n = n_agents
        self.types: list[AgentType] = [AGENT_TYPES[i % len(AGENT_TYPES)] for i in range(n_agents)]
        self.pos = self.layout.starts.copy()
        self.vel = np.zeros((n_agents, 2), dtype=float)
        self.accel = np.zeros((n_agents, 2), dtype=float)
        self.goal = self.layout.goals.copy()
        self.done = np.zeros(n_agents, dtype=bool)
        self.t = 0
        self.path_length = np.zeros(n_agents)
        self.energy = np.zeros(n_agents)
        self.history: list[dict[str, np.ndarray | float]] = []

    @property
    def radii(self) -> np.ndarray:
        return np.asarray([t.radius for t in self.types])

    @property
    def vmax(self) -> np.ndarray:
        return np.asarray([t.vmax for t in self.types])

    @property
    def amax(self) -> np.ndarray:
        return np.asarray([t.amax for t in self.types])

    def state(self) -> dict[str, object]:
        lag = np.asarray([t.lag for t in self.types])
        nominal_gain = np.asarray([t.actuator_gain for t in self.types])
        return {
            "pos": self.pos.copy(), "vel": self.vel.copy(), "goal": self.goal.copy(),
            "done": self.done.copy(), "radii": self.radii.copy(), "vmax": self.vmax.copy(),
            "amax": self.amax.copy(), "priority": np.linspace(1.0, 0.15, self.n),
            # Exact coefficient of the current command in the implemented
            # acceleration recursion: lambda_i * beta_i.
            "control_effectiveness": lag * nominal_gain * float(self.params["gain"]),
            "accel_drift": (1.0 - lag)[:, None] * self.accel.copy(),
            "obstacles": self.layout.obstacles, "time": self.t * self.cfg.controller.dt,
        }

    def _disturbance(self) -> np.ndarray:
        p = self.params
        base = self.rng.normal(0.0, p["noise"], (self.n, 2))
        if p["tail"] > 0:
            selector = self.rng.random((self.n, 1)) < 0.12 * p["tail"]
            base += selector * self.rng.standard_t(3.2, (self.n, 2)) * p["noise"] * 1.8
        if p["burst"] and (23 <= self.t % 47 <= 30):
            direction = np.column_stack((np.sin(np.arange(self.n)), np.cos(np.arange(self.n))))
            base += 0.12 * direction
        return base

    def step(self, command: np.ndarray) -> None:
        dt = self.cfg.controller.dt
        command = np.asarray(command, dtype=float).copy()
        command[self.done] = 0.0
        norms = np.linalg.norm(command, axis=1) + 1e-12
        command *= np.minimum(1.0, self.amax / norms)[:, None]
        disturbance = self._disturbance()
        realized = np.zeros_like(command)
        for i, typ in enumerate(self.types):
            gain = typ.actuator_gain * self.params["gain"]
            realized[i] = (1.0 - typ.lag) * self.accel[i] + typ.lag * gain * command[i] + disturbance[i]
        old = self.pos.copy()
        self.vel += realized * dt
        speed = np.linalg.norm(self.vel, axis=1) + 1e-12
        self.vel *= np.minimum(1.0, self.vmax / speed)[:, None]
        self.pos += self.vel * dt
        self.accel = realized
        self.path_length += np.linalg.norm(self.pos - old, axis=1)
        self.energy += np.sum(command**2, axis=1) * dt
        reached = np.linalg.norm(self.pos - self.goal, axis=1) <= self.cfg.goal_tolerance
        self.done |= reached
        self.vel[self.done] *= 0.25
        self.t += 1
        self.history.append({"pos": self.pos.copy(), "vel": self.vel.copy(), "u": command.copy(), "a": realized.copy()})


def minimum_clearance(pos: np.ndarray, radii: np.ndarray, obstacles: tuple[CircleObstacle, ...]) -> tuple[float, float]:
    pair = np.inf
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            pair = min(pair, float(np.linalg.norm(pos[i] - pos[j]) - radii[i] - radii[j]))
    obs = np.inf
    for i in range(len(pos)):
        for obstacle in obstacles:
            obs = min(obs, float(np.linalg.norm(pos[i] - np.asarray(obstacle.center)) - radii[i] - obstacle.radius))
    return float(pair), float(obs if np.isfinite(obs) else 99.0)
